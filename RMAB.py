import numpy as np
from itertools import product

def get_stable_dist(p):
    return p[:,0] / (p[:,0] + 1.0 - p[:,1])

def get_next_belief(p01, p11, w, k):
    if k == np.inf:
        return p01 / (p01 + 1.0 - p11)
    else:
        tmp = np.power(p11 - p01, k) * (p01 - (1.0 + p01 - p11)*w)
        return (p01 - tmp) / (1 + p01 - p11)

def crossing_time(p01, p11, w1, w2):
    p10 = 1.0 - p11
    w0 = p01 / (p01 + p10)
    if p11 >= p01:
        # postively correlated arm
        if w1 > w2:
            return 0
        elif w2 < w0:
            tmp = (p01 - w2*(p10+p01)) / (p01 - w1*(p10+p01))
            return np.floor(np.log(tmp) / np.log(p11-p01)) + 1
        else:
            return np.inf
    else:
        # negatively correlated arm
        if w1 > w2:
            return 0
        else:
            Tw1 = get_next_belief(p01, p11, w1, 1)
            if Tw1 > w2:
                return 1
            else:
                return np.inf
            
def update_posterior(supp, dist, n, r0, r1):
    if r0 < 0:
        n = np.inf
        r0 = 0

    for i, p in enumerate(product(supp, supp)):
        p01, p11 = p
        w = get_next_belief(p01, p11, r0, n)
        w = max(w, 1e-3)
        if r1 == 1:
            dist[i] *= w
        else:
            dist[i] *= 1-w

    dist /= dist.sum()

def get_whittle_indices(p_hat, w_array):
    ret = []
    for k in xrange(len(w_array)):
        p01 = p_hat[k, 0]
        p11 = p_hat[k, 1]
        w0 = p01 / (p01 + 1.0 - p11)
        w = w_array[k]
        Tw = get_next_belief(p01, p11, w, 1)
        if p11 >= p01:
            # postively correlated arm
            if w <= p01 or w >= p11:
                ret.append(w)
            elif w < w0:
                tmp = crossing_time(p01, p11, p01, w)
                tmp1 = (w - Tw) * (1 + tmp) + get_next_belief(p01, p11, p01, tmp)
                tmp2 = 1 - p11 + (w - Tw)*tmp + get_next_belief(p01, p11, p01, tmp)
                tmp2 = max(tmp2, 1e-6)
                ret.append(tmp1/tmp2)
            else:
                ret.append(w/(1.0 - p11 + w))
        else:
            # negatively correlated arm
            tmp = get_next_belief(p01, p11, p11, 1)
            if w <= p11 or w >= p01:
                ret.append(w)
            elif w < w0:
                ret.append((w + p01 - Tw) / (1.0 + p01 - tmp + Tw - w))
            elif w < tmp:
                ret.append(p01 / (1.0 + p01 - tmp))
            else:
                ret.append(p01 / (1.0 + p01 - w))
    return np.array(ret)

class RestlessMAB:
    def __init__(self, K):
        self.K = K
        self.p = np.zeros((K, 2)) # transition probs, p_01 and p_11
        self.w = np.zeros(K) # initial condition, \omega
        self.states = np.zeros(K, dtype=int)
    
    def initialize(self, p=None, w=None):
        if p is not None:
            self.p = p
        if w is not None:
            self.w = w
        self.states = np.array([np.random.binomial(1, _w) for _w in self.w], dtype=int)
    
    def evolve_states(self):
        for k in xrange(self.K):
            s = self.states[k]
            p = self.p[k, s]
            self.states[k] = int(np.random.binomial(1, p))

    def get_rewards(self, A):
        ret = np.ones(self.K) * -1.0
        ret[A == 1] = self.states[A == 1]
        return ret