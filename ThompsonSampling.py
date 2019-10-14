import numpy as np
from RMAB import *

class ThompsonSampling:
    def __init__(self, rmab, N, supp=None, posterior=None):
        self.rmab = rmab
        self.K = rmab.K
        self.N = N
        self.supp = supp
        self.posterior = None
        if posterior is not None:
            self.posterior = posterior.copy()
        self.r = np.ones(self.K) * -1.0
        self.n = np.ones(self.K, dtype=int)

    def reset_suff_stats(self):
        self.r = np.ones(self.K) * -1.0
        self.n = np.ones(self.K, dtype=int)

    def update_posterior(self, rewards):
        if self.posterior is None or self.supp is None:
            return
        for k, r in enumerate(rewards):
            if r < 0:
                continue
            update_posterior(self.supp, self.posterior[k], self.n[k], self.r[k], r)

    def update_suff_stats(self, rewards):
        self.r = rewards
        self.n[rewards < 0] += 1
        self.n[rewards >= 0] = 1

    def draw_p(self):
        ret = np.zeros((self.K, 2))
        len_supp = len(self.supp)
        for k in xrange(self.K):
            dist = self.posterior[k]
            i = np.random.choice(range(len_supp**2), p=dist)
            i0 = i / len_supp
            i1 = i % len_supp
            ret[k, 0] = self.supp[i0]
            ret[k, 1] = self.supp[i1]
        return ret

    def get_posterior_summary(self):
        len_supp = len(self.supp)
        for k in range(self.K):
            dist = self.posterior[k]
            i = np.argmax(dist)
            i0 = i/len_supp
            i1 = i%len_supp
            print [round(self.supp[i0], 2), round(self.supp[i1], 2)], 
            print round(dist[i], 2)

    def run_baseline(self, p_hat, w_hat, L, baseline='whittle', beta=1):
        if baseline == 'whittle':
            return self.run_whittle(p_hat, w_hat, L, beta)
        elif baseline == 'fixed':
            return self.run_best_fixed_arm(p_hat, w_hat, L, beta)
        elif baseline == 'myopic':
            return self.run_myopic(p_hat, w_hat, L, beta)
        elif baseline == 'bonus':
            return self.run_bonus(p_hat, w_hat, L, beta)
        else:
            print 'Unrecognized baseline.'
            return 

    def get_action(self, p_hat, w, baseline='whittle'):
        if baseline == 'whittle':
            return self.get_action_whittle(p_hat, w)
        elif baseline == 'fixed':
            return self.get_action_best_fixed_arm(p_hat, w)
        elif baseline == 'myopic':
            return self.get_action_myopic(p_hat, w)
        elif baseline == 'bonus':
            return self.get_action_bonus(p_hat, w)
        else:
            print 'Unrecognized baseline.'
            return         

    def get_action_whittle(self, p_hat, w):
        indices = get_whittle_indices(p_hat, w)
        sorted_indices = np.argsort(indices)[::-1]
        A = np.zeros(self.K)
        for k in sorted_indices[:self.N]:
            A[k] = 1
        return A

    def get_action_best_fixed_arm(self, p_hat, w):
        w_hat = get_stable_dist(p_hat)
        sorted_indices = np.argsort(w_hat)[::-1]
        A = np.zeros(self.K)

        for k in sorted_indices[:self.N]:
            A[k] = 1
        return A
            
    def get_action_myopic(self, p_hat, w):
        sorted_indices = np.argsort(w)[::-1]
        A = np.zeros(self.K)
        for k in sorted_indices[:self.N]:
            A[k] = 1
        return A

    def get_action_bonus(self, p_hat, w):
        std = np.sqrt(w*(1-w))
        sorted_indices = np.argsort(w + 4*std)[::-1]
        A = np.zeros(self.K)
        for k in sorted_indices[:self.N]:
            A[k] = 1
        return A

    def run_whittle(self, p_hat, w_hat, L, beta=1):
        self.rmab.initialize()
        self.reset_suff_stats()
        w = np.array(w_hat).copy()
        cum_reward = 0
        for i in xrange(L):
            A = self.get_action_whittle(p_hat, w)
            rewards = self.rmab.get_rewards(A)
            self.update_posterior(rewards)
            self.update_suff_stats(rewards)
            for k in xrange(self.K):
                p01 = p_hat[k, 0]
                p11 = p_hat[k, 1]
                if A[k] == 1:
                    w[k] = get_next_belief(p01, p11, rewards[k], 1)
                    cum_reward += rewards[k] * np.power(beta, i)
                else:
                    w[k] = get_next_belief(p01, p11, w[k], 1)
            self.rmab.evolve_states()
        return cum_reward    

    def run_best_fixed_arm(self, p_hat, w_hat, L, beta=1):
        self.rmab.initialize()
        self.reset_suff_stats()
        w = np.array(w_hat).copy()
        cum_reward = 0
        A = self.get_action_best_fixed_arm(p_hat, w)
            
        for i in xrange(L):
            rewards = self.rmab.get_rewards(A)
            self.update_posterior(rewards)
            self.update_suff_stats(rewards)
            cum_reward += rewards[A == 1].sum() * np.power(beta, i)
            self.rmab.evolve_states()
        return cum_reward    

    def run_myopic(self, p_hat, w_hat, L, beta=1):
        self.rmab.initialize()
        self.reset_suff_stats()
        w = np.array(w_hat).copy()
        cum_reward = 0
        for i in xrange(L):
            A = self.get_action_myopic(p_hat, w)
            rewards = self.rmab.get_rewards(A)
            self.update_posterior(rewards)
            self.update_suff_stats(rewards)
            for k in xrange(self.K):
                p01 = p_hat[k, 0]
                p11 = p_hat[k, 1]
                if A[k] == 1:
                    w[k] = get_next_belief(p01, p11, rewards[k], 1)
                    cum_reward += rewards[k] * np.power(beta, i)
                else:
                    w[k] = get_next_belief(p01, p11, w[k], 1)
            self.rmab.evolve_states()
        return cum_reward    
    
    def run_bonus(self, p_hat, w_hat, L, beta=1):
        self.rmab.initialize()
        self.reset_suff_stats()
        w = np.array(w_hat).copy()
        cum_reward = 0
        for i in xrange(L):
            A = self.get_action_bonus(p_hat, w)
            rewards = self.rmab.get_rewards(A)
            self.update_posterior(rewards)
            self.update_suff_stats(rewards)
            for k in xrange(self.K):
                p01 = p_hat[k, 0]
                p11 = p_hat[k, 1]
                if A[k] == 1:
                    w[k] = get_next_belief(p01, p11, rewards[k], 1)
                    cum_reward += rewards[k] * np.power(beta, i)
                else:
                    w[k] = get_next_belief(p01, p11, w[k], 1)
            self.rmab.evolve_states()
        return cum_reward    
    
    def runTS(self, L, m, baseline='whittle', true_indices=None):
        results = []
        if true_indices is not None:
            self.weight_changes = [[] for _ in xrange(self.K)]
            
        for i in xrange(m):
            if true_indices is not None:
                for k in range(self.K):
                    dist = self.posterior[k]
                    self.weight_changes[k].append(dist[true_indices[k]])
            p_hat = self.draw_p()
            w_hat = get_stable_dist(p_hat)
            results.append(self.run_baseline(p_hat, w_hat, L, baseline))
        return np.array(results)
