import numpy as np
from ThompsonSampling import ThompsonSampling
from RMAB import *

"""
To Do
- define N(k, s, n)
- inject artificial exploration to the best fixed arm policy
- Finish TSDE
- inject artificial exploration to the myopic policy
- inject artificial exploration to the whittle index policy
"""


class TSDE(ThompsonSampling):
    def reset_suff_stats(self):
        self.r = np.ones(self.K, dtype=int) # Assume the initial states are all ones
        self.n = np.ones(self.K, dtype=int)

    def update_suff_stats(self, rewards):
        self.r[rewards >= 0] = rewards[rewards >= 0]
        self.n[rewards < 0] += 1
        self.n[rewards >= 0] = 1

    def runTS(self, L, baseline='whittle', verbose=False, min_len=0, 
        true_indices=None):
        self.rmab.initialize()
        self.reset_suff_stats()
        results = []
        if true_indices is not None:
            self.weight_changes = [[] for _ in xrange(self.K)]
        self.num_samples = np.ones((self.K, 2, L), dtype=int) * min_len
        thresholds = self.num_samples * 2
        self.num_ep = 1
        ep_len = 0
        prev_ep_len = min_len
        p_hat = self.draw_p()
        w = np.ones(self.K)
        for i in xrange(L):
            if true_indices is not None:
                for k in range(self.K):
                    dist = self.posterior[k]
                    self.weight_changes[k].append(dist[true_indices[k]])
            ep_len += 1
            is_new_ep = ep_len > prev_ep_len
            A = self.get_action(p_hat, w, baseline)
            rewards = self.rmab.get_rewards(A)
            self.update_posterior(rewards)
            results.append(sum(rewards[A==1]))
            for k in xrange(self.K):
                p01 = p_hat[k, 0]
                p11 = p_hat[k, 1]
                if A[k] == 1:
                    w[k] = get_next_belief(p01, p11, rewards[k], 1)
                    self.num_samples[k, self.r[k], self.n[k]] += 1
                    if self.num_samples[k, self.r[k], self.n[k]] >= \
                    thresholds[k, self.r[k], self.n[k]]:
                        is_new_ep = True
                else:
                    w[k] = get_next_belief(p01, p11, w[k], 1)
            self.update_suff_stats(rewards)
            self.rmab.evolve_states()
            if is_new_ep:
                if verbose:
                    print self.num_ep, ep_len
                prev_ep_len = ep_len
                ep_len = 0
                self.num_ep += 1
                p_hat = self.draw_p()
                thresholds = self.num_samples * 2
        return np.array(results)
