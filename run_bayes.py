import numpy as np
import pickle
from RMAB import *
from ThompsonSamplingDE import TSDE

def main():
	###########################
	##### Bayesian regret #####
	###########################

	K = 8 
	N = 3
	M = 100 # size of MCMC iterations
	M2 = 30
	LM = 300
	L = 2000
	min_len = 0
	baselines = ['whittle', 'fixed', 'myopic', 'bonus']

	supp = np.linspace(0.1, 0.9, 9)
	len_supp = len(supp)
	posterior = np.ones((K, len_supp**2)) / len_supp**2

	regrets = {baseline: np.zeros(L) for baseline in baselines}
	avg_value_baselines = {baseline: 0.0 for baseline in baselines}

	for i in xrange(M):
	    print i
	    # Draw true p
	    p = np.random.choice(supp, 2*K).reshape(K, 2)
	    w = np.zeros(K)

	    rmab = RestlessMAB(K)
	    rmab.initialize(p, w)
	    policies = {baseline: TSDE(rmab, N) for baseline in baselines}

	    # Estimate the value of WIP
	    value_baselines = {baseline: 0.0 for baseline in baselines}
	    for _ in xrange(M2):
	        for baseline in baselines:
	            value_baselines[baseline] += policies[baseline].run_baseline(p, w, LM, baseline) / float(LM*M2)
	    for baseline in baselines:
	    	avg_value_baselines[baseline] += value_baselines[baseline] / float(M)
	    results = {baseline: np.zeros(L) for baseline in baselines}
	    for j in xrange(M2):
	        for baseline in baselines:
	            posterior = np.ones((K, len_supp**2)) / len_supp**2
	            policy = TSDE(rmab, N, supp, posterior)
	            results[baseline] += policy.runTS(L, baseline=baseline, min_len=min_len) / float(M2)

	    for baseline in baselines:
	        regrets[baseline] += ((1 + np.arange(L)) * value_baselines[baseline] - np.cumsum(results[baseline])) / float(M)

   	##################################
	##### Bayesian regret pickle #####
	##################################

	bayes_data = {'regrets': regrets, 'avg_value_baselines': avg_value_baselines}

	pickle.dump(bayes_data, open('TSDE_bayes_K_8_N_3_M_100_M2_30_L_2000_LM_300_with_bonus.pk', 'wb'))
	
if __name__ == '__main__':
	main()