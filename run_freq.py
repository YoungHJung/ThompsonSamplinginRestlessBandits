import numpy as np
import pickle
from RMAB import *
from ThompsonSamplingDE import TSDE

def main():
	####################################
	##### Frequentist regret (1/2) #####
	####################################

	# Fix true p

	K = 4
	N = 2
	M = 100 # size of MCMC iterations
	L = 10000
	min_len = 0
	baselines = ['whittle', 'fixed', 'myopic', 'bonus']

	supp = np.linspace(0.1, 0.9, 9)
	len_supp = len(supp)
	posterior = np.ones((K, len_supp**2)) / len_supp**2


	# Draw true p
	p = np.array([[.3, .7],
	             [.4, .6],
	             [.5, .5],
	             [.6, .4]])
	true_indices = [24, 32, 40, 48]
	w = np.ones(K)

	print "True p"
	print p
	print 

	rmab = RestlessMAB(K)
	rmab.initialize(p, w)
	policies = {baseline: TSDE(rmab, N) for baseline in baselines}

	# Estimate the value of WIP
	print "Estimating the value of baselines..."
	value_baselines = {baseline: 0.0 for baseline in baselines}
	for _ in xrange(M):
	    for baseline in baselines:
	        value_baselines[baseline] += policies[baseline].run_baseline(p, w, L, baseline) / float(L*M)
	for baseline in baselines:
	    print "Value({0}) = ".format(baseline), value_baselines[baseline]
	print 

	####################################
	##### Frequentist regret (2/2) #####
	####################################

	results = {baseline: np.zeros(L) for baseline in baselines}
	weight_changes = np.zeros(K*L).reshape((K, L))
	for j in xrange(M):
	    if j % 10 == 0:
	        print j
	    for baseline in baselines:
	        posterior = np.ones((K, len_supp**2)) / len_supp**2
	        policy = TSDE(rmab, N, supp, posterior)
	        results[baseline] += policy.runTS(L, baseline=baseline, min_len=min_len, true_indices=true_indices) / float(M)
	        if baseline == 'whittle':
	            weight_changes += np.array(policy.weight_changes) / float(M)

	#####################################
	##### Frequentist regret pickle #####
	#####################################

	freq_data = {"value_baselines": value_baselines,
	       "results": results,
	       "weight_changes": weight_changes}

	pickle.dump(freq_data, open('TSDE_freq_K_4_N_2_with_bonus.pk', 'wb'))

if __name__ == '__main__':
	main()