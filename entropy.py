# PGM's source/ and utilities/ are put on sys.path by src/config.py, which is
# imported before this module.

# import packages
import os

import numpy as np
import rbm, utilities, RBM_utils
from itertools import product


# basic functions
curr_float = np.float32
curr_int = np.int16

def convert_number(seqs):
	'''
	convert letters to numbers
	'''
	aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V',  'W', 'Y','-']
	aadict = {aa[k]: k for k in range(len(aa))} 
	
	msa_num = np.array(list(map(lambda x: [aadict[y] for y in x], seqs[0:])), dtype=curr_int, order="c")
	return msa_num

def convert_letter(seqs_n):
	'''
	convert numbers to letters
	'''
	aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V',  'W', 'Y','-']
	aadictinv = {k: aa[k] for k in range(len(aa))} 
	seqs=[]
	if type(seqs_n[0]) == curr_int:
		seqs.append(''.join([aadictinv[e] for e in seqs_n]))
	else:
		for t in range(len(seqs_n)):
			seqs.append(''.join([aadictinv[e] for e in seqs_n[t]]))
	return seqs

def average_n(config,q):
	'''
	probability of PWM, size out is L x q
	'''
	N = config.shape[0]  # number of data points
	L = config.shape[1]  # sequence length
	out = np.zeros((L,q), dtype = curr_float)
	for i in list(range(N)):
		for j in list(range(L)):
			out[j, config[i,j]] +=1
	out/=N
	return out


# PWM functions
def pwm_train(df_train, ifpseudo=True, a=None):
	'''
	Train a PWN (local field) model based on the provided training data.

	Args:
		df_train (np.array): The training data used to train the PWM.
		ifpseudo (bool): Whether to add a pseudo-count to the PWM.
		a (float): The value of the pseudo-count to be added. If `None`, a value of 1/N is used, where N is the number of samples.

	Returns:
		pwm_ind (np.array): A (9,20) matrix representing the trained PWM.
	'''
	if df_train.dtype != curr_int:
		df_train = convert_number(list(df_train))
	pwm_ind = average_n(df_train, 20)
	if ifpseudo:
		if a == None:
			a = 1/len(df_train)
		pwm_ind = (1-a)*pwm_ind + a/20
	return pwm_ind


def entropy_pwm(df_train, ifpseudo=True, a=None, ifprint=False):
	'''
	Calculate the entropy of PWM (local field) based on the provided training data.

	Args:
		df_train (np.array): The training data used to train the PWM.
		ifpseudo (bool): Whether to add a pseudo-count to the empirical PWM.
		a (float): The value of the pseudo-count to be added. If `None`, a value of 1/N is used, where N is the number of samples.
		ifprint (bool): Whether to print the results of the entropy calculation.

	Returns:
		E_pwm (float): The true entropy of PWM.
		E_anl (float): The analytic entropy of PWM.
		var_anl (float): The analytic variance of PWM.
	'''
	pwm_ind = pwm_train(df_train, ifpseudo=ifpseudo, a=a)

	non_zero_mask = pwm_ind > 0
	log_pwm_ind = np.zeros_like(pwm_ind)
	log_pwm_ind[non_zero_mask] = np.log(pwm_ind[non_zero_mask])
	E_pwm = np.sum(-pwm_ind * log_pwm_ind)

	L, q = np.shape(pwm_ind)
	E_anl = E_pwm + (L*q-L)/(2*len(df_train))

	E_ind = np.sum(pwm_ind * log_pwm_ind, axis=1)
	var_anl = np.sum(pwm_ind * log_pwm_ind**2) - np.sum(E_ind**2)
	var_anl = var_anl/len(df_train)

	if ifprint:
		print(f'Entropy of PWM is {E_pwm:.3f}')
		print(f'Analytic entropy of PWM is {E_anl:.3f} and analytic variance is {var_anl:.3}')

	return E_pwm, E_anl, var_anl


def gen_data_pwm(df_train, N=None, ifpseudo=True, a=None):
	'''
	Generate N new samples of based on the trained PWM model.

	Args:
		df_train (np.array): The training data used to train the PWM model.
		N (int): The number of samples to generate.
		ifpseudo (bool): Indicates whether to add a pseudo-count to the empirical PWM for smoothing.
		a (float): The value of the pseudo-count to be added. If `None`, a value of 1/N is used, where N is the number of training samples.

	Returns:
		gen_data (np.array): An array of generated samples represented as integers.
	'''
	if df_train.dtype != curr_int:
		df_train = convert_number(list(df_train))
	pwm_ind = pwm_train(df_train, ifpseudo=ifpseudo, a=a)
	L, q = np.shape(pwm_ind)
	if N == None:
		N = len(df_train)
	else:
		N = int(N)
	gen_data = np.array([np.random.choice(q,N,p=pwm_ind[j]) for j in range(L)]).T
	return gen_data.astype(curr_int)


def pwm_train_NP(df_train, B, ind=0, frac=1, replace=True,
				 out_fold=None, out_par=None):
	'''
	Generate and save B non-parametric bootstrap batches for PWM.

	Args:
		df_train (np.array): The training data used to train the empirical PWM.
		B (int): The number of bootstrap batches.
		ind (int): Starting index of bootstrap batches.
		frac (float): The fraction of data used for training.
		replace (bool): Whether to sample with replacement.
		out_fold (str): The output folder where the text file will be saved.
		out_par (str): The output name of the text file.
	'''
	if out_fold == None:
		raise Exception('Output folder is not specified')
	if os.path.exists(out_fold) is False:
		os.makedirs(out_fold)
	if df_train.dtype != curr_int:
		df_train = convert_number(list(df_train))
	N = len(df_train)
	for i in range(ind,ind+B):
		# non-parametric sampling
		batch = df_train[np.random.choice(np.arange(N), int(frac*N), replace=replace)]
		if out_par == None:
			out_par1 = f'/sample_np_{i+1}.txt'
		else:
			out_par1 = f'/sample_np_{out_par}_{i+1}.txt'
		np.savetxt(out_fold+out_par1, batch.astype(curr_int), fmt='%d')


def pwm_train_P(df_train, B, ind=0, ifpseudo=True, a=None,
				out_fold=None, out_par=None):
	'''
	Generate and save B parametric bootstrap batches for PWM.
	'''
	if out_fold == None:
		raise Exception('Output folder is not specified')
	if os.path.exists(out_fold) is False:
		os.makedirs(out_fold)

	pwm_ind = pwm_train(df_train, ifpseudo=ifpseudo, a=a)
	N = len(df_train)
	L, q = np.shape(pwm_ind)
	for i in range(ind,ind+B):
		# parametric sampling
		batch = np.array([np.random.choice(q,N,p=pwm_ind[j]) for j in range(L)]).T
		if out_par == None:
			out_par1 = f'/sample_p_{i+1}.txt'
		else:
			out_par1 = f'/sample_p_{out_par}_{i+1}.txt'
		np.savetxt(out_fold+out_par1, batch.astype(curr_int), fmt='%d')


def entropy_pwm_NP(df_train, B, ind=0, frac=1, replace=True,
				   ifpseudo=True, a=None, ifload=False, ifsave=False,
				   out_fold=None, out_par=None, ifprint=False):
	'''
	Calculate the non-parametric entropy of PWM using bootstrap resampling.

	Args:
		df_train (np.array): The training data used to train the empirical PWM.
		B (int): The number of bootstrap batches to run.
		frac (float): The fraction of data used for training.
		replace (bool): Whether to sample with replacement.
		ifpseudo (bool): Whether to add a pseudo-count to the empirical PWM.
		a (float): The value of the pseudo-count to be added. If `None`, a value of 1/N is used, where N is the number of samples.
		ifsave (bool): Whether to save the training samples as a text file.
		out_fold (str): The output folder where the text file will be saved.
		out_par (str): The output name of the text file.
		ifprint (bool): Whether to print the results of the entropy calculation.

	Returns:
		E_NP (float): The non-parametric entropy of PWM.
		var_NP (float): The variance of the non-parametric entropy over the bootstrap samples.
		list_E_NP (np.array): The entropies calculated for each bootstrap sample.
	'''
	if ifload or ifsave:
		if out_fold == None:
			raise Exception('Output folder is not specified')
		if os.path.exists(out_fold) is False:
			os.makedirs(out_fold)
	if df_train.dtype != curr_int:
		df_train = convert_number(list(df_train))
	N = len(df_train)

	list_E_NP = []
	for i in range(ind,ind+B):
		if out_par == None:
			out_par1 = f'/sample_np_{i+1}.txt'
		else:
			out_par1 = f'/sample_np_{out_par}_{i+1}.txt'
		if ifload:  # load data
			if os.path.exists(out_fold+out_par1):
				batch = np.loadtxt(out_fold+out_par1, dtype=curr_int)
			else:
				raise Exception(f'No file to be loaded for {i}-th batch.')
		else:
			batch = df_train[np.random.choice(np.arange(N), int(frac*N), replace=replace)]
			if ifsave:
				np.savetxt(out_fold+out_par1, batch.astype(curr_int), fmt='%d')

		pwm_ind = pwm_train(batch.astype(curr_int), ifpseudo=ifpseudo, a=a)
		non_zero_mask = pwm_ind > 0
		log_pwm_ind = np.zeros_like(pwm_ind)
		log_pwm_ind[non_zero_mask] = np.log(pwm_ind[non_zero_mask])
		E = np.sum(-pwm_ind * log_pwm_ind)
		list_E_NP.append(E)

	E_NP = np.mean(list_E_NP)
	var_NP = np.var(list_E_NP, ddof=1)
	if ifprint:
		print(f'Entropy of PWM with non-parametric bootstrapping is {E_NP:.3f} and variance is {var_NP:.3}')

	return E_NP, var_NP, list_E_NP


def entropy_pwm_P(df_train, B, ind=0,
				  ifpseudo=True, ifpseudo2=True, a=None,
				  ifload=False, ifsave=False,
				  out_fold=None, out_par=None, ifprint=False):
	'''
	Calculate the parametric entropy of PWM using bootstrap resampling.
	'''
	if ifload or ifsave:
		if out_fold == None:
			raise Exception('Output folder is not specified')
		if os.path.exists(out_fold) is False:
			os.makedirs(out_fold)
	if df_train.dtype != curr_int:
		df_train = convert_number(list(df_train))
	N = len(df_train)

	# empirical PWM
	pwm_ind = pwm_train(df_train, ifpseudo=ifpseudo, a=a)
	L, q = np.shape(pwm_ind)

	list_E_P = []
	for i in range(ind,ind+B):
		if out_par == None:
			out_par1 = f'/sample_p_{i+1}.txt'
		else:
			out_par1 = f'/sample_p_{out_par}_{i+1}.txt'
		if ifload:  # load data
			if os.path.exists(out_fold+out_par1):
				batch = np.loadtxt(out_fold+out_par1, dtype=curr_int)
			else:
				raise Exception(f'No file to be loaded for {i}-th batch.')
		else:
			batch = np.array([np.random.choice(q,N,p=pwm_ind[j]) for j in range(L)]).T
			if ifsave:
				np.savetxt(out_fold+out_par1, batch.astype(curr_int), fmt='%d')
		
		# bootstrap PWM
		pwm_ind1 = pwm_train(batch.astype(curr_int), ifpseudo=ifpseudo2, a=a)
		non_zero_mask = pwm_ind1 > 0
		log_pwm_ind1 = np.zeros_like(pwm_ind1)
		log_pwm_ind1[non_zero_mask] = np.log(pwm_ind1[non_zero_mask])
		E = np.sum(-pwm_ind1 * log_pwm_ind1)
		list_E_P.append(E)

	E_P = np.mean(list_E_P)
	var_P = np.var(list_E_P, ddof=1)
	if ifprint:
		print(f'Entropy of PWM with parametric bootstrapping is {E_P:.3f} and variance is {var_P:.3}')

	return E_P, var_P, list_E_P


# RBM functions
def rbm_train(df_train, n_v, n_h, n_cv, l12, lr, n_iter,
			  visible='Potts', hidden='dReLU',
			  ifsave=False, out_fold=None, out_par=None, verbose=0):
	'''
	Train a RBM for a single allele.
	'''
	if ifsave:
		if out_fold == None:
			raise Exception('Output folder is not specified')
		if os.path.exists(out_fold) is False:
			os.makedirs(out_fold)
	if df_train.dtype != curr_int:
		df_train = convert_number(df_train)
	
	seed = utilities.check_random_state(0) # shuffle data
	batch_size = 100 # Size of mini-batches (and number of Markov chains used).
	decay_after = 0.5 # Decay learning rate after 50% of iterations
	N_MC = 15 # Number of Monte Carlo steps between each update
	l2f = 1/len(df_train)

	# train a new RBM model
	all_weights = np.ones(len(df_train))
	RBM = rbm.RBM(visible = visible, hidden = hidden, n_v = n_v, n_h = n_h,
				  n_cv = n_cv, random_state = seed)
	RBM.fit(df_train, weights = all_weights, batch_size = batch_size,
			n_iter = n_iter, l1b = l12, l2_fields = l2f, learning_rate = lr,
			N_MC = N_MC, decay_after = decay_after, verbose = verbose)
	RBM.AIS(M=20,n_betas=20000)
	
	if ifsave:
		if not out_par:
			out_par = f'hu_{n_h}_l12_{l12}_lr_{lr}_niter_{n_iter}'
		name_r = out_fold + '/model_' + out_par + '.data'
		RBM_utils.saveRBM(name_r,RBM)

	return RBM


def entropy_rbm(RBM, N, Nc=100, Lc=1000, iffull=False, ifprint=False):
	'''
	Compute the entropy for the input RBM model (global field).

	Args:
		RBM: The trained RBM model.
		N (int): The number of samples used in the training.
		Nc (int): The number of chains (Nchains) used to generate data.
		Lc (int): The length of chains (Lchains) used to generate data.
		iffull (bool): If True, computes the full entropy. If False, computes the average log probability (log(p)) of the generated data.
		ifprint (bool): Whether to print the results.

	Returns:
		E_rbm (float): The full/estimated entropy of the RBM model.
		var_rbm (float): The variance of log(p) for the full/generated data.
	'''
	if iffull:
		L = int(RBM.n_v)
		full_pep = np.array(list(product(range(20), repeat=L))).astype(curr_int)
		full_E = RBM.free_energy(full_pep)
		Z = np.sum(np.exp(-full_E))
		p_list = np.exp(-full_E)/Z
		
		E_rbm = -np.sum(p_list*np.log(p_list))
		sum_err = np.sum(p_list*np.log(p_list)**2)
		var_rbm = abs(sum_err - E_rbm**2) / N
	else:
		gen_data = RBM.gen_data(Nchains = int(Nc), Lchains = int(Lc), Nthermalize = 500, Nstep = 10, reshape = True)
		E_rbm = -np.mean(RBM.likelihood(gen_data[0].astype(curr_int)))
		sum_err = np.mean(RBM.likelihood(gen_data[0].astype(curr_int))**2)
		var_rbm = abs(sum_err - E_rbm**2) / N
	
	if ifprint:
		print(f'Entropy of RBM is {E_rbm:.3f} and variance of log(p) is {var_rbm:.3}')
		print(f'The Z partition function is {RBM.log_Z_AIS:.3f} with std = {RBM.log_Z_AIS_std:.3}')
	return E_rbm, var_rbm


def gen_data_rbm(RBM, df_train, N=None, Nc=100):
	'''
	Generate N new samples from the trained RBM model.
	'''
	if N == None:
		N = len(df_train)
	Lc = np.ceil(N/Nc).astype(curr_int)
	gen_list = RBM.gen_data(Nchains = int(Nc), Lchains = int(Lc), Nthermalize = 500, Nstep = 10, reshape = True)
	gen_data = gen_list[0]
	for j in reversed(range(1, len(gen_data)-N+1)):
		gen_data = np.delete(gen_data, -Lc*j, axis=0)
	if len(gen_data) != N:
		raise IndexError('Gen_list is not the same size as N')
	return gen_data.astype(curr_int)


def rbm_train_NP(df_train, B, ind, n_v, n_h, n_cv, l12, lr, n_iter,
				 visible='Potts', hidden='dReLU', frac=1, replace=True,
				 out_fold=None, out_par=None, ifprint=False):
	'''
	Train non-parametric bootstrap batches of RBM (global) with training data.
	'''
	if out_fold == None:
		raise Exception('Output folder is not specified')
	if os.path.exists(out_fold) is False:
		os.makedirs(out_fold)
	if df_train.dtype != curr_int:
		df_train = convert_number(df_train)
	N = len(df_train)

	for i in range(ind,ind+B):
		# generate and save data
		batch = df_train[np.random.choice(np.arange(N), int(frac*N), replace=replace)]
		if out_par == None:
			out_par1 = f'/sample_{i+1}.txt'
		else:
			out_par1 = f'/sample_{out_par}_{i+1}'
		np.savetxt(out_fold+out_par1, batch, fmt='%d')

		# train bootstrap RBM
		rbm_train(batch, n_v, n_h, n_cv, l12, lr, n_iter, visible, hidden,
			ifsave=True, out_fold=out_fold, out_par=out_par1)
		if ifprint:
			print(f'Batch {i+1} is done.')


def rbm_train_P(RBM, B, ind, N, Nc, n_v, n_h, n_cv, l12, lr, n_iter,
				visible = 'Potts', hidden = 'dReLU', 
				out_fold = None, out_par = None, ifprint =False):
	'''
	Train parametric bootstrap batches of RBM (global) with empirical RBM.
	'''
	if out_fold == None:
		raise Exception('Output folder is not specified')
	if os.path.exists(out_fold) is False:
		os.makedirs(out_fold)

	for i in range(ind,ind+B):
		# generate and save data
		Lc = np.ceil(N/Nc).astype(curr_int)
		gen_list = RBM.gen_data(Nchains = Nc, Lchains = Lc, Nthermalize = 500, Nstep = 10, reshape = True)
		gen_data = gen_list[0]
		for j in reversed(range(1, len(gen_data)-N+1)):
			gen_data = np.delete(gen_data, -Lc*j, axis=0).astype(curr_int)
		if len(gen_data) != N:
			raise IndexError('Gen_data is not the same size as df_train')
		if out_par == None:
			out_par1 = f'/sample_{i+1}.txt'
		else:
			out_par1 = f'/sample_{out_par}_{i+1}'
		np.savetxt(out_fold+out_par1, gen_data, fmt='%d')
		
		# train bootstrap RBM
		rbm_train(gen_data, n_v, n_h, n_cv, l12, lr, n_iter, visible, hidden,
			ifsave=True, out_fold=out_fold, out_par=out_par1)
		if ifprint:
			print(f'Batch {i+1} is done.')


def entropy_rbm_NP(df_train, B, ind, n_v, n_h, n_cv, l12, lr, n_iter,
				   visible='Potts', hidden='dReLU', frac=1, replace=True,
				   ifload=False, ifsave=False, out_fold=None, out_par=None):
	'''
	Compute the entropy for RBM (global) using non-parametric bootstrapping.
	'''
	if ifload or ifsave:
		if out_fold == None:
			raise Exception('Output folder is not specified')
		if os.path.exists(out_fold) is False:
			os.makedirs(out_fold)
	if df_train.dtype != curr_int:
		df_train = convert_number(df_train)
	N = len(df_train)

	list_E_NP = []
	for i in range(ind,ind+B):
		batch = df_train[np.random.choice(np.arange(N), int(frac*N), replace=replace)]
		if out_par == None:
			out_par1 = f'/sample_{i+1}.txt'
		else:
			out_par1 = f'/sample_{out_par}_{i+1}'
		if ifload:  # load data
			if os.path.exists(out_fold+out_par1):
				batch = np.loadtxt(out_fold+out_par1, dtype=curr_int)
			else:
				raise Exception(f'No file to be loaded for {i}-th batch.')
		else:
			batch = df_train[np.random.choice(np.arange(N), int(frac*N), replace=replace)]
			if ifsave:
				np.savetxt(out_fold+out_par1, batch.astype(curr_int), fmt='%d')

		# train the NP bootstrap RBM
		RBM = rbm_train(batch, n_v, n_h, n_cv, l12, lr, n_iter, visible, hidden, ifsave, out_fold, out_par1)    
		gen_data = RBM.gen_data(Nchains = 100, Lchains = 1000, Nthermalize = 500, Nstep = 10, reshape = True)
		E = -RBM.likelihood(gen_data[0].astype(curr_int)).mean()
		list_E_NP.append(E)

	E_NP = np.mean(list_E_NP)
	var_NP = np.var(list_E_NP, ddof=1)
	return E_NP, var_NP, list_E_NP


def entropy_rbm_P(RBM, B, ind, N, Nc, n_v, n_h, n_cv, l12, lr, n_iter,
				  visible = 'Potts', hidden = 'dReLU',
				  ifload=False, ifsave=False, out_fold=None, out_par=None):
	'''
	Compute the entropy of RBM (global) using parametric bootstrapping.
	'''
	if ifload or ifsave:
		if out_fold == None:
			raise Exception('Output folder is not specified')
		if os.path.exists(out_fold) is False:
			os.makedirs(out_fold)
	if df_train.dtype != curr_int:
		df_train = convert_number(df_train)
	N = len(df_train)

	list_E_P = []
	for i in range(ind,ind+B):
		# generate new batch of training data
		if out_par == None:
			out_par1 = f'/sample_{i+1}.txt'
		else:
			out_par1 = f'/sample_{out_par}_{i+1}'
		if ifload:  # load data
			if os.path.exists(out_fold+out_par1):
				batch = np.loadtxt(out_fold+out_par1, dtype=curr_int)
			else:
				raise Exception(f'No file to be loaded for {i}-th batch.')
		else:
			Lc = np.ceil(N/Nc).astype(curr_int)
			gen_data = RBM.gen_data(Nchains = Nc, Lchains = Lc, Nthermalize = 500, Nstep = 10, reshape = True)
			batch = gen_data[0]
			for j in reversed(range(1, len(batch)-N+1)):
				batch = np.delete(batch, -Lc*j, axis=0).astype(curr_int)
			if len(batch) != N:
				raise IndexError('Gen_list is not the same size as df_train')
			if ifsave:
				np.savetxt(out_fold+out_par1, batch.astype(curr_int), fmt='%d')
		
		# train the P bootstrap RBM
		RBM1 = rbm_train(batch, n_v, n_h, n_cv, l12, lr, n_iter, visible, hidden, ifsave, out_fold, out_par1)
		gen_data = RBM1.gen_data(Nchains = 100, Lchains = 1000, Nthermalize = 500, Nstep = 10, reshape = True)
		E = -RBM1.likelihood(gen_data[0].astype(curr_int)).mean()
		list_E_P.append(E)

	E_P = np.mean(list_E_P)
	var_P = np.var(list_E_P, ddof=1)
	return E_P, var_P, list_E_P


# def entropy_rbm_ens(df_train, batches, n_v, n_h, n_cv, l12, lr, n_iter,
#                     ifsave = False, out_fold = None,
#                     visible = 'Potts', hidden = 'dReLU', ifprint = False):
#     '''
#     Compute entropy - RBM (global) ensemble with a given batches
#     '''
#     list_E = []
#     gen_history = []
#     rbm_history = []
#     for i in range(batches):
#         RBM = rbm_train(df_train, n_v, n_h, n_cv, l12, lr, n_iter, ifsave, out_fold, str(i+1), visible, hidden)
#         RBM.AIS(M=20,n_betas=20000)
#         GEN_data_rbm = RBM.gen_data(Nchains = 100, Lchains = 1000, Nthermalize = 500, Nstep = 10, reshape = True)

#         E = -RBM.likelihood(GEN_data_rbm[0].astype(curr_int)).mean()
#         list_E.append(E)
#         rbm_history.append(RBM)
#         gen_history.append(GEN_data_rbm[0])

#     E_ens = np.mean(list_E)
#     err_ens = np.var(list_E, ddof=1)
#     if ifprint:
#         print(f'Entropy mean of RBM ensemble is {E_ens:.3f} and variance is {err_ens:.3}')

#     return E_ens, err_ens, list_E, gen_history, rbm_history


# def entropy_rbm_bootstrap(df_train, batches, N, L, frac, n_v, n_h, n_cv, l12, lr, n_iter,
#                           ifsave = False, out_fold = None, out_par = None,
#                           visible = 'Potts', hidden = 'dReLU', ifprint = False):
#     list_E = []
#     RBM = rbm_train(df_train, n_v, n_h, n_cv, l12, lr, n_iter, ifsave, out_fold, out_par, visible, hidden)
#     RBM.AIS(M=20,n_betas=20000)
#     GEN_data_rbm = RBM.gen_data(Nchains = N, Lchains = L, Nthermalize = 500, Nstep = 10, reshape = True)

#     gen_history = []
#     for _ in range(batches):
#         batch = GEN_data_rbm[0].sample(frac=frac, replace=True)
#         E = -RBM.likelihood(batch.astype(curr_int)).mean()
#         list_E.append(E)
#         gen_history.append(batch)

#     E_boot = np.mean(list_E)
#     err_boot = np.var(list_E)
#     if ifprint:
#         print(f'Entropy mean of RBM (gen_data bootstrap) is {E_boot:.3f} and variance is {err_boot:.3}')
#         print(f'The Z partition function is {RBM.log_Z_AIS:.3f} with std = {RBM.log_Z_AIS_std:.3}')
	
#     return E_boot, err_boot, list_E, gen_history
