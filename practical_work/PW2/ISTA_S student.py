import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
# from fichiers_utils import prox_oblique
#%%
def prox_l1(x,thrd=0): # Fill this function with the code you already implemented
	...
	return res

#%%
def ISTA_S(set_loader,lamb = 0, itmax = 10000):
    # set_loader : a data loader object
    # lamb : the threshold value used in the l1 proximal operator
    # itmax : maximum number of iterations. You can keep this value and not modify it.
    
    eps = 0.01
    S_est_list = [] # Each element of S_est_list should be a mini-batch of estimated S matrices.
    itmax_list = []# Each element of itmax_list should be a mini-batch of the maximal number of iterations reached by ISTA (only used for display).
    
    
    it = 0

    for ... :# Here, you should loop over all the mini-batches in set_loader. You can use the enumerate python function and define :
        # X = Data set mini-batch
        # A = Ground truth A  mini-batch, used for comupting the gradient
        # S = Ground truth S  mini-batch, should be unused
        
        

        S_est_mb = torch.zeros([X.size()[0], A.size()[2],X.size()[2]], dtype=torch.double) # Initialization of a mini-batch of estimated sources
        
        it_max_mb = torch.zeros([X.size()[0]], dtype=torch.double)
            
        for ... :# Loop over all the elements in the current mini-batch
            S_est = ... # A single estimated mixing matrix.
            S_est_prev = ... # For the stopping criterion, keep in memory the previous source matrix.
            
            L = ... # Lispschitz constant. You can use the np.linalg.norm function of numpy
            
            it = 0 
            
            while(torch.norm(S_est-S_est_prev, p = 'fro') > 1e-6 or it < 2) and it < itmax: # ISTA iterations
                if it>0:
                    S_est_prev = ...
                
                S_est = ... # Proximal gradient step
                
                it += 1
            
            S_est_mb[j] = ... # Put the estimated source matrix inside of the mini-batch
            it_max_mb[j] = ...
            
        S_est_list.append(...)# List containing all the mini-batches of estimated sources.
        itmax_list.append(...)
        
    return S_est_list, itmax_list
