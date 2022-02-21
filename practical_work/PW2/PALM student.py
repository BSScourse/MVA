#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 16:56:48 2022

@author: ckervazo
"""

import numpy as np
import copy as cp
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
#%%
def prox_l1(x,thrd=0):
	...
	return res

def prox_oblique(A_pred):
    ...
    return A_pred


#%%
def PALM(set_loader, lamb = 0, itmax = 100,Ainit=1):    
    eps = 0.01
    S_est_list = []
    A_est_list = []
    itmax_list = []
    
    
    
    
    it=0

    for ... # For all mini-batches
        # X = Data set mini-batch
        # A = Ground truth A  mini-batch, used only for metric and sizes
        # S = Ground truth S  mini-batch, unused
        print("example %d",i)
        
        

        S_est_mb = torch.zeros([X.size()[0], S.size()[1],S.size()[2]], dtype=torch.double) # S_est_mb contains all the different samples of the estimated sources in a mini-batch
        A_est_mb = torch.zeros([X.size()[0], A.size()[1],A.size()[2]], dtype=torch.double) # A_est_mb contains all the different samples of A in a mini-batch
        it_max_mb = torch.zeros([X.size()[0]], dtype=torch.double)
    
        for ... :# For all samples in the current mini-batch

            S_est = ... # A single source matrix
            A_est = Ainit[0]# Initialization of A

            S_est_prev = ... # For the stopping criterion
            A_est_prev = ... # For the stopping criterion
            
            
            it = 0
            
            while(torch.norm(S_est-S_est_prev, p = 'fro') > 1e-6 or torch.norm(A_est-A_est_prev, p = 'fro') > 1e-6 or it < 2) and it < itmax: # PALM iterations
                if it>0:
                    S_est_prev = ...
                    A_est_prev = ...
                
                gamma = ... # Lipschitz constant for S update
                S_est = ... # Proximal gradient step for S
                
                eta = ... # Lipschitz constant for A update
                A_est = ...# Proximal gradient step for A
                it += 1
            
            print('itmax %s'%it)
            S_est_mb[j] = ... # Put the estimated S matrix inside of the mini-batch
            A_est_mb[j] = ... # Put the estimated A matrix inside of the mini-batch
            it_max_mb[j] = it


        S_est_list.append(...) # List containing all the mini-batches of estimated S matrices.
        A_est_list.append(...) # List containing all the mini-batches of estimated A matrices.
        itmax_list.append(...)
        
    return A_est_list,S_est_list,itmax_list
