# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 20:53:15 2024

@author: abhij
"""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def legendre_gauss(N_points):
    
    x, w = np.polynomial.legendre.leggauss(N_points)
    
    return x, w

if __name__ == "__main__":
    
    #number of points for integration
    N_points = 3
    truncation_sigma = 1e-2
    scalar_file = "Fs_sparse_V1.npy"
    scaling_file = "scaling_factors_sparse_V1.npy"
    parameter_file = "sample_data_sparse_V1.npy"
    
    scalings = np.load(scaling_file)
    parameters = np.load(parameter_file)
    Fs = np.load(scalar_file)
    
    num_sols, num_timesteps, M, _ = Fs.shape

    Fs = Fs[:]/scalings.reshape(num_sols, 1, 1, 1)
    Fs = Fs.reshape(num_sols, num_timesteps, M*M)
    
    ti = np.linspace(-1, 1, num_timesteps)

    #set number of gauss points for quadrature
    num_points = 10  
    #integrate over time
    points, weights = legendre_gauss(num_points)
    #fs at collocation points
    Fs_col= np.zeros((num_sols, num_points, M*M))
    
    for sol_idx in range(num_sols):
        print("generating interpolations for solution index "+str(sol_idx +1)+"/"+str(num_sols)+"...")
        for loc_idx in range(M*M):
            interp_func = interp1d(ti, Fs[sol_idx, :, loc_idx], kind = 'cubic', axis = 0)
            Fs_col[sol_idx, :, loc_idx] = interp_func(points)
    
    #generate integrated Fs
    sqrt_weights = np.sqrt(weights)
    S = np.zeros((num_sols, M*M))
    
    for sol_idx in range(num_sols):
        print("integrating with square root of weights to generate reduced order basis...")
        for loc_idx in range(M*M):
            S[sol_idx, loc_idx] = np.dot(sqrt_weights, Fs_col[sol_idx, :, loc_idx]) 
        
    U, sigmas, V = np.linalg.svd(S.T)

    num_valid_sv = 0
    for sigma_idx in range(sigmas.shape[0]):
        if sigmas[sigma_idx] > truncation_sigma:
            num_valid_sv += 1
    
    print("Number of singular values considered: "+str(num_valid_sv))
    U_trunc = U[:, 0:num_valid_sv] 
    np.save("V_sparse_V1.npy", U_trunc)
    print("done")
    
    
    

    