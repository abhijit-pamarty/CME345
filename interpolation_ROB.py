# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 22:53:55 2024

@author: abhij
"""
import numpy as np
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt

def calculate_s(V_i, V_0):
    
    I = np.identity(V_0.shape[0])
    S = np.matmul(np.matmul((I- np.matmul(V_0, V_0.T)), V_i), np.linalg.inv(np.matmul(V_0.T, V_i)))
    
    return S
    
def interpolate_V(Vs, samples, sample_point):
    
    #here samples is a 1d function representing the peclet number
    
    debug_GUI = True
    print("Interpolating Vs...")
    
    #put in tangent space
    
    num_Vs, num_points, num_SVs = Vs.shape
    V_0 = Vs[0, :, :]
    
    #base Ss
    Ss = np.zeros(Vs.shape)
    #tangent space Ss
    S_tan = np.zeros(Vs.shape)
    
    #find the logarithmic map of all S
    sigma_base = np.zeros((num_SVs, num_points))
    for V_idx in range(num_Vs):
        
        #find the S's associated with the Vs
        
        Ss[V_idx, :, :] = calculate_s(Vs[V_idx, :, :], V_0)
        U, sigmas, Z = np.linalg.svd(Ss[V_idx, :, :], full_matrices = False)
        
        sigmas = np.arctan(sigmas)
        # diag_sigmas = np.diag(sigmas) 
        # sigma_base[0:(num_SVs), 0:(num_SVs)] = diag_sigmas
        S_tan[V_idx, :, :] = np.matmul(U, np.matmul(sigmas, Z.T)).reshape(num_points, 1)
        
    
    
    #interpolate in the tangent manifold
    
    S_tan_interpolated = np.zeros(V_0.shape)
    current_sample = np.zeros((1, 2))
    
    for loc_idx in range(num_points):
        for SV_idx in range(num_SVs):
            interp_func = RBFInterpolator((samples), S_tan[: ,loc_idx, SV_idx], kernel='linear')
            current_sample[:, 0] = sample_point[0]
            current_sample[:, 1] = sample_point[1]
            S_tan_interpolated[loc_idx, SV_idx] = interp_func(current_sample)
            
    #convert back to normal space with exponential map
    
    U, sigmas, Z = np.linalg.svd(S_tan_interpolated, full_matrices = False)

    V_interpolated = (np.matmul(V_0, np.matmul(Z, np.cos(sigmas))) + np.matmul(U, np.sin(sigmas))).reshape(num_points, 1)
    
    if debug_GUI:
        plt.imshow(V_interpolated.reshape(50, 50))
        plt.show()
    
    return V_interpolated
    
        
    
        
    
    
    