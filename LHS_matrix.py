# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 19:34:35 2024

@author: abhij
"""

import numpy as np

def generate_LHS_matrix(sample, M):
    
    print("generating LHS...")
    h = 1/M
    
    dt = 2                       #timestep size
    #create the del matrix with upwinding
    
    du = 1/h
    d = -1/h
    dl = 0
    
    #create the matrix in one dimension
    exu = np.arange(M - 1)
    exl = np.arange(M - 1) + 1
    eyu = np.arange(M - 1)
    eyl = np.arange(M - 1) + 1
    
    I = np.identity(M)
    
    Ax = I*d
    Ax[exu, exu] /= exu + 1
    Ax[exu, exu+1] += du/(exu + 1)
    Ax[exl, exl-1] += dl/(exl + 1)
    Ax[-1, -1] = -1/(h*M)
    Ax[-1, -2] = 1/(h*M)

    del1 = np.kron(I, Ax) 
    
    #create the del^2 matrix with central differences
    
    du= 1/h**2                       #upper diagonal value
    d = (-2/h**2)                    #middle diagonal value
    dl = du                         #lower diagonal value
    
    #create the matrix in one dimension
    exu = np.arange(M - 1)
    exl = np.arange(M - 1) + 1
    
    eyu = np.arange(M - 1)
    eyl = np.arange(M - 1) + 1
    

    #set the diagonals for the base matrices
    Ax = I*d
    Ax[exu, exu + 1] += du
    Ax[exl, exl- 1] += dl
    Ax[-1,-1] = 0
    Ax[-1,-2] = 0
    Ax[0,0] = 0
    Ax[0,1] = 0
    
    Ay = I*d
    Ay[eyu, eyu + 1] += du
    Ay[eyl, eyl - 1] += dl
    
    #set the neumann boundary conditions

    Ay[-1,-1] = 0
    Ay[-1, -2] = 0
    Ay[0,0] = 0
    Ay[0,1] = 0

    del2_y = np.kron(I, Ax)
    del2_x = np.kron(Ay, I)
    
    del2 = (del2_y + del2_x)
    
    #set the time stepping matrix with upwinding
    
    d = -1/dt
    
    At = I*d
    delt_x = np.kron(I, At)
    delt_y = np.kron(At, I)
    delt = delt_x + delt_y 
    
    #set the time stepping matrix with upwinding
    
    d = -1/dt
    
    At = I*d
    delt_x = np.kron(I, At)
    delt_y = np.kron(At, I)
    delt = delt_x + delt_y 
    
    #set the time stepping matrix with upwinding
    
    d = -1/dt
    
    At = I*d
    delt_x = np.kron(I, At)
    delt_y = np.kron(At, I)
    delt = delt_x + delt_y 

    #create the right hand side matrix and implement dirichlet boundary conditions
        
    LHS = (sample[0]*del2 + sample[1]*del1 + delt) #LHS reduced order array
    
    return LHS