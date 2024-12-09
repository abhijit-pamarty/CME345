
import numpy as np
import matplotlib.pyplot as plt


def PROM(sample, V, M = 50):
    
    #same as diffusion code
    print("starting...\n")
    h = 1/M
    
    num_timesteps = 100            #timesteps
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
    

    
    Fs = np.zeros((1, num_timesteps, M, M))
    F = np.zeros((M, M))
    F_prev = np.zeros((M, M))
    scaling_factors = []



    #create the right hand side matrix and implement dirichlet boundary conditions

        
    #dirichlet boundary conditions
    scaling_factor = sample[0]/((h**2)) + sample[1]/(2*h)
    scaling_factors.append(scaling_factor)
        
    LHS = np.matmul(np.matmul(V.T, (sample[0]*del2 + sample[1]*del1 + delt)), V)              #LHS reduced order array
        
    t = 0
    F_prev = np.zeros((M, M))
    F_prev[:,:] = scaling_factor
    F_prev[int(np.floor(M/2 -5)):int(np.floor(M/2 + 5)), -1] = 2*scaling_factor 
        
    while t < num_timesteps:
            
        print("timestep :", t)
        F_eval = F_prev.reshape(M*M,1)
        RHS = np.matmul(V.T, F_eval)
        F = np.matmul(V, np.linalg.solve(-LHS,RHS))
        F = F.reshape(M, M)
            
        Fs[0, t, :, :] = F
        F[0,:] = scaling_factor
        F[int(np.floor(M/2 -5)):int(np.floor(M/2 + 5)),-1] = 2*scaling_factor 
            
        F_prev = F
        t += 1

        
    return Fs

