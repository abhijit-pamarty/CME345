

import numpy as np

import matplotlib.pyplot as plt
import PROM
import interpolation_ROB

if __name__ == "__main__":
    
    interpolate = False
    
    num_Vs = 5
    V = np.load("V_sparse_V1.npy")
    num_SVs = 1
    num_points, _ = V.shape
    interpolation_Vs = np.zeros((num_Vs, num_points, num_SVs))

    
    if interpolate:
        
        interpolation_Vs[0, :, :] = np.load("V_sparse_V1.npy")
        interpolation_Vs[1, :, :] = np.load("V_sparse_V2.npy")
        interpolation_Vs[2, :, :] = np.load("V_sparse_V3.npy")
        interpolation_Vs[3, :, :] = np.load("V_sparse_V4.npy")
        interpolation_Vs[4, :, :] = np.load("V_sparse_V5.npy")
        

        
    
    #calculate for things not in dataset
    test_samples = np.load("sample_data.npy")
    test_Fs_HDM = np.load("Fs.npy")
    
    num_test_samples, num_sample_params = test_samples.shape
    
    interpolation_samples = np.zeros((num_Vs, num_sample_params))
    
    if interpolate:
        
        interpolation_samples[0, :] = np.load("sample_data_sparse_V1.npy")
        interpolation_samples[1, :] = np.load("sample_data_sparse_V2.npy")
        interpolation_samples[2, :] = np.load("sample_data_sparse_V3.npy")
        interpolation_samples[3, :] = np.load("sample_data_sparse_V4.npy")
        interpolation_samples[4, :] = np.load("sample_data_sparse_V5.npy")
        
        
    
    test_Fs_pred = np.zeros(test_Fs_HDM.shape)
    
    num_test_samples, num_timesteps, M, _ = test_Fs_HDM.shape
    for sample_idx in range(num_test_samples):
        
        if interpolate:
            V = interpolation_ROB.interpolate_V(interpolation_Vs, interpolation_samples, test_samples[sample_idx, :])
            
        test_Fs_pred[sample_idx, :, :, :] = PROM.PROM(test_samples[sample_idx, :], V)
    
    test_errors = np.zeros((num_test_samples, num_timesteps))
    for sample_idx in range(num_test_samples):
        for time_idx in range(num_timesteps):
            test_errors[sample_idx, time_idx] = np.mean(np.abs((test_Fs_pred[sample_idx, time_idx, :, :] - test_Fs_HDM[sample_idx, time_idx, :, :]))/np.mean(test_Fs_HDM[sample_idx, time_idx, :, :]))
        
    #calculate for things in dataset
    train_samples = np.load("sample_data_sparse.npy")
    train_Fs_HDM = np.load("Fs_sparse.npy")
    
    
    train_Fs_pred = np.zeros(train_Fs_HDM.shape)
    
    num_train_samples, num_timesteps, M, _ = train_Fs_HDM.shape
    for sample_idx in range(num_train_samples):
        
        if interpolate:
            V = interpolation_ROB.interpolate_V(interpolation_Vs, interpolation_samples, train_samples[sample_idx, :])
            
        train_Fs_pred[sample_idx, :, :, :] = PROM.PROM(train_samples[sample_idx, :], V)
    
    train_errors = np.zeros((num_train_samples, num_timesteps))
    for sample_idx in range(num_train_samples):
        for time_idx in range(num_timesteps):
            train_errors[sample_idx, time_idx] = np.mean(np.abs((train_Fs_pred[sample_idx, time_idx, :, :] - train_Fs_HDM[sample_idx, time_idx, :, :]))/np.mean(train_Fs_HDM[sample_idx, time_idx, :, :]))
        
            
    test_peclet_numbers = np.zeros((num_test_samples, 1))
    test_maximum_errors = np.zeros((num_test_samples, 1))
    for sample_idx in range(num_test_samples):
        sample_peclet_number = test_samples[sample_idx, 1]/test_samples[sample_idx,0]
        sample_maximum_error = np.max(abs(test_errors[sample_idx, :]))*100
        print("sample peclet number: " +str(sample_peclet_number)+" sample max error :"+str(sample_maximum_error)+"%")
        test_peclet_numbers[sample_idx, 0] = sample_peclet_number
        test_maximum_errors[sample_idx, 0] = sample_maximum_error
    
    train_peclet_numbers = np.zeros((num_train_samples, 1))
    train_maximum_errors = np.zeros((num_train_samples, 1))
    for sample_idx in range(num_train_samples):
        sample_peclet_number = train_samples[sample_idx, 1]/train_samples[sample_idx,0]
        sample_maximum_error = np.max(abs(train_errors[sample_idx, :]))*100
        print("sample peclet number: " +str(sample_peclet_number)+" sample max error :"+str(sample_maximum_error)+"%")
        train_peclet_numbers[sample_idx, 0] = sample_peclet_number
        train_maximum_errors[sample_idx, 0] = sample_maximum_error
        
    plt.scatter(test_peclet_numbers, np.log10(test_maximum_errors), marker="+", color = 'red', label = "test data")
    plt.scatter(train_peclet_numbers, np.log10(train_maximum_errors), marker="v", color = 'k', label = "train data")
    plt.title("Single ROB, σ = 3")
    plt.xlabel("Pe")
    plt.ylabel("log(ε)")
    plt.legend()
    plt.show()
    
    plt.plot(np.linspace(0, 1, num_timesteps), np.log10(abs(test_errors[11, :]*100)), color = 'red', label = "Pe = 869.3")
    
    plt.plot(np.linspace(0, 1, num_timesteps), np.log10(abs(test_errors[0, :]*100)), color = 'grey', label = "Pe = 176.9")
    plt.plot(np.linspace(0, 1, num_timesteps), np.log10(abs(test_errors[9, :]*100)), color = 'k', label = "Pe = 13.4")
    plt.title("error vs peclet number")
    plt.xlabel("Pe")
    plt.ylabel("log(ε)")
    plt.legend()
    plt.show()
