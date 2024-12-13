
#general dependencies
import numpy as np
import matplotlib.pyplot as plt
import PROM
import interpolation_ROB

#pade neural operator dependencies
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
import random as r
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import LHS_matrix

#the man himself :3
import pade_operator_2D

if __name__ == "__main__":

    interpolate = False
    use_PNO = True
    use_CUDA = False
    num_Vs = 5
    V = np.load("V_sparse.npy")
    num_SVs = 1
    num_points, _ = V.shape
    interpolation_Vs = np.zeros((num_Vs, num_points, num_SVs))
    
    #pade neural operator controls
    pade_num_order = 9
    pade_denom_order = 8
    run_to_load = 13
    epoch_to_load = 1000
    
    
    if interpolate:
        
        interpolation_Vs[0, :, :] = np.load("V_sparse_V1.npy")
        interpolation_Vs[1, :, :] = np.load("V_sparse_V2.npy")
        interpolation_Vs[2, :, :] = np.load("V_sparse_V3.npy")
        interpolation_Vs[3, :, :] = np.load("V_sparse_V4.npy")
        interpolation_Vs[4, :, :] = np.load("V_sparse_V5.npy")
        

        
    
    #calculate for things not in dataset
    test_samples = np.load("sample_data_full_scale.npy")
    test_Fs_HDM = np.load("Fs_full_scale.npy")
    
    num_test_samples, num_sample_params = test_samples.shape
    
    interpolation_samples = np.zeros((num_Vs, num_sample_params))
    
    if interpolate:
        
        interpolation_samples[0, :] = np.load("sample_data_sparse_V1.npy")
        interpolation_samples[1, :] = np.load("sample_data_sparse_V2.npy")
        interpolation_samples[2, :] = np.load("sample_data_sparse_V3.npy")
        interpolation_samples[3, :] = np.load("sample_data_sparse_V4.npy")
        interpolation_samples[4, :] = np.load("sample_data_sparse_V5.npy")
        
        
    
    test_Fs_pred = np.zeros(test_Fs_HDM.shape)
    
    num_test_samples, num_timesteps, M_test, _ = test_Fs_HDM.shape
    
    pade_neural_operator_high_dim = pade_operator_2D.Pade_Neural_Operator(num_sample_params, M_test, pade_num_order, pade_denom_order, 1, 1e-7)
    pade_neural_operator_high_dim.load_state_dict(torch.load("PNO_state_run_"+str(run_to_load)+"_"+str(epoch_to_load)+".pth"))
    
    if torch.cuda.is_available() and use_CUDA:
        device = torch.device("cuda:0")  # Specify the GPU device
        print("CUDA available")
        pade_neural_operator_high_dim = pade_neural_operator_high_dim.to(device)
    
    
    for sample_idx in range(num_test_samples):
        
        if interpolate:
            V = interpolation_ROB.interpolate_V(interpolation_Vs, interpolation_samples, test_samples[sample_idx, :])
        
        elif use_PNO:
            Xs = torch.linspace(0, 1, M_test)                      #X variable to create pade approximant
            Ys = torch.linspace(0, 1, M_test)                      #Y variable to create pade approximant
            sample_tensor = torch.from_numpy(test_samples[sample_idx, :]).float()
            
            if torch.cuda.is_available() and use_CUDA:
                print("CUDA available")
                device = torch.device("cuda:0")  # Specify the GPU device
                sample_tensor = sample_tensor.to(device)
                Xs = Xs.to(device)
                Ys = Ys.to(device)

            
            prediction = pade_neural_operator_high_dim(sample_tensor, Xs, Ys)
            pred = prediction.detach().cpu().numpy()
            plt.imshow(pred)
            plt.title("Sample basis")
            plt.show()
            V = torch.reshape(prediction, (M_test*M_test, 1)).detach().cpu().numpy()
        
            
        test_Fs_pred[sample_idx, :, :, :] = PROM.PROM(test_samples[sample_idx, :], V, M_test)
    
    test_errors = np.zeros((num_test_samples, num_timesteps))
    for sample_idx in range(num_test_samples):
        for time_idx in range(num_timesteps):
            test_errors[sample_idx, time_idx] = np.mean(np.abs((test_Fs_pred[sample_idx, time_idx, :, :] - test_Fs_HDM[sample_idx, time_idx, :, :]))/np.mean(test_Fs_HDM[sample_idx, time_idx, :, :]))
    
    
    sample_idx = 1
    time_idx = 50
    data1 = test_Fs_HDM[sample_idx, time_idx, :, :]/(np.max(test_Fs_HDM[sample_idx, time_idx, :, :])*0.5)
    data2 = test_Fs_pred[sample_idx, time_idx, :, :]/(np.max(test_Fs_pred[sample_idx, time_idx, :, :])*0.5)
    data3 = np.abs(data1 - data2)
    
    # Create a figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot the images and add colorbars
    im1 = axs[0].imshow(data1, cmap='viridis')
    fig.colorbar(im1, ax=axs[0])
    axs[0].set_title('HDM concentration',fontsize=12)
    im1.set_clim(1, 2)
    
    im2 = axs[1].imshow(data2, cmap='viridis')
    fig.colorbar(im2, ax=axs[1])
    axs[1].set_title('Interpolated basis',fontsize=12)
    im2.set_clim(1, 2)
    
    im3 = axs[2].imshow(np.log10(np.abs(data3*100)), cmap='inferno')
    fig.colorbar(im3, ax=axs[2])
    axs[2].set_title('Prediction error (log(e))',fontsize=12)
    im3.set_clim(-3, 2)

    plt.show()
    
    
    #calculate for things in dataset
    train_samples = np.load("sample_data_sparse.npy")
    train_Fs_HDM = np.load("Fs_sparse.npy")
    
    num_train_samples, num_timesteps, M_train, _ = train_Fs_HDM.shape
    
    pade_neural_operator_low_dim = pade_operator_2D.Pade_Neural_Operator(num_sample_params, M_train, pade_num_order, pade_denom_order, 1, 1e-7)
    pade_neural_operator_low_dim.load_state_dict(torch.load("PNO_state_run_"+str(run_to_load)+"_"+str(epoch_to_load)+".pth"))
    
    train_Fs_pred = np.zeros(train_Fs_HDM.shape)
    
    num_train_samples, num_timesteps, M_train, _ = train_Fs_HDM.shape
    for sample_idx in range(num_train_samples):
        
        if interpolate:
            V = interpolation_ROB.interpolate_V(interpolation_Vs, interpolation_samples, train_samples[sample_idx, :])
        
        elif use_PNO:
            Xs = torch.linspace(0, 1, M_train)                      #X variable to create pade approximant
            Ys = torch.linspace(0, 1, M_train)                      #Y variable to create pade approximant
            sample_tensor = torch.from_numpy(train_samples[sample_idx, :]).float()
            
            if torch.cuda.is_available() and use_CUDA:
                print("CUDA available")
                device = torch.device("cuda:0")  # Specify the GPU device
                sample_tensor = sample_tensor.to(device)
                Xs = Xs.to(device)
                Ys = Ys.to(device)

            prediction = pade_neural_operator_low_dim(sample_tensor, Xs, Ys)
            V = torch.reshape(prediction, (M_train*M_train, 1)).detach().cpu().numpy()
            
            
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
    mean_error = np.mean(np.vstack((np.log10(test_maximum_errors), np.log10(train_maximum_errors))))
    mean_array = mean_error*np.ones((10))
    Xs = np.linspace(0, 1000, 10)
    plt.plot(Xs, mean_array, linestyle = "dashed", color = "k")
    plt.ylim([0.8, 2.2])
    if use_PNO:
        plt.title("PNO generated ROBs, σ = 1, M = 50")
    if interpolate:
        plt.title("Interpolated ROBs, σ = 1, M = 50")
    plt.xlabel("Pe")
    plt.ylabel("log(ε)")
    plt.legend()
    plt.show()
    
    plt.plot(np.linspace(0, 1, num_timesteps), np.log10(abs(test_errors[8, :]*100)), color = 'red', label = "Pe = 156.9")
    
    plt.plot(np.linspace(0, 1, num_timesteps), np.log10(abs(test_errors[0, :]*100)), color = 'grey', label = "Pe = 176.9")
    plt.plot(np.linspace(0, 1, num_timesteps), np.log10(abs(test_errors[9, :]*100)), color = 'k', label = "Pe = 13.4")
    plt.title("error vs peclet number")
    plt.xlabel("Pe")
    plt.ylabel("log(ε)")
    plt.legend()
    plt.show()
