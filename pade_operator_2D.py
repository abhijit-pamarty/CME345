# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 19:06:24 2024

@author: abhij
"""

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np
import random as r
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import LHS_matrix

#define the pade layer
class Pade_Layer(nn.Module):
    
    def __init__(self, parameter_dim, num_X, pade_num_order, pade_denom_order, batch_size, epsilon):
        
        super(Pade_Layer, self).__init__()
        
        #fc layers
        self.fc_1 = 4
        self.fc_2 = 8
        self.fc_3 = 16
        self.fc_4 = 32
        
        self.fc_p_nc = pade_num_order                                #pade approximant numerator coefficients
        self.fc_p_np = pade_num_order                               #pade approximant numerator powers
        self.fc_p_dc = pade_denom_order                            #pade approximant denominator coefficients
        self.fc_p_dp = pade_denom_order                            #pade approximant denominator powers
        
        
        #create the decoder
        self.fc1 = nn.Linear(in_features=parameter_dim, out_features=self.fc_1)
        self.fc2 = nn.Linear(in_features=self.fc_1, out_features=self.fc_2)
        self.fc3 = nn.Linear(in_features=self.fc_2, out_features=self.fc_3)
        self.fc4 = nn.Linear(in_features=self.fc_3, out_features=self.fc_4)
        self.fc5 = nn.Linear(in_features=self.fc_4, out_features=(self.fc_p_nc + self.fc_p_np + self.fc_p_dc + self.fc_p_dp))
        

        
        
        self.fc6 = nn.Linear(in_features = parameter_dim, out_features = (self.fc_p_nc + self.fc_p_np + self.fc_p_dc + self.fc_p_dp))
        
        
    def forward(self, x, time):
        
        
        #FC layers
        x1 = f.leaky_relu(self.fc1(x))
        x2 = f.leaky_relu(self.fc2(x1))
        x3 = f.leaky_relu(self.fc3(x2))
        x4 = f.leaky_relu(self.fc4(x3))
        x5 = f.sigmoid(self.fc5(x4))
    
        
        #PNO layer
        num_coeffs = 2*(x5[ 0:(self.fc_p_nc)] - 0.5)
        num_powers = pade_num_order*x5[ self.fc_p_nc:(self.fc_p_nc + self.fc_p_np)]
        denom_coeffs = 2*(x5[(self.fc_p_nc + self.fc_p_np):(self.fc_p_nc + self.fc_p_np + self.fc_p_dc)] - 0.5)
        denom_powers = pade_denom_order*x5[(self.fc_p_nc + self.fc_p_np + self.fc_p_dc):(self.fc_p_nc + self.fc_p_np + self.fc_p_dc + self.fc_p_dc)]
        
        time_num = time.reshape(num_X, 1)
        time_denom = time.reshape(num_X, 1)
        time_num = time_num.repeat(1, pade_num_order)
        time_denom = time_denom.repeat(1, pade_denom_order)
        
        pade = torch.sum((num_coeffs*((time_num + epsilon)**num_powers)), dim = 1)/(torch.sum(((denom_coeffs*((time_denom + epsilon)**denom_powers))), dim =1)) 
        
        short = torch.sum(f.tanh(self.fc6(x)))
        
        output = pade + short
        
        return output
    
class Pade_Neural_Operator(nn.Module):
    
    def __init__(self, parameter_dim, num_points, pade_num_order, pade_denom_order, batch_size, epsilon):
        
        super(Pade_Neural_Operator, self).__init__()
        
        #pade layers
        
        self.pade1 = Pade_Layer(parameter_dim, num_points, pade_num_order, pade_denom_order, batch_size, epsilon)
        self.pade2 = Pade_Layer(parameter_dim, num_points, pade_num_order, pade_denom_order, batch_size, epsilon)
        self.pade3 = Pade_Layer(parameter_dim, num_points, pade_num_order, pade_denom_order, batch_size, epsilon)
        self.pade4 = Pade_Layer(parameter_dim, num_points, pade_num_order, pade_denom_order, batch_size, epsilon)
        
        self.weights_layer_X = nn.Linear(in_features= parameter_dim, out_features= 1)
        self.weights_layer_Y = nn.Linear(in_features= parameter_dim, out_features= 1)
        
        
    def forward(self, x, X, Y):
        
        
        #pade layers
        output1 = self.pade1(x, X)
        output2 = self.pade2(x, Y)
        output3 = self.pade3(x, X)
        output4 = self.pade4(x, Y)
        weights_X = self.weights_layer_X(x)
        weights_Y = self.weights_layer_Y(x)
        
        
        return weights_X[0]*torch.outer(output1, output2) + weights_Y[0]*torch.transpose(torch.outer(output3, output4), 0, 1)
    
# Create a custom dataset that dynamically computes slices
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, Fs, samples, LHSs, num_timesteps):
        self.Fs = Fs
        self.samples = samples
        self.LHSs = LHSs
        self.num_timesteps = num_timesteps
        self.num_samples = len(samples)

    def __len__(self):
        return (self.num_samples * (self.num_timesteps - 1))

    def __getitem__(self, index):
        sample_idx = index // (self.num_timesteps - 1)
        timestep_idx = index % (self.num_timesteps - 1)

        F_current = self.Fs[sample_idx * self.num_timesteps + timestep_idx + 1]
        F_previous = self.Fs[sample_idx * self.num_timesteps + timestep_idx]
        sample = self.samples[sample_idx]
        LHS = self.LHSs[sample_idx]

        return F_current, F_previous, sample, LHS

def train_model(pade_neural_operator, criterion, optimizer, sample_data, Fs_data, run, learn_rate, batch_size, LHSs, scalings, num_epochs=1, batchsize=20, use_GPU = True):
    
    # Reshape data into (num_samples * num_timesteps, X_size, Y_size)
    num_samples, num_timesteps, num_X, num_Y = Fs_data.shape
    _, parameter_dim = sample_data.shape
    
    
    # Convert to tensor and prepare DataLoader
    Fs_tensor = torch.from_numpy(Fs_data).float()  
    sample_tensor = torch.from_numpy(sample_data).float()
    LHSs_tensor = torch.from_numpy(LHSs).float()
    
    # LHSs_tensor.unsqueeze_(1)
    # LHSs_tensor = LHSs_tensor.expand(-1, num_timesteps, -1, -1).to_sparse()
    
    Xs = torch.linspace(0, 1, num_X)                      #X variable to create pade approximant
    Ys = torch.linspace(0, 1, num_Y)                      #Y variable to create pade approximant
    
    if torch.cuda.is_available() and use_GPU:
        print("CUDA available")
        device = torch.device("cuda:0")  # Specify the GPU device
        Fs_tensor = Fs_tensor.to(device)
        sample_tensor = sample_tensor.to(device)
        Xs = Xs.to(device)
        Ys = Ys.to(device)
        LHSs_tensor = LHSs_tensor.to(device)
    
    #reshape to make time dimension same
    
    Fs_tensor = torch.reshape(Fs_tensor, (num_samples*num_timesteps, num_X, num_Y))
    #sample_tensor = torch.reshape(sample_tensor, (num_samples*num_timesteps, parameter_dim))
    #LHSs_tensor = torch.reshape(LHSs_tensor, (num_samples*num_timesteps, num_X*num_Y,  num_X*num_Y))

    dataset = CustomDataset(Fs_tensor, sample_tensor, LHSs_tensor, num_timesteps)
    
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1 / 1.2)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_num = 0
        
        for batch_outputs_cur_t, batch_outputs_prev_t, batch_inputs, batch_LHS in dataloader:
            
            # Get a batch of data
            sample = batch_inputs[0]  # Shape: (batchsize, 1, X_size, Y_size)
            Fs = batch_outputs_cur_t[0]
            Fs_prev = batch_outputs_prev_t[0]
            LHS = batch_LHS[0]

            # Forward pass
            prediction = pade_neural_operator(sample, Xs, Ys)
            
            if batch_num % 20 == 0 and epoch % 200 == 0: 
                pred = prediction.detach().cpu().numpy()
                plt.imshow(pred)
                plt.show()
            
            V = torch.reshape(prediction, (num_X*num_Y, 1))
            
            
            Fs_reshaped = torch.reshape(Fs, (1, num_X*num_Y))
            Fs_prev_reshaped = torch.reshape(Fs_prev, ((num_X*num_Y), 1))
            loss = residual_loss(V, LHS, Fs_prev_reshaped, Fs_reshaped)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(pade_neural_operator.parameters(), 0.2)
            
            optimizer.step()
            
            # Accumulate loss
            epoch_loss += loss.item()
            batch_num += 1
        
        # Scheduler step every 1000 epochs
        if epoch % 10000 == 0 and epoch > 0:
            scheduler.step()
        
        # Print epoch loss
        avg_loss = epoch_loss / len(dataloader)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")
    
        
        # Save model periodically
        if (epoch + 1) % 10000 == 0:
            print("Saving model...\n")
            torch.save(pade_neural_operator.state_dict(), "PNO_state_run_"+str(run)+"_"+str(epoch+1)+".pth")

def residual_loss(V, LHS, RHS, X):
    
    q = torch.matmul(torch.linalg.inv(torch.matmul(torch.matmul(torch.transpose(V, 0, 1), LHS), V)), torch.matmul(torch.transpose(V, 0, 1), RHS))
    r = torch.matmul(torch.matmul(LHS, V), q) -  RHS
    
    return torch.linalg.norm(r, dim=0, ord = 2)
    
#Latent space trajectory finder
if __name__ == "__main__":
    
    Fs_data_file = "Fs_sparse.npy"              #dataset variable for the latent space (outputs)
    sample_data_file = "sample_data_sparse.npy"                    #dataset variable for the sample data (inputs)
    scaling_factors_file = "scaling_factors_sparse.npy"
    batch_size = 50                                         #batch size
    epsilon = 1e-7                                          #small coefficient for pade neural operator
    
    load_model = False
    restart_training = False
    use_CUDA =  True
    run_to_load = 11
    epoch_to_load = 10000
    learn_rate = 1e-3
    batch_size = 1
    run = 12
    num_epochs = 200000
    
    #pade neural operator controls
    pade_num_order = 9
    pade_denom_order = 8
                                                    #index of latent space variable to train


    print("Loading dataset and sample dataset...")
    Fs_data = np.load(Fs_data_file).astype(np.float32)
    sample_data = np.load(sample_data_file).astype(np.float32)
    scalings = np.load(scaling_factors_file).astype(np.float32)
    num_samples, num_timesteps, num_X, num_Y = Fs_data.shape
    _, parameter_dim = sample_data.shape
    
    #create pade neural operator 
    pade_neural_operator = Pade_Neural_Operator(parameter_dim, num_X, pade_num_order, pade_denom_order, batch_size, epsilon)
    
    if torch.cuda.is_available() and use_CUDA:
        device = torch.device("cuda:0")  # Specify the GPU device
        print("CUDA available")
        pade_neural_operator = pade_neural_operator.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(pade_neural_operator.parameters())  , lr=learn_rate)
    
    LHSs = np.zeros((num_samples, num_X*num_Y, num_X*num_Y))
    
    for sample_idx in range(num_samples):
        
        LHSs[sample_idx, :, :] = LHS_matrix.generate_LHS_matrix(sample_data[sample_idx, :], num_X)
        
    
        

    # Train the model
    if (load_model):
        print("Loading model...\n")
        pade_neural_operator.load_state_dict(torch.load("PNO_state_run_"+str(run_to_load)+"_"+str(epoch_to_load)+".pth"))

    elif (restart_training):
        print("Starting training with restart...\n")
        pade_neural_operator.load_state_dict(torch.load("PNO_state_run_"+str(run_to_load)+"_"+str(epoch_to_load)+".pth"))
        train_model(pade_neural_operator, criterion, optimizer, sample_data, Fs_data, run, learn_rate, batch_size, LHSs, scalings, num_epochs )
    else:
        print("Starting training...\n")
        train_model(pade_neural_operator, criterion, optimizer, sample_data, Fs_data, run, learn_rate, batch_size, LHSs, scalings, num_epochs)

    # Test with a new sample

    