#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 19:47:03 2025

my autoencoder for modelling swarm


@author: tjards
"""

# import stuff
import torch
import numpy as np

# configs
eps = 1e-9

#%% custom class for autoencoder
# ------------------------------

class Model(torch.nn.Module):
    
    def __init__(self, d_in, d_out, d_u = 0):
        
        # good practice to pull down the parent class methods
        super().__init__()
        
        # encode
        self.encode = torch.nn.Sequential(
            torch.nn.Linear(d_in,     64), torch.nn.ReLU(),
            torch.nn.Linear(64,       32), torch.nn.ReLU(),
            torch.nn.Linear(32,       d_out),
            )
        
        # decode
        self.decode = torch.nn.Sequential(
            torch.nn.Linear(d_out,    32), torch.nn.ReLU(),
            torch.nn.Linear(32,       64), torch.nn.ReLU(),
            torch.nn.Linear(64,       d_in),
            )
        
        # control inialization: x_dot = A * x + B * u 
        self.use_control = (d_u > 0) # if I feed control, in will use the control
        self.A = torch.nn.Parameter(torch.eye(d_out)) # system matrix
        if self.use_control:
            self.B = torch.nn.Parameter(0.01 * torch.randn(d_out, d_u)) # input matrix
        else:
            self.B = None
        
        
    def forward(self, x):
        
        z       = self.encode(x)
        x_hat   = self.decode(z)
        
        return x_hat, z

    def fit(self, dl_train, dl_valid, epochs = 100, device = 'cpu', learning_rate = 1e-3, patience = 10, weight_err = 1.0, weight_dyn = 1.0):
        
        self.to(device)
        opt             = torch.optim.Adam(self.parameters(), lr = learning_rate)
        loss_fn         = torch.nn.MSELoss()
        training_loss   = 0.0
        validation_loss = 0.0
        
        # initialize
        wait        = 0
        history = {"train": [], "valid": []}
        best_val    = float('inf')
        best_state  = {k: v.detach().cpu().clone() 
                       for k, v in self.state_dict().items()
                       } # a detached, CPU-based, cloned copy of all model parameters 

        # move through the epochs
        for epoch in range(epochs):
            
            # === TRAIN === #
            
            # set model to train mode (does not actually do training)
            self.train()
            
            # initialize 
            training_loss_accumulated   = 0.0
            training_sample_count       = 0
            
            # iterate through minibatches
            #for x_batch, y_batch in dl_train:
            for batch in dl_train:
                
                opt.zero_grad(set_to_none = True) # reset all grads to None
                
                # just states
                if not self.use_control:
                
                    x_batch, y_batch = batch
                    
                    x_batch = x_batch.to(device, non_blocking  = True) # nonblocking allows asynch ops (faster)
                    y_batch = y_batch.to(device, non_blocking  = True)
                    #opt.zero_grad(set_to_none = True) # reset all grads to None
                    
                    # move forward
                    x_hat, _ = self(x_batch)
                    
                    # compute loss
                    loss = loss_fn(x_hat, y_batch) 
                    
                    batch_size = x_batch.size(0)
                
                # also with control (dropping _batch notion for convenience )
                else:
                    
                    x_t, x_tp1, u_t = batch

                    x_t    = x_t.to(device, non_blocking=True)
                    x_tp1  = x_tp1.to(device, non_blocking=True)
                    u_t    = u_t.to(device, non_blocking=True)
                    
                    # move forward
                    x_hat_t,   z_t   = self(x_t)
                    x_hat_tp1, z_tp1 = self(x_tp1)
                    
                    # compute loss (error)
                    loss_t      = loss_fn(x_hat_t, x_t)
                    loss_tp1    = loss_fn(x_hat_tp1, x_tp1)
                    loss_err        = 0.5 * (loss_t + loss_tp1)
                    
                    # compute loss (dynamics): make a linear prediction
                    z_tp1_pred = (z_t @ self.A.T)
                    if self.B is not None:
                        z_tp1_pred = z_tp1_pred + (u_t @ self.B.T)
                    loss_dyn = torch.mean((z_tp1 - z_tp1_pred) ** 2)
                    
                    # total loss
                    loss = weight_err*loss_err + weight_dyn*loss_dyn
                    
                    batch_size = x_t.size(0)
                    
                loss.backward()
                opt.step()
                
                #training_loss_accumulted    += loss.item() * x_batch.size(0)
                training_loss_accumulated    += loss.item() * batch_size
                #training_sample_count       += x_batch.size(0)
                training_sample_count       += batch_size
                
            training_loss = training_loss_accumulated / max(1, training_sample_count)
            
            # === VALIDATE === #
            
            # set model to evaluation mode
            self.eval() 
            
            # initialize 
            validation_loss_accumulated    = 0.0
            validation_sample_count       = 0
            
            # don't need to use up memory for predictions here
            with torch.no_grad():
                
                #for x_batch, y_batch in dl_valid:
                for batch in dl_valid:
                 
                    if not self.use_control:  
                        
                        x_batch, y_batch = batch
                 
                        x_batch = x_batch.to(device, non_blocking  = True) # nonblocking allows asynch ops (faster)
                        y_batch = y_batch.to(device, non_blocking  = True) 
                        
                        x_hat, _ = self(x_batch)
                        
                        loss = loss_fn(x_hat, y_batch)
                        
                        batch_size = x_batch.size(0)
                        
                    else:
                        
                        x_t, x_tp1, u_t = batch
                        
                        x_t    = x_t.to(device, non_blocking=True)
                        x_tp1  = x_tp1.to(device, non_blocking=True)
                        u_t    = u_t.to(device, non_blocking=True)
                        
                        # move forward
                        x_hat_t,   z_t   = self(x_t)
                        x_hat_tp1, z_tp1 = self(x_tp1)
                        
                        # compute loss (error)
                        loss_t      = loss_fn(x_hat_t, x_t)
                        loss_tp1    = loss_fn(x_hat_tp1, x_tp1)
                        loss_err        = 0.5 * (loss_t + loss_tp1)
                        
                        # compute loss (dynamics): make a linear prediction
                        z_tp1_pred = (z_t @ self.A.T)
                        if self.B is not None:
                            z_tp1_pred = z_tp1_pred + (u_t @ self.B.T)
                        loss_dyn = torch.mean((z_tp1 - z_tp1_pred) ** 2)
                        
                        # total loss
                        loss = weight_err*loss_err + weight_dyn*loss_dyn
                        
                        batch_size = x_t.size(0)
                        
                        
                    #validation_loss_accumulated += loss.item() * x_batch.size(0)
                    validation_loss_accumulated += loss.item() * batch_size
                    #validation_sample_count += x_batch.size(0)
                    validation_sample_count += batch_size
                    
            validation_loss = validation_loss_accumulated / max(1, validation_sample_count)
            
            # === STORE === #
            
            history['train'].append(training_loss)
            history['valid'].append(validation_loss)
            print(f'epoch {epoch:03d} train {training_loss:.6f} valid {validation_loss:.6f}')
            
            # === EARLY STOPPING === #
            
            if validation_loss + eps < best_val:
                best_val = validation_loss 
                wait = 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
            else:
                wait += 1
                print(f'growing impatient, waiting: {wait} of {patience}')
                if wait > patience:
                    print('early stopping')
                    break 
        
        self.load_state_dict(best_state)
        return history
                
#%% helpers (mainly for analysis)
# -------------------------------

# Note: I use decorator with no_grad() to ensure these functions run with
#   gradient tracking disabled, thereby saving memory

@torch.no_grad()
def encode_all(model: torch.nn.Module, 
               X_norm: np.ndarray,
               device: str,
               batch: int):
    
    model.eval()
    Z = []
    
    for i in range(0, len(X_norm), batch):
        x_batch = torch.from_numpy(X_norm[i: i + batch]).to(device)
        _, z = model(x_batch)
        Z.append(z.cpu().numpy())
        
    return np.concatenate(Z, axis = 0)

@torch.no_grad()
def reconstruct_all(model: torch.nn.Module, 
               X_norm: np.ndarray,
               device: str,
               batch: int,
               mean: np.ndarray,
               std: np.ndarray
               ):
    
    model.eval()
    X_hat_norm = []
    
    for i in range(0, len(X_norm), batch):
        x_batch = torch.from_numpy(X_norm[i: i + batch]).to(device)
        x_hat, _ = model(x_batch)
        X_hat_norm.append(x_hat.cpu().numpy())
    
    X_hat_norm  = np.concatenate(X_hat_norm, axis = 0)
    X_hat       = X_hat_norm * std + mean
        
    return X_hat_norm, X_hat
            
def per_sample_mse(X_true, X_hat):
    
    return ((X_true - X_hat)**2).mean(axis = 1)

                 
        
        
        
        
        
