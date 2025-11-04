#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 19:15:03 2025

Why this step?

build Loader objects so that PyTorch can handle mini-batching, 
shuffling, and (if needed later) efficient GPU feeding automatically.

note: if using GPU, make sure to set pin_memory flag to True

@author: tjards
"""

# import stuff
import torch

# custom class for my static data subset
class StaticSubset(torch.utils.data.Dataset):
    
    def __init__(self, X_norm, idx):
        
        # good practice to pull down the parent class methods
        super().__init__()
        self.X = torch.from_numpy(X_norm[idx])
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, i):
        x = self.X[i]
        return x, x
    
# returns the actual Pytorch Loaders 
def build_loader(X_norm, idx_train, idx_valid, batch_size):
    
    # training
    ds_train = StaticSubset(X_norm, idx_train)
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size = batch_size, shuffle = True, pin_memory = False)

    # validation
    ds_valid = StaticSubset(X_norm, idx_valid)
    dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size = batch_size, shuffle = False, pin_memory = False)

    return dl_train, dl_valid
    
# this is for when I use sequences
def build_seq_loader(X_t, X_tp1, U_t, idx_train, idx_valid, batch_size):
    
    # training
    X_t_train    = torch.from_numpy(X_t[idx_train]).float()
    X_tp1_train  = torch.from_numpy(X_tp1[idx_train]).float()
    U_t_train    = torch.from_numpy(U_t[idx_train]).float()
    
    ds_train = torch.utils.data.TensorDataset(X_t_train, X_tp1_train, U_t_train)
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)

    # validation
    X_t_valid    = torch.from_numpy(X_t[idx_valid]).float()
    X_tp1_valid  = torch.from_numpy(X_tp1[idx_valid]).float()
    U_t_valid    = torch.from_numpy(U_t[idx_valid]).float()

    ds_valid = torch.utils.data.TensorDataset(X_t_valid, X_tp1_valid, U_t_valid)
    dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=batch_size, shuffle=False)

    return dl_train, dl_valid   
    
    
    
    