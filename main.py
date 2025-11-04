#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 19:19:21 2025

@author: tjards


"""

#%% imports
# ---------
import os
import numpy as np
import data.data_manager as dm
import neighbour_finder as nf
import preprocess 
import loader 
import torch
import autoencoder as ae
import json
from visualization import plotting

#%% config
# --------

# locations
DIR_DATA    = 'data/data/'
FILE_DATA   = os.path.join(DIR_DATA, 'data.h5') 

DIR_OUT     = 'models'
os.makedirs(DIR_OUT, exist_ok = True)

DIR_VIZ     = 'visualization'
DIR_PLOTS   = os.path.join(DIR_VIZ, 'plots') 
os.makedirs(DIR_PLOTS, exist_ok = True)

# learning params
TRAIN_FRAC      = 0.5
DEVICE          = 'cpu'     # train on cpu, cuda, ... etc
LATENT_DIM      = 2         # how many dimensions in the latent space (nominally 2)
BATCH_SIZE      = 512
EPOCHS          = 100
LEARNING_RATE   = 1e-4
PATIENCE        = 10        # how many epochs to tolerate no improvements in loss 
USE_CONTROL     = True      # uses control inputs to enforce some kind of linearization
WEIGHT_ERR      = 0.5
WEIGHT_DYN      = 0.5       # for case when USE_CONTROL (i.e., linearization enforcement) 

# other
SELECTED_AGENT  = 0       # None = train in all agent (global); else, train selected agent (local)

#%% load and proprocess
# ---------------------

states_all, cmds_all, _     = dm.extract_features(FILE_DATA)
nSamples, nStates, nAgents  = states_all.shape
assert nStates == 6, f'[assert] Expected 6 states dimensions (pos, vel), got {nStates}'

# find the neighbours 
ahead_idx, behind_idx = nf.compute_neighbourhoods(states_all)
plot_neighbors = False 
if plot_neighbors:
    nf.plot_neighbor_timelines(ahead_idx, behind_idx, title_prefix="")
    
# build features
X = preprocess.build_features(states_all, ahead_idx, behind_idx)

# split data 
idx_train_time, idx_valid_time = preprocess.split_indices(X, nSamples, TRAIN_FRAC, shuffle = True, normalize=True)
train_rows_all = preprocess.time_to_rows(idx_train_time, nAgents)
valid_rows_all = preprocess.time_to_rows(idx_valid_time, nAgents)

# if we are training on a selected agent (i.e., local)
if SELECTED_AGENT is not None:

    train_rows, valid_rows = preprocess.pull_selected_agent(
        idx_train_time, 
        idx_valid_time, 
        nAgents, 
        SELECTED_AGENT)
else:
    train_rows, valid_rows = train_rows_all, valid_rows_all
    
# normalize (states)
X_norm, mean_X, std_X = preprocess.normalize(X, norm_set = train_rows_all)  

# if we are using the control inputs to shape the model (nominally, yes)
if not USE_CONTROL:

    # loader (custom for Pytorch)
    dl_train, dl_valid = loader.build_loader(X_norm, 
                                             train_rows, 
                                             valid_rows, 
                                             BATCH_SIZE)    
    # build and train model
    model = ae.Model(d_in=X_norm.shape[1], d_out=LATENT_DIM, d_u = 0)
    history = model.fit(
        dl_train, 
        dl_valid, 
        epochs = EPOCHS, 
        device = DEVICE, 
        learning_rate = LEARNING_RATE, 
        patience = PATIENCE)
    
else:
    
    # build the control features
    U = preprocess.build_controls(cmds_all)

    # normalize them             
    U_norm, mean_U, std_U = preprocess.normalize(U, norm_set = train_rows)
    
    d_in = X_norm.shape[1]
    d_u  = U_norm.shape[1]   
    
    X_reshaped = X_norm.reshape(nSamples, nAgents, d_in)
    U_reshaped = U_norm.reshape(nSamples, nAgents, d_u)
    
    # shift in time (lose last sample)
    X_t    = X_reshaped[:-1, :, :].reshape((nSamples - 1) * nAgents, d_in)
    X_tp1  = X_reshaped[1:,  :, :].reshape((nSamples - 1) * nAgents, d_in)
    U_t    = U_reshaped[:-1, :, :].reshape((nSamples - 1) * nAgents, d_u)
    
    # clip time indices as well
    train_time_seq = idx_train_time[idx_train_time < (nSamples - 1)]
    valid_time_seq = idx_valid_time[idx_valid_time < (nSamples - 1)]
    
    # find rows in the sequence
    train_rows_seq = preprocess.time_to_rows(train_time_seq, nAgents)
    valid_rows_seq = preprocess.time_to_rows(valid_time_seq, nAgents)
    
    if SELECTED_AGENT is not None:
        
        train_rows_seq, valid_rows_seq = preprocess.pull_selected_agent(
            train_time_seq,
            valid_time_seq,
            nAgents,
            selected_agent=SELECTED_AGENT,
            )
    
    # build sequence loaders
    dl_train, dl_valid = loader.build_seq_loader(X_t, 
                                                 X_tp1, 
                                                 U_t, 
                                                 train_rows_seq, 
                                                 valid_rows_seq, 
                                                 BATCH_SIZE)
    
    # model that knows it has control
    model = ae.Model(d_in  = d_in, d_out = LATENT_DIM, d_u = d_u)
    
    history = model.fit(
        dl_train,
        dl_valid,
        epochs        = EPOCHS,
        device        = DEVICE,
        learning_rate = LEARNING_RATE,
        patience      = PATIENCE,
        weight_err    = WEIGHT_ERR,
        weight_dyn    = WEIGHT_DYN,    
    )

#%% save
# ------

# weights
torch.save(model.state_dict(), os.path.join(DIR_OUT, 'weights.pth'))

# normalization values
np.savez(os.path.join(DIR_OUT, 'normalizer_states.npz'), mean = mean_X, std = std_X)
if USE_CONTROL:
    np.savez(os.path.join(DIR_OUT, 'normalizer_cmds.npz'), mean = mean_U, std  = std_U)

# metadata
metadata = {
    'input_dim': int(X_norm.shape[1]),
    'output_dim': int(LATENT_DIM),
    'history': history,
    'nSamples': nSamples, 
    'nStates': nStates, 
    'nAgents': nAgents,
    'learning_rate': LEARNING_RATE,
    'training_fraction': TRAIN_FRAC,  
    'device': DEVICE,            
    'batch_size': BATCH_SIZE,  
    'epochs': EPOCHS,     
    'patience': PATIENCE,  
    'use_control': USE_CONTROL,         
    'weight_err': WEIGHT_ERR,           
    'weight_dyn': WEIGHT_DYN,                
    }
with open(os.path.join(DIR_OUT, "meta.json"), "w") as f:
    json.dump(metadata, f, indent=2)

# neighbours
np.savez(os.path.join(DIR_OUT, 'neighbours.npz'), ahead_idx = ahead_idx, behind_idx = behind_idx)

print(f"[save] artifacts written to /{DIR_OUT}/")

#%% analysis/plots
# ----------------

selected_agent_plots = SELECTED_AGENT or 0

# encode whole dataset with model
Z = ae.encode_all(model, X_norm, DEVICE, BATCH_SIZE)

# reconstruct with model
X_hat_norm, X_hat = ae.reconstruct_all(model, X_norm, DEVICE, BATCH_SIZE, mean_X, std_X)

# compute error
mse = ae.per_sample_mse(X, X_hat)

# reshape Z
Z_reshaped = Z.reshape(nSamples, nAgents, LATENT_DIM)

# plots
plotting.plot_history(history, save_path=os.path.join(DIR_PLOTS, "history.png"))
if LATENT_DIM == 2:
    plotting.plot_polar(Z_reshaped, save_path=os.path.join(DIR_PLOTS, "polar.png"))
plotting.plot_latent_over_time(Z_reshaped, 
                               agent_idx=selected_agent_plots, 
                               save_path=os.path.join(DIR_PLOTS, f"latent_over_time_agent{selected_agent_plots}.png"))



