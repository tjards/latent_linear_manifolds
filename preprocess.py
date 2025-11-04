#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 19:55:25 2025

@author: tjards
"""

# import stuff
import numpy as np

# configs
eps = 1e-6

def build_features(states_all, aheads, behinds):
    
    nSamples, nStates, nAgents = states_all.shape
    assert nStates == 6, f"Expected 6 state dims (pos+vel); got {nStates}"
    
    pos     = np.transpose(states_all[:, 0:3, :], (0, 2, 1)) # nSamples, nAgents, 3
    vel     = np.transpose(states_all[:, 3:6, :], (0, 2, 1)) 
    
    feats = []
    
    for t in range(nSamples):
        
        #if not selected_agent:
        
        # self 
        pi = pos[t]
        vi = vel[t]

        # neighbors
        pj = pos[t, aheads[t]]
        vj = vel[t, aheads[t]]
        pk = pos[t, behinds[t]]
        vk = vel[t, behinds[t]]

        # else:
        #     assert 0 <= selected_agent < nAgents, f"agent {selected_agent} out of range (0..{nAgents-1})"
            
        #     pi = pos[t, selected_agent:selected_agent+1, :]  # (1, 3)
        #     vi = vel[t, selected_agent:selected_agent+1, :]
            
        #     a_idx = aheads[t, selected_agent]
        #     b_idx = behinds[t, selected_agent]
            
        #     pj = pos[t, a_idx:a_idx+1, :]  # (1, 3)
        #     vj = vel[t, a_idx:a_idx+1, :]
        #     pk = pos[t, b_idx:b_idx+1, :]
        #     vk = vel[t, b_idx:b_idx+1, :]

        # deltas
        dp_a = pj - pi
        dv_a = vj - vi
        dp_b = pk - pi
        dv_b = vk - vi

        # build a feature matrix
        x_t = np.concatenate([dp_a, dv_a, dp_b, dv_b], axis=1)  
        
        # append this timestep
        feats.append(x_t)

    
    X = np.concatenate(feats, axis=0).astype(np.float32)  
    
    return X        

# for when I want to model with the controls
def build_controls(cmds_all):
    
    nSamples, nControls, _ = cmds_all.shape
    assert nControls == 3, f"Expected 3 control dims (x,y,z); got {nControls}"
    
    control_feats = []
    
    for t in range(nSamples):
        
        u_t = cmds_all[t].T
        control_feats.append(u_t)
        
    U = np.concatenate(control_feats, axis = 0).astype(np.float32)
    
    return U
        
def split_indices(X, nSamples, train_frac, shuffle = True, normalize = True):
    
    #random_seed = 0.0 
    assert 0.0 < train_frac < 1.0, f"Expected training frac [0,1]; got {train_frac}"
    nTrain = int(train_frac*nSamples)
    
    if shuffle:
        
        rng = np.random.default_rng(42)
        perm = rng.permutation(nSamples)
        idx_train = perm[:nTrain]
        idx_valid = perm[nTrain:]
    
    else:
        
        idx_train = np.arange(0, nTrain)
        idx_valid = np.arange(nTrain, nSamples)
        
    return idx_train, idx_valid

# given time indices, find rows when stacked agent-wise
def time_to_rows(idx_time, nAgents):
    
    if idx_time is None or len(idx_time) == 0:
        return np.array([], dtype=np.int64)

    row_idx = np.concatenate([np.arange(t * nAgents, (t + 1) * nAgents) for t in idx_time]).astype(np.int64)
    
    return row_idx
  
def normalize(X, norm_set = None):
    
    # accepts norm_set, if you want to normalize across a subset of the data
    if norm_set is None:
        idx = np.arange(len(X))
    else:
        idx = norm_set
    
    mean    = X[idx].mean(0, keepdims = True)
    std     = np.maximum(X[idx].std(0, keepdims = True), eps)
    X_norm  = ((X - mean) / std)  
    
    return X_norm, mean, std
        
def pull_selected_agent(idx_train_time, idx_valid_time, nAgents, selected_agent):
    
    train_rows_agent = np.array(
        [t * nAgents + selected_agent for t in idx_train_time],
        dtype=np.int64
        )
    valid_rows_agent = np.array(
        [t * nAgents + selected_agent for t in idx_valid_time],
        dtype=np.int64
        )
    
    return train_rows_agent, valid_rows_agent

            
        
        
        
    
    
    
    
    