#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 19:06:06 2025

@author: tjards

consistency-based method
to infer who are neighbours in the trajectory dataset

"""

# import stuff
import numpy as np
import matplotlib.pyplot as plt

# configs
eps             = 1e-9
window_mode     = 'fixed'   # 'fixed' (full dataset), sliding', 'gathering' 
outer_window_0  = 100           # sweeps across dataset (initial)
inner_window    = [0.25, 1.0]   # within the outer window, we may further window
max_gather      = 1000          # we can cap the window (for 'gather')
w_pers          = 1.0           # weight of persistence
w_vari          = 1.0           # wright of variability

# build history of proximities
def build_proximity_history(traj):
    
    nSamples, _, nAgents = traj.shape
    
    pos     = np.transpose(traj[:, 0:3, :], (0, 2, 1)) # nSamples, nAgents, 3
    vel     = np.transpose(traj[:, 3:6, :], (0, 2, 1)) 
    speed   = np.linalg.norm(vel, axis=2, keepdims=True)
    u       = vel/(speed + eps) 

    proximity_history = []
    
    for t in range(nSamples):
        
        relative_distances  = pos[t][None, :, :] - pos[t][:, None, :]
        
        # build a matrix with elements ij = projection of relatives distances onto u
        offset_matrix       = np.einsum('ijk,ik->ij', relative_distances, u[t])
        
        # ensure diagonals are NaN
        np.fill_diagonal(offset_matrix, np.nan)
        proximity_history.append(offset_matrix)
        
    return proximity_history

# infer neighbours based on consistency of proximity
def infer_consistent_neighbours(traj, proximity_history):
    
    nSamples, _, _ = traj.shape
    
    # find the edges of the window and clamp
    left    = int(np.floor(inner_window[0]*nSamples))
    left    = max(0, min(left, nSamples-1))
    right   = int(np.floor(inner_window[1]*nSamples))
    right   = max(left+1, min(right, nSamples))

    window_stack = np.stack(proximity_history[left:right], axis=0)
        
    # masks
    ahead_mask = window_stack > 0
    behind_mask = window_stack < 0
    
    # scoring criteria
    
    # gaps 
    gap_ahead       = np.where(ahead_mask, window_stack, np.nan)
    gap_behind      = np.where(behind_mask, -window_stack, np.nan)
    
    # persistence
    persist_ahead   = np.nanmean(ahead_mask, axis = 0)
    persist_behind  = np.nanmean(behind_mask, axis = 0)
    
    # variability  
    variability_ahead     = med_abs_dev(gap_ahead, axis = 0)
    variability_behind    = med_abs_dev(gap_behind, axis = 0)
    
    # proximity    
    _count_ahead = np.sum(ahead_mask, axis=0)            
    _count_behind = np.sum(behind_mask, axis=0)           
    
    # sums over valid samples
    _sum_ahead = np.nansum(gap_ahead,  axis=0)         
    _sum_behind = np.nansum(gap_behind, axis=0)         
    
    # safe means: NaN when count == 0, no RuntimeWarning
    mean_ahead  = np.divide(_sum_ahead, _count_ahead, out=np.full_like(_sum_ahead, np.nan), where=_count_ahead > 0)
    mean_behind = np.divide(_sum_behind, _count_behind, out=np.full_like(_sum_behind, np.nan), where=_count_behind > 0)
    
    # exclude selves
    for criterion in (persist_ahead, persist_behind, variability_ahead, variability_behind, mean_ahead, mean_behind):
        np.fill_diagonal(criterion, np.nan)
        
    # compute weighted score
    score_ahead     = w_pers * persist_ahead    - w_vari * (variability_ahead / (mean_ahead + eps))
    score_behind    = w_pers * persist_behind   - w_vari * (variability_behind / (mean_behind + eps))

    # force NaNs when no persistence
    score_ahead     = np.where(persist_ahead > 0, score_ahead, np.nan)
    score_behind    = np.where(persist_behind > 0, score_behind, np.nan)
    
    # pick best (for each agent)
    ahead_best  = np.nanargmax(score_ahead, axis=1)  
    behind_best = np.nanargmax(score_behind, axis=1)   
    
    return ahead_best, behind_best

# median absolute deviation 
def med_abs_dev(x, axis=None):

    # used masked (ma) arrays
    mx  = np.ma.masked_invalid(x)                       # mask NaNs and Infs                      
    med = np.ma.median(mx, axis=axis, keepdims=True)    # median    
    mad = np.ma.median(np.ma.abs(mx - med), axis=axis)  # abs value of deviation
    
    return mad.filled(np.nan) 

def compute_neighbourhoods(states_all):
    
    nSamples, nStates, nAgents = states_all.shape
    
    ahead_idx  = np.full((nSamples, nAgents), -1, dtype=np.int32)
    behind_idx = np.full((nSamples, nAgents), -1, dtype=np.int32)

    proximity_history = build_proximity_history(states_all)
    
    if window_mode == 'fixed' :
        
        consistent_ahead, consistent_behind = infer_consistent_neighbours(states_all, proximity_history)
        
        ahead_idx[:]    =  consistent_ahead[None, :]  
        behind_idx[:]   =  consistent_behind[None, :] 
                    
    else:
        
        # start with initial value
        outer_window = outer_window_0
        
        # for each timestep
        for t in range(nSamples):
            
            # ensure we have enough samples
            if t + 1 < outer_window_0:
                continue
            
            # we can grow this outer window if gather data along the way
            if window_mode == 'gathering':
                outer_window = max(outer_window_0, min(t + 1, max_gather))

            # define start and end of data
            start   = max(0, t - outer_window + 1)
            end     = t + 1
            
            # grab subset of data and the proximity history
            states_subset   = states_all[start:end, :, :]          
            proximity_subset     = proximity_history[start:end]  
            
            try:            
                consistent_ahead, consistent_behind = infer_consistent_neighbours(states_subset, proximity_subset)
            # if maps to empty slices
            except ValueError:
                continue 
            # any other errors
            except Exception:
                continue
    
            ahead_idx[t,:]    =  consistent_ahead 
            behind_idx[t,:]   =  consistent_behind 

    return ahead_idx, behind_idx
        
 #%% plots

def plot_neighbor_timelines(ahead_idx, behind_idx, title_prefix=""):
 
    T, N = ahead_idx.shape

    # Ahead raster
    plt.figure()
    plt.imshow(ahead_idx.T, aspect='auto', interpolation='nearest')
    plt.xlabel("time index")
    plt.ylabel("agent (i)")
    plt.title(f"{title_prefix}Ahead neighbor index per agent over time")
    plt.colorbar(label="neighbor j")
    plt.tight_layout()

    # Behind raster
    plt.figure()
    plt.imshow(behind_idx.T, aspect='auto', interpolation='nearest')
    plt.xlabel("time index")
    plt.ylabel("agent (i)")
    plt.title(f"{title_prefix}Behind neighbor index per agent over time")
    plt.colorbar(label="neighbor j")
    plt.tight_layout()

    plt.show()
    
#%% test

'''
import data.data_manager as dm
import os

# locations
DIR_DATA    = 'data'
FILE_DATA   = os.path.join(DIR_DATA, 'data.h5') 
DIR_OUT     = 'models'
os.makedirs(DIR_OUT, exist_ok=True)

# import data
states_all, _, _            = dm.extract_features(FILE_DATA)

# pull params
nSamples, nStates, nAgents  = states_all.shape
assert nStates == 6, f'Expected 6 states dimensions (pos, vel), got {nStates}'

# run
ahead_idx, behind_idx       = compute_neighbourhoods(states_all)

# plot
plot_neighbor_timelines(ahead_idx, behind_idx, title_prefix='')
''' 
        
        
        