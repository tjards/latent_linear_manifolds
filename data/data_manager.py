#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 15:18:08 2025

@author: tjards
"""


import h5py
import copy
import numpy as np

def load_data_HDF5(group, key, file_path_h5):    
    # open the HDF5 file
    with h5py.File(file_path_h5, 'r') as file:
        # check if group exists in the file
        if group in file:
            # access group
            history_group = file[group]
            # check if this key exists in group
            if key in history_group:
                # pull the data for that key
                dataset = history_group[key]
                # pull the values 
                values = dataset[:]  
            else:
                print("Key not found within group.")
        else:
            print("Group not found in the HDF5 file.")  
    # return the key and values
    return key, values

def extract_features(data_file_path):
    
    start_time = 1500
    
    _, t_all          = load_data_HDF5('History', 't_all', data_file_path)
    _, states_all     = load_data_HDF5('History', 'states_all', data_file_path)
    _, cmds_all       = load_data_HDF5('History', 'cmds_all', data_file_path)
    _, targets_all    = load_data_HDF5('History', 'targets_all', data_file_path)
    _, lemni_all      = load_data_HDF5('History', 'lemni_all', data_file_path) 
    
    states_all_centered = states_all.copy()
    
    # ensure states are all centered on origin
    states_all_centered[:,0:3,:] = states_all_centered[:,0:3,:] - targets_all[:,0:3,:] 
    
    #nVeh = states_all.shape[2]
    #x_all = states_all[:, 0, :]
    #y_all = states_all[:, 1, :]
    #z_all = states_all[:, 2, :]
    
    return states_all_centered[start_time::,:,:], cmds_all[start_time::,:,:], lemni_all[start_time::,:,:]


        
        
        
        
        
    
    
    
    
    
    
    
