#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 13:34:31 2025

@author: tjards
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_history(history, save_path=None):
   
    plt.figure(figsize=(7, 5))
    plt.plot(history['train'], label='train')
    plt.plot(history['valid'],   label='val')
    plt.xlabel('epoch')
    plt.ylabel('MSE loss')
    plt.title('Training history')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
    
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
        
    plt.show()

def plot_polar(Z, save_path=None):

    assert Z.shape[2] == 2, "plot_polar requires latent_dim == 2"

    z1 = Z[:, :, 0]
    z2 = Z[:, :, 1]
    R = np.sqrt(z1**2 + z2**2)
    theta = np.arctan2(z2, z1)

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection="polar")
    for i in range(Z.shape[1]):
        ax.plot(theta[:, i], R[:, i], alpha=0.6)
    ax.set_title("Latent polar trajectories (R, θ)")
    plt.tight_layout()
    
    if save_path:
    
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
        
    plt.show()

def plot_latent_over_time(Z, agent_idx=None, save_path=None, title="Latent over time"):

    assert Z.ndim == 3, "Z_TND must have shape (T, N, d_latent)"
    T, N, d = Z.shape

    plt.figure(figsize=(10, 5))
    if agent_idx is None:
        # plot dim 0 for all agents, dotted dim 1
        for i in range(N):
            t = np.arange(T)
            plt.plot(t, Z[:, i, 0], lw=1.0, alpha=0.4)
            if d > 1:
                plt.plot(t, Z[:, i, 1], lw=1.0, alpha=0.25, linestyle="dotted")
        plt.title(f"{title} (all agents)")
    else:
        t = np.arange(T)
        for k in range(d):
            plt.plot(t, Z[:, agent_idx, k], lw=1.5, label=f"z{k+1}")
        plt.title(f"{title} — agent {agent_idx}")
        plt.legend()

    plt.xlabel("time step")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
    
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
        
    plt.show()