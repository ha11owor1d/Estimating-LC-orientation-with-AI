# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 13:50:14 2026

@author: user
"""

import torch

path = "C:/Users/user/Downloads/FYP/Director_simulations/lc_orientation_model.pth"          # ← change if saved elsewhere

state_dict = torch.load(path, map_location="cpu")

print("Keys and shapes in the .pth file:\n")
for key, value in state_dict.items():
    print(f"{key: <45} {value.shape if hasattr(value, 'shape') else type(value)}")
    
    
state = torch.load("C:/Users/user/Downloads/FYP/Director_simulations/lc_orientation_model.pth", map_location="cpu")

# Example: look at first conv layer weights
print("First 5 values of conv1.weight.flatten():")
print(state['conv1.weight'].flatten()[:5])