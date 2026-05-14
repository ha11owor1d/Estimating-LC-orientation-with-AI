# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 13:50:14 2026

@author: user
"""

import torch
try:
    import cv2
    print("Open CV version:", cv2.__version__)
except Exception as e:
    print("Open CV failed:", str(e))
    
import numpy as np
print("Numpy:", np.__version__)

# path = "C:/Users/user/Downloads/FYP/director_model1.pth"          # ← change if saved elsewhere

# state_dict = torch.load(path, map_location="cpu")

print(torch.__version__)
# for key, value in state_dict.items():
#     print(f"{key: <45} {value.shape if hasattr(value, 'shape') else type(value)}")
    
    
# state = torch.load("C:/Users/user/Downloads/FYP/director_model1.pth", map_location="cpu")

# # Example: look at first conv layer weights
# print("First 5 values of conv1.weight.flatten():")
# print(state['conv1.weight'].flatten()[:5])