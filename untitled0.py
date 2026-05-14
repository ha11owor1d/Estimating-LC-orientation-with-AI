# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 18:06:45 2026

@author: user
"""
import numpy as np
print("NumPy early import OK:", np.__version__)  # should print 2.0.2

import sys
print("sys.path:", sys.path)

import sys
import traceback

print("Python version:", sys.version)
print("NumPy version:", sys.modules.get('numpy', 'not imported').__version__ if 'numpy' in sys.modules else "not yet")
print("OpenCV version:", sys.modules.get('cv2', 'not imported').__version__ if 'cv2' in sys.modules else "not yet")
print("PyTorch version:", sys.modules.get('torch', 'not imported').__version__ if 'torch' in sys.modules else "not yet")

print("\n--- Trying imports one by one ---")

try:
    import numpy as np
    print("NumPy imported OK")
    # Basic test
    arr = np.array([1, 2, 3])
    print("NumPy array works:", arr)
except Exception as e:
    print("NumPy import FAILED:")
    traceback.print_exc()

try:
    import cv2
    print("OpenCV imported OK (version:", cv2.__version__, ")")
except Exception as e:
    print("OpenCV import FAILED:")
    traceback.print_exc()

try:
    import albumentations
    print("Albumentations imported OK")
except Exception as e:
    print("Albumentations import FAILED:")
    traceback.print_exc()

# Add suspects here — try one at a time or all
suspects = ['insightface', 'numba', 'bottleneck', 'scipy', 'skimage', 'sklearn']

for lib in suspects:
    try:
        __import__(lib)
        print(f"{lib} imported OK")
    except ImportError:
        print(f"{lib} not installed")
    except Exception as e:
        print(f"{lib} import FAILED:")
        traceback.print_exc()

# If the error happens later (e.g., during usage), try:
if 'albumentations' in sys.modules:
    try:
        import albumentations as A
        transform = A.Compose([A.RandomCrop(100, 100)])
        print("Albumentations basic transform OK")
    except Exception as e:
        print("Albumentations usage FAILED:")
        traceback.print_exc()