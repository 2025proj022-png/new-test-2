import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from Dictionary_Causal_Estimator import causal_direction

mat_data = loadmat("Dataset/causal_symbolic_data.mat")

keys = [k for k in mat_data.keys() if not k.startswith('__')]

print("--- Top-Level Variables in the .mat file ---")
for key in keys[:1]:
    data = mat_data[key]
    print(f"Variable Name: {key}")
    print(f"  Shape: {data.shape}")
    print(f"  Dtype: {data.dtype}")
    if data.size > 0:
        print(f"  Snippet: {data.flat[:1].tolist()[0][0]}")
    print("-" * 30)