# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 12:51:01 2025

@author: CSU5KOR
"""

import os
os.chdir(r"C:\Users\CSU5KOR\OneDrive - Bosch Group\Coursera\pytorch_training")
import torch
import numpy as np
import pandas as pd

x = torch.tensor([1, 2, 3])

print("FROM PYTHON LISTS:", x)
print("TENSOR DATA TYPE:", x.dtype)

numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
torch_tensor_from_numpy = torch.from_numpy(numpy_array)

print("TENSOR FROM NUMPY:\n\n", torch_tensor_from_numpy)
df=pd.read_csv("data.csv")

all_values=df.values
tensors_from_df=torch.from_numpy(all_values)
zeros = torch.zeros(2, 3)

print("TENSOR WITH ZEROS:\n\n", zeros)
ones = torch.ones(2, 3)

print("TENSOR WITH ONES:\n\n", ones)
random = torch.rand(2, 3)

print("RANDOM TENSOR:\n\n", random)

# Range of numbers
range_tensor = torch.arange(0, 10, step=1)

print("ARANGE TENSOR:", range_tensor)

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

print("ORIGINAL TENSOR:\n\n", x)
print("\nTENSOR SHAPE:", x.shape)