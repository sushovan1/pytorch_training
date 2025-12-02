# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 11:42:01 2025

@author: CSU5KOR
"""
import os
import torch
import helper_utils
data_dir=r"C:\Users\CSU5KOR\OneDrive - Bosch Group\Coursera\pytorch_training"
os.chdir(data_dir)
import helper_utils
if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"
import torch.nn as nn
import torch.optim as optim

distances = torch.tensor([
    [1.0], [1.5], [2.0], [2.5], [3.0], [3.5], [4.0], [4.5], [5.0], [5.5],
    [6.0], [6.5], [7.0], [7.5], [8.0], [8.5], [9.0], [9.5], [10.0], [10.5],
    [11.0], [11.5], [12.0], [12.5], [13.0], [13.5], [14.0], [14.5], [15.0], [15.5],
    [16.0], [16.5], [17.0], [17.5], [18.0], [18.5], [19.0], [19.5], [20.0]
], dtype=torch.float32)

# Corresponding delivery times in minutes
times = torch.tensor([
    [6.96], [9.67], [12.11], [14.56], [16.77], [21.7], [26.52], [32.47], [37.15], [42.35],
    [46.1], [52.98], [57.76], [61.29], [66.15], [67.63], [69.45], [71.57], [72.8], [73.88],
    [76.34], [76.38], [78.34], [80.07], [81.86], [84.45], [83.98], [86.55], [88.33], [86.83],
    [89.24], [88.11], [88.16], [91.77], [92.27], [92.13], [90.73], [90.39], [92.98]
], dtype=torch.float32)
torch.manual_seed(27)
model=nn.Sequential(nn.Linear(1, 3),
                    nn.ReLU(),
                    nn.Linear(3,1)
                )

model_paramaeters=sum(p.numel() for p in model.parameters() if p.requires_grad)
"""
A New Step: Normalizing the Data
Before building your model, you will apply a quick data preparation step called normalization. This is a standard technique that makes the training process more stable and effective by adjusting the scale of the data. This adjustment helps prevent large distance values from dominating the learning process and keeps gradients stable during training. You will explore this topic in greater detail in a later module.

You will calculate the mean and standard deviation for the distances and times tensors.
You will then apply standardization to each tensor using its respective mean and standard deviation, which creates new normalized tensors named distances_norm and times_norm.
This specific technique is called standardization (or z-score normalization), which converts the original data from 1.0 to 20.0 miles and approximately 7 to 93 minutes into a new, normalized scale.
"""
distances_mean = distances.mean()
distances_std = distances.std()

# Calculate the mean and standard deviation for the 'times' tensor
times_mean = times.mean()
times_std = times.std()

# Apply standardization to the distances.
distances_norm = (distances - distances_mean) / distances_std

# Apply standardization to the times.
times_norm = (times - times_mean) / times_std

loss_function=nn.MSELoss()
optimizer=optim.SGD(model.parameters(),lr=0.01)

for epoch in range(3000):
    optimizer.zero_grad()
    outputs=model(distances_norm)
    loss=loss_function(outputs,times_norm)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        helper_utils.plot_training_progress(
            epoch=epoch,
            loss=loss,
            model=model,
            distances_norm=distances_norm,
            times_norm=times_norm
        )
print("\nTraining Complete.")
print(f"\nFinal Loss: {loss.item()}")

#predict
distance_to_predict = 5.1
distance_torch=torch.tensor([[distance_to_predict]],dtype=torch.float32)
distance_torch_norm=(distance_torch-distances_mean)/distances_std

with torch.no_grad():
    output=model(distance_torch_norm)
    time_output=output*times_std+times_mean
    print(f"predicted_value is {time_output.item():.1f}")