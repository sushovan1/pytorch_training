# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 11:03:57 2025

@author: CSU5KOR
"""

"""Building a Simple Neural Network
Welcome to the first lab of this course!

In this lab, you'll build and train your first neural network, a single neuron 
that learns patterns from data to make predictions.

You'll work with the delivery scenario from the lecture videos: 
You're a bike delivery person with a 7-mile delivery order. 
Your company promises delivery in under 30 minutes, 
and one more late delivery could put your job at risk. 
Can you make this delivery on time? Y
our neural network will learn from historical delivery data to help you decide.

Following the Machine Learning (ML) pipeline from the lecture videos, you will:

Prepare delivery data, the distances and times from past orders.
Build a simple neural network using PyTorch (just one neuron!).
Train it to find the relationship between distance and delivery time.
Predict whether you can make that 7-mile delivery in time."""
import os
import torch
if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"
    
import torch.nn as nn
import torch.optim as optim
path=r"C:\Users\CSU5KOR\OneDrive - Bosch Group\Coursera\pytorch_training"
os.chdir(path)
import helper_utils

# This line ensures that your results are reproducible and consistent every time.
torch.manual_seed(42)
####################################################################################
distances = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)

# Corresponding delivery times in minutes
times = torch.tensor([[6.96], [12.11], [16.77], [22.21]], dtype=torch.float32)
"""
Use nn.Sequential(nn.Linear(1, 1)) to create a linear model.
nn.Linear(1, 1): The first 1 means it takes one input (distance), and the second 1 means one neuron that is producing one output (predicted time).
This single linear layer will automatically manage the weight and bias parameters for you.
# Create a model with one input (distance) and one output (time)
"""
model = nn.Sequential(nn.Linear(1, 1))
# Define the loss function and optimizer
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
"""
You'll train for 500 epochs (complete passes through your data). During each epoch, these steps occur:

optimizer.zero_grad(): Clears gradients from the previous round. Without this, PyTorch would accumulate adjustments, which could break the learning process.

outputs = model(distances): Performs the "forward pass", where the model makes predictions based on the input distances.

loss = loss_function(outputs, times): Calculates how wrong the predicted outputs are by comparing them to the actual delivery times.

loss.backward(): The "backward pass" (backpropagation) is performed, which calculates exactly how to adjust the weight and bias to reduce the error.

optimizer.step(): Updates the model's parameters using those calculated adjustments.

The loss is printed every 50 epochs to allow you to track the model's learning progress as the error decreases.
"""
# Training loop
for epoch in range(500):
    # Reset the optimizer's gradients
    optimizer.zero_grad()
    # Make predictions (forward pass)
    outputs = model(distances)
    # Calculate the loss
    loss = loss_function(outputs, times)
    # Calculate adjustments (backward pass)
    loss.backward()
    # Update the model's parameters
    optimizer.step()
    # Print loss every 50 epochs
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}: Loss = {loss.item()}")
helper_utils.plot_results(model, distances, times)

distance_to_predict = 7.0
with torch.no_grad():
    # Convert the Python variable into a 2D PyTorch tensor that the model expects
    new_distance = torch.tensor([[distance_to_predict]], dtype=torch.float32)
    
    # Pass the new data to the trained model to get a prediction
    predicted_time = model(new_distance)
    
    # Use .item() to extract the scalar value from the tensor for printing
    print(f"Prediction for a {distance_to_predict}-mile delivery: {predicted_time.item():.1f} minutes")

    # Use the scalar value in a conditional statement to make the final decision
    if predicted_time.item() > 30:
        print("\nDecision: Do NOT take the job. You will likely be late.")
    else:
        print("\nDecision: Take the job. You can make it!")