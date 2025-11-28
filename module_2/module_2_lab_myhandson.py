# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 14:21:23 2025

@author: CSU5KOR
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device=torch.device("cpu")
device_name=torch.cuda.get_device_name()
os.chdir(r"C:\Users\CSU5KOR\OneDrive - Bosch Group\Coursera\pytorch_training\module_2")

data_path="./data"
train_dataset_without_transform=torchvision.datasets.MNIST(
    root=data_path,
    train=True,
    download=True
    )

image_pil, label = train_dataset_without_transform[0]
print(f"Image type:        {type(image_pil)}")
# Since `image_pil` is a PIL Image object, its dimensions are accessed using the .size attribute.
print(f"Image Dimensions:  {image_pil.size}")
print(f"Label Type:        {type(label)}")
print(f"Label value:       {label}")
###############################################################
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
train_dataset=torchvision.datasets.MNIST(
    root=data_path,
    train=True,
    download=True,
    transform=transform
   )

image_tensor, label = train_dataset[0]
print(f"Image Type:                   {type(image_tensor)}")
# Since the `image` is now a PyTorch Tensor, its dimensions are accessed using the .shape attribute.
print(f"Image Shape After Transform:  {image_tensor.shape}")
print(f"Label Type:                   {type(label)}")
print(f"Label value:                  {label}")
#################################################################
test_dataset=torchvision.datasets.MNIST(
    root=data_path,
    train=False,
    download=True,
    transform=transform
    )

train_dataloader=DataLoader(train_dataset,batch_size=64,shuffle=True)
test_dataloader=DataLoader(test_dataset,batch_size=1000,shuffle=False)
####################################################################
#model

class SimpleMINISTDNN(nn.Module):
    def __init__(self):
        super(SimpleMINISTDNN,self).__init__()
        self.flatten = nn.Flatten()
        self.layers=nn.Sequential(
            nn.Linear(784,128),
            nn.ReLU(),
            nn.Linear(128,10)
            )
    def forward(self,x):
        x=self.flatten(x)
        x=self.layers(x)
        return(x)
###################################################################
model=SimpleMINISTDNN()
loss_function=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.01)
def train_epoch(model,loss_function,optimizer,train_loader,device):
    model.to(device)
    model.train()
    epoch_loss=0.0
    running_loss = 0.0
    #num_correct_predictions = 0
    #total_predictions = 0
    total_batches = len(train_loader)
    for batch_idx,(inputs,targets) in enumerate(train_loader):
        inputs,targets=inputs.to(device),targets.to(device)
        optimizer.zero_grad()
        outputs=model(inputs)
        loss=loss_function(outputs,targets)
        loss.backward()
        optimizer.step()
        loss_value=loss.item()
        epoch_loss+=loss_value
        running_loss+=loss_value
        if (batch_idx+1) % 100==0 or (batch_idx+1)==total_batches:
            avg_running_loss = running_loss / 100
            print(f'\tStep {batch_idx + 1}/{total_batches} - Loss: {avg_running_loss:.3f}')
        running_loss = 0.0
        #num_correct_predictions = 0
    avg_epoch_loss = epoch_loss / total_batches
    return model, avg_epoch_loss
def evaluate(model,test_loader,device):
    model.eval()
    num_correct_prediction=0
    total_prediction=0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs,targets=inputs.to(device),targets.to(device)
            outputs = model(inputs)
            _, predicted_indices = outputs.max(1)
            batch_size=targets.size(0)
            total_prediction=total_prediction+batch_size
            correct_prediction=predicted_indices.eq(targets)
            num_correct_in_batch = correct_prediction.sum().item()
            num_correct_prediction = num_correct_prediction + num_correct_in_batch
        accuracy_percentage = (num_correct_prediction / total_prediction) * 100
    # Prints the calculated accuracy to the console.
    print((f'\tAccuracy - {accuracy_percentage:.2f}%'))
    
    return accuracy_percentage
start=time.time_ns()
epochs=5
train_loss=[]
test_acc=[]

for epoch in range(epochs):
    trained_model,loss_value=train_epoch(model,loss_function,optimizer,train_dataloader,device)
    train_loss.append(loss_value)
    accuracy = evaluate(trained_model, test_dataloader, device)
    test_acc.append(accuracy)
end=time.time_ns()
time_taken=(end-start)/10**9
print(f"time taken (in secs) {time_taken:.2f}")
    