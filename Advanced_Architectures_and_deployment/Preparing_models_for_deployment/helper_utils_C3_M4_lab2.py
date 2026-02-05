import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from tqdm.auto import tqdm


def dataset_images_per_class(dataset_path):
    """
    Counts and prints the number of images for each class in a dataset directory.
    """
    # Specify valid image file extensions.
    valid_exts = ('.jpeg', '.jpg', '.JPG', '.Jpg')
    
    # Initialize a dictionary to store class counts.
    class_counts = defaultdict(int)
    
    # Iterate through each class directory.
    for class_name in sorted(os.listdir(dataset_path)):
        # Get the full path of the class directory.
        class_dir = os.path.join(dataset_path, class_name)
        
        # Check if the path is a directory.
        if os.path.isdir(class_dir):
            # List all valid image files in the directory.
            image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(valid_exts)]
            # Store the image count for the class.
            class_counts[class_name] = len(image_files)
    
    # Display the results.
    print(f"Total number of classes: {len(class_counts)}\n")
    print("Number of images per class:\n")
    for cls, count in class_counts.items():
        print(f"{cls:25}: {count}")


class TransformedDataset(Dataset):
    """
    A custom dataset class that applies a given transformation to an existing dataset subset.
    """
    # Initialize the dataset and transformations.
    def __init__(self, subset, transform):
        # Store the data subset.
        self.subset = subset
        # Store the transformations.
        self.transform = transform
    
    # Return the size of the dataset.
    def __len__(self):
        return len(self.subset)

    # Get a single item from the dataset.
    def __getitem__(self, idx):
        # Retrieve an image and label from the subset.
        img, label = self.subset[idx]
        # Apply transform and return the image with its label.
        return self.transform(img), label

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.630, 0.554, 0.489],
                         std=[0.248, 0.271, 0.319]),
])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.630, 0.554, 0.489],
                         std=[0.248, 0.271, 0.319]),
])

def get_dataloaders(dataset_path, transformations=[transform_train, transform_val], batch_size=32):
    """
    Creates and returns training and validation DataLoaders from an ImageFolder dataset.
    """
    # Load the full image dataset from the specified path.
    full_dataset = ImageFolder(root=dataset_path)

    # Set the training set ratio.
    train_ratio = 0.8
    # Calculate the number of training samples.
    train_size = int(train_ratio * len(full_dataset))
    # Calculate the number of validation samples.
    val_size = len(full_dataset) - train_size
    # Split the dataset into training and validation subsets.
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    # Display the number of samples in each subset.
    print(f"Train samples (80%):\t\t{len(train_subset)}")
    print(f"Validation samples (20%):\t{len(val_subset)}\n")

    # Apply transformations to the subsets using the custom dataset.
    train_dataset = TransformedDataset(train_subset, transformations[0])
    val_dataset = TransformedDataset(val_subset, transformations[1])

    # Create data loaders for training and validation.
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Confirm data loader creation.
    print(f"DataLoaders created with {len(train_dataset)} training images and {len(val_dataset)} validation images.")

    # Return the created data loaders.
    return train_loader, val_loader

def show_image_grid(dataloader, class_names):
    """
    Displays a 3x3 grid of images, reversing the normalization for display.
    """
    # Get one batch of training images
    images, labels = next(iter(dataloader))
    
    # Define the mean and std used for normalization to reverse the process
    mean = torch.tensor([0.630, 0.554, 0.489])
    std = torch.tensor([0.248, 0.271, 0.319])

    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    
    for i, ax in enumerate(axes.flatten()):
        # Ensure we don't go out of bounds if the batch size is less than 9
        if i < len(images):
            image = images[i]
            label = labels[i]
            
            # --- Reverse the normalization ---
            # Reshape mean and std to (C, 1, 1) to allow broadcasting across image dimensions (C, H, W)
            # The operation is: image = (image * std) + mean
            unnormalized_image = image * std[:, None, None] + mean[:, None, None]
            
            # PyTorch tensors are (C, H, W), but matplotlib expects (H, W, C).
            img_display = unnormalized_image.permute(1, 2, 0).numpy()
            
            # Clip the values to the valid range [0, 1] to handle floating point inaccuracies
            img_display = np.clip(img_display, 0, 1)
            
            # Display the image and set the title with the true class name
            ax.imshow(img_display)
            ax.set_title(class_names[label])
            ax.axis("off") # Hide the axes
    
    plt.tight_layout()
    plt.show()


def adapt_model_for_transfer_learning(model, num_classes):
    """
    Freezes pre-trained layers and replaces the final classifier layer
    to adapt a model for transfer learning on a new dataset.
    """
    # Freeze all the parameters in the pre-trained layers.
    for param in model.parameters():
        param.requires_grad = False
    
    # Get the number of input features for the classifier.
    num_ftrs = model.fc.in_features
    
    # Replace the final layer with a new, untrained fully-connected layer.
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model


def training_loop(model, train_loader, val_loader, num_epochs, device):
    """
    Executes the complete training and validation loop for a PyTorch model.
    """
    # --- Initialization ---
    
    # Define the loss function.
    loss_function = nn.CrossEntropyLoss()
    
    # Initialize the optimizer for the classifier layer only.
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    # Initialize a learning rate scheduler to reduce LR on plateau.
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    
    # Move the model to the specified device (CPU or GPU).
    model = model.to(device)

    # --- Start Loop ---
    
    # Announce the start of the training process.
    print(f"--- Starting Training ---")
    
    # Begin the main training loop for the specified number of epochs.
    for epoch in range(num_epochs):
        
        # --- Training Phase ---
        
        # Set the model to training mode.
        model.train()
        # Initialize training loss for the epoch.
        train_loss = 0.0
        # Create a progress bar for the training phase.
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        
        # Iterate over batches in the training data.
        for inputs, labels in train_progress_bar:
            # Move data to the specified device.
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Clear previous gradients.
            optimizer.zero_grad()
            
            # Perform a forward pass.
            outputs = model(inputs)
            # Calculate the loss.
            loss = loss_function(outputs, labels)
            
            # Perform a backward pass to compute gradients.
            loss.backward()
            # Update the model weights.
            optimizer.step()
            
            # Accumulate the batch loss.
            train_loss += loss.item()
            # Update the training progress bar display.
            train_progress_bar.set_postfix(loss=f"{(train_loss / (train_progress_bar.n + 1)):.4f}")
            
        # --- Validation Phase ---
        
        # Set the model to evaluation mode.
        model.eval()
        # Initialize validation metrics.
        val_loss = 0.0
        correct = 0
        total = 0
        # Create a progress bar for the validation phase.
        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
        
        # Disable gradient calculations for validation.
        with torch.no_grad():
            # Iterate over batches in the validation data.
            for inputs, labels in val_progress_bar:
                # Move data to the specified device.
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Perform a forward pass.
                outputs = model(inputs)
                # Calculate the loss.
                loss = loss_function(outputs, labels)
                
                # Accumulate the validation batch loss.
                val_loss += loss.item()
                # Get the model's predictions.
                _, predicted = torch.max(outputs.data, 1)
                # Update the total sample count.
                total += labels.size(0)
                # Update the count of correct predictions.
                correct += (predicted == labels).sum().item()
                
                # Update the validation progress bar display.
                val_progress_bar.set_postfix(loss=f"{(val_loss / (val_progress_bar.n + 1)):.4f}", acc=f"{(100 * correct / total):.2f}%")
                
        # --- Epoch Summary ---
        
        # Calculate average losses and accuracy for the epoch.
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        # Adjust the learning rate based on validation loss.
        scheduler.step(avg_val_loss)
        
        # Print a summary for the epoch.
        print(f"Epoch {epoch+1}/{num_epochs} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.2f}%")

    # Announce the end of the training process.
    print("\n--- Finished Training ---")
    
    # Return the trained model.
    return model


def show_prediction_grid(images, labels, predictions, classes):
    """
    Displays a grid of images, their true labels, and predicted labels.
    
    Args:
        images (np.array): A batch of images to display.
        labels (list/np.array): The true labels for the images.
        predictions (list/np.array): The model's output predictions.
        classes (list): A list of class names for label lookup.
    """
    # Denormalization parameters
    mean = np.array([0.630, 0.554, 0.489])
    std = np.array([0.248, 0.271, 0.319])
    
    # Create a 3x3 grid for displaying images
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    fig.subplots_adjust(hspace=0.4, wspace=0.2)
    
    for i, ax in enumerate(axes.flat):
        # Display images only up to the number of images available
        if i < len(images):
            # Transpose and denormalize the image
            img = images[i].transpose((1, 2, 0))
            img = std * img + mean
            img = np.clip(img, 0, 1) # Ensure pixel values are between 0 and 1
            
            # Display the image
            ax.imshow(img)
            
            # Get true and predicted labels from the classes list
            true_label = classes[labels[i]]
            pred_label = classes[np.argmax(predictions[i])]
            
            # Set the title with color-coding for correctness
            ax.set_title(
                f"True: {true_label}\nPred: {pred_label}",
                color=("green" if true_label == pred_label else "red")
            )
            
            # Remove ticks from axes
            ax.set_xticks([])
            ax.set_yticks([])
            
    plt.show()