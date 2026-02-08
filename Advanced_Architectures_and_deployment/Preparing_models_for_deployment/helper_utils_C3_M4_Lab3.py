import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
import torchvision.models as tv_models
import torchvision.transforms as transforms
from tqdm.auto import tqdm
from torchmetrics import Accuracy, Precision, Recall, F1Score

# Define the global random seed value
RANDOM_SEED = 42

# Set seed for PyTorch CPU operations
torch.manual_seed(RANDOM_SEED)

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    # Set seed for PyTorch GPU operations on all available GPUs
    torch.cuda.manual_seed_all(RANDOM_SEED)


def show_weights(model, layer_names):
    """
    Shows a sample of weights for given layer names.
    Accepts a single layer name as a string or multiple as a list of strings.
    - For a Conv2d layer, it shows a 6x6 grid of kernels.
    - For a Linear layer, it shows the top-left 6x6 corner of the weight matrix.
    """
    # If a single layer name string is passed, convert it to a list
    if isinstance(layer_names, str):
        layer_names = [layer_names]

    # Iterate over the list of layer names
    for layer_name in layer_names:
        # Use a try-except block to handle cases where a layer name doesn't exist.
        try:
            # Get the layer object from the model using its name string.
            layer = getattr(model, layer_name)

            # Check if the layer has a 'weight' attribute to prevent errors.
            if not hasattr(layer, 'weight'):
                print(f"Layer '{layer_name}' has no 'weight' attribute.")
                continue

            print(f"--- Weights for '{layer_name}' ---")
            # Get the weights as a NumPy array for processing.
            weights = layer.weight.detach().cpu().numpy()

            # Check the layer's type to apply the correct visualization.
            if isinstance(layer, nn.Conv2d):
                if weights.shape[0] >= 2 and weights.shape[1] >= 2:
                    # To create the 6x6 view, we select four specific 3x3 kernels.
                    kernel_0_0 = weights[0, 0]
                    kernel_0_1 = weights[0, 1]
                    kernel_1_0 = weights[1, 0]
                    kernel_1_1 = weights[1, 1]
                    grid_6x6 = np.block([[kernel_0_0, kernel_0_1], [kernel_1_0, kernel_1_1]])
                    print(grid_6x6)
                else:
                    # If the layer is too small, just show the first kernel.
                    print(weights[0, 0])

            elif isinstance(layer, nn.Linear):
                rows, cols = weights.shape
                # Dynamically determine the slice size to be no larger than the matrix itself.
                slice_rows = min(6, rows)
                slice_cols = min(6, cols)
                # Print the slice, which will be up to 6x6.
                print(weights[:slice_rows, :slice_cols])

            else:
                print(f"Visualization for layer type {type(layer)} is not implemented.")

            print("-" * 50)
            print() # To add a line space

        except AttributeError:
            # Catch the error if the layer_name string doesn't match any layer in the model.
            print(f"Layer '{layer_name}' not found in the model.")


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


def load_resnet18():
    # Load a pre-trained ResNet18 model architecture
    model = tv_models.resnet18(weights=None)
    
    # Path to the local pre-trained weights file
    weights_path = './pretrained_resnet18_weights/resnet18-f37072fd.pth'
    # Load the weights from the file
    state_dict = torch.load(weights_path)
    
    # Apply the loaded weights to the model instance
    model.load_state_dict(state_dict)

    return model


def replace_final_layer(model, num_classes):
    """
    Replaces the final classifier layer
    to adapt a model for transfer learning on a new dataset.
    """
    # Get the number of input features for the classifier.
    num_ftrs = model.fc.in_features
    
    # Replace the final layer with a new, untrained fully-connected layer.
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model


def training_loop(model, train_loader, val_loader, num_epochs, device, num_classes):
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
    
    # Initialize the classification metrics from torchmetrics.
    accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    precision_metric = Precision(task="multiclass", num_classes=num_classes, average='macro').to(device)
    recall_metric = Recall(task="multiclass", num_classes=num_classes, average='macro').to(device)
    f1_score_metric = F1Score(task="multiclass", num_classes=num_classes, average='macro').to(device)
    
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
                
                # Update the state of the metrics with the new data.
                accuracy_metric.update(predicted, labels)
                precision_metric.update(predicted, labels)
                recall_metric.update(predicted, labels)
                f1_score_metric.update(predicted, labels)
                
                # Update the validation progress bar display.
                val_progress_bar.set_postfix(loss=f"{(val_loss / (val_progress_bar.n + 1)):.4f}")
                
        # --- Epoch Summary ---
        
        # Calculate average losses for the epoch.
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Compute the final metrics for the epoch.
        accuracy = accuracy_metric.compute().item() * 100
        precision = precision_metric.compute().item()
        recall = recall_metric.compute().item()
        f1_score = f1_score_metric.compute().item()
        
        # Reset the metrics for the next epoch.
        accuracy_metric.reset()
        precision_metric.reset()
        recall_metric.reset()
        f1_score_metric.reset()
        
        # Adjust the learning rate based on validation loss.
        scheduler.step(avg_val_loss)
        
        # Print a summary for the epoch.
        print(f"Epoch {epoch+1}/{num_epochs} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.2f}%")

    # Announce the end of the training process.
    print("\n--- Finished Training ---")
    
    # Store the final computed metrics in a dictionary.
    final_metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }
    
    # Return the trained model and the final validation metrics.
    return model, final_metrics

def save_unpruned_model_and_metrics(model, metrics, state_dict_filename="unpruned_model_state_dict.pth", metrics_filename="unpruned_metrics.pkl"):
    """
    Saves the state dictionary and performance metrics of the trained unpruned model.

    Args:
        model (torch.nn.Module): The trained unpruned PyTorch model.
        metrics (dict): Dictionary containing the performance metrics.
        state_dict_filename (str): File path to save the model's state dictionary (.pth).
        metrics_filename (str): File path to save the metrics dictionary (.pkl).
    """
    torch.save(model.state_dict(), state_dict_filename)
    print(f"Unpruned model state dictionary saved to {state_dict_filename}")

    with open(metrics_filename, 'wb') as f:
        pickle.dump(metrics, f)
    print(f"Unpruned model metrics saved to {metrics_filename}")


def save_pruned_model_and_metrics(model, metrics, state_dict_filename="pruned_model_permanent_state_dict.pth", metrics_filename="pruned_metrics.pkl"):
    """
    Saves the state dictionary and performance metrics of the trained, permanently pruned model.

    Args:
        model (torch.nn.Module): The trained, permanently pruned PyTorch model.
        metrics (dict): Dictionary containing the performance metrics.
        state_dict_filename (str): File path to save the model's state dictionary (.pth).
        metrics_filename (str): File path to save the metrics dictionary (.pkl).
    """
    torch.save(model.state_dict(), state_dict_filename)
    print(f"Permanently pruned model state dictionary saved to {state_dict_filename}")

    with open(metrics_filename, 'wb') as f:
        pickle.dump(metrics, f)
    print(f"Permanently pruned model metrics saved to {metrics_filename}")


def comparison_report(unpruned_state_dict_path: str, unpruned_metrics_path: str,
                      pruned_state_dict_path: str, pruned_metrics_path: str,
                      num_epochs: int, device: str):
    """
    Generates and prints a comparison report for unpruned and pruned models.

    Args:
        unpruned_state_dict_path (str): Path to the saved state_dict of the unpruned model (.pth).
        unpruned_metrics_path (str): Path to the saved metrics of the unpruned model (.pkl).
        pruned_state_dict_path (str): Path to the saved state_dict of the permanently pruned model (.pth).
        pruned_metrics_path (str): Path to the saved metrics of the permanently pruned model (.pkl).
        num_epochs (int): Number of epochs the models were trained for.
        device (str): Device to load model state_dicts onto ('cuda' or 'cpu').
    """
    print(f"\n--- Final Comparison Report After Running Training for {num_epochs} Epoch(s) ---")

    def count_total_parameters_from_state_dict(state_dict):
        """Counts the total number of parameters from a model's state_dict."""
        # Sums numel() for all tensors in the state_dict
        return sum(p.numel() for p in state_dict.values())

    def count_nonzero_parameters_from_state_dict(state_dict):
        """Counts the total number of NON-ZERO parameters from a model's state_dict."""
        # Counts non-zero elements for all tensors in the state_dict
        return sum(torch.count_nonzero(p).item() for p in state_dict.values())

    # --- Load Data for Unpruned Model ---
    unpruned_state_dict = torch.load(unpruned_state_dict_path, map_location=device)
    with open(unpruned_metrics_path, 'rb') as f:
        unpruned_metrics = pickle.load(f)

    # --- Metrics for Unpruned Model ---
    unpruned_total_params = count_total_parameters_from_state_dict(unpruned_state_dict)
    unpruned_nonzero_params = count_nonzero_parameters_from_state_dict(unpruned_state_dict)
    unpruned_size_bytes = os.path.getsize(unpruned_state_dict_path) # Uses .pth file size
    unpruned_size_mb = unpruned_size_bytes / (1024 * 1024)

    # --- Load Data for Pruned Model ---
    pruned_state_dict = torch.load(pruned_state_dict_path, map_location=device)
    with open(pruned_metrics_path, 'rb') as f:
        pruned_metrics = pickle.load(f)

    # --- Metrics for Pruned Model ---
    pruned_total_params = count_total_parameters_from_state_dict(pruned_state_dict)
    pruned_nonzero_params = count_nonzero_parameters_from_state_dict(pruned_state_dict)
    pruned_size_bytes = os.path.getsize(pruned_state_dict_path) # Uses .pth file size
    pruned_size_mb = pruned_size_bytes / (1024 * 1024)

    # --- Generate and Print the Final Report ---
    report = f"""
--- Unpruned Model ---
Total Parameters:         {unpruned_total_params:,}
Non-Zero Parameters:      {unpruned_nonzero_params:,}
Saved model size:         {unpruned_size_mb:.2f} MB
Final Accuracy:           {unpruned_metrics['accuracy']:.2f}%
Final Precision (Macro):  {unpruned_metrics['precision']:.4f}
Final Recall (Macro):     {unpruned_metrics['recall']:.4f}
Final F1-Score (Macro):   {unpruned_metrics['f1_score']:.4f}

--- Pruned Model ---
Total Parameters:         {pruned_total_params:,}
Non-Zero Parameters:      {pruned_nonzero_params:,} (Effective Parameters: weights retained for computation)
Saved model size:         {pruned_size_mb:.2f} MB
Final Accuracy:           {pruned_metrics['accuracy']:.2f}%
Final Precision (Macro):  {pruned_metrics['precision']:.4f}
Final Recall (Macro):     {pruned_metrics['recall']:.4f}
Final F1-Score (Macro):   {pruned_metrics['f1_score']:.4f}
"""
    print(report)
