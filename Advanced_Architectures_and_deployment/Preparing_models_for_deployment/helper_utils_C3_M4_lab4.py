import os
import tempfile
import time

from IPython.display import display, HTML, Markdown
import ipywidgets as widgets
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm.auto import tqdm
from transformers import BlipForQuestionAnswering, BlipProcessor


def load_cifar10():
    # Define a series of transformations to apply to the training images.
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # Define a simpler set of transformations for the test images.
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # Specify the local directory to store the dataset.
    data_path = './CIFAR10_data'
    # Check if the data directory already exists to avoid re-downloading.
    if os.path.exists(data_path) and os.path.isdir(data_path):
        download = False  # If the folder exists, set download to False.
        print("CIFAR10 Data folder found locally. Loading from local.\n")
    else:
        download = True   # If the folder doesn't exist, set download to True.
        print("CIFAR10 Data folder not found locally. Downloading data.\n")

    # Set the number of images to be processed in one batch.
    batch_size = 128

    # Load the CIFAR10 training dataset and apply the defined training transformations.
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=download, transform=transform_train)
    # Create a data loader for the training set, which will shuffle the data and serve it in batches.
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Load the CIFAR10 test dataset and apply the defined test transformations.
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=download, transform=transform_test)
    # Create a data loader for the test set. Shuffling is not necessary for the test set.
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    return trainloader, testloader
    

def training_loop(model, trainloader, testloader, num_epochs, DEVICE):
    """
    Trains and validates a PyTorch model.

    Args:
        model (nn.Module): The neural network model to train.
        trainloader (DataLoader): The DataLoader for the training set.
        testloader (DataLoader): The DataLoader for the validation set.
        num_epochs (int): The number of epochs to train for.
        DEVICE (torch.device): The device to run the training on (e.g., 'cuda' or 'cpu').
    """

    # Set model to configured device for training
    model = model.to(DEVICE)
    
    # Define loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    # Initialize best accuracy
    best_accuracy = 0.0

    # Wrap the epoch loop with tqdm
    epoch_loop = tqdm(range(num_epochs), desc="Training Progress")

    for epoch in epoch_loop:
        model.train()
        train_loss = 0.0

        # Wrap trainloader with tqdm for inner progress
        train_inner_loop = tqdm(trainloader, total=len(trainloader), leave=False, desc=f"Epoch [{epoch + 1}/{num_epochs}] Training")
        for data in train_inner_loop:
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Update tqdm description with the training loss
            train_inner_loop.set_postfix(train_loss=loss.item())

        # Evaluate the model
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            # Wrap testloader with tqdm for evaluation progress
            test_inner_loop = tqdm(testloader, total=len(testloader), leave=False, desc=f"Epoch [{epoch + 1}/{num_epochs}] Validation")
            for data in test_inner_loop:
                images, labels = data
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = loss_function(outputs, labels)  # Calculate validation loss
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                test_inner_loop.set_postfix(val_loss=loss.item())

        # Calculate average losses and accuracy
        avg_train_loss = train_loss / len(trainloader)
        avg_val_loss = val_loss / len(testloader)
        accuracy = 100 * correct / total

        # Use tqdm.write to print epoch statistics
        tqdm.write(f'Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')

        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'cifar10_cnn_best.pt')
            tqdm.write(f'Best model saved with accuracy: {best_accuracy:.2f}%')

        # Update learning rate
        scheduler.step(avg_val_loss)

        # Update the main progress bar's postfix
        epoch_loop.set_postfix(train_loss=avg_train_loss, val_loss=avg_val_loss, accuracy=f"{accuracy:.2f}%")

        # Add a line space for readability using tqdm.write
        tqdm.write("")

    print(f'Finished Training with best accuracy: {best_accuracy:.2f}%')

    # Save the final model
    torch.save(model.state_dict(), 'cifar10_cnn_final.pt')
    print('Final model saved!')


def train_qat(model_to_train, trainloader, device, epochs=5):
    """
    Performs Quantization-Aware Training (QAT) fine-tuning for a given model.

    Args:
        model_to_train (torch.nn.Module): The QAT-prepared model.
        trainloader (torch.utils.data.DataLoader): The data loader for training.
        device (torch.device): The device to train on ('cuda' or 'cpu').
        epochs (int, optional): The number of epochs to train. Defaults to 5.
        
    Returns:
        torch.nn.Module: The fine-tuned model.
    """
    
    # Move the model to the specified device
    model_to_train.to(device)
    # Set the model to training mode
    model_to_train.train()

    # Define loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_to_train.parameters(), lr=0.001, momentum=0.9)
    
    epoch_loop = tqdm(range(epochs), desc="QAT Training Progress")
    
    for epoch in epoch_loop:
        running_loss = 0.0
        correct = 0
        total = 0
        batch_loop = tqdm(trainloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        
        for data in batch_loop:
            inputs, labels = data
            # Move data to the specified device
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model_to_train(inputs)
            loss = loss_function(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            batch_loop.set_postfix(loss=f"{loss.item():.3f}")

        avg_epoch_loss = running_loss / len(trainloader)
        train_accuracy = 100 * correct / total
        print(f'Epoch {epoch + 1}, Training Loss: {avg_epoch_loss:.4f}, Accuracy: {train_accuracy:.2f}%')
        epoch_loop.set_postfix(avg_loss=f"{avg_epoch_loss:.3f}")

    print('\nQAT Training finished.')
    
    # Return the fine-tuned model
    return model_to_train


def evaluate_qat(model_to_eval, dataloader):
    """
    Evaluates the accuracy of a QAT-quantized model on a given dataset.

    Args:
        model_to_eval (torch.nn.Module): The quantized model to evaluate.
        dataloader (torch.utils.data.DataLoader): The data loader for evaluation.
    """
    model_to_eval.eval() 
    
    correct = 0
    total = 0
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating QAT Model")
        for images, labels in progress_bar:
            outputs = model_to_eval(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the QAT model on the test set: {accuracy:.2f}%')


def get_model_size(model):
    """Calculates the size of a model's state_dict in megabytes (MB) using a temporary file."""

    # Create a temporary file using the tempfile module.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".p") as temp_file:
        temp_file_path = temp_file.name
        # Save the model's state_dict to the temporary file.
        torch.save(model.state_dict(), temp_file_path)

    # Get the size of the saved file in bytes.
    size_bytes = os.path.getsize(temp_file_path)

    # Convert the size from bytes to megabytes (1 MB = 1024*1024 bytes).
    size_mb = size_bytes / (1024 * 1024)

    # Clean up by deleting the temporary file.
    os.remove(temp_file_path)

    # Return the calculated size.
    return size_mb


def measure_average_inference_time_ms(model, input_shape=(1, 3, 32, 32), num_runs=100):
    """
    Measures the average inference time of a PyTorch model in milliseconds,
    forcing execution on the CPU to ensure compatibility with quantized models.
    """
    
    # Force the entire measurement to run on the CPU
    device = torch.device("cpu")
    model.to(device)
    
    # Create a random input tensor on the CPU
    input_tensor = torch.rand(input_shape).to(device)
    
    # Set the model to evaluation mode
    model.eval()

    # Warm-up run to cache operations
    with torch.no_grad():
        for _ in range(10):
            model(input_tensor)

    # Measure inference time
    timings = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            model(input_tensor)
            end_time = time.time()
            timings.append((end_time - start_time) * 1000) # Convert to ms

    # Calculate the average time
    avg_time = sum(timings) / len(timings)
    
    return avg_time


def comparison_table(baseline_model_size, baseline_model_time, quantized_model_size, quantized_model_time, quantization_type):
    """Displays a model comparison table in a Jupyter-like environment."""
    # Calculate the differences
    size_diff = baseline_model_size - quantized_model_size
    time_diff = baseline_model_time - quantized_model_time

    # Dynamically create the header for the quantized model column
    quantized_header = f"Quantized Model ({quantization_type})"

    # Create the formatted Markdown table string with the requested changes
    markdown_table_string = f"""
| | Baseline Model | {quantized_header} | Change |
|:---|:---|:---|:---|
| **Model Size (MB)** | {baseline_model_size:.2f} | {quantized_model_size:.2f} | {size_diff:.2f} |
| **Inference Latency (ms)** | {baseline_model_time:.2f} | {quantized_model_time:.2f} | {time_diff:.2f} |
"""

    # Directly display the rendered Markdown table
    display(Markdown(markdown_table_string))


def display_full_comparison(stats_dict):
    """
    Displays a markdown table comparing the size and inference time of all models.

    Args:
        stats_dict (dict): A dictionary where keys are model names and 
                           values are tuples of (size, time).
    """
    
    # Start building the markdown table string
    table = "| Model Type | Size (MB) | Inference Time (ms) |\n"
    table += "|--------------------------|-----------|-----------------------|\n"
    
    # Add a row for each model in the dictionary
    for model_name, (size, time) in stats_dict.items():
        table += f"| {model_name:<24} | {size:<9.2f} | {time:<21.2f} |\n"

    # Display the rendered markdown table
    display(Markdown(table))


def get_blip_vqa_model_and_processor():
    """
    Ensures the 'blip-vqa-base' model is available locally, downloading it if necessary.
    Then, loads and returns the model and its processor, specifying the use of the
    fast processor to address compatibility warnings.

    Returns:
        tuple: A tuple containing the loaded model and processor.
               (model, processor)
    """
    # --- Configuration ---
    # Define the model identifier from the Hugging Face Hub.
    model_id = "Salesforce/blip-vqa-base"
    # Define the local directory where the model will be stored and loaded from.
    local_path = "./blip-vqa-base-local"

    # --- Step 1: Check for Local Model and Download if Necessary ---
    # Check if the model directory does not exist on the local filesystem.
    if not os.path.exists(local_path):
        # If the model is not found locally, print a message and begin the download process.
        print(f"Local model not found. Downloading '{model_id}' to local path: '{local_path}'")
        # Create the target directory. `exist_ok=True` prevents an error if the directory already exists.
        os.makedirs(local_path, exist_ok=True)
        
        # Use a try-except block to handle potential network errors during download.
        try:
            # --- Download from Hugging Face Hub ---
            # Download the pre-trained model weights.
            temp_model = BlipForQuestionAnswering.from_pretrained(model_id)
            # Download the processor. `use_fast=True` is set to use the faster tokenizer
            # and resolve future compatibility warnings from the transformers library.
            temp_processor = BlipProcessor.from_pretrained(model_id, use_fast=True)
            
            # --- Save for Offline Use ---
            # Save the downloaded model weights to the local directory.
            temp_model.save_pretrained(local_path)
            # Save the processor's configuration and files to the same local directory.
            temp_processor.save_pretrained(local_path)
            
            print("Model and processor downloaded and saved successfully.")
        except Exception as e:
            # If any error occurs during download (e.g., network issue), print the error and exit.
            print(f"An error occurred during download: {e}")
            return None, None
    else:
        # If the directory already exists, print a confirmation message.
        print(f"Model already exists at local path: '{local_path}'.")

    # --- Step 2: Load Model from Local Path ---
    # This block runs regardless of whether the model was just downloaded or already existed.
    try:
        # Print a status message indicating that the model is being loaded from the disk.
        print(f"Loading model and processor from local path: {local_path}")
        
        # Load the model's weights and configuration from the local directory.
        model = BlipForQuestionAnswering.from_pretrained(local_path)
        # Load the processor from the local directory, again specifying `use_fast=True`.
        processor = BlipProcessor.from_pretrained(local_path, use_fast=True)
        
        print("Model and processor loaded successfully.")
        # Return the loaded model and processor objects, ready for use.
        return model, processor
    except Exception as e:
        # If loading from the local path fails for any reason, print the error and exit.
        print(f"Failed to load model from local path: {e}")
        return None, None


def upload_jpg_widget():
    """
    Displays a file upload widget in a Jupyter/IPython environment.
    1. User uploads a file.
    2. Checks if the file is .jpg format. If not, displays an error.
    3. If .jpg, checks if size is <= 5MB. If larger, displays an error.
    4. If valid format and size, saves to './images/'.
    5. Displays a success message with the Python file path.
    """
    # Define the target directory for uploaded images
    output_image_folder = "./images"
    # Ensure the target directory exists, create it if it doesn't
    os.makedirs(output_image_folder, exist_ok=True)

    # Create the file upload widget
    uploader = widgets.FileUpload(
        accept='.jpg',  # Browser-side hint to filter for .jpg files
        multiple=False, # Allow only a single file upload
        description='Upload JPG (Max 5MB)' # Description for the user
    )

    # Create an output widget to display messages (errors or success)
    output_area = widgets.Output()

    def on_file_uploaded(change):
        """Callback function triggered when a file is uploaded."""
        
        # Get the new value from the change event object.
        # For FileUpload, 'change['new']' is a tuple of dictionaries.
        # If multiple=False, it's a tuple with one dictionary: ({'name': ..., 'content': ...},)
        current_uploaded_value_tuple = change['new']

        # If the new value is empty, it means the uploader was cleared programmatically.
        # In this case, do nothing further to avoid clearing a previous valid message.
        if not current_uploaded_value_tuple:
            return

        # Only proceed to clear output and process if there's actual file data
        with output_area:
            output_area.clear_output() # Clear messages from any previous upload attempt

            # Extract the file data dictionary from the tuple
            file_data_dict = current_uploaded_value_tuple[0]
            filename = file_data_dict['name']
            file_content = file_data_dict['content'] # File content as bytes

            # Requirement 2: Check file format
            if not filename.lower().endswith('.jpg'):
                error_msg_format = (
                    f"<p style='color:red;'>Error: Please upload a file with a ‘.jpg’ format. "
                    f"You uploaded: '{filename}'</p>"
                )
                display(HTML(error_msg_format))
                uploader.value = () # Clear the invalid upload from the widget
                return

            # Requirement 3: Check file size (if format is correct)
            file_size_bytes = len(file_content)
            max_size_bytes = 5 * 1024 * 1024  # 5 MB

            if file_size_bytes > max_size_bytes:
                file_size_mb = file_size_bytes / (1024 * 1024)
                error_msg_size = (
                    f"<p style='color:red;'>Error: File '{filename}' is too large ({file_size_mb:.2f} MB). "
                    f"Please upload a file less than or equal to 5 MB.</p>"
                )
                display(HTML(error_msg_size))
                uploader.value = () # Clear the oversized upload from the widget
                return

            # Requirement 4: If format and size are valid, try to save the file
            try:
                # Construct the full path to save the file
                save_path = os.path.join(output_image_folder, filename)

                # Save the file (in binary write mode)
                with open(save_path, 'wb') as f:
                    f.write(file_content)

                # Requirement 5: Display successful upload message and valid file path
                # Prepare the python path string, using repr() for proper quoting
                python_code_path = repr(save_path)

                # Construct the success message as per the specified format
                success_message = f"""
                <p style='color:green;'>File successfully uploaded!</p>
                <p>Please use the path as <code>image_path = {python_code_path}</code></p>
                """
                display(HTML(success_message))

            except Exception as e:
                # Handle potential errors during file saving
                error_msg_save = f"<p style='color:red;'>Error saving file '{filename}': {e}</p>"
                display(HTML(error_msg_save))
            finally:
                # Always clear the uploader value after processing an attempt.
                # This resets the widget for the next upload and triggers the
                # callback again, which is handled by the initial check for empty value.
                uploader.value = ()

    # Observe changes in the uploader's 'value' trait and call on_file_uploaded
    uploader.observe(on_file_uploaded, names='value')

    # Display the uploader widget and the output area where messages will appear
    display(uploader)
    display(output_area)


def perform_vqa(model, processor, image_path, question_text):
    """
    Opens an image from a file path, prepares inputs, generates an answer
    from the VQA model, and measures the inference time.

    Args:
        model: The VQA model object.
        processor: The model's processor object.
        image_path (str): The file path to the image.
        question_text (str): The question to ask about the image.

    Returns:
        tuple: A tuple containing:
               - answer (str): The generated answer.
               - inference_time (float): The time taken for inference in seconds.
    """
    # Open the image file from the provided path and convert it to RGB.
    raw_image = Image.open(image_path).convert('RGB')
    
    # Prepare the model inputs using the opened image.
    inputs = processor(raw_image, question_text, return_tensors="pt")
    
    # Record the start time.
    start_time = time.perf_counter()
    
    # Generate the output from the model.
    output = model.generate(**inputs)
    
    # Record the end time.
    end_time = time.perf_counter()
    
    # Calculate the inference time.
    inference_time = end_time - start_time
    
    # Decode the generated tokens to get the text answer.
    answer = processor.decode(output[0], skip_special_tokens=True)
    
    # Return both the answer and the calculated inference time.
    return answer, inference_time


def blip_comparison_table(
    question,
    baseline_answer,
    quantized_answer,
    baseline_size,
    quantized_size,
    baseline_time_s,
    quantized_time_s
):
    """
    Displays a formatted HTML table comparing the performance of the baseline
    and quantized BLIP model.
    """
    # --- Calculations ---
    size_change = baseline_size - quantized_size
    size_change_text = "N/A"
    if baseline_size > 0:
        size_percent_change = (size_change / baseline_size) * 100
        size_change_label = "reduction" if size_change >= 0 else "increase"
        size_change_text = f"{size_change:.2f} ({abs(size_percent_change):.1f}% {size_change_label})"

    time_change_s = baseline_time_s - quantized_time_s
    time_change_text = "N/A"
    if baseline_time_s > 0:
        time_percent_change = (time_change_s / baseline_time_s) * 100
        time_change_label = "reduction" if time_change_s >= 0 else "increase"
        time_change_text = f"{time_change_s:.4f} ({abs(time_percent_change):.1f}% {time_change_label})"
        
    # --- Build the HTML String for the table ---
    html_string = f"""
    <p><b>Question:</b> {question}</p>
    <table border="1" style="width:100%; border-collapse: collapse; text-align: left;">
      <tr style="background-color: #4A5568; color: white;">
        <th style="padding: 8px;"></th>
        <th style="padding: 8px;">Model Size (MB)</th>
        <th style="padding: 8px;">Inference Time (s)</th>
        <th style="padding: 8px;">Answer</th>
      </tr>
      <tr>
        <td style="padding: 8px;"><b>Baseline Model</b></td>
        <td style="padding: 8px;">{baseline_size:.2f}</td>
        <td style="padding: 8px;">{baseline_time_s:.4f}</td>
        <td style="padding: 8px;">{baseline_answer}</td>
      </tr>
      <tr>
        <td style="padding: 8px;"><b>Quantized Model (Dynamic)</b></td>
        <td style="padding: 8px;">{quantized_size:.2f}</td>
        <td style="padding: 8px;">{quantized_time_s:.4f}</td>
        <td style="padding: 8px;">{quantized_answer}</td>
      </tr>
      <tr>
        <td style="padding: 8px;"><b>Change</b></td>
        <td style="padding: 8px;"><b>{size_change_text}</b></td>
        <td style="padding: 8px;"><b>{time_change_text}</b></td>
        <td style="padding: 8px;">---</td>
      </tr>
    </table>
    """

    # --- Render the HTML in the Notebook ---
    display(HTML(html_string))