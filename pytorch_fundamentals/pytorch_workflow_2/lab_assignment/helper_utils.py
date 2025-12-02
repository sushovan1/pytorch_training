import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
from torchvision.transforms import functional as F

# region = Message =
letter_ref = [
    "Dear Laurence",
    "Hope the PyTorch course is going well",
    "Do notforget to keep the labs interesting and engaging",
    "Maybe the students could decode my messy handwriting",
    "That might be a bit too challenging though",
    "I am impressed you are able to read this",
]


path_data = "./EMNIST_data"


def load_hidden_message_images(file_name="hidden_message_images.pkl"):
    with open(file_name, "rb") as f:
        import pickle

        message_imgs = pickle.load(f)
    return message_imgs


def decode_word_imgs(word_imgs, model, device):
    model.eval()
    decoded_chars = []
    with torch.no_grad():
        for char_img in word_imgs:
            char_img = char_img.unsqueeze(0).to(
                device
            )  # Add batch dimension and move to device
            output = model(char_img)
            _, predicted = output.max(1)
            predicted_label = predicted.item()
            # uppercase_char = chr(ord("A") + predicted_label)
            lowercase_char = chr(ord("a") + predicted_label)
            # decoded_chars.append(f"{uppercase_char}/{lowercase_char}")
            decoded_chars.append(f"{lowercase_char}")
    decoded_word = "".join(decoded_chars)
    # print("Decoded word:", decoded_word)
    # print("Predicted characters:", " ".join(decoded_chars))
    # return decoded_word, decoded_chars
    return decoded_word


def visualize_image(img, label=None, ax=None):
    """
    Visualizes an EMNIST image with its label. If an axis is provided, plots on that axis; otherwise, creates a new figure.

    Args:
        img (np.ndarray or torch.Tensor): The image to display.
        label (int, optional): The EMNIST label (1-26). If None, no title is shown.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. If None, creates a new figure.
    """
    # Convert to numpy array if needed
    if isinstance(img, torch.Tensor):
        img = img.numpy().squeeze()
    elif isinstance(img, np.ndarray):
        if img.ndim == 3:
            img = img[:, :, 0]

    # Prepare title if label is provided
    if label is not None:
        uppercase_char, lowercase_char = convert_emnist_label_to_char(label)
        title = f"EMNIST Letter: {uppercase_char}/{lowercase_char}"
    else:
        title = None

    # Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
        show_colorbar = True
    else:
        show_colorbar = False

    im = ax.imshow(img, cmap="gray")
    ax.set_xticks(np.arange(0, 28, 1))
    ax.set_yticks(np.arange(0, 28, 1))
    ax.tick_params(labelsize=6)
    ax.grid(True, color="red", alpha=0.3)
    if title:
        ax.set_title(title)

    if show_colorbar:
        plt.colorbar(im, ax=ax)
        plt.show()


def display_data_loader_contents(data_loader):
    """
    Displays the contents of the data loader.

    Args:
        data_loader (torch.utils.data.DataLoader): The data loader to display.
    """
    try:
        print("Total number of images in dataset:", len(data_loader.dataset))
        print("Total number of batches:", len(data_loader))
        for batch_idx, (data, labels) in enumerate(data_loader):
            print(f"--- Batch {batch_idx + 1} ---")
            print(f"Data shape: {data.shape}")
            print(f"Labels shape: {labels.shape}")
            break  # display only the first batch.
    except StopIteration:
        print("data loader is empty.")
    except Exception as e:
        print(f"An error occurred: {e}")


def evaluate_per_class(model, test_loader, device):
    """
    Evaluates the model's accuracy for each class (letter).

    Args:
        model: The trained PyTorch model.
        test_loader: DataLoader for the test dataset.
        device: Device to run the model on (e.g., 'cpu' or 'cuda').

    Returns:
        dict: class_accuracies - Dictionary containing accuracy for each class (letter).
    """

    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Shift target labels down by 1
            targets = targets - 1

            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    class_accuracies = {}

    for class_idx in range(26):  # 26 classes for A-Z
        class_targets = [
            t for t, p in zip(all_targets, all_predictions) if t == class_idx
        ]
        class_predictions = [
            p for t, p in zip(all_targets, all_predictions) if t == class_idx
        ]

        if len(class_targets) > 0:
            class_accuracies[chr(65 + class_idx)] = accuracy_score(
                class_targets, class_predictions
            )
        else:
            class_accuracies[chr(65 + class_idx)] = 0.0  # Handle empty classes

    return class_accuracies


def save_student_model(model, filename="trained_student_model.pth"):
    """
    Saves the student's trained model and metadata.

    Args:
        model (nn.Module): The student's trained model.
        filename (str): The filename to save to.
    """
    save_dict = {"model": model}
    torch.save(save_dict, filename)
    print(f"Model saved to {filename}")


def convert_emnist_label_to_char(label):
    """
    Converts an EMNIST label to its corresponding uppercase and lowercase letters.

    Args:
        label (int): The EMNIST label (1-26).

    Returns:
        tuple: A tuple containing the uppercase and lowercase letters.
    """
    if not (1 <= label <= 26):
        raise ValueError("Label must be between 1 and 26 inclusive.")

    uppercase_char = chr(64 + label)  # 'A' is at 65, 'B' is at 66, etc.
    lowercase_char = chr(96 + label)  # 'a' is at 97, 'b' is at 98, etc.

    return uppercase_char, lowercase_char