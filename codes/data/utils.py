import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.signal import resample
from torchvision import transforms
from PIL import Image
import yaml

# Load configuration
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def load_data_split(dataset, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split the dataset into training, validation, and test sets.
    
    Args:
        dataset (Dataset): The dataset to split.
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the training set to include in the validation split.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        test_dataset (Dataset): Test dataset.
    """
    # Split into training+validation and test sets
    indices = np.arange(len(dataset))
    train_val_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state)

    # Further split into training and validation sets
    train_indices, val_indices = train_test_split(train_val_indices, test_size=val_size/(1-test_size), random_state=random_state)

    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset

def loso_split(dataset, subject_ids):
    """
    Generate Leave-One-Subject-Out (LOSO) cross-validation splits.
    
    Args:
        dataset (Dataset): The dataset to split.
        subject_ids (list or np.array): List of subject IDs corresponding to each sample in the dataset.
    
    Returns:
        folds (list of tuples): List of (train_indices, val_indices) for each fold.
    """
    unique_subjects = np.unique(subject_ids)
    folds = []
    for subject in unique_subjects:
        val_indices = np.where(subject_ids == subject)[0]
        train_indices = np.where(subject_ids != subject)[0]
        folds.append((train_indices, val_indices))
    return folds

def augment_image(image, augment=True):
    """
    Apply data augmentation techniques to an image.
    
    Args:
        image (PIL.Image): The image to augment.
        augment (bool): Whether to apply augmentation.
    
    Returns:
        augmented_image (PIL.Image): The augmented image.
    """
    if augment:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(size=image.size, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])
        image = transform(image)
    return image

def normalize_data(data, scaler=None):
    """
    Normalize data using StandardScaler.
    
    Args:
        data (np.array): The data to normalize.
        scaler (StandardScaler): Fitted scaler. If None, a new scaler is fitted.
    
    Returns:
        normalized_data (np.array): The normalized data.
        scaler (StandardScaler): The fitted scaler.
    """
    if scaler is None:
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(data)
    else:
        normalized_data = scaler.transform(data)
    return normalized_data, scaler

def resample_signal(signal, original_length, new_length):
    """
    Resample a signal to a new length.
    
    Args:
        signal (np.array): The signal to resample.
        original_length (int): Original length of the signal.
        new_length (int): New length of the signal.
    
    Returns:
        resampled_signal (np.array): The resampled signal.
    """
    resampled_signal = resample(signal, new_length)
    if new_length > original_length:
        resampled_signal = resampled_signal[:original_length]
    else:
        resampled_signal = np.pad(resampled_signal, (0, original_length - new_length), mode='constant')
    return resampled_signal

def encode_labels(labels, encoder=None):
    """
    Encode labels using LabelEncoder.
    
    Args:
        labels (list or np.array): The labels to encode.
        encoder (LabelEncoder): Fitted encoder. If None, a new encoder is fitted.
    
    Returns:
        encoded_labels (np.array): The encoded labels.
        encoder (LabelEncoder): The fitted encoder.
    """
    if encoder is None:
        encoder = LabelEncoder()
        encoded_labels = encoder.fit_transform(labels)
    else:
        encoded_labels = encoder.transform(labels)
    return encoded_labels, encoder

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save model checkpoint.
    
    Args:
        state (dict): Model state to save.
        filename (str): Name of the checkpoint file.
    """
    torch.save(state, filename)

def load_checkpoint(filename, model, optimizer=None):
    """
    Load model checkpoint.
    
    Args:
        filename (str): Name of the checkpoint file.
        model (nn.Module): Model to load the state into.
        optimizer (torch.optim.Optimizer): Optimizer to load the state into.
    
    Returns:
        model (nn.Module): Model with loaded state.
        optimizer (torch.optim.Optimizer): Optimizer with loaded state.
        epoch (int): The epoch from which the checkpoint was saved.
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch

# Example usage
if __name__ == "__main__":
    # Example of using the utility functions
    data = np.random.rand(100, 10)
    labels = np.random.randint(0, 3, 100)
    subject_ids = np.random.randint(0, 10, 100)  # Example subject IDs

    # Normalize data
    normalized_data, scaler = normalize_data(data)

    # Encode labels
    encoded_labels, encoder = encode_labels(labels)

    # Split data using LOSO
    folds = loso_split(normalized_data, subject_ids)
    for fold_idx, (train_indices, val_indices) in enumerate(folds):
        print(f"Fold {fold_idx + 1}: Train indices {train_indices}, Validation indices {val_indices}")

    # Example of resampling a signal
    signal = np.random.rand(100)
    resampled_signal = resample_signal(signal, original_length=100, new_length=80)

    print("Normalized data shape:", normalized_data.shape)
    print("Encoded labels:", encoded_labels)
    print("Resampled signal length:", len(resampled_signal))