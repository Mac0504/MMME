import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torchvision import transforms
from PIL import Image
import yaml

# Load configuration
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class MEDataset(Dataset):
    def __init__(self, data_path, transform=None, augment=False):
        """
        Args:
            data_path (str): Path to the microexpression data directory.
            transform (callable, optional): Optional transform to be applied on a sample.
            augment (bool): Whether to apply data augmentation.
        """
        self.data_path = data_path
        self.transform = transform
        self.augment = augment
        self.image_paths, self.labels = self._load_data()
        self.label_encoder = LabelEncoder()
        self._preprocess_labels()

    def _load_data(self):
        """
        Load microexpression data from the specified path.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data directory not found at {self.data_path}")

        image_paths = []
        labels = []
        for label in os.listdir(self.data_path):
            label_dir = os.path.join(self.data_path, label)
            if os.path.isdir(label_dir):
                for image_name in os.listdir(label_dir):
                    image_path = os.path.join(label_dir, image_name)
                    image_paths.append(image_path)
                    labels.append(label)

        return image_paths, labels

    def _preprocess_labels(self):
        """
        Encode labels.
        """
        self.labels = self.label_encoder.fit_transform(self.labels)

    def _augment_data(self, image):
        """
        Apply data augmentation techniques such as random cropping, flipping, and rotation.
        """
        if self.augment:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomResizedCrop(size=(64, 64), scale=(0.8, 1.0)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])
            image = transform(image)
        return image

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Apply data augmentation
        if self.augment:
            image = self._augment_data(image)

        # Convert to PyTorch tensors
        image = transforms.ToTensor()(image)
        label = torch.tensor(label, dtype=torch.long)

        return image, label

def load_data_split(me_dataset, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split the dataset into training, validation, and test sets.
    """
    # Split into training+validation and test sets
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        me_dataset.image_paths, me_dataset.labels, test_size=test_size, random_state=random_state)

    # Further split into training and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=val_size/(1-test_size), random_state=random_state)

    # Create datasets
    train_dataset = MEDataset(train_paths, train_labels, transform=me_dataset.transform, augment=me_dataset.augment)
    val_dataset = MEDataset(val_paths, val_labels, transform=me_dataset.transform, augment=False)
    test_dataset = MEDataset(test_paths, test_labels, transform=me_dataset.transform, augment=False)

    return train_dataset, val_dataset, test_dataset

def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=32, shuffle=True):
    """
    Create DataLoader instances for training, validation, and test sets.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# Example usage
if __name__ == "__main__":
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # Load dataset
    me_dataset = MEDataset(config['data']['MEs'], transform=transform, augment=True)

    # Split dataset
    train_dataset, val_dataset, test_dataset = load_data_split(me_dataset)

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset)

    # Example of iterating through the data loader
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}, Images shape: {images.shape}, Labels: {labels}")
        if batch_idx == 2:  # Just print the first few batches
            break
