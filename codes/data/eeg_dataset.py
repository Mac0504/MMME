import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.signal import butter, lfilter, resample
import yaml

# Load configuration
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class EEGDataset(Dataset):
    def __init__(self, data_path, transform=None, augment=False):
        """
        Args:
            data_path (str): Path to the EEG data file.
            transform (callable, optional): Optional transform to be applied on a sample.
            augment (bool): Whether to apply data augmentation.
        """
        self.data_path = data_path
        self.transform = transform
        self.augment = augment
        self.data, self.labels = self._load_data()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self._preprocess_data()

    def _load_data(self):
        """
        Load EEG data from the specified path.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")

        # Assuming the data is in CSV format with columns as features and the last column as labels
        data_df = pd.read_csv(self.data_path)
        data = data_df.iloc[:, :-1].values  # Features
        labels = data_df.iloc[:, -1].values  # Labels

        return data, labels

    def _preprocess_data(self):
        """
        Preprocess the EEG data: normalize, filter, and encode labels.
        """
        # Normalize data
        self.data = self.scaler.fit_transform(self.data)

        # Apply bandpass filter
        self.data = self._apply_bandpass_filter(self.data, lowcut=config['data']['lowcut'], highcut=config['data']['highcut'], fs=config['data']['sampling_rate'], order=config['data']['filter_order'])

        # Encode labels
        self.labels = self.label_encoder.fit_transform(self.labels)

    def _apply_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        """
        Apply a bandpass filter to the EEG data.
        """
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_data = np.apply_along_axis(lambda x: lfilter(b, a, x), 1, data)
        return filtered_data

    def _augment_data(self, data):
        """
        Apply data augmentation techniques such as noise addition and time warping.
        """
        if self.augment:
            # Add Gaussian noise
            noise = np.random.normal(0, 0.1, data.shape)
            data = data + noise

            # Time warping by resampling
            original_length = data.shape[1]
            new_length = int(original_length * np.random.uniform(0.9, 1.1))
            data = resample(data, new_length, axis=1)
            data = resample(data, original_length, axis=1)

        return data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        if self.augment:
            sample = self._augment_data(sample)

        # Convert to PyTorch tensors
        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return sample, label

def load_data_split(eeg_dataset, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split the dataset into training, validation, and test sets.
    """
    # Split into training+validation and test sets
    train_val_data, test_data, train_val_labels, test_labels = train_test_split(
        eeg_dataset.data, eeg_dataset.labels, test_size=test_size, random_state=random_state)

    # Further split into training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_val_data, train_val_labels, test_size=val_size/(1-test_size), random_state=random_state)

    # Create datasets
    train_dataset = EEGDataset(train_data, train_labels)
    val_dataset = EEGDataset(val_data, val_labels)
    test_dataset = EEGDataset(test_data, test_labels)

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
    # Load dataset
    eeg_dataset = EEGDataset(config['data']['eeg_path'])

    # Split dataset
    train_dataset, val_dataset, test_dataset = load_data_split(eeg_dataset)

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset)

    # Example of iterating through the data loader
    for batch_idx, (data, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}, Data shape: {data.shape}, Labels: {labels}")
        if batch_idx == 2:  # Just print the first few batches
            break