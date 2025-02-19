import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat

class PeriDataset(Dataset):
    def __init__(self, data_paths, labels, transform=None, max_length=1000):
        """
        Initialize the PeriDataset class.

        Parameters:
            - data_paths: List of file paths to the peripheral physiological signal data.
            - labels: Labels for each signal data (e.g., emotion classification labels).
            - transform: Optional transformation operation for data augmentation or normalization.
            - max_length: The maximum length of each signal sequence. Sequences longer than this will be truncated, and those shorter will be padded.
        """
        self.data_paths = data_paths
        self.labels = labels
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        """
        Returns the size of the dataset (the number of signal samples).
        """
        return len(self.data_paths)

    def __getitem__(self, idx):
        """
        Get the peripheral physiological signal data and its label at the specified index.

        Parameters:
            - idx: Index of the dataset.

        Returns:
            - signal_tensor: Processed tensor of the peripheral physiological signal.
            - label: The corresponding label.
        """
        # Get the signal data path and label
        data_path = self.data_paths[idx]
        label = self.labels[idx]
        
        # Load the peripheral physiological signal data
        signal = self.load_signal(data_path)
        
        # Process the signal data
        signal_tensor = self.process_signal(signal)
        
        # Apply transformation if any (e.g., normalization)
        if self.transform:
            signal_tensor = self.transform(signal_tensor)

        return signal_tensor, label

    def load_signal(self, path):
        """
        Load peripheral physiological signal data, supporting various formats (e.g., .mat, .csv).

        Parameters:
            - path: Path to the peripheral physiological signal data file.

        Returns:
            - signal: Loaded signal data, typically a multichannel time series.
        """
        # Here we assume the data is in .mat format (can be modified for other formats)
        data = loadmat(path)
        
        # Assume the signal data is stored under the key 'signal_data' in the .mat file
        signal = data['signal_data']
        
        # Return the signal data
        return signal

    def process_signal(self, signal):
        """
        Process the peripheral physiological signal data to ensure uniform length.
        If the signal length is greater than max_length, it will be truncated; if shorter, it will be padded.

        Parameters:
            - signal: Peripheral physiological signal data, typically of shape (channels, time_steps)

        Returns:
            - signal_tensor: A tensor of shape (C, T), where C is the number of channels, and T is the time steps.
        """
        # Get the length of the signal
        num_channels, num_time_steps = signal.shape

        if num_time_steps > self.max_length:
            # If the signal time steps exceed the maximum length, truncate
            signal = signal[:, :self.max_length]
        elif num_time_steps < self.max_length:
            # If the signal time steps are less than the maximum length, pad with zeros
            padding_time_steps = self.max_length - num_time_steps
            signal = np.pad(signal, ((0, 0), (0, padding_time_steps)), mode='constant', constant_values=0)  # Zero-padding

        # Convert to PyTorch tensor
        signal_tensor = torch.tensor(signal).float()

        return signal_tensor

    def collate_fn(self, batch):
        """
        Custom collate function for handling batch processing of signal data.

        Parameters:
            - batch: A batch of samples, each is a tuple (signal_tensor, label)

        Returns:
            - signals: A batch of signal tensors, shape (batch_size, C, T)
            - labels: A batch of labels, shape (batch_size,)
        """
        signals, labels = zip(*batch)
        
        # Stack the signal tensors into one large tensor, shape (batch_size, C, T)
        signals = torch.stack(signals, dim=0)
        
        # Convert the labels to a tensor, shape (batch_size,)
        labels = torch.tensor(labels, dtype=torch.long)
        
        return signals, labels


if __name__ == "__main__":
    # Example data paths and labels
    data_paths = ["./data/peri_signal1.mat", "./data/peri_signal2.mat"]  # File paths for peripheral physiological signals
    labels = [0, 1]  # Example labels, 0 for happy, 1 for sad

    # Create dataset object
    dataset = PeriDataset(data_paths, labels)

    # Get the first sample
    signal_tensor, label = dataset[0]
    print("Signal tensor shape:", signal_tensor.shape)  # Should print (C, T)
    print("Label:", label)

    # Create a DataLoader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)
    for batch_signals, batch_labels in dataloader:
        print("Batch signals shape:", batch_signals.shape)
        print("Batch labels:", batch_labels)
