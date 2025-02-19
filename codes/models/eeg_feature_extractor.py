import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import scipy.signal as signal

# Power Spectral Density (PSD) Extraction from EEG signal
def compute_psd(eeg_data, fs=250, nperseg=256):
    """
    Compute the Power Spectral Density (PSD) of an EEG signal.
    
    Parameters:
        - eeg_data: The EEG signal (numpy array of shape [channels, time_steps])
        - fs: Sampling frequency (default 250 Hz)
        - nperseg: Length of each segment for FFT computation (default 256)
        
    Returns:
        - psd: The computed PSD features (numpy array of shape [channels, frequency_bins, time_steps])
    """
    psd_features = []
    for channel_data in eeg_data:
        f, Pxx = signal.welch(channel_data, fs=fs, nperseg=nperseg)
        psd_features.append(Pxx)
    
    # Stack all channels' PSD features
    psd_features = np.array(psd_features)
    return psd_features

class TCN(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, num_layers=3):
        super(TCN, self).__init__()
        self.num_layers = num_layers
        
        # Define temporal convolution layers
        layers = []
        for i in range(num_layers):
            in_channels = input_channels if i == 0 else output_channels
            layers.append(
                nn.Conv1d(in_channels, output_channels, kernel_size=kernel_size, padding=kernel_size//2)
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        
        self.tcn = nn.Sequential(*layers)

    def forward(self, x):
        return self.tcn(x)

class NAM(nn.Module):
    def __init__(self, channels):
        super(NAM, self).__init__()
        # Normalization-based Attention Module (NAM)
        self.attention = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_map = self.attention(x)
        return x * attention_map  # Apply attention to the features

class EEGFeatureExtractor(nn.Module):
    def __init__(self, input_channels, output_channels, fs=250, nperseg=256):
        super(EEGFeatureExtractor, self).__init__()
        # Power Spectral Density (PSD) Feature Extraction
        self.psd_extractor = self.extract_psd
        
        # Temporal Convolutional Network (TCN)
        self.tcn = TCN(input_channels, output_channels)
        
        # Normalization-based Attention Module (NAM)
        self.nam = NAM(output_channels)

        # PSD computation parameters
        self.fs = fs
        self.nperseg = nperseg

    def extract_psd(self, eeg_data):
        """
        Extract Power Spectral Density (PSD) features from the EEG data.
        
        Parameters:
            - eeg_data: The EEG signal (numpy array of shape [channels, time_steps])
        
        Returns:
            - psd_features: The extracted PSD features (tensor).
        """
        # Compute PSD for each channel in the EEG signal
        psd = compute_psd(eeg_data, fs=self.fs, nperseg=self.nperseg)
        
        # Convert PSD features to a PyTorch tensor
        psd_tensor = torch.tensor(psd, dtype=torch.float32)
        return psd_tensor

    def forward(self, eeg_data):
        """
        Forward pass through the EEG feature extractor.
        
        Parameters:
            - eeg_data: Input EEG signal (numpy array of shape [channels, time_steps])

        Returns:
            - feature_map: Extracted features after passing through TCN and NAM.
        """
        # Step 1: Extract Power Spectral Density (PSD) features from EEG data
        psd_features = self.extract_psd(eeg_data)

        # Step 2: Pass the PSD features through the TCN (Temporal Convolutional Network)
        psd_features = psd_features.unsqueeze(0)  # Add batch dimension: [1, channels, time_steps]
        tcn_output = self.tcn(psd_features)

        # Step 3: Apply NAM (Normalization-based Attention Module)
        attention_features = self.nam(tcn_output)
        
        return attention_features

    def load_eeg_data(self, mat_file_path, eeg_key="EEG"):
        """
        Load EEG data from a .mat file.
        
        Parameters:
            - mat_file_path: Path to the .mat file containing EEG data.
            - eeg_key: The key under which the EEG data is stored in the .mat file.
        
        Returns:
            - eeg_data: The loaded EEG signal (numpy array of shape [channels, time_steps])
        """
        # Load the .mat file
        mat_data = sio.loadmat(mat_file_path)
        
        # Extract EEG data from the .mat file (assuming it's stored under the key `eeg_key`)
        eeg_data = mat_data[eeg_key]
        
        # Return EEG data (make sure it's in the shape [channels, time_steps])
        return eeg_data

if __name__ == "__main__":
    
    # Load EEG data from the .mat file
    eeg_feature_extractor = EEGFeatureExtractor(input_channels=5, output_channels=64)  # 5 channels, 64 output channels
    eeg_data = eeg_feature_extractor.load_eeg_data(mat_file_path, eeg_key="EEG")

    # Forward pass through the model
    output_features = eeg_feature_extractor(eeg_data)
    
    print("Extracted features shape:", output_features.shape)
