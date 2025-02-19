import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

# LSTM Model to process time-series data (e.g., PPG, RSP, SKT, EDA, ECG)
class LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTM_Model, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, (hn, cn) = self.lstm(x)
        # We use the last hidden state for output
        output = self.fc(hn[-1])
        return output

# Normalization-based Attention Module (NAM)
class NAM(nn.Module):
    def __init__(self, channels):
        super(NAM, self).__init__()
        # Normalization-based Attention Module (NAM)
        self.attention = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_map = self.attention(x)
        return x * attention_map  # Apply attention to the features

# Feature Extractor for Peripheral Signals (PPG, RSP, SKT, EDA, ECG)
class PeriFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PeriFeatureExtractor, self).__init__()
        # LSTM for sequential processing of peripheral signals
        self.lstm = LSTM_Model(input_size, hidden_size)
        
        # NAM (Normalization-based Attention Module)
        self.nam = NAM(hidden_size)
    
    def forward(self, x):
        """
        Forward pass through the PeriFeatureExtractor.
        
        Parameters:
            - x: Input peripheral signal data (tensor) of shape [batch_size, seq_len, input_size]
        
        Returns:
            - F': The final extracted features after LSTM and NAM (tensor).
        """
        # Step 1: Pass input through the LSTM model
        lstm_output = self.lstm(x)
        
        # Step 2: Apply NAM to the output of the LSTM
        features_with_attention = self.nam(lstm_output)
        
        return features_with_attention

    def load_data(self, file_path):
        """
        Load peripheral physiological signals from a CSV file.
        
        Parameters:
            - file_path: Path to the CSV file containing the signal data.
        
        Returns:
            - data: Loaded signal data (numpy array).
        """
        # Load data from CSV
        df = pd.read_csv(file_path)
        
        # Assuming the signals are stored as columns in the CSV file, with rows as time steps
        # For example: PPG, RSP, SKT, EDA, ECG columns
        data = df.values.T  # Transpose to shape [channels, time_steps]
        
        # Convert to PyTorch tensor
        data_tensor = torch.tensor(data, dtype=torch.float32)
        
        # Normalize the data (optional, depending on your data's scale)
        data_tensor = (data_tensor - data_tensor.mean()) / data_tensor.std()
        
        return data_tensor

if __name__ == "__main__":
    
    # Create PeriFeatureExtractor instance
    model = PeriFeatureExtractor(input_size=1, hidden_size=64, output_size=64)  # 1 input channel, 64 hidden size
    
    # Load the peripheral signal data (PPG, RSP, SKT, EDA, ECG)
    peri_data = model.load_data(file_path)
    
    # Add batch dimension: [batch_size, seq_len, input_size]
    # For example, batch_size=1 (processing one sample at a time)
    peri_data = peri_data.unsqueeze(0)  # Shape becomes [1, time_steps, 5] (assuming 5 signals)
    
    # Forward pass through the model
    output_features = model(peri_data)
    
    print("Extracted features shape:", output_features.shape)
