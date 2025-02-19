import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

# Optical Flow Extraction (using OpenCV to compute optical flow)
def compute_optical_flow(frame1, frame2):
    flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

class CA_I3D(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CA_I3D, self).__init__()
        # Inflated 3D Convolutions with Coordinate Attention
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv3 = nn.Conv3d(128, output_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        
        # Coordinate Attention (Placeholder for the actual logic)
        self.coord_attention = nn.Sequential(
            nn.Conv3d(output_channels, output_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        attention_map = self.coord_attention(x)  # Apply attention
        x = x * attention_map + x  # Residual connection
        return x

class NAM(nn.Module):
    def __init__(self, channels):
        super(NAM, self).__init__()
        # Normalization-based Attention Module (NAM)
        self.attention = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Apply normalization-based attention
        attention_map = self.attention(x)
        return x * attention_map  # Apply attention to the features

class VideoFeatureExtractor(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(VideoFeatureExtractor, self).__init__()
        # Optical Flow Extraction
        self.flow_extractor = self.extract_optical_flow
        
        # CA-I3D Model (Inflated 3D Convolutions with Coordinate Attention)
        self.ca_i3d = CA_I3D(input_channels, output_channels)
        
        # NAM (Normalization-based Attention Module)
        self.nam = NAM(output_channels)

    def extract_optical_flow(self, video_path):
        # Open video file using OpenCV
        cap = cv2.VideoCapture(video_path)
        
        # Initialize variables
        frames = []
        flow_features = []
        
        ret, prev_frame = cap.read()
        if not ret:
            raise ValueError("Could not read the video file.")
        
        # Convert to grayscale (for optical flow calculation)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Read frames and compute optical flow between consecutive frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert the current frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Compute optical flow between previous and current frames
            flow = compute_optical_flow(prev_gray, gray)
            flow_features.append(flow)
            
            # Set the current frame as previous for the next iteration
            prev_gray = gray

        cap.release()
        
        # Convert list of flow features into a tensor (shape: [num_frames, height, width, 2])
        flow_features = np.array(flow_features)
        flow_features = torch.tensor(flow_features, dtype=torch.float32)
        
        return flow_features

    def forward(self, video_path):
        # Step 1: Extract optical flow from the video
        optical_flow = self.extract_optical_flow(video_path)

        # Step 2: Pass the optical flow features through the CA-I3D model
        flow_features = self.ca_i3d(optical_flow)
        
        # Step 3: Apply NAM (Normalization-based Attention Module)
        features_with_attention = self.nam(flow_features)
        
        return features_with_attention


if __name__ == "__main__":
  
    # Create VideoFeatureExtractor instance
    model = VideoFeatureExtractor(input_channels=3, output_channels=128)  # RGB input (3 channels), 128 output channels
    
    # Forward pass
    output_features = model(video_path)
    
    print("Extracted features shape:", output_features.shape)
