import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class OpticalFlowExtractor:
    """
    Optical flow extraction module.
    Computes optical flow between two frames and generates optical flow features.
    """
    def __init__(self):
        super(OpticalFlowExtractor, self).__init__()
        # Future: Add advanced optical flow computation methods

    def forward(self, onset_frame, apex_frame):
        """
        Compute optical flow features.

        Args:
            onset_frame (torch.Tensor): Onset frame image, shape (batch_size, C, H, W)
            apex_frame (torch.Tensor): Apex frame image, shape (batch_size, C, H, W)

        Returns:
            torch.Tensor: Optical flow features (batch_size, 2, H, W)
        """
        flow = apex_frame - onset_frame  # Simple optical flow approximation, can be replaced with more complex methods
        return flow

class SpatialTransformerNetwork(nn.Module):
    """
    Spatial Transformer Network (STN) for aligning input images.
    """
    def __init__(self):
        super(SpatialTransformerNetwork, self).__init__()
        
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 5 * 5, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image, shape (batch_size, 3, H, W)

        Returns:
            torch.Tensor: Aligned image.
        """
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 5 * 5)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

class MEFeatureExtractor(nn.Module):
    """
    Micro-expression feature extraction module.
    Combines optical flow features and apex frame image features.
    """
    def __init__(self, pretrained=True):
        super(MEFeatureExtractor, self).__init__()

        # Optical flow extraction module
        self.optical_flow_extractor = OpticalFlowExtractor()

        # Spatial Transformer Network
        self.stn = SpatialTransformerNetwork()

        # Use pre-trained ResNet for apex frame feature extraction
        self.apex_feature_extractor = models.resnet18(pretrained=pretrained)
        self.apex_feature_extractor.fc = nn.Linear(self.apex_feature_extractor.fc.in_features, 256)

        # Optical flow feature extraction
        self.flow_feature_extractor = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 256),
            nn.ReLU()
        )

        # Final fusion
        self.fc_fusion = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU()
        )

    def forward(self, onset_frame, apex_frame):
        """
        Forward pass.

        Args:
            onset_frame (torch.Tensor): Onset frame image, shape (batch_size, 3, H, W)
            apex_frame (torch.Tensor): Apex frame image, shape (batch_size, 3, H, W)

        Returns:
            torch.Tensor: Micro-expression feature vector, shape (batch_size, 64)
        """
        # Optical flow features
        flow = self.optical_flow_extractor(onset_frame, apex_frame)
        flow_features = self.flow_feature_extractor(flow)

        # Spatial transformation and apex frame feature extraction
        aligned_apex = self.stn(apex_frame)
        apex_features = self.apex_feature_extractor(aligned_apex)

        # Feature fusion
        combined_features = torch.cat((flow_features, apex_features), dim=1)
        final_features = self.fc_fusion(combined_features)

        return final_features

if __name__ == "__main__":
    # Test code
    batch_size = 8
    channels, height, width = 3, 224, 224

    onset_frame = torch.rand(batch_size, channels, height, width)
    apex_frame = torch.rand(batch_size, channels, height, width)

    model = MEFeatureExtractor(pretrained=False)
    features = model(onset_frame, apex_frame)

    print("Output shape:", features.shape)