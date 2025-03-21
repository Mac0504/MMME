import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from models.eeg_feature_extractor import EEGFeatureExtractor
from models.me_feature_extractor import MEFeatureExtractor
from models.peri_feature_extractor import PeriFeatureExtractor  # Import peripheral signal feature extractor
from models.transformer_fusion import TransformerFusion
from data.eeg_dataset import EEGDataset
from data.me_dataset import MEDataset
from data.peri_dataset import PeriDataset  # Import peripheral signal dataset
from data.utils import load_data_split, create_data_loaders
from train.loss import MultiModalLoss
import yaml
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

# Load configuration
with open('../configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set up logging
log_dir = os.path.join(config['logs']['log_dir'], datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
writer = SummaryWriter(log_dir)

# Load datasets
eeg_dataset = EEGDataset(config['data']['eeg_path'])
me_dataset = MEDataset(config['data']['me_path'])
peri_dataset = PeriDataset(config['data']['peri_path'])  # Load peripheral signal dataset

# Assuming that the dataset has a method to get subject IDs
# For example, eeg_dataset.get_subject_ids() returns a list of subject IDs for each sample
subject_ids = eeg_dataset.get_subject_ids()  # This should be a list of subject IDs corresponding to each sample

# Initialize LOSO cross-validator
logo = LeaveOneGroupOut()

# Initialize models
eeg_feature_extractor = EEGFeatureExtractor().to(device)
me_feature_extractor = MEFeatureExtractor().to(device)
peri_feature_extractor = PeriFeatureExtractor().to(device)  # Initialize peripheral signal feature extractor
fusion_model = TransformerFusion().to(device)

# Initialize loss function and optimizer
criterion = MultiModalLoss(num_classes=config['model']['num_classes'], feature_dim=config['model']['feature_dim'])
optimizer = optim.Adam([ 
    {'params': eeg_feature_extractor.parameters()},
    {'params': me_feature_extractor.parameters()},
    {'params': peri_feature_extractor.parameters()},  # Add peripheral feature extractor to optimizer
    {'params': fusion_model.parameters()}
], lr=config['train']['learning_rate'])

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['train']['step_size'], gamma=config['train']['gamma'])

def train_epoch(model, data_loader, criterion, optimizer, device, epoch):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        data_loader (DataLoader): Data loader for the training dataset.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        device (torch.device): Device to run the training on.
        epoch (int): Current epoch number.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0

    for batch_idx, (eeg_data, me_data, peri_data, labels) in enumerate(data_loader):
        eeg_data, me_data, peri_data, labels = eeg_data.to(device), me_data.to(device), peri_data.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        eeg_features = eeg_feature_extractor(eeg_data)
        me_features = me_feature_extractor(me_data)
        peri_features = peri_feature_extractor(peri_data)  # Process peripheral signals
        outputs = fusion_model(eeg_features, me_features, peri_features)

        # Compute loss
        loss = criterion(outputs, eeg_features, me_features, peri_features, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Log training loss
        if batch_idx % config['train']['log_interval'] == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(eeg_data)}/{len(data_loader.dataset)} "
                  f"({100. * batch_idx / len(data_loader):.0f}%)]\tLoss: {loss.item():.6f}")
            writer.add_scalar('Train/Loss', loss.item(), epoch * len(data_loader) + batch_idx)

    avg_loss = total_loss / len(data_loader)
    return avg_loss

def validate_epoch(model, data_loader, criterion, device, epoch):
    """
    Validate the model for one epoch.

    Args:
        model (nn.Module): The model to validate.
        data_loader (DataLoader): Data loader for the validation dataset.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run the validation on.
        epoch (int): Current epoch number.

    Returns:
        float: Average loss for the epoch.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, (eeg_data, me_data, peri_data, labels) in enumerate(data_loader):
            eeg_data, me_data, peri_data, labels = eeg_data.to(device), me_data.to(device), peri_data.to(device), labels.to(device)

            # Forward pass
            eeg_features = eeg_feature_extractor(eeg_data)
            me_features = me_feature_extractor(me_data)
            peri_features = peri_feature_extractor(peri_data)  # Process peripheral signals
            outputs = fusion_model(eeg_features, me_features, peri_features)

            # Compute loss
            loss = criterion(outputs, eeg_features, me_features, peri_features, labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    print(f"Validation Epoch: {epoch}\tLoss: {avg_loss:.6f}")
    writer.add_scalar('Validation/Loss', avg_loss, epoch)
    return avg_loss

def save_checkpoint(model, optimizer, epoch, save_path):
    """
    Save the model checkpoint.

    Args:
        model (nn.Module): The model to save.
        optimizer (optim.Optimizer): The optimizer to save.
        epoch (int): Current epoch number.
        save_path (str): Path to save the checkpoint.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)

def main():
    """
    Main function to train the model with LOSO validation.
    """
    best_val_loss = float('inf')

    # Perform LOSO cross-validation
    for fold, (train_idx, val_idx) in enumerate(logo.split(np.arange(len(eeg_dataset)), groups=subject_ids)):
        print(f"Starting Fold {fold + 1}")

        # Create subsets for the current fold
        train_subset = Subset(eeg_dataset, train_idx)
        val_subset = Subset(eeg_dataset, val_idx)

        # Create data loaders for the current fold
        train_loader = DataLoader(train_subset, batch_size=config['train']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=config['train']['batch_size'], shuffle=False)

        for epoch in range(1, config['train']['num_epochs'] + 1):
            # Train for one epoch
            train_loss = train_epoch(fusion_model, train_loader, criterion, optimizer, device, epoch)

            # Validate for one epoch
            val_loss = validate_epoch(fusion_model, val_loader, criterion, device, epoch)

            # Step the learning rate scheduler
            scheduler.step()

            # Save the model checkpoint if it's the best so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(fusion_model, optimizer, epoch, os.path.join(log_dir, f'best_model_fold_{fold + 1}.pth'))

            # Save the latest model checkpoint
            save_checkpoint(fusion_model, optimizer, epoch, os.path.join(log_dir, f'latest_model_fold_{fold + 1}.pth'))

    # Close the TensorBoard writer
    writer.close()

if __name__ == "__main__":
    main()
