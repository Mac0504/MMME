import os
import argparse
import yaml
from data.eeg_dataset import EEGDataset
from data.me_dataset import MEDataset
from data.utils import load_data_split, create_data_loaders
from models.eeg_feature_extractor import EEGFeatureExtractor
from models.me_feature_extractor import MEFeatureExtractor
from models.transformer_fusion import TransformerFusion
from train.train import train_epoch, validate_epoch, save_checkpoint
from train.loss import MultiModalLoss
from train.evaluate import evaluate_model
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Load configuration
def load_config(config_path):
    """
    Load the configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Set up argument parser
def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Multimodal Emotion Recognition")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'infer'], required=True, help='Mode to run the script in')
    parser.add_argument('--checkpoint', type=str, help='Path to the model checkpoint for evaluation or inference')
    return parser.parse_args()

# Main function
def main():
    """
    Main function to run the multimodal emotion recognition pipeline.
    """
    args = parse_args()
    config = load_config(args.config)

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load datasets
    eeg_dataset = EEGDataset(config['data']['eeg_path'])
    me_dataset = MEDataset(config['data']['me_path'])
    train_dataset, val_dataset, test_dataset = load_data_split(eeg_dataset, me_dataset)

    # Create data loaders
    batch_size = config['train']['batch_size']
    train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=batch_size)

    # Initialize models
    eeg_feature_extractor = EEGFeatureExtractor().to(device)
    me_feature_extractor = MEFeatureExtractor().to(device)
    fusion_model = TransformerFusion().to(device)

    # Initialize loss function and optimizer
    criterion = MultiModalLoss(num_classes=config['model']['num_classes'], feature_dim=config['model']['feature_dim'])
    optimizer = optim.Adam([
        {'params': eeg_feature_extractor.parameters()},
        {'params': me_feature_extractor.parameters()},
        {'params': fusion_model.parameters()}
    ], lr=config['train']['learning_rate'])

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['train']['step_size'], gamma=config['train']['gamma'])

    # Set up logging
    log_dir = os.path.join(config['logs']['log_dir'], 'logs')
    writer = SummaryWriter(log_dir)

    if args.mode == 'train':
        # Training mode
        best_val_loss = float('inf')

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
                save_checkpoint(fusion_model, optimizer, epoch, os.path.join(log_dir, 'best_model.pth'))

            # Save the latest model checkpoint
            save_checkpoint(fusion_model, optimizer, epoch, os.path.join(log_dir, 'latest_model.pth'))

        # Close the TensorBoard writer
        writer.close()

    elif args.mode == 'eval':
        # Evaluation mode
        if args.checkpoint is None:
            raise ValueError("Checkpoint path must be provided for evaluation mode.")

        # Load the model checkpoint
        checkpoint = torch.load(args.checkpoint)
        fusion_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

        # Evaluate the model
        metrics = evaluate_model(fusion_model, test_loader)

        # Print evaluation metrics
        print(f"Evaluation Metrics at Epoch {epoch}:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")

    elif args.mode == 'infer':
        # Inference mode
        if args.checkpoint is None:
            raise ValueError("Checkpoint path must be provided for inference mode.")

        # Load the model checkpoint
        checkpoint = torch.load(args.checkpoint)
        fusion_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

        # Perform inference (example with a single batch)
        fusion_model.eval()
        with torch.no_grad():
            for batch_idx, (eeg_data, me_data, labels) in enumerate(test_loader):
                eeg_data, me_data, labels = eeg_data.to(device), me_data.to(device), labels.to(device)

                # Forward pass
                eeg_features = eeg_feature_extractor(eeg_data)
                me_features = me_feature_extractor(me_data)
                outputs = fusion_model(eeg_features, me_features)

                # Print predictions
                _, preds = torch.max(outputs, 1)
                print(f"Batch {batch_idx + 1} Predictions: {preds.cpu().numpy()}")
                print(f"Batch {batch_idx + 1} Ground Truth: {labels.cpu().numpy()}")

                # Break after the first batch for demonstration
                break

if __name__ == "__main__":
    main()