import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from models.eeg_feature_extractor import EEGFeatureExtractor
from models.me_feature_extractor import MEFeatureExtractor
from models.transformer_fusion import TransformerFusion
from data.eeg_dataset import EEGDataset
from data.me_dataset import MEDataset
from data.utils import load_data_split, create_data_loaders
import yaml

# Load configuration
with open('../configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load datasets
eeg_dataset = EEGDataset(config['data']['eeg_path'])
me_dataset = MEDataset(config['data']['me_path'])
train_dataset, val_dataset, test_dataset = load_data_split(eeg_dataset, me_dataset)

# Create data loaders
batch_size = config['train']['batch_size']
test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=batch_size, shuffle=False)[2]

# Load models
eeg_feature_extractor = EEGFeatureExtractor().to(device)
me_feature_extractor = MEFeatureExtractor().to(device)
fusion_model = TransformerFusion().to(device)

# Load trained weights
eeg_feature_extractor.load_state_dict(torch.load(config['model']['eeg_weights']))
me_feature_extractor.load_state_dict(torch.load(config['model']['me_weights']))
fusion_model.load_state_dict(torch.load(config['model']['fusion_weights']))

def evaluate_model(model, data_loader):
    """
    Evaluate the model on the given data loader.

    Args:
        model (nn.Module): The model to evaluate.
        data_loader (DataLoader): Data loader for the evaluation dataset.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for eeg_data, me_data, labels in data_loader:
            eeg_data, me_data, labels = eeg_data.to(device), me_data.to(device), labels.to(device)

            # Extract features
            eeg_features = eeg_feature_extractor(eeg_data)
            me_features = me_feature_extractor(me_data)

            # Fusion and classification
            outputs = fusion_model(eeg_features, me_features)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }

def plot_confusion_matrix(cm, class_names):
    """
    Plot the confusion matrix.

    Args:
        cm (np.array): Confusion matrix.
        class_names (list): List of class names.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_roc_curve(all_labels, all_probs, class_names):
    """
    Plot the ROC curve for multi-class classification.

    Args:
        all_labels (list): True labels.
        all_probs (list): Predicted probabilities.
        class_names (list): List of class names.
    """
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc

    # Binarize the labels
    y_true = label_binarize(all_labels, classes=np.arange(len(class_names)))
    y_score = np.array(all_probs)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'orange']
    for i, color in zip(range(len(class_names)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

def main():
    # Evaluate the model
    metrics = evaluate_model(fusion_model, test_loader)

    # Print metrics
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")

    # Plot confusion matrix
    class_names = ['Happy', 'Sad', 'Angry', 'Neutral']
    plot_confusion_matrix(metrics['confusion_matrix'], class_names)

    # Plot ROC curve
    plot_roc_curve(metrics['confusion_matrix'], class_names)

if __name__ == "__main__":
    main()