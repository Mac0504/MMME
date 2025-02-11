import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from itertools import cycle
import pandas as pd
import yaml
import torch
from torch.utils.data import DataLoader
from models.eeg_feature_extractor import EEGFeatureExtractor
from models.me_feature_extractor import MEFeatureExtractor
from models.transformer_fusion import TransformerFusion
from data.eeg_dataset import EEGDataset
from data.me_dataset import MEDataset
from data.utils import load_data_split

# Load configuration
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load datasets
eeg_dataset = EEGDataset(config['data']['eeg_path'])
me_dataset = MEDataset(config['data']['me_path'])
train_loader, val_loader, test_loader = load_data_split(eeg_dataset, me_dataset, config['data']['split_ratio'])

# Load models
eeg_feature_extractor = EEGFeatureExtractor().to(device)
me_feature_extractor = MEFeatureExtractor().to(device)
fusion_model = TransformerFusion().to(device)

# Load trained weights
eeg_feature_extractor.load_state_dict(torch.load(config['model']['eeg_weights']))
me_feature_extractor.load_state_dict(torch.load(config['model']['me_weights']))
fusion_model.load_state_dict(torch.load(config['model']['fusion_weights']))

# Function to get predictions
def get_predictions(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for eeg_data, me_data, labels in data_loader:
            eeg_data, me_data, labels = eeg_data.to(device), me_data.to(device), labels.to(device)
            eeg_features = eeg_feature_extractor(eeg_data)
            me_features = me_feature_extractor(me_data)
            outputs = fusion_model(eeg_features, me_features)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)

# Get predictions for the test set
test_preds, test_labels = get_predictions(fusion_model, test_loader)

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Plot confusion matrix
class_names = ['Happy', 'Sad', 'Fear', 'Surprise', 'Disgust']
plot_confusion_matrix(test_labels, test_preds, classes=class_names, normalize=True)

# Function to plot ROC curve
def plot_roc_curve(y_true, y_pred, n_classes):
    y_true = label_binarize(y_true, classes=np.arange(n_classes))
    y_pred = label_binarize(y_pred, classes=np.arange(n_classes))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    colors = cycle(['blue', 'red', 'green', 'orange'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

# Plot ROC curve
plot_roc_curve(test_labels, test_preds, n_classes=len(class_names))

# Function to plot precision-recall curve
def plot_precision_recall_curve(y_true, y_pred, n_classes):
    y_true = label_binarize(y_true, classes=np.arange(n_classes))
    y_pred = label_binarize(y_pred, classes=np.arange(n_classes))
    precision = dict()
    recall = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
    colors = cycle(['blue', 'red', 'green', 'orange'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2, label=f'Precision-Recall curve of class {i}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()

# Plot precision-recall curve
plot_precision_recall_curve(test_labels, test_preds, n_classes=len(class_names))

# Function to plot feature distributions
def plot_feature_distributions(features, labels, class_names):
    df = pd.DataFrame(features)
    df['label'] = labels
    sns.pairplot(df, hue='label', palette='Set1', diag_kind='kde', vars=df.columns[:-1])
    plt.suptitle('Feature Distributions by Class', y=1.02)
    plt.show()

# Extract features for visualization
def extract_features(data_loader):
    eeg_feature_extractor.eval()
    me_feature_extractor.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for eeg_data, me_data, labels in data_loader:
            eeg_data, me_data, labels = eeg_data.to(device), me_data.to(device), labels.to(device)
            eeg_features = eeg_feature_extractor(eeg_data)
            me_features = me_feature_extractor(me_data)
            features = torch.cat((eeg_features, me_features), dim=1)
            all_features.extend(features.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_features), np.array(all_labels)

# Extract features from the test set
test_features, test_labels = extract_features(test_loader)

# Plot feature distributions
plot_feature_distributions(test_features, test_labels, class_names)

# Function to save visualizations
def save_visualizations():
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    plt.savefig('visualizations/confusion_matrix.png')
    plt.savefig('visualizations/roc_curve.png')
    plt.savefig('visualizations/precision_recall_curve.png')
    plt.savefig('visualizations/feature_distributions.png')

# Save all visualizations
save_visualizations()