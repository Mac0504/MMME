import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, alpha1=0.5, alpha2=0.5, alpha3=0.5):
        super(EmotionClassifier, self).__init__()
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(input_size, hidden_size)  # First hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Second hidden layer
        self.fc3 = nn.Linear(hidden_size, num_classes)  # Output layer (emotion categories)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Store hyperparameters for contrastive losses
        self.alpha1 = alpha1  # Weight for L_DCL
        self.alpha2 = alpha2  # Weight for L_CCL
        self.alpha3 = alpha3  # Weight for L_TRL

    def forward(self, F_vep):
        """
        Forward pass for emotion classification.
        
        Parameters:
            - F_vep: The concatenated multi-modal features (tensor of shape [batch_size, input_size])
        
        Returns:
            - output: Predicted class probabilities (tensor of shape [batch_size, num_classes])
        """
        # Apply first fully connected layer and activation
        x = F.relu(self.fc1(F_vep))
        x = self.dropout(x)
        
        # Apply second fully connected layer and activation
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Apply output layer
        output = self.fc3(x)
        
        return output
    
    def contrastive_loss(self, F_vep, P, N):
        """
        Computes the sample contrastive loss (L_TRL) to bring positive pairs closer and negative pairs farther apart.
        
        Parameters:
            - F_vep: The multi-modal fused features (tensor of shape [batch_size, feature_size])
            - P: Positive sample representations (tensor of shape [batch_size, feature_size])
            - N: Negative sample representations (tensor of shape [batch_size, feature_size])
        
        Returns:
            - L_TRL: The sample contrastive loss (scalar)
        """
        # Cosine similarity between the multi-modal representation and positive/negative samples
        similarity_pos = F.cosine_similarity(F_vep, P)
        similarity_neg = F.cosine_similarity(F_vep, N)
        
        # Calculate the contrastive loss
        L_TRL_pos = torch.mean((1 - similarity_pos) ** 2)  # Minimize the difference for positive pairs
        L_TRL_neg = torch.mean((similarity_neg + 1) ** 2)  # Maximize the difference for negative pairs
        
        # Combine positive and negative contrastive losses
        L_TRL = self.alpha3 * (L_TRL_pos + L_TRL_neg) / 2
        return L_TRL
    
    def cross_entropy_loss(self, outputs, labels):
        """
        Computes the cross-entropy loss for emotion classification.
        
        Parameters:
            - outputs: The predicted class scores (tensor of shape [batch_size, num_classes])
            - labels: The ground truth emotion labels (tensor of shape [batch_size,])
        
        Returns:
            - L_pred: The prediction loss (scalar)
        """
        # Convert labels to long format (required for nn.CrossEntropyLoss)
        labels = labels.long()
        
        # Calculate the cross-entropy loss
        L_pred = F.cross_entropy(outputs, labels)
        return L_pred
    
    def diversity_contrast_loss(self, F_m, N_m):
        """
        Computes the Diversity Contrast Loss (L_DCL) to encourage learning diverse representations.
        
        Parameters:
            - F_m: The feature representations from one modality
            - N_m: The negative sample representations from another modality
        
        Returns:
            - L_DCL: The diversity contrast loss (scalar)
        """
        # Cosine similarity between features
        cosine_sim = F.cosine_similarity(F_m, N_m)
        L_DCL = 0.5 * torch.mean(((cosine_sim + 1) / 2 - self.alpha1) ** 2)
        return L_DCL

    def consistency_contrast_loss(self, F_m, P_m):
        """
        Computes the Consistency Contrast Loss (L_CCL) to enforce consistency within the same modality.
        
        Parameters:
            - F_m: The feature representations from one modality
            - P_m: The positive sample representations from the same modality
        
        Returns:
            - L_CCL: The consistency contrast loss (scalar)
        """
        # Cosine similarity between features
        cosine_sim = F.cosine_similarity(F_m, P_m)
        L_CCL = 0.5 * torch.mean(((cosine_sim + 1) / 2 - self.alpha2) ** 2)
        return L_CCL
    
    def total_loss(self, F_vep, P, N, outputs, labels, F_v, F_e, F_p):
        """
        Computes the total loss including prediction loss, diversity contrast loss, consistency contrast loss, and triplet contrast loss.
        
        Parameters:
            - F_vep: The fused multi-modal features
            - P: Positive sample representations
            - N: Negative sample representations
            - outputs: The predicted emotion class probabilities
            - labels: The ground truth emotion labels
            - F_v: Video features
            - F_e: EEG features
            - F_p: Peripheral features
        
        Returns:
            - total_loss: The total loss combining all individual losses
        """
        # Calculate individual losses
        L_pred = self.cross_entropy_loss(outputs, labels)
        L_DCL_vp = self.diversity_contrast_loss(F_v, F_p)  # Video vs Peripheral
        L_DCL_ve = self.diversity_contrast_loss(F_v, F_e)  # Video vs EEG
        L_CCL_ve = self.consistency_contrast_loss(F_v, F_e)  # Video vs EEG
        L_CCL_vp = self.consistency_contrast_loss(F_v, F_p)  # Video vs Peripheral
        L_TRL = self.contrastive_loss(F_vep, P, N)  # Triplet loss
        
        # Total loss
        total_loss = L_pred + self.alpha1 * L_DCL_vp + self.alpha1 * L_DCL_ve + self.alpha2 * L_CCL_ve + self.alpha2 * L_CCL_vp + self.alpha3 * L_TRL
        return total_loss

if __name__ == "__main__":
    batch_size = 2
    input_size = 192  # Example input size, replace with actual feature size
    hidden_size = 128  # Hidden layer size
    num_classes = 7    # Number of emotion classes (e.g., 7 emotions: happy, sad, angry, etc.)
    
    # Create the EmotionClassifier instance
    model = EmotionClassifier(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
    
    # Example input features (replace with actual concatenated features from previous modules)
    target_labels = torch.tensor([0, 1])  # Example labels: 0 -> Happy, 1 -> Sad, etc.
    F_v = torch.randn(batch_size, input_size)
    F_e = torch.randn(batch_size, input_size)
    F_p = torch.randn(batch_size, input_size)
    
    # Forward pass
    output = model(example_input)
    
    # Calculate the total loss
    L_DCL = model.diversity_contrast_loss(F_v, F_p)
    L_CCL = model.consistency_contrast_loss(F_v, F_e)
    L_TRL = model.contrastive_loss(example_input, example_input, example_input)  # Example triplet loss
    total_loss = model.total_loss(example_input, example_input, example_input, output, target_labels, F_v, F_e, F_p)
    
    print("Emotion classification output:", output)
    print("Total Loss:", total_loss.item())
