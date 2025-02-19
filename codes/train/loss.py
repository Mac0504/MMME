import torch
import torch.nn.functional as F

# Cosine Similarity
def cosine_similarity(x1, x2):
    """
    Compute cosine similarity between two feature sets.

    Parameters:
        - x1: First set of features
        - x2: Second set of features

    Returns:
        - cosine_similarity: The cosine similarity between the two feature sets
    """
    return F.cosine_similarity(x1, x2, dim=-1)

# **1. Diversity Contrast Loss (L_DCL)**
def diversity_contrast_loss(Fm, Nm, alpha=0.5):
    """
    Compute the Diversity Contrast Loss (L_DCL) based on the similarity between 
    different modality feature sets.
    
    Parameters:
        - Fm: Feature set (e.g., fused video features, EEG features, peripheral features)
        - Nm: Negative feature set from another modality
        - alpha: Margin for similarity, typically between 0 and 1
    
    Returns:
        - L_DCL: The Diversity Contrast Loss
    """
    # Compute cosine similarity
    similarity = cosine_similarity(Fm, Nm)
    # Compute the diversity contrast loss (L_DCL)
    L_DCL = 0.5 * torch.mean(((similarity + 1) / 2 - alpha) ** 2)
    return L_DCL

# **2. Consistency Contrast Loss (L_CCL)**
def consistency_contrast_loss(Fm, Pm, alpha=0.5):
    """
    Compute the Consistency Contrast Loss (L_CCL) between two feature sets 
    from the same modality.
    
    Parameters:
        - Fm: Feature set (e.g., fused features from different modalities)
        - Pm: Positive feature set from the same modality
        - alpha: Margin for similarity, typically between 0 and 1
    
    Returns:
        - L_CCL: The Consistency Contrast Loss
    """
    # Compute cosine similarity
    similarity = cosine_similarity(Fm, Pm)
    # Compute the consistency contrast loss (L_CCL)
    L_CCL = 0.5 * torch.mean(((similarity + 1) / 2 - alpha) ** 2)
    return L_CCL

# **3. Triplet Contrastive Loss (L_TRL)**
def triplet_loss(Fvep, P, N, alpha=0.5):
    """
    Compute the Triplet Contrastive Loss (L_TRL) to push positive pairs closer 
    and negative pairs farther apart based on cosine similarity.
    
    Parameters:
        - Fvep: Fused multi-modal representation (e.g., video, EEG, peripheral features)
        - P: Positive sample representations
        - N: Negative sample representations
        - alpha: Margin for similarity, typically between 0 and 1
    
    Returns:
        - L_TRL: The Triplet Contrastive Loss
    """
    # Cosine similarity between the multi-modal representation and positive/negative samples
    similarity_pos = cosine_similarity(Fvep, P)
    similarity_neg = cosine_similarity(Fvep, N)
    
    # Compute the triplet contrastive loss
    L_TRL_pos = torch.mean((1 - similarity_pos) ** 2)  # Minimize the difference for positive pairs
    L_TRL_neg = torch.mean((similarity_neg + 1) ** 2)  # Maximize the difference for negative pairs
    
    # Combine positive and negative contrastive losses
    L_TRL = alpha * (L_TRL_pos + L_TRL_neg) / 2
    return L_TRL

# **4. Cross-Entropy Loss (L_pred)**
def cross_entropy_loss(outputs, labels):
    """
    Compute the cross-entropy loss for emotion classification.
    
    Parameters:
        - outputs: The predicted emotion class scores (tensor of shape [batch_size, num_classes])
        - labels: The ground truth emotion labels (tensor of shape [batch_size,])
    
    Returns:
        - L_pred: The prediction loss (scalar)
    """
    # Convert labels to long format (required for nn.CrossEntropyLoss)
    labels = labels.long()
    
    # Calculate the cross-entropy loss
    L_pred = F.cross_entropy(outputs, labels)
    return L_pred

# **5. Total Loss Function**
def total_loss(Fvep, P, N, outputs, labels, F_v, F_e, F_p, alpha1=0.5, alpha2=0.5, alpha3=0.5):
    """
    Compute the total loss, including prediction loss, diversity contrast loss, consistency contrast loss, and triplet contrast loss.
    
    Parameters:
        - Fvep: Fused multi-modal features
        - P: Positive sample representations
        - N: Negative sample representations
        - outputs: Predicted emotion class probabilities
        - labels: Ground truth emotion labels
        - F_v: Video features
        - F_e: EEG features
        - F_p: Peripheral features
        - alpha1: Weight for Diversity Contrast Loss
        - alpha2: Weight for Consistency Contrast Loss
        - alpha3: Weight for Triplet Contrast Loss
    
    Returns:
        - total_loss: The total loss combining all individual losses
    """
    # Calculate individual losses
    L_pred = cross_entropy_loss(outputs, labels)
    L_DCL_vp = diversity_contrast_loss(F_v, F_p, alpha=alpha1)  # Video vs Peripheral
    L_DCL_ve = diversity_contrast_loss(F_v, F_e, alpha=alpha1)  # Video vs EEG
    L_CCL_ve = consistency_contrast_loss(F_v, F_e, alpha=alpha2)  # Video vs EEG
    L_CCL_vp = consistency_contrast_loss(F_v, F_p, alpha=alpha2)  # Video vs Peripheral
    L_TRL = triplet_loss(Fvep, P, N, alpha=alpha3)  # Triplet loss
    
    # Total loss
    total_loss = L_pred + alpha1 * (L_DCL_vp + L_DCL_ve) + alpha2 * (L_CCL_ve + L_CCL_vp) + alpha3 * L_TRL
    return total_loss

if __name__ == "__main__":
    # Example input (replace with actual features)
    batch_size = 2
    input_size = 192  # Example input size, replace with actual feature size
    hidden_size = 128  # Hidden layer size
    num_classes = 7    # Number of emotion classes (e.g., 7 emotions: happy, sad, angry, etc.)
    
    # Simulating features (replace with actual feature extraction)
    Fvep = torch.randn(batch_size, input_size)  # Example multi-modal fused features
    P = torch.randn(batch_size, input_size)     # Positive sample features
    N = torch.randn(batch_size, input_size)     # Negative sample features
    outputs = torch.randn(batch_size, num_classes)  # Example outputs (predicted probabilities)
    labels = torch.tensor([0, 1])  # Example labels: 0 -> Happy, 1 -> Sad, etc.
    F_v = torch.randn(batch_size, input_size)
    F_e = torch.randn(batch_size, input_size)
    F_p = torch.randn(batch_size, input_size)
    
    # Calculate the total loss
    loss = total_loss(Fvep, P, N, outputs, labels, F_v, F_e, F_p, alpha1=0.5, alpha2=0.5, alpha3=0.5)
    
    print("Total Loss:", loss.item())
