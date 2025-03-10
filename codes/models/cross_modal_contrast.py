import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Multihead Attention Layer
class MultiheadAttentionLayer(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiheadAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        # x shape: [batch_size, seq_len, embed_size]
        attention_output, _ = self.attention(x, x, x)
        return attention_output

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

# Diversity Contrast Loss (L_DCL) based on formula (9)
def diversity_contrast_loss(Fm, Nm, alpha=0.5):
    """
    Compute the Diversity Contrast Loss (L_DCL) based on formula (9).
    
    Parameters:
        - Fm: Feature set (e.g., fused video features, EEG features, peripheral features)
        - Nm: Negative features
        - alpha: Margin for similarity, typically between 0 and 1
    
    Returns:
        - L_DCL: The Diversity Contrast Loss
    """
    # Compute cosine similarity
    similarity = cosine_similarity(Fm, Nm)
    # Compute the diversity contrast loss (L_DCL)
    L_DCL = 0.5 * torch.mean(((similarity + 1) / 2 - alpha) ** 2)
    return L_DCL

# Consistency Contrast Loss (L_CCL) based on formula (10)
def consistency_contrast_loss(Fm, Pm, alpha=0.5):
    """
    Compute the Consistency Contrast Loss (L_CCL) based on formula (10).
    
    Parameters:
        - Fm: Feature set (e.g., fused features from different modalities)
        - Pm: Positive features
        - alpha: Margin for similarity, typically between 0 and 1
    
    Returns:
        - L_CCL: The Consistency Contrast Loss
    """
    # Compute cosine similarity
    similarity = cosine_similarity(Fm, Pm)
    # Compute the consistency contrast loss (L_CCL)
    L_CCL = 0.5 * torch.mean(((similarity + 1) / 2 - alpha) ** 2)
    return L_CCL

# Cross-modal Contrastive Learning Module
class CrossModalContrastiveLearning(nn.Module):
    def __init__(self, embed_size, num_heads, alpha=0.5):
        super(CrossModalContrastiveLearning, self).__init__()
        
        # Multihead Attention for each modality
        self.attention_v = MultiheadAttentionLayer(embed_size, num_heads)
        self.attention_e = MultiheadAttentionLayer(embed_size, num_heads)
        self.attention_p = MultiheadAttentionLayer(embed_size, num_heads)
        
        self.alpha = alpha

    def forward(self, F_v, F_e, F_p):
        """
        Forward pass through the Cross-modal Contrastive Learning module.
        
        Parameters:
            - F_v: Features from video modality (tensor of shape [batch_size, seq_len, embed_size])
            - F_e: Features from EEG modality (tensor of shape [batch_size, seq_len, embed_size])
            - F_p: Features from peripheral signals modality (tensor of shape [batch_size, seq_len, embed_size])
        
        Returns:
            - F_vep: Concatenated features from all modalities after attention and contrastive learning
            - L_DCL: The Diversity Contrast Loss
            - L_CCL: The Consistency Contrast Loss
        """
        # Step 1: Apply Multihead Attention for each modality
        F_v_att = self.attention_v(F_v)
        F_e_att = self.attention_e(F_e)
        F_p_att = self.attention_p(F_p)
        
        # Step 2: Compute Diversity Contrast Loss (L_DCL)
        L_DCL_ve = diversity_contrast_loss(F_v_att, F_e_att, alpha=self.alpha)  # Video vs EEG
        L_DCL_vp = diversity_contrast_loss(F_v_att, F_p_att, alpha=self.alpha)  # Video vs Peripheral
        L_DCL_ep = diversity_contrast_loss(F_e_att, F_p_att, alpha=self.alpha)  # EEG vs Peripheral
        
        L_DCL = (L_DCL_ve + L_DCL_vp + L_DCL_ep) / 3
        
        # Step 3: Compute Consistency Contrast Loss (L_CCL)
        L_CCL_ve = consistency_contrast_loss(F_v_att, F_e_att, alpha=self.alpha)  # Video vs EEG
        L_CCL_vp = consistency_contrast_loss(F_v_att, F_p_att, alpha=self.alpha)  # Video vs Peripheral
        L_CCL_ep = consistency_contrast_loss(F_e_att, F_p_att, alpha=self.alpha)  # EEG vs Peripheral
        
        L_CCL = (L_CCL_ve + L_CCL_vp + L_CCL_ep) / 3
        
        # Step 4: Concatenate the features from all modalities
        F_vep = torch.cat((F_v_att, F_e_att, F_p_att), dim=-1)  # Concatenate along feature dimension
        
        return F_vep, L_DCL, L_CCL

if __name__ == "__main__":
    # Example feature vectors (you should replace these with actual features)
    batch_size = 2
    seq_len = 10
    embed_size = 64
    
    F_v = np.load(r"..data/MEs/me_data.npy")  # Video features
    F_e = np.load(r"..data/EEG/eeg_data.npy")  # EEG features
    F_p = np.load(r"..data/PERI/peri_data.npy")  # Peripheral features
    
    # Create the CrossModalContrastiveLearning module
    model = CrossModalContrastiveLearning(embed_size=embed_size, num_heads=4, alpha=0.5)
    
    # Forward pass
    F_vep, L_DCL, L_CCL = model(F_v, F_e, F_p)
    
    # Output results
    print("Concatenated Features Shape:", F_vep.shape)
    print("Diversity Contrast Loss (L_DCL):", L_DCL.item())
    print("Consistency Contrast Loss (L_CCL):", L_CCL.item())
