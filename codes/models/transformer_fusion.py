import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerFusion(nn.Module):
    """
    Transformer-based fusion network for EEG and ME features.
    """
    def __init__(self, 
                 eeg_input_dim, me_input_dim, 
                 d_model=128, 
                 num_heads=8, 
                 num_layers=2, 
                 fusion_hidden_dim=256, 
                 num_classes=3, 
                 dropout=0.1):
        """
        Args:
            eeg_input_dim (int): Input dimension for EEG features.
            me_input_dim (int): Input dimension for Micro-Expression (ME) features.
            d_model (int): Hidden size of the Transformer model.
            num_heads (int): Number of attention heads in Multi-Head Attention.
            num_layers (int): Number of Transformer layers.
            fusion_hidden_dim (int): Hidden dimension of the fusion layer.
            num_classes (int): Number of output classes for classification.
            dropout (float): Dropout rate.
        """
        super(TransformerFusion, self).__init__()
        
        # EEG Transformer Encoder
        self.eeg_embedding = nn.Linear(eeg_input_dim, d_model)
        self.eeg_positional_encoding = PositionalEncoding(d_model, dropout)
        self.eeg_transformer = TransformerEncoder(
            TransformerEncoderLayer(d_model, num_heads, dim_feedforward=4 * d_model, dropout=dropout),
            num_layers
        )
        
        # ME Transformer Encoder
        self.me_embedding = nn.Linear(me_input_dim, d_model)
        self.me_positional_encoding = PositionalEncoding(d_model, dropout)
        self.me_transformer = TransformerEncoder(
            TransformerEncoderLayer(d_model, num_heads, dim_feedforward=4 * d_model, dropout=dropout),
            num_layers
        )
        
        # Cross-Transformer (for multi-modal interaction)
        self.cross_eeg_to_me = CrossTransformer(d_model, num_heads, num_layers, dropout)
        self.cross_me_to_eeg = CrossTransformer(d_model, num_heads, num_layers, dropout)
        
        # Fusion and Classification
        self.fusion_layer = nn.Linear(2 * d_model, fusion_hidden_dim)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_classes)
        )
    
    def forward(self, eeg_tokens, me_tokens):
        """
        Args:
            eeg_tokens (torch.Tensor): EEG input tokens of shape (batch_size, seq_len_eeg, eeg_input_dim).
            me_tokens (torch.Tensor): ME input tokens of shape (batch_size, seq_len_me, me_input_dim).
        
        Returns:
            torch.Tensor: Classification logits of shape (batch_size, num_classes).
        """
        # EEG Encoding
        eeg_tokens = self.eeg_embedding(eeg_tokens)  # Shape: (batch_size, seq_len_eeg, d_model)
        eeg_tokens = self.eeg_positional_encoding(eeg_tokens)
        eeg_encoded = self.eeg_transformer(eeg_tokens)  # Shape: (batch_size, seq_len_eeg, d_model)
        
        # ME Encoding
        me_tokens = self.me_embedding(me_tokens)  # Shape: (batch_size, seq_len_me, d_model)
        me_tokens = self.me_positional_encoding(me_tokens)
        me_encoded = self.me_transformer(me_tokens)  # Shape: (batch_size, seq_len_me, d_model)
        
        # Cross-modal Interaction
        eeg_to_me = self.cross_eeg_to_me(eeg_encoded, me_encoded)  # Shape: (batch_size, seq_len_me, d_model)
        me_to_eeg = self.cross_me_to_eeg(me_encoded, eeg_encoded)  # Shape: (batch_size, seq_len_eeg, d_model)
        
        # Fusion of Cross-modal Features
        eeg_fusion = eeg_to_me.mean(dim=1)  # Global average pooling over sequence dimension
        me_fusion = me_to_eeg.mean(dim=1)
        fusion_features = torch.cat([eeg_fusion, me_fusion], dim=-1)  # Shape: (batch_size, 2 * d_model)
        
        # Classification
        fusion_output = self.fusion_layer(fusion_features)
        logits = self.classifier(fusion_output)  # Shape: (batch_size, num_classes)
        
        return logits


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer Inputs.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant positional encodings
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
        
        Returns:
            torch.Tensor: Positionally encoded tensor of the same shape as input.
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CrossTransformer(nn.Module):
    """
    Cross-Transformer for multi-modal feature interaction.
    """
    def __init__(self, d_model, num_heads, num_layers, dropout):
        super(CrossTransformer, self).__init__()
        self.cross_transformer = TransformerEncoder(
            TransformerEncoderLayer(d_model, num_heads, dim_feedforward=4 * d_model, dropout=dropout),
            num_layers
        )

    def forward(self, query, key):
        """
        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, seq_len_query, d_model).
            key (torch.Tensor): Key tensor of shape (batch_size, seq_len_key, d_model).
        
        Returns:
            torch.Tensor: Cross-attended features of shape (batch_size, seq_len_query, d_model).
        """
        # Concatenate query and key along the sequence dimension
        combined = torch.cat([query, key], dim=1)
        attended = self.cross_transformer(combined)
        # Return the attended query part
        return attended[:, :query.size(1)]


if __name__ == "__main__":
    # Example usage
    batch_size = 8
    seq_len_eeg = 50
    seq_len_me = 20
    eeg_input_dim = 64
    me_input_dim = 128
    num_classes = 5

    model = TransformerFusion(eeg_input_dim, me_input_dim, num_classes=num_classes)
    eeg_tokens = torch.rand(batch_size, seq_len_eeg, eeg_input_dim)
    me_tokens = torch.rand(batch_size, seq_len_me, me_input_dim)
    logits = model(eeg_tokens, me_tokens)
    print("Output logits shape:", logits.shape)