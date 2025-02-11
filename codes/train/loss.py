import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    """
    Standard Cross-Entropy Loss for multi-class classification tasks.
    """
    def __init__(self, weight=None, reduction='mean'):
        """
        Initialize the Cross-Entropy Loss.

        Args:
            weight (torch.Tensor, optional): Weight for each class. Default is None.
            reduction (str, optional): Reduction method for the loss, either 'mean' or 'sum'. Default is 'mean'.
        """
        super(CrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, labels):
        """
        Compute the Cross-Entropy Loss.

        Args:
            logits (torch.Tensor): Model output logits with shape (batch_size, num_classes).
            labels (torch.Tensor): Ground truth labels with shape (batch_size,).

        Returns:
            torch.Tensor: Computed Cross-Entropy Loss.
        """
        return F.cross_entropy(logits, labels, weight=self.weight, reduction=self.reduction)


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for learning feature representations, ensuring that features of similar samples are closer
    and features of dissimilar samples are farther apart.
    """
    def __init__(self, margin=1.0):
        """
        Initialize the Contrastive Loss.

        Args:
            margin (float, optional): Minimum distance between dissimilar pairs. Default is 1.0.
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, features1, features2, labels):
        """
        Compute the Contrastive Loss.

        Args:
            features1 (torch.Tensor): Features of the first sample with shape (batch_size, feature_dim).
            features2 (torch.Tensor): Features of the second sample with shape (batch_size, feature_dim).
            labels (torch.Tensor): Ground truth labels indicating whether the pairs are similar (1) or dissimilar (0),
                                   with shape (batch_size,).

        Returns:
            torch.Tensor: Computed Contrastive Loss.
        """
        # Compute the Euclidean distance between the features
        distance = F.pairwise_distance(features1, features2)

        # Compute the contrastive loss
        loss = (labels * distance.pow(2) +  # Loss for similar pairs
                (1 - labels) * F.relu(self.margin - distance).pow(2))  # Loss for dissimilar pairs

        return loss.mean()


class CenterLoss(nn.Module):
    """
    Center Loss for learning feature representations by penalizing the distance between features and their class centers.
    """
    def __init__(self, num_classes, feature_dim):
        """
        Initialize the Center Loss.

        Args:
            num_classes (int): Number of classes.
            feature_dim (int): Dimension of the feature space.
        """
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))

    def forward(self, features, labels):
        """
        Compute the Center Loss.

        Args:
            features (torch.Tensor): Extracted features with shape (batch_size, feature_dim).
            labels (torch.Tensor): Ground truth labels with shape (batch_size,).

        Returns:
            torch.Tensor: Computed Center Loss.
        """
        # Gather the centers for each sample in the batch
        batch_size = features.size(0)
        centers_batch = self.centers[labels]

        # Compute the squared Euclidean distance between features and their corresponding centers
        loss = torch.sum((features - centers_batch).pow(2), dim=1).mean()

        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance by focusing on hard-to-classify samples.
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        Initialize the Focal Loss.

        Args:
            gamma (float, optional): Focusing parameter. Default is 2.0.
            alpha (torch.Tensor, optional): Weighting factor for each class. Default is None.
            reduction (str, optional): Reduction method for the loss, either 'mean' or 'sum'. Default is 'mean'.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, labels):
        """
        Compute the Focal Loss.

        Args:
            logits (torch.Tensor): Model output logits with shape (batch_size, num_classes).
            labels (torch.Tensor): Ground truth labels with shape (batch_size,).

        Returns:
            torch.Tensor: Computed Focal Loss.
        """
        # Compute the Cross-Entropy Loss
        ce_loss = F.cross_entropy(logits, labels, weight=self.alpha, reduction='none')

        # Compute the Focal Loss
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt).pow(self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MultiModalLoss(nn.Module):
    """
    Multi-Modal Loss combining Cross-Entropy Loss and Contrastive Loss for joint training.
    """
    def __init__(self, num_classes, feature_dim, alpha=1.0, beta=1.0):
        """
        Initialize the Multi-Modal Loss.

        Args:
            num_classes (int): Number of classes.
            feature_dim (int): Dimension of the feature space.
            alpha (float, optional): Weight for Cross-Entropy Loss. Default is 1.0.
            beta (float, optional): Weight for Contrastive Loss. Default is 1.0.
        """
        super(MultiModalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = CrossEntropyLoss()
        self.contrastive_loss = ContrastiveLoss()
        self.center_loss = CenterLoss(num_classes, feature_dim)

    def forward(self, logits, features1, features2, labels):
        """
        Compute the Multi-Modal Loss.

        Args:
            logits (torch.Tensor): Model output logits with shape (batch_size, num_classes).
            features1 (torch.Tensor): Features of the first modality with shape (batch_size, feature_dim).
            features2 (torch.Tensor): Features of the second modality with shape (batch_size, feature_dim).
            labels (torch.Tensor): Ground truth labels with shape (batch_size,).

        Returns:
            torch.Tensor: Computed Multi-Modal Loss.
        """
        # Compute Cross-Entropy Loss
        ce_loss = self.ce_loss(logits, labels)

        # Compute Contrastive Loss
        contrastive_loss = self.contrastive_loss(features1, features2, labels)

        # Compute Center Loss
        center_loss = self.center_loss(features1, labels)

        # Combine the losses
        total_loss = self.alpha * ce_loss + self.beta * contrastive_loss + center_loss

        return total_loss


# Example usage
if __name__ == "__main__":
    # Example data
    batch_size = 16
    num_classes = 4
    feature_dim = 128

    logits = torch.randn(batch_size, num_classes)
    features1 = torch.randn(batch_size, feature_dim)
    features2 = torch.randn(batch_size, feature_dim)
    labels = torch.randint(0, num_classes, (batch_size,))

    # Initialize loss functions
    ce_loss = CrossEntropyLoss()
    contrastive_loss = ContrastiveLoss()
    center_loss = CenterLoss(num_classes, feature_dim)
    focal_loss = FocalLoss()
    multi_modal_loss = MultiModalLoss(num_classes, feature_dim)

    # Compute losses
    print("Cross-Entropy Loss:", ce_loss(logits, labels))
    print("Contrastive Loss:", contrastive_loss(features1, features2, labels))
    print("Center Loss:", center_loss(features1, labels))
    print("Focal Loss:", focal_loss(logits, labels))
    print("Multi-Modal Loss:", multi_modal_loss(logits, features1, features2, labels))