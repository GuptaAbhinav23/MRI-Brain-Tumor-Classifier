"""
Utility Functions for Brain Tumor Segmentation
===============================================
Contains loss functions, metrics, visualization tools, and helper functions.

Author: AI Research Engineer
Date: 2026
"""

import os
from typing import Tuple, List, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation.
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Loss = 1 - Dice
    
    Args:
        smooth: Smoothing factor to avoid division by zero
    """
    
    def __init__(self, smooth: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice Loss.
        
        Args:
            pred: Predicted logits (B, C, H, W)
            target: Ground truth mask (B, C, H, W)
        
        Returns:
            Dice loss value
        """
        pred = torch.sigmoid(pred)
        
        # Flatten spatial dimensions
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        
        # Calculate Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return loss (1 - Dice)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Dice Loss.
    
    This combination often works better than either loss alone:
    - BCE provides pixel-wise supervision
    - Dice focuses on overlap between prediction and target
    
    Args:
        bce_weight: Weight for BCE loss component
        dice_weight: Weight for Dice loss component
        smooth: Smoothing factor for Dice calculation
    """
    
    def __init__(
        self, 
        bce_weight: float = 0.5, 
        dice_weight: float = 0.5,
        smooth: float = 1e-6
    ):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined BCE + Dice loss.
        
        Args:
            pred: Predicted logits
            target: Ground truth mask
        
        Returns:
            Combined loss value
        """
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Useful when tumor pixels are much fewer than background pixels.
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection"
    
    Args:
        alpha: Weighting factor for positive class
        gamma: Focusing parameter (higher = more focus on hard examples)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate Focal Loss."""
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()


# ============================================================================
# METRICS
# ============================================================================

def dice_score(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> float:
    """
    Calculate Dice Score (Dice Coefficient / F1 Score).
    
    Args:
        pred: Predicted logits or probabilities
        target: Ground truth mask
        threshold: Threshold for binarizing predictions
        smooth: Smoothing factor
    
    Returns:
        Dice score between 0 and 1
    """
    # Apply sigmoid if needed and threshold
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)
    
    pred_binary = (pred > threshold).float()
    
    # Flatten
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    
    # Calculate Dice
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    return dice.item()


def iou_score(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> float:
    """
    Calculate Intersection over Union (IoU / Jaccard Index).
    
    Args:
        pred: Predicted logits or probabilities
        target: Ground truth mask
        threshold: Threshold for binarizing predictions
        smooth: Smoothing factor
    
    Returns:
        IoU score between 0 and 1
    """
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)
    
    pred_binary = (pred > threshold).float()
    
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()


def pixel_accuracy(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    threshold: float = 0.5
) -> float:
    """
    Calculate pixel-wise accuracy.
    
    Args:
        pred: Predicted logits or probabilities
        target: Ground truth mask
        threshold: Threshold for binarizing predictions
    
    Returns:
        Accuracy between 0 and 1
    """
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)
    
    pred_binary = (pred > threshold).float()
    
    correct = (pred_binary == target).float().sum()
    total = target.numel()
    
    return (correct / total).item()


def sensitivity(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> float:
    """
    Calculate Sensitivity (Recall / True Positive Rate).
    
    Sensitivity = TP / (TP + FN)
    """
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)
    
    pred_binary = (pred > threshold).float()
    
    tp = (pred_binary * target).sum()
    fn = ((1 - pred_binary) * target).sum()
    
    return ((tp + smooth) / (tp + fn + smooth)).item()


def specificity(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    threshold: float = 0.5,
    smooth: float = 1e-6
) -> float:
    """
    Calculate Specificity (True Negative Rate).
    
    Specificity = TN / (TN + FP)
    """
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)
    
    pred_binary = (pred > threshold).float()
    
    tn = ((1 - pred_binary) * (1 - target)).sum()
    fp = (pred_binary * (1 - target)).sum()
    
    return ((tn + smooth) / (tn + fp + smooth)).item()


def calculate_all_metrics(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.
    
    Args:
        pred: Predicted logits or probabilities
        target: Ground truth mask
        threshold: Threshold for binarizing predictions
    
    Returns:
        Dictionary of metric names and values
    """
    return {
        'dice': dice_score(pred, target, threshold),
        'iou': iou_score(pred, target, threshold),
        'accuracy': pixel_accuracy(pred, target, threshold),
        'sensitivity': sensitivity(pred, target, threshold),
        'specificity': specificity(pred, target, threshold)
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    val_dice_scores: List[float],
    save_path: Optional[str] = None
):
    """
    Plot training history with loss and Dice score curves.
    
    Args:
        train_losses: Training loss per epoch
        val_losses: Validation loss per epoch
        val_dice_scores: Validation Dice score per epoch
        save_path: Path to save figure (if None, displays plot)
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Dice score plot
    axes[1].plot(epochs, val_dice_scores, 'g-', label='Validation Dice', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Dice Score', fontsize=12)
    axes[1].set_title('Validation Dice Score', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_prediction(
    image: np.ndarray,
    mask_true: np.ndarray,
    mask_pred: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Visualize original image, ground truth, and prediction.
    
    Args:
        image: Input image (H, W) or (C, H, W)
        mask_true: Ground truth mask (H, W)
        mask_pred: Predicted mask (H, W)
        save_path: Path to save figure
    """
    # Handle multi-channel image
    if len(image.shape) == 3:
        image = image[0]  # Use first channel (FLAIR)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original MRI', fontsize=12)
    axes[0].axis('off')
    
    # Ground truth
    axes[1].imshow(mask_true, cmap='Reds')
    axes[1].set_title('Ground Truth', fontsize=12)
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(mask_pred, cmap='Reds')
    axes[2].set_title('Prediction', fontsize=12)
    axes[2].axis('off')
    
    # Overlay
    overlay = image.copy()
    overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min() + 1e-8)
    overlay_rgb = np.stack([overlay] * 3, axis=-1)
    
    # Add red color for prediction
    mask_overlay = mask_pred > 0.5
    overlay_rgb[mask_overlay, 0] = 1.0  # Red channel
    overlay_rgb[mask_overlay, 1] *= 0.3
    overlay_rgb[mask_overlay, 2] *= 0.3
    
    axes[3].imshow(overlay_rgb)
    axes[3].set_title('Overlay', fontsize=12)
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def create_overlay_image(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    color: Tuple[int, int, int] = (255, 0, 0)
) -> np.ndarray:
    """
    Create overlay of mask on image.
    
    Args:
        image: Input image (H, W) grayscale
        mask: Binary mask (H, W)
        alpha: Transparency of overlay
        color: RGB color for mask overlay
    
    Returns:
        RGB overlay image
    """
    # Normalize image to 0-255
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    # Convert to RGB
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.copy()
    
    # Create colored mask
    mask_colored = np.zeros_like(image_rgb)
    mask_binary = mask > 0.5
    mask_colored[mask_binary] = color
    
    # Blend
    overlay = cv2.addWeighted(image_rgb, 1 - alpha, mask_colored, alpha, 0)
    
    # Add contour
    mask_uint8 = (mask_binary * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)
    
    return overlay


# ============================================================================
# MODEL UTILITIES
# ============================================================================

def save_model(
    model: nn.Module,
    optimizer,
    epoch: int,
    val_dice: float,
    save_path: str,
    additional_info: Dict = None
):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        val_dice: Validation Dice score
        save_path: Path to save checkpoint
        additional_info: Additional info to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_dice': val_dice,
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")


def load_model(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    load_optimizer: bool = False,
    optimizer = None
) -> Tuple[nn.Module, Optional[Dict]]:
    """
    Load model from checkpoint.
    
    Args:
        model: PyTorch model architecture
        checkpoint_path: Path to checkpoint file
        device: Device to load model to
        load_optimizer: Whether to load optimizer state
        optimizer: Optimizer (required if load_optimizer=True)
    
    Returns:
        Tuple of (model, checkpoint_info)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    if load_optimizer and optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    info = {
        'epoch': checkpoint.get('epoch'),
        'val_dice': checkpoint.get('val_dice')
    }
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"  Epoch: {info['epoch']}, Val Dice: {info['val_dice']:.4f}")
    
    return model, info


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Returns:
        Dictionary with total, trainable, and non-trainable parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable
    }


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for loss, 'max' for metrics like Dice
    """
    
    def __init__(self, patience: int = 7, min_delta: float = 0, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
        
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test loss functions
    pred = torch.randn(2, 1, 128, 128)
    target = torch.randint(0, 2, (2, 1, 128, 128)).float()
    
    dice_loss = DiceLoss()
    bce_dice_loss = BCEDiceLoss()
    focal_loss = FocalLoss()
    
    print(f"Dice Loss: {dice_loss(pred, target):.4f}")
    print(f"BCE+Dice Loss: {bce_dice_loss(pred, target):.4f}")
    print(f"Focal Loss: {focal_loss(pred, target):.4f}")
    
    # Test metrics
    metrics = calculate_all_metrics(pred, target)
    print(f"Metrics: {metrics}")
    
    print("Utility tests passed!")
