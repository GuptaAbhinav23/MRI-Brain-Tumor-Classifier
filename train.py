"""
Training Script for Brain Tumor Segmentation
=============================================
Complete training pipeline for 2D Multi-Modal Attention U-Net brain tumor segmentation.

Features:
- Patient-wise data splitting (no data leakage)
- Multi-GPU support
- Mixed precision training (AMP)
- Learning rate scheduling (ReduceLROnPlateau)
- Early stopping with patience
- Gradient clipping for stability
- Model checkpointing (best Dice score)
- Comprehensive metrics tracking
- Confusion matrix computation
- Training visualization

Author: AI Research Engineer
Date: 2026
"""

import os
import sys
import argparse
import time
from datetime import datetime
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.unet import UNet, get_model, count_parameters
from model.dataset import (
    BraTSDataset, 
    get_patient_ids, 
    patient_wise_split, 
    create_data_loaders
)
from model.utils import (
    BCEDiceLoss,
    DiceLoss,
    dice_score,
    iou_score,
    sensitivity,
    specificity,
    pixel_accuracy,
    calculate_all_metrics,
    plot_training_history,
    save_model,
    load_model,
    get_device,
    EarlyStopping,
    AverageMeter
)


# ============================================================================
# CONFIGURATION
# ============================================================================

class TrainingConfig:
    """Training configuration with best practices."""
    
    # Model
    MODEL_NAME = "2D Multi-Modal Attention U-Net"
    IN_CHANNELS = 4
    OUT_CHANNELS = 1
    USE_ATTENTION = True
    
    # Training
    NUM_EPOCHS = 20
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # Optimization
    GRADIENT_CLIP_VALUE = 1.0
    EARLY_STOPPING_PATIENCE = 10
    LR_SCHEDULER_PATIENCE = 5
    LR_SCHEDULER_FACTOR = 0.5
    
    # Data
    IMG_SIZE = 128
    NUM_WORKERS = 0
    
    # Segmentation
    THRESHOLD = 0.5
    
    # Seeds
    RANDOM_SEED = 42


# ============================================================================
# CONFUSION MATRIX
# ============================================================================

def compute_confusion_matrix(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    threshold: float = 0.5
) -> Dict[str, int]:
    # Apply sigmoid if needed
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)
    
    pred_binary = (pred > threshold).float()
    
    # Flatten tensors
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    
    # Compute confusion matrix elements
    tp = ((pred_flat == 1) & (target_flat == 1)).sum().item()
    tn = ((pred_flat == 0) & (target_flat == 0)).sum().item()
    fp = ((pred_flat == 1) & (target_flat == 0)).sum().item()
    fn = ((pred_flat == 0) & (target_flat == 1)).sum().item()
    
    return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}


def print_confusion_matrix(cm: Dict[str, int], title: str = "Confusion Matrix"):
    """Print confusion matrix in a formatted way."""
    print(f"\n{title}")
    print("-" * 40)
    print(f"                Predicted")
    print(f"                Neg       Pos")
    print(f"Actual  Neg     {cm['TN']:>10,}  {cm['FP']:>10,}")
    print(f"        Pos     {cm['FN']:>10,}  {cm['TP']:>10,}")
    print("-" * 40)
    
    total = cm['TP'] + cm['TN'] + cm['FP'] + cm['FN']
    accuracy = (cm['TP'] + cm['TN']) / total if total > 0 else 0
    precision = cm['TP'] / (cm['TP'] + cm['FP']) if (cm['TP'] + cm['FP']) > 0 else 0
    recall = cm['TP'] / (cm['TP'] + cm['FN']) if (cm['TP'] + cm['FN']) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")


def train_one_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    scaler: GradScaler = None,
    use_amp: bool = True,
    gradient_clip: float = 1.0
) -> Tuple[float, float, float]:
    """
    Train model for one epoch with gradient clipping.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        scaler: Gradient scaler for mixed precision
        use_amp: Whether to use automatic mixed precision
        gradient_clip: Maximum gradient norm for clipping
    
    Returns:
        Tuple of (average loss, average dice score, average iou score)
    """
    model.train()
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with optional mixed precision
        if use_amp and scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            # Backward pass with gradient clipping
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            batch_dice = dice_score(outputs.detach(), masks)
            batch_iou = iou_score(outputs.detach(), masks)
        
        # Update meters
        loss_meter.update(loss.item(), images.size(0))
        dice_meter.update(batch_dice, images.size(0))
        iou_meter.update(batch_iou, images.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'dice': f'{dice_meter.avg:.4f}',
            'iou': f'{iou_meter.avg:.4f}'
        })
    
    return loss_meter.avg, dice_meter.avg, iou_meter.avg


def validate(
    model: nn.Module,
    val_loader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5
) -> Tuple[float, float, Dict, Dict[str, int]]:
    """
    Validate model on validation set with comprehensive metrics.
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        threshold: Binarization threshold
    
    Returns:
        Tuple of (average loss, average dice score, metrics dict, confusion matrix)
    """
    model.eval()
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()
    
    all_metrics = {
        'dice': [],
        'iou': [],
        'accuracy': [],
        'sensitivity': [],
        'specificity': []
    }
    
    # Confusion matrix accumulator
    total_cm = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    
    pbar = tqdm(val_loader, desc="Validation", leave=False)
    
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate all metrics
            metrics = calculate_all_metrics(outputs, masks, threshold)
            
            # Calculate confusion matrix for this batch
            batch_cm = compute_confusion_matrix(outputs, masks, threshold)
            for key in total_cm:
                total_cm[key] += batch_cm[key]
            
            # Update meters
            loss_meter.update(loss.item(), images.size(0))
            dice_meter.update(metrics['dice'], images.size(0))
            iou_meter.update(metrics['iou'], images.size(0))
            
            # Store metrics
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
            
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'dice': f'{dice_meter.avg:.4f}',
                'iou': f'{iou_meter.avg:.4f}'
            })
    
    # Average all metrics
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    std_metrics = {f"{key}_std": np.std(values) for key, values in all_metrics.items()}
    avg_metrics.update(std_metrics)
    
    return loss_meter.avg, dice_meter.avg, avg_metrics, total_cm


def train(
    data_dir: str,
    output_dir: str,
    num_epochs: int = 20,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    img_size: int = 128,
    in_channels: int = 4,
    use_attention: bool = True,
    use_amp: bool = True,
    num_workers: int = 4,
    patience: int = 10,
    random_seed: int = 42
):
    """
    Main training function.
    
    Args:
        data_dir: Path to BRATS dataset
        output_dir: Directory to save outputs
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        img_size: Input image size
        in_channels: Number of input channels
        use_attention: Whether to use attention U-Net
        use_amp: Whether to use automatic mixed precision
        num_workers: Number of data loading workers
        patience: Early stopping patience
        random_seed: Random seed for reproducibility
    """
    # Set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get device
    device = get_device()
    
    # Create data loaders
    print("\n" + "="*60)
    print("Loading Dataset...")
    print("="*60)
    
    modalities = ['flair', 't1', 't1ce', 't2'] if in_channels == 4 else ['flair']
    
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers,
        modalities=modalities,
        random_seed=random_seed
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print("\n" + "="*60)
    print("Creating Model...")
    print("="*60)
    
    model = get_model(
    in_channels=in_channels,
    out_channels=1,
    use_attention=use_attention
    )
    model = model.to(device)
        
    params = count_parameters(model)
    print(f"Model parameters: {params:,}")
    
    # Loss function and optimizer
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=5,
    )
    
    # Gradient scaler for mixed precision
    scaler = GradScaler() if use_amp and device.type == 'cuda' else None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, mode='max')
    
    # Training history
    history = {
        'train_loss': [],
        'train_dice': [],
        'train_iou': [],
        'val_loss': [],
        'val_dice': [],
        'val_iou': [],
        'learning_rate': []
    }
    
    # Best model tracking
    best_dice = 0.0
    best_iou = 0.0
    best_epoch = 0
    
    # Configuration
    gradient_clip = 1.0
    
    # Training loop
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60)
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Using attention: {use_attention}")
    print(f"Using AMP: {use_amp and device.type == 'cuda'}")
    print("="*60 + "\n")
    
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 40)
        
        # Train
        train_loss, train_dice, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, 
            device, scaler, use_amp and device.type == 'cuda',
            gradient_clip=gradient_clip
        )
        
        # Validate
        val_loss, val_dice, val_metrics, val_cm = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step(val_dice)
        
        # Record history
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_metrics['iou'])
        history['learning_rate'].append(current_lr)
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary (as requested: Train Loss, Val Loss, Val Dice, Val IoU)
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{num_epochs} Summary | Time: {epoch_time:.1f}s")
        print(f"{'='*50}")
        print(f"  Train Loss:   {train_loss:.4f}")
        print(f"  Train Dice:   {train_dice:.4f}")
        print(f"  Train IoU:    {train_iou:.4f}")
        print(f"  Val Loss:     {val_loss:.4f}")
        print(f"  Val Dice:     {val_dice:.4f}")
        print(f"  Val IoU:      {val_metrics['iou']:.4f}")
        print(f"  Sensitivity:  {val_metrics['sensitivity']:.4f}")
        print(f"  Specificity:  {val_metrics['specificity']:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            best_iou = val_metrics['iou']
            best_epoch = epoch
            
            save_model(
                model, optimizer, epoch, val_dice,
                os.path.join(output_dir, 'model.pth'),
                additional_info={
                    'metrics': val_metrics,
                    'in_channels': in_channels,
                    'use_attention': use_attention,
                    'img_size': img_size,
                    'best_iou': best_iou
                }
            )
            print(f"\n*** New best model saved (Dice: {val_dice:.4f}, IoU: {best_iou:.4f}) ***")
        
        # Early stopping
        if early_stopping(val_dice):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
    
    total_time = time.time() - start_time
    
    # Model Summary
    model_name = "2D Multi-Modal Attention U-Net" if use_attention else "2D Multi-Modal U-Net"
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel Name: {model_name}")
    print(f"Total Parameters: {params:,}")
    print(f"Total Training Time: {total_time/60:.1f} minutes")
    print(f"Best Validation Dice: {best_dice:.4f}")
    print(f"Best Validation IoU: {best_iou:.4f}")
    print(f"Best Epoch: {best_epoch}")
    
    # Save training history plot
    plot_path = os.path.join(output_dir, 'training_history.png')
    plot_training_history(
        history['train_loss'],
        history['val_loss'],
        history['val_dice'],
        save_path=plot_path
    )
    
    # Final evaluation on test set
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    # Load best model
    model, _ = load_model(model, os.path.join(output_dir, 'model.pth'), device)
    
    test_loss, test_dice, test_metrics, test_cm = validate(
        model, test_loader, criterion, device
    )
    
    print(f"\nTest Set Results:")
    print(f"  Loss:        {test_loss:.4f}")
    print(f"  Dice Score:  {test_dice:.4f} ± {test_metrics.get('dice_std', 0):.4f}")
    print(f"  IoU:         {test_metrics['iou']:.4f} ± {test_metrics.get('iou_std', 0):.4f}")
    print(f"  Accuracy:    {test_metrics['accuracy']:.4f} ± {test_metrics.get('accuracy_std', 0):.4f}")
    print(f"  Sensitivity: {test_metrics['sensitivity']:.4f} ± {test_metrics.get('sensitivity_std', 0):.4f}")
    print(f"  Specificity: {test_metrics['specificity']:.4f} ± {test_metrics.get('specificity_std', 0):.4f}")
    
    # Print confusion matrix
    print("\nTest Set Confusion Matrix:")
    print_confusion_matrix(test_cm)
    
    # Final Summary Message
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"\n  Model Name:       {model_name}")
    print(f"  Total Parameters: {params:,}")
    print(f"  Best Dice Score:  {best_dice:.4f}")
    print(f"  Best IoU Score:   {best_iou:.4f}")
    print(f"  Test Dice Score:  {test_dice:.4f}")
    print(f"  Test IoU Score:   {test_metrics['iou']:.4f}")
    print(f"  Model saved to:   {os.path.join(output_dir, 'model.pth')}")
    print(f"  History plot:     {plot_path}")
    print("\n" + "="*60)
    
    # Save test results
    results = {
        'model_name': model_name,
        'total_parameters': params,
        'best_epoch': best_epoch,
        'best_val_dice': best_dice,
        'best_val_iou': best_iou,
        'test_loss': test_loss,
        'test_dice': test_dice,
        'test_metrics': test_metrics,
        'test_confusion_matrix': test_cm,
        'history': history
    }
    
    torch.save(results, os.path.join(output_dir, 'results.pth'))
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train Brain Tumor Segmentation Model')
    
    parser.add_argument('--data_dir', type=str, 
                        default=r"S:\Brain Tumar Detection\dataset\MICCAI_BraTS2020_TrainingData",
                        help='Path to BRATS dataset')
    parser.add_argument('--output_dir', type=str, 
                        default='outputs',
                        help='Directory to save outputs')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--img_size', type=int, default=128,
                        help='Image size')
    parser.add_argument('--in_channels', type=int, default=4,
                        help='Number of input channels (1 or 4)')
    parser.add_argument('--no_attention', action='store_true',
                        help='Disable attention mechanism')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable automatic mixed precision')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    
    # Run training
    results = train(
        data_dir=args.data_dir,
        output_dir=output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        img_size=args.img_size,
        in_channels=args.in_channels,
        use_attention=not args.no_attention,
        use_amp=not args.no_amp,
        num_workers=args.workers,
        patience=args.patience,
        random_seed=args.seed
    )
    
    return results


if __name__ == "__main__":
    main()
