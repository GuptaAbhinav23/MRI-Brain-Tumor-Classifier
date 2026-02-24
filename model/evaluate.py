"""
Model Evaluation Script for Brain Tumor Segmentation
=====================================================
Comprehensive evaluation of trained model on test set.

Author: AI Research Engineer
Date: 2026
"""

import os
import sys
import argparse
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.unet import get_model
from model.dataset import BraTSDataset, get_patient_ids, patient_wise_split
from model.utils import (
    BCEDiceLoss,
    calculate_all_metrics,
    visualize_prediction,
    load_model,
    get_device
)


def evaluate_model(
    model,
    test_loader: DataLoader,
    criterion,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to evaluate on
    
    Returns:
        Dictionary of averaged metrics
    """
    model.eval()
    
    all_metrics = {
        'loss': [],
        'dice': [],
        'iou': [],
        'accuracy': [],
        'sensitivity': [],
        'specificity': []
    }
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate metrics for each sample
            for i in range(images.size(0)):
                metrics = calculate_all_metrics(
                    outputs[i:i+1], 
                    masks[i:i+1]
                )
                
                all_metrics['loss'].append(loss.item())
                for key in metrics:
                    all_metrics[key].append(metrics[key])
    
    # Average all metrics
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    std_metrics = {f"{key}_std": np.std(values) for key, values in all_metrics.items()}
    
    return {**avg_metrics, **std_metrics}


def evaluate_per_patient(
    model,
    data_dir: str,
    test_patient_ids: List[str],
    img_size: int,
    modalities: List[str],
    device: torch.device
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model on a per-patient basis.
    
    Returns:
        Dictionary mapping patient_id to metrics
    """
    model.eval()
    criterion = BCEDiceLoss()
    
    patient_metrics = {}
    
    for patient_id in tqdm(test_patient_ids, desc="Per-patient evaluation"):
        # Create dataset for single patient
        dataset = BraTSDataset(
            data_dir=data_dir,
            patient_ids=[patient_id],
            img_size=img_size,
            use_augmentation=False,
            modalities=modalities
        )
        
        if len(dataset) == 0:
            continue
        
        loader = DataLoader(dataset, batch_size=8, shuffle=False)
        
        patient_dice = []
        patient_iou = []
        
        with torch.no_grad():
            for images, masks in loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                
                for i in range(images.size(0)):
                    metrics = calculate_all_metrics(outputs[i:i+1], masks[i:i+1])
                    patient_dice.append(metrics['dice'])
                    patient_iou.append(metrics['iou'])
        
        patient_metrics[patient_id] = {
            'dice': np.mean(patient_dice),
            'dice_std': np.std(patient_dice),
            'iou': np.mean(patient_iou),
            'iou_std': np.std(patient_iou),
            'num_slices': len(patient_dice)
        }
    
    return patient_metrics


def generate_visualizations(
    model,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: str,
    num_samples: int = 10
):
    """
    Generate visualization images for model predictions.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    sample_count = 0
    
    with torch.no_grad():
        for images, masks in test_loader:
            if sample_count >= num_samples:
                break
            
            images = images.to(device)
            outputs = model(images)
            
            probs = torch.sigmoid(outputs)
            pred_masks = (probs > 0.5).float()
            
            for i in range(images.size(0)):
                if sample_count >= num_samples:
                    break
                
                img = images[i].cpu().numpy()
                mask_true = masks[i, 0].cpu().numpy()
                mask_pred = pred_masks[i, 0].cpu().numpy()
                
                save_path = os.path.join(output_dir, f'sample_{sample_count:03d}.png')
                visualize_prediction(img, mask_true, mask_pred, save_path)
                
                sample_count += 1
    
    print(f"Generated {sample_count} visualization images in {output_dir}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate Brain Tumor Segmentation Model')
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to BRATS dataset')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='eval_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--img_size', type=int, default=128,
                        help='Image size')
    parser.add_argument('--in_channels', type=int, default=4,
                        help='Number of input channels')
    parser.add_argument('--num_vis', type=int, default=20,
                        help='Number of visualization samples')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Get device
    device = get_device()
    
    # Load model
    print("\n" + "="*60)
    print("Loading Model...")
    print("="*60)
    
    model = get_model(
        in_channels=args.in_channels,
        out_channels=1,
        use_attention=True
    )
    model, checkpoint_info = load_model(model, args.model_path, device)
    
    # Get test patients
    print("\n" + "="*60)
    print("Loading Test Data...")
    print("="*60)
    
    modalities = ['flair', 't1', 't1ce', 't2'] if args.in_channels == 4 else ['flair']
    
    patient_ids = get_patient_ids(args.data_dir)
    _, _, test_ids = patient_wise_split(patient_ids, random_seed=args.seed)
    
    test_dataset = BraTSDataset(
        data_dir=args.data_dir,
        patient_ids=test_ids,
        img_size=args.img_size,
        use_augmentation=False,
        modalities=modalities
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Test patients: {len(test_ids)}")
    print(f"Test slices: {len(test_dataset)}")
    
    # Evaluate
    print("\n" + "="*60)
    print("Evaluating Model...")
    print("="*60)
    
    criterion = BCEDiceLoss()
    metrics = evaluate_model(model, test_loader, criterion, device)
    
    print("\nTest Results:")
    print("-" * 40)
    print(f"  Loss:        {metrics['loss']:.4f} ± {metrics['loss_std']:.4f}")
    print(f"  Dice Score:  {metrics['dice']:.4f} ± {metrics['dice_std']:.4f}")
    print(f"  IoU:         {metrics['iou']:.4f} ± {metrics['iou_std']:.4f}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f} ± {metrics['accuracy_std']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f} ± {metrics['sensitivity_std']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f} ± {metrics['specificity_std']:.4f}")
    
    # Per-patient evaluation
    print("\n" + "="*60)
    print("Per-Patient Evaluation...")
    print("="*60)
    
    patient_metrics = evaluate_per_patient(
        model, args.data_dir, test_ids, args.img_size, modalities, device
    )
    
    # Calculate statistics
    patient_dice_scores = [m['dice'] for m in patient_metrics.values()]
    print(f"\nPer-Patient Dice Score:")
    print(f"  Mean: {np.mean(patient_dice_scores):.4f}")
    print(f"  Std:  {np.std(patient_dice_scores):.4f}")
    print(f"  Min:  {np.min(patient_dice_scores):.4f}")
    print(f"  Max:  {np.max(patient_dice_scores):.4f}")
    
    # Generate visualizations
    print("\n" + "="*60)
    print("Generating Visualizations...")
    print("="*60)
    
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    generate_visualizations(model, test_loader, device, vis_dir, args.num_vis)
    
    # Save results
    results = {
        'test_metrics': metrics,
        'patient_metrics': patient_metrics,
        'test_patient_ids': test_ids
    }
    
    torch.save(results, os.path.join(args.output_dir, 'evaluation_results.pth'))
    print(f"\nResults saved to {args.output_dir}")
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
