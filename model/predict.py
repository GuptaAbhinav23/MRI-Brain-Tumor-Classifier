"""
Command-Line Prediction Script for Brain Tumor Segmentation
============================================================
Perform inference on single images or directories from command line.

Author: AI Research Engineer
Date: 2026
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.unet import get_model
from model.utils import create_overlay_image, load_model, get_device


def preprocess_image(
    image_path: str, 
    img_size: int = 128, 
    in_channels: int = 4
) -> torch.Tensor:
    """
    Preprocess a single image for inference.
    """
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Store original for overlay
    original = image.copy()
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize
    image = cv2.resize(image, (img_size, img_size))
    
    # Normalize
    image = image.astype(np.float32)
    if image.max() > 1:
        image = image / 255.0
    
    mask = image > 0.01
    if mask.sum() > 0:
        mean = image[mask].mean()
        std = image[mask].std()
        if std > 0:
            image = np.where(mask, (image - mean) / std, 0)
    
    # Stack channels
    image = np.stack([image] * in_channels, axis=0)
    
    # Convert to tensor
    tensor = torch.from_numpy(image).unsqueeze(0)
    
    return tensor, original


def predict_single(
    model,
    image_path: str,
    device: torch.device,
    img_size: int = 128,
    in_channels: int = 4,
    threshold: float = 0.5
) -> dict:
    """
    Predict segmentation for a single image.
    
    Returns:
        Dictionary with prediction results
    """
    # Preprocess
    input_tensor, original = preprocess_image(image_path, img_size, in_channels)
    input_tensor = input_tensor.to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    
    # Postprocess
    prob_map = torch.sigmoid(output).squeeze().cpu().numpy()
    binary_mask = (prob_map > threshold).astype(np.uint8)
    
    # Resize mask to original size
    original_size = (original.shape[1], original.shape[0])
    mask_resized = cv2.resize(binary_mask, original_size, interpolation=cv2.INTER_NEAREST)
    
    # Calculate metrics
    confidence = float(prob_map.max())
    tumor_ratio = float(mask_resized.sum()) / mask_resized.size
    tumor_detected = tumor_ratio > 0.001
    
    # Create overlay
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original
    
    overlay = create_overlay_image(original_gray, mask_resized, alpha=0.4, color=(255, 50, 50))
    
    return {
        'mask': mask_resized,
        'prob_map': cv2.resize(prob_map, original_size),
        'overlay': overlay,
        'confidence': confidence,
        'tumor_ratio': tumor_ratio,
        'tumor_detected': tumor_detected,
        'original': original
    }


def predict_directory(
    model,
    input_dir: str,
    output_dir: str,
    device: torch.device,
    img_size: int = 128,
    in_channels: int = 4,
    threshold: float = 0.5
):
    """
    Process all images in a directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image files
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
    image_files = [
        f for f in Path(input_dir).iterdir() 
        if f.suffix.lower() in extensions
    ]
    
    print(f"Found {len(image_files)} images")
    
    results = []
    
    for img_path in image_files:
        try:
            print(f"Processing: {img_path.name}")
            
            result = predict_single(
                model, str(img_path), device, 
                img_size, in_channels, threshold
            )
            
            # Save output
            base_name = img_path.stem
            
            # Save overlay
            overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
            cv2.imwrite(overlay_path, cv2.cvtColor(result['overlay'], cv2.COLOR_RGB2BGR))
            
            # Save mask
            mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
            cv2.imwrite(mask_path, result['mask'] * 255)
            
            results.append({
                'file': img_path.name,
                'tumor_detected': result['tumor_detected'],
                'confidence': result['confidence'],
                'tumor_ratio': result['tumor_ratio']
            })
            
            status = "TUMOR DETECTED" if result['tumor_detected'] else "No tumor"
            print(f"  -> {status} (confidence: {result['confidence']:.2%})")
            
        except Exception as e:
            print(f"  -> Error: {e}")
    
    # Print summary
    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    
    tumors_found = sum(1 for r in results if r['tumor_detected'])
    print(f"Total images: {len(results)}")
    print(f"Tumors detected: {tumors_found}")
    print(f"No tumors: {len(results) - tumors_found}")
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Brain Tumor Segmentation - Command Line Prediction'
    )
    
    parser.add_argument('input', type=str,
                        help='Input image file or directory')
    parser.add_argument('--output', '-o', type=str, default='predictions',
                        help='Output directory')
    parser.add_argument('--model', '-m', type=str, default='model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--img_size', type=int, default=128,
                        help='Model input size')
    parser.add_argument('--in_channels', type=int, default=4,
                        help='Number of input channels')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                        help='Segmentation threshold')
    parser.add_argument('--no_attention', action='store_true',
                        help='Use model without attention')
    
    args = parser.parse_args()
    
    # Get device
    device = get_device()
    
    # Load model
    print("\nLoading model...")
    model = get_model(
        in_channels=args.in_channels,
        out_channels=1,
        use_attention=not args.no_attention
    )
    
    if os.path.exists(args.model):
        model, _ = load_model(model, args.model, device)
    else:
        print(f"Warning: Model file not found at {args.model}")
        print("Using randomly initialized model (for demo purposes)")
        model = model.to(device)
    
    # Check if input is file or directory
    if os.path.isfile(args.input):
        # Single file prediction
        print(f"\nProcessing: {args.input}")
        
        result = predict_single(
            model, args.input, device,
            args.img_size, args.in_channels, args.threshold
        )
        
        # Save output
        os.makedirs(args.output, exist_ok=True)
        base_name = Path(args.input).stem
        
        overlay_path = os.path.join(args.output, f"{base_name}_overlay.png")
        cv2.imwrite(overlay_path, cv2.cvtColor(result['overlay'], cv2.COLOR_RGB2BGR))
        
        mask_path = os.path.join(args.output, f"{base_name}_mask.png")
        cv2.imwrite(mask_path, result['mask'] * 255)
        
        print("\nResults:")
        print(f"  Tumor detected: {result['tumor_detected']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Tumor coverage: {result['tumor_ratio']:.4%}")
        print(f"\nOutput saved to: {args.output}")
        
    elif os.path.isdir(args.input):
        # Directory prediction
        predict_directory(
            model, args.input, args.output, device,
            args.img_size, args.in_channels, args.threshold
        )
        
    else:
        print(f"Error: Input not found: {args.input}")
        sys.exit(1)


if __name__ == "__main__":
    main()
