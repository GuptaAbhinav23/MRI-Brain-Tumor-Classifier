"""
Dataset Loader for BRATS Brain Tumor Segmentation
==================================================
Custom PyTorch Dataset class for loading and preprocessing BRATS MRI data.

Features:
- Patient-wise data splitting (no data leakage)
- Multi-modal MRI input (FLAIR, T1, T1ce, T2)
- 3D to 2D slice conversion
- Intensity normalization
- Image resizing to 128x128

Author: AI Research Engineer
Date: 2026
"""

import os
import glob
import random
from typing import Tuple, List, Dict, Optional

import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import cv2


class BraTSDataset(Dataset):
    """
    BRATS Dataset for Brain Tumor Segmentation.
    
    Loads multi-modal MRI volumes (FLAIR, T1, T1ce, T2) and corresponding
    segmentation masks. Converts 3D volumes to 2D slices for training.
    
    Args:
        data_dir: Path to BRATS dataset directory
        patient_ids: List of patient folder names to include
        img_size: Target image size (default: 128x128)
        use_augmentation: Whether to apply data augmentation
        modalities: List of modalities to use ('flair', 't1', 't1ce', 't2')
        slice_range: Range of slices to extract (start, end) or None for all
        min_tumor_pixels: Minimum tumor pixels required for slice inclusion
    """
    
    def __init__(
        self,
        data_dir: str,
        patient_ids: List[str],
        img_size: int = 128,
        use_augmentation: bool = False,
        modalities: List[str] = None,
        slice_range: Tuple[int, int] = (70, 110),
        min_tumor_pixels: int = 50
    ):
        self.data_dir = data_dir
        self.patient_ids = patient_ids
        self.img_size = img_size
        self.use_augmentation = use_augmentation
        self.modalities = modalities or ['flair', 't1', 't1ce', 't2']
        
        self.slice_range = slice_range
        self.min_tumor_pixels = min_tumor_pixels
        
        # Build slice index
        self.slices = self._build_slice_index()
        
        print(f"Dataset initialized with {len(self.patient_ids)} patients, "
              f"{len(self.slices)} slices")
    
    def _build_slice_index(self) -> List[Tuple[str, int]]:
        """
        Build index of (patient_id, slice_idx) tuples.
        Only includes slices with sufficient tumor content.
        """
        slices = []
        
        for patient_id in self.patient_ids:
            patient_dir = os.path.join(self.data_dir, patient_id)
            
            # Load segmentation mask to determine valid slices
            seg_path = os.path.join(patient_dir, f"{patient_id}_seg.nii")
            
            if not os.path.exists(seg_path):
                continue
            
            try:
                seg_data = nib.load(seg_path).get_fdata()
                num_slices = seg_data.shape[2]
                
                # Determine slice range
                start = self.slice_range[0] if self.slice_range else 0
                end = min(self.slice_range[1], num_slices) if self.slice_range else num_slices
                
                for slice_idx in range(start, end):
                    # Count tumor pixels in this slice
                    tumor_pixels = np.sum(seg_data[:, :, slice_idx] > 0)
                    
                    if tumor_pixels >= self.min_tumor_pixels:
                        slices.append((patient_id, slice_idx))
                    elif random.random() < 0.1:  # Include some negative samples
                        slices.append((patient_id, slice_idx))
                        
            except Exception as e:
                print(f"Error loading {patient_id}: {e}")
                continue
        
        return slices
    
    def _normalize_intensity(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize MRI intensity values using z-score normalization.
        Only normalizes non-zero voxels (brain region).
        """
        mask = img > 0
        if mask.sum() == 0:
            return img
        
        mean = img[mask].mean()
        std = img[mask].std()
        
        if std > 0:
            img = np.where(mask, (img - mean) / std, 0)
        
        return img
    
    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        """Resize image to target size."""
        return cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
    
    def _resize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Resize mask to target size using nearest neighbor interpolation."""
        return cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
    
    def _apply_augmentation(
        self, 
        images: np.ndarray, 
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply data augmentation.
        
        Augmentations:
        - Random horizontal flip
        - Random vertical flip
        - Random rotation (0, 90, 180, 270 degrees)
        """
        # Random horizontal flip
        if random.random() > 0.5:
            images = np.flip(images, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()
        
        # Random vertical flip
        if random.random() > 0.5:
            images = np.flip(images, axis=1).copy()
            mask = np.flip(mask, axis=0).copy()
        
        # Random rotation
        k = random.randint(0, 3)
        if k > 0:
            images = np.rot90(images, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k).copy()
        
        return images, mask
    
    def __len__(self) -> int:
        return len(self.slices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            image: Tensor of shape (C, H, W) where C is number of modalities
            mask: Tensor of shape (1, H, W) with binary tumor mask
        """
        patient_id, slice_idx = self.slices[idx]
        patient_dir = os.path.join(self.data_dir, patient_id)
        
        # Load all modalities
        images = []
        for modality in self.modalities:
            img_path = os.path.join(patient_dir, f"{patient_id}_{modality}.nii")
            img_data = nib.load(img_path).get_fdata()
            slice_data = img_data[:, :, slice_idx]
            
            # Normalize and resize
            slice_data = self._normalize_intensity(slice_data)
            slice_data = self._resize_image(slice_data)
            images.append(slice_data)
        
        # Stack modalities
        images = np.stack(images, axis=0).astype(np.float32)
        
        # Load segmentation mask
        seg_path = os.path.join(patient_dir, f"{patient_id}_seg.nii")
        seg_data = nib.load(seg_path).get_fdata()
        mask = seg_data[:, :, slice_idx]
        
        # Binarize mask (any tumor class -> 1)
        # Original labels: 0=background, 1=necrotic, 2=edema, 4=enhancing
        mask = (mask > 0).astype(np.float32)
        mask = self._resize_mask(mask)
        
        # Apply augmentation
        if self.use_augmentation:
            images, mask = self._apply_augmentation(images, mask)
        
        # Convert to tensors
        images = torch.from_numpy(images)
        mask = torch.from_numpy(mask).unsqueeze(0)  # Add channel dimension
        
        return images, mask
    
    def get_sample_info(self, idx: int) -> Dict:
        """Get metadata about a sample."""
        patient_id, slice_idx = self.slices[idx]
        return {
            'patient_id': patient_id,
            'slice_idx': slice_idx,
            'modalities': self.modalities
        }


def get_patient_ids(data_dir: str) -> List[str]:
    
    patient_dirs = glob.glob(os.path.join(data_dir, "BraTS20_Training_*"))
    patient_ids = [os.path.basename(d) for d in patient_dirs if os.path.isdir(d)]
    patient_ids.sort()
    return patient_ids


def patient_wise_split(
    patient_ids: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split patients into train/val/test sets.
    
    IMPORTANT: This ensures no data leakage - all slices from a patient
    belong to the same split.
    
    Args:
        patient_ids: List of patient identifiers
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_ids, val_ids, test_ids)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # First split: train vs (val + test)
    train_ids, temp_ids = train_test_split(
        patient_ids,
        train_size=train_ratio,
        random_state=random_seed
    )
    
    # Second split: val vs test
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=relative_test_ratio,
        random_state=random_seed
    )
    
    print(f"Patient-wise split:")
    print(f"  Train: {len(train_ids)} patients")
    print(f"  Val: {len(val_ids)} patients")
    print(f"  Test: {len(test_ids)} patients")
    
    return train_ids, val_ids, test_ids


def create_data_loaders(
    data_dir: str,
    batch_size: int = 8,
    img_size: int = 128,
    num_workers: int = 4,
    modalities: List[str] = None,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    modalities = ['flair', 't1', 't1ce', 't2']
    
    # Get patient IDs and split
    patient_ids = get_patient_ids(data_dir)
    train_ids, val_ids, test_ids = patient_wise_split(
        patient_ids, 
        random_seed=random_seed
    )
    
    # Create datasets
    train_dataset = BraTSDataset(
        data_dir=data_dir,
        patient_ids=train_ids,
        img_size=img_size,
        use_augmentation=True,
        modalities=modalities
    )
    
    val_dataset = BraTSDataset(
        data_dir=data_dir,
        patient_ids=val_ids,
        img_size=img_size,
        use_augmentation=False,
        modalities=modalities
    )
    
    test_dataset = BraTSDataset(
        data_dir=data_dir,
        patient_ids=test_ids,
        img_size=img_size,
        use_augmentation=False,
        modalities=modalities
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


class BraTSInferenceDataset(Dataset):
    """
    Dataset for inference on single images (PNG/JPG format).
    Used by Flask API for web-uploaded images.
    """
    
    def __init__(self, img_size: int = 128):
        self.img_size = img_size
    
    def preprocess_image(self, image: np.ndarray, num_channels: int = 4) -> torch.Tensor:
        """
        Preprocess a single image for inference.
        
        Args:
            image: Input image (grayscale or RGB)
            num_channels: Number of input channels expected by model
        
        Returns:
            Preprocessed tensor of shape (1, C, H, W)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Resize
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # Normalize
        image = image.astype(np.float32)
        if image.max() > 0:
            mean = image[image > 0].mean()
            std = image[image > 0].std()
            if std > 0:
                image = np.where(image > 0, (image - mean) / std, 0)
        
        # Replicate for multi-channel input
        image = np.stack([image] * num_channels, axis=0)
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(image).unsqueeze(0)
        
        return tensor


if __name__ == "__main__":
    # Test dataset loading
    import sys
    
    # Update this path to your data directory
    data_dir = r"S:\Brain Tumar Detection\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)
    
    print("Testing dataset...")
    
    # Get patient IDs
    patient_ids = get_patient_ids(data_dir)
    print(f"Found {len(patient_ids)} patients")
    
    # Test split
    train_ids, val_ids, test_ids = patient_wise_split(patient_ids[:20])  # Use subset for testing
    
    # Create test dataset
    test_dataset = BraTSDataset(
        data_dir=data_dir,
        patient_ids=train_ids[:5],
        img_size=128,
        use_augmentation=True
    )
    
    # Test loading a sample
    if len(test_dataset) > 0:
        image, mask = test_dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Image dtype: {image.dtype}")
        print(f"Mask unique values: {torch.unique(mask)}")
        
        info = test_dataset.get_sample_info(0)
        print(f"Sample info: {info}")
    
    print("Dataset test passed!")
