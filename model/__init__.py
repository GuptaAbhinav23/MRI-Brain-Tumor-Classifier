"""
Brain Tumor Segmentation Model Package
======================================
Contains the U-Net architecture, dataset loader, training utilities, and helper functions.
"""

from .unet import UNet, get_model, count_parameters
from .dataset import BraTSDataset, create_data_loaders, get_patient_ids, patient_wise_split
from .utils import (
    DiceLoss,
    BCEDiceLoss,
    FocalLoss,
    dice_score,
    iou_score,
    calculate_all_metrics,
    plot_training_history,
    visualize_prediction,
    create_overlay_image,
    save_model,
    load_model,
    get_device,
    EarlyStopping,
    AverageMeter
)

__version__ = "1.0.0"
__author__ = "AI Research Engineer"
