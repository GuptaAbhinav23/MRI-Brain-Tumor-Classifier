"""
Flask API for Brain Tumor Segmentation
======================================
REST API for brain tumor detection and segmentation.

Endpoints:
- GET /: Web interface
- POST /predict: Upload MRI image and get segmentation
- GET /health: Health check

Author: AI Research Engineer
Date: 2026
"""

import os
import io
import base64
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import cv2
from PIL import Image
import torch
from flask import Flask, request, jsonify, render_template, send_from_directory

# Import model components
from model.unet import UNet, get_model
from model.utils import create_overlay_image, get_device

import os
MODEL_PATH = os.environ.get("MODEL_PATH", "outputs/latest/model.pth")

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Application configuration."""
    MODEL_PATH = os.environ.get('MODEL_PATH', 'model.pth')
    IMG_SIZE = 128
    IN_CHANNELS = 4
    USE_ATTENTION = True
    THRESHOLD = 0.5
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}


# ============================================================================
# Flask App Setup
# ============================================================================

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

# Global model and device
model = None
device = None


def load_trained_model():
    """Load the trained model."""
    global model, device
    
    device = get_device()
    
    # Create model architecture
    model = get_model(
        in_channels=Config.IN_CHANNELS,
        out_channels=1,
        use_attention=Config.USE_ATTENTION
    )
    
    # Load weights if model file exists
    if os.path.exists(Config.MODEL_PATH):
        try:
            checkpoint = torch.load(Config.MODEL_PATH, map_location=device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Model loaded from {Config.MODEL_PATH}")
                print(f"  Best validation Dice: {checkpoint.get('val_dice', 'N/A')}")
            else:
                model.load_state_dict(checkpoint)
                print(f"Model weights loaded from {Config.MODEL_PATH}")
                
        except Exception as e:
            print(f"Warning: Could not load model weights: {e}")
            print("Using randomly initialized model (for demo purposes)")
    else:
        print(f"Warning: Model file not found at {Config.MODEL_PATH}")
        print("Using randomly initialized model (for demo purposes)")
    
    model = model.to(device)
    model.eval()
    
    return model


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """
    Preprocess uploaded image for model inference.
    
    Args:
        image: Input image (H, W, C) or (H, W)
    
    Returns:
        Preprocessed tensor (1, C, H, W)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        if image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        elif image.shape[2] == 3:  # RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Resize to model input size
    image = cv2.resize(image, (Config.IMG_SIZE, Config.IMG_SIZE))
    
    # Normalize using z-score
    image = image.astype(np.float32)
    if image.max() > 1:
        image = image / 255.0
    
    mask = image > 0.01
    if mask.sum() > 0:
        mean = image[mask].mean()
        std = image[mask].std()
        if std > 0:
            image = np.where(mask, (image - mean) / std, 0)
    
    # Replicate to match expected input channels
    image = np.stack([image] * Config.IN_CHANNELS, axis=0)
    
    # Convert to tensor and add batch dimension
    tensor = torch.from_numpy(image).unsqueeze(0)
    
    return tensor


def postprocess_prediction(
    prediction: torch.Tensor,
    original_size: Tuple[int, int]
) -> Tuple[np.ndarray, float, float]:
    """
    Postprocess model prediction.
    
    Args:
        prediction: Model output tensor
        original_size: Original image size (width, height)
    
    Returns:
        Tuple of (binary mask, confidence score, tumor ratio)
    """
    # Apply sigmoid and threshold
    prob_map = torch.sigmoid(prediction).squeeze().cpu().numpy()
    binary_mask = (prob_map > Config.THRESHOLD).astype(np.uint8)
    
    # Resize to original size
    binary_mask = cv2.resize(binary_mask, original_size, interpolation=cv2.INTER_NEAREST)
    
    # Calculate confidence score (mean probability of positive predictions)
    if prob_map.max() > Config.THRESHOLD:
        confidence = float(prob_map[prob_map > Config.THRESHOLD].mean())
    else:
        confidence = float(prob_map.max())
    
    # Calculate tumor ratio (percentage of image classified as tumor)
    tumor_ratio = float(binary_mask.sum()) / binary_mask.size
    
    return binary_mask, confidence, tumor_ratio


def create_result_image(
    original: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    Create visualization with original image and mask overlay.
    
    Args:
        original: Original input image
        mask: Predicted binary mask
    
    Returns:
        Combined result image
    """
    # Ensure original is grayscale
    if len(original.shape) == 3:
        if original.shape[2] == 4:
            original_gray = cv2.cvtColor(original, cv2.COLOR_RGBA2GRAY)
        else:
            original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    else:
        original_gray = original
    
    # Resize mask to match original
    mask_resized = cv2.resize(mask, (original_gray.shape[1], original_gray.shape[0]), 
                              interpolation=cv2.INTER_NEAREST)
    
    # Create overlay
    overlay = create_overlay_image(
        original_gray, 
        mask_resized, 
        alpha=0.4, 
        color=(255, 50, 50)  # Red for tumor
    )
    
    return overlay


def image_to_base64(image: np.ndarray) -> str:
    """Convert numpy image to base64 string."""
    # Ensure image is uint8
    if image.dtype != np.uint8:
        if image.max() <= 1:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Convert to PIL Image
    if len(image.shape) == 2:
        pil_image = Image.fromarray(image, mode='L')
    else:
        pil_image = Image.fromarray(image, mode='RGB')
    
    # Convert to base64
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)
    
    return base64.b64encode(buffer.read()).decode('utf-8')


# ============================================================================
# Routes
# ============================================================================

@app.route('/')
def index():
    """Serve the main web interface."""
    return render_template('index.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    return send_from_directory('static', filename)


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if device else 'not initialized',
        'timestamp': datetime.utcnow().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict tumor segmentation from uploaded MRI image.
    
    Expects:
        - POST request with 'file' in form-data
    
    Returns:
        JSON with:
        - success: boolean
        - tumor_detected: boolean
        - confidence: float (0-1)
        - tumor_ratio: float (0-1)
        - original_image: base64 encoded
        - mask_image: base64 encoded
        - overlay_image: base64 encoded
    """
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'Invalid file type. Allowed: {", ".join(Config.ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Read image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        original_image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        
        if original_image is None:
            return jsonify({
                'success': False,
                'error': 'Could not decode image'
            }), 400
        
        original_size = (original_image.shape[1], original_image.shape[0])
        
        # Preprocess
        input_tensor = preprocess_image(original_image).to(device)
        
        # Inference
        with torch.no_grad():
            prediction = model(input_tensor)
        
        # Postprocess
        mask, confidence, tumor_ratio = postprocess_prediction(prediction, original_size)
        
        # Determine if tumor is detected
        tumor_detected = tumor_ratio > 0.001  # At least 0.1% of image is tumor
        
        # Create visualization
        overlay = create_result_image(original_image, mask)
        
        # Convert images to base64
        # Original image
        if len(original_image.shape) == 2:
            original_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        elif original_image.shape[2] == 4:
            original_rgb = cv2.cvtColor(original_image, cv2.COLOR_RGBA2RGB)
        else:
            original_rgb = original_image
        
        original_b64 = image_to_base64(original_rgb)
        mask_b64 = image_to_base64((mask * 255).astype(np.uint8))
        overlay_b64 = image_to_base64(overlay)
        
        # Return results
        return jsonify({
            'success': True,
            'tumor_detected': tumor_detected,
            'confidence': round(confidence, 4),
            'confidence_percent': round(confidence * 100, 2),
            'tumor_ratio': round(tumor_ratio, 6),
            'tumor_ratio_percent': round(tumor_ratio * 100, 4),
            'original_image': f'data:image/png;base64,{original_b64}',
            'mask_image': f'data:image/png;base64,{mask_b64}',
            'overlay_image': f'data:image/png;base64,{overlay_b64}'
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/model-info')
def model_info():
    """Get information about the loaded model."""
    if model is None:
        return jsonify({
            'loaded': False,
            'error': 'Model not loaded'
        })
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return jsonify({
        'loaded': True,
        'architecture': 'U-Net with Attention',
        'input_channels': Config.IN_CHANNELS,
        'input_size': f'{Config.IMG_SIZE}x{Config.IMG_SIZE}',
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'threshold': Config.THRESHOLD,
        'device': str(device)
    })


# ============================================================================
# Error Handlers
# ============================================================================

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB.'
    }), 413


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error."""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


# ============================================================================
# Main
# ============================================================================

def create_app():
    """Create and configure the Flask application."""
    load_trained_model()
    return app


if __name__ == '__main__':
    # Load model
    load_trained_model()
    
    # Run server
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    
    print(f"\n{'='*60}")
    print("Brain Tumor Segmentation API")
    print(f"{'='*60}")
    print(f"Running on: http://localhost:{port}")
    print(f"Debug mode: {debug}")
    print(f"{'='*60}\n")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
