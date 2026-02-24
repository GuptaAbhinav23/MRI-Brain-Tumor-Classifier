"""
U-Net Architecture for Brain Tumor Segmentation
================================================
A 2D U-Net implementation with optional attention mechanism for medical image segmentation.

Author: AI Research Engineer
Date: 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Double convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    Standard building block for U-Net architecture.
    """
    
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    Attention Gate module for focusing on relevant features.
    Helps the model focus on tumor regions during segmentation.
    
    Reference: Attention U-Net (Oktay et al., 2018)
    """
    
    def __init__(self, gate_channels: int, features_channels: int, intermediate_channels: int):
        super(AttentionGate, self).__init__()
        
        # Transform gating signal
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, intermediate_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(intermediate_channels)
        )
        
        # Transform skip connection features
        self.W_x = nn.Sequential(
            nn.Conv2d(features_channels, intermediate_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(intermediate_channels)
        )
        
        # Final attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(intermediate_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, gate: torch.Tensor, skip_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gate: Gating signal from decoder (lower resolution)
            skip_features: Features from encoder (skip connection)
        
        Returns:
            Attention-weighted features
        """
        # Upsample gate if needed to match skip features size
        gate_transformed = self.W_gate(gate)
        if gate_transformed.shape[2:] != skip_features.shape[2:]:
            gate_transformed = F.interpolate(
                gate_transformed, 
                size=skip_features.shape[2:], 
                mode='bilinear', 
                align_corners=True
            )
        
        x_transformed = self.W_x(skip_features)
        
        # Compute attention coefficients
        combined = self.relu(gate_transformed + x_transformed)
        attention = self.psi(combined)
        
        # Apply attention to skip features
        return skip_features * attention


class UNet(nn.Module):
    """
    2D U-Net Architecture for Brain Tumor Segmentation.
    
    Features:
    - Encoder-Decoder structure with skip connections
    - Optional attention gates for improved segmentation
    - Dropout for regularization
    - Multi-channel input support (FLAIR, T1, T1ce, T2)
    
    Args:
        in_channels: Number of input channels (1 for single modality, 4 for all modalities)
        out_channels: Number of output classes (1 for binary, 4 for multi-class)
        features: List of feature channels at each level
        use_attention: Whether to use attention gates
        dropout_rate: Dropout rate for regularization
    """
    
    def __init__(
        self, 
        in_channels: int = 4,
        out_channels: int = 1,
        features: list = None,
        use_attention: bool = True,
        dropout_rate: float = 0.1
    ):
        super(UNet, self).__init__()
        
        if features is None:
            features = [64, 128, 256, 512]
        
        self.use_attention = use_attention
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Attention gates (if enabled)
        if use_attention:
            self.attention_gates = nn.ModuleList()
        
        # Encoder path (Contracting)
        for feature in features:
            self.encoder.append(ConvBlock(in_channels, feature, dropout_rate))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2, dropout_rate)
        
        # Decoder path (Expanding)
        reversed_features = features[::-1]
        for idx, feature in enumerate(reversed_features):
            # Transposed convolution for upsampling
            self.decoder.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            # Convolution block after concatenation
            self.decoder.append(ConvBlock(feature * 2, feature, dropout_rate))
            
            # Attention gate
            if use_attention:
                self.attention_gates.append(
                    AttentionGate(feature * 2, feature, feature // 2)
                )
        
        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
        
        Returns:
            Output segmentation map of shape (batch, out_channels, height, width)
        """
        skip_connections = []
        
        # Encoder path
        for encoder_block in self.encoder:
            x = encoder_block(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Reverse skip connections for decoder
        skip_connections = skip_connections[::-1]
        
        # Decoder path
        for idx in range(0, len(self.decoder), 2):
            # Upsample
            x = self.decoder[idx](x)
            skip = skip_connections[idx // 2]
            
            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            
            # Apply attention if enabled
            if self.use_attention:
                skip = self.attention_gates[idx // 2](x, skip)
            
            # Concatenate skip connection
            x = torch.cat((skip, x), dim=1)
            
            # Convolution block
            x = self.decoder[idx + 1](x)
        
        return self.final_conv(x)
    
    def get_confidence_score(self, output: torch.Tensor) -> torch.Tensor:
        """
        Calculate confidence score for predictions.
        
        Args:
            output: Raw model output (logits)
        
        Returns:
            Confidence score between 0 and 1
        """
        probs = torch.sigmoid(output)
        # Confidence is the mean of max probabilities across spatial dimensions
        confidence = probs.max(dim=2)[0].max(dim=2)[0].mean(dim=1)
        return confidence


class UNetSmall(nn.Module):
    """
    Smaller U-Net variant for faster training and inference.
    Suitable for resource-constrained environments.
    """
    
    def __init__(self, in_channels: int = 4, out_channels: int = 1):
        super(UNetSmall, self).__init__()
        
        features = [32, 64, 128, 256]
        self.unet = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
            use_attention=False,
            dropout_rate=0.05
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet(x)


def get_model(
    in_channels: int = 4,
    out_channels: int = 1,
    use_attention: bool = True,
    model_size: str = 'normal'
) -> nn.Module:
    """
    Factory function to create U-Net model.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output classes
        use_attention: Whether to use attention gates
        model_size: 'small' or 'normal'
    
    Returns:
        U-Net model instance
    """
    if model_size == 'small':
        return UNetSmall(in_channels, out_channels)
    else:
        return UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            use_attention=use_attention
        )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    print("Testing U-Net Architecture...")
    
    # Create model
    model = UNet(in_channels=4, out_channels=1, use_attention=True)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(2, 4, 128, 128)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test confidence score
    confidence = model.get_confidence_score(output)
    print(f"Confidence scores: {confidence}")
    
    print("Model test passed!")
