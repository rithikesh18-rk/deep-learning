"""Dual-Stream Forensic Neural Network Architecture.

Combines Spatial Domain Features (via ConvNeXt-Tiny) and Frequency Domain Features
(via a 4-Layer Spectrum CNN) to detect AI-generated and deepfake imagery.
"""

import torch
import torch.nn as nn
import timm


class FrequencyStream(nn.Module):
    """4-Layer Convolutional Neural Network for 1-Channel FFT Spectrum Processing.

    Architecture:
      - Layer 1: Conv2d(1, 32) -> BatchNorm2d -> ReLU -> MaxPool2d (112x112)
      - Layer 2: Conv2d(32, 64) -> BatchNorm2d -> ReLU -> MaxPool2d (56x56)
      - Layer 3: Conv2d(64, 128) -> BatchNorm2d -> ReLU -> MaxPool2d (28x28)
      - Layer 4: Conv2d(128, 256) -> BatchNorm2d -> ReLU -> MaxPool2d (14x14)
      - AdaptiveAvgPool2d((1, 1)) -> 256-dim feature vector
    """

    def __init__(self, in_channels: int = 1, out_features: int = 256):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 4
            nn.Conv2d(128, out_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv_layers(x)
        pooled = self.global_pool(features)
        return self.flatten(pooled)


class DualStreamForensicNet(nn.Module):
    """Dual-Stream Forensic Neural Network.

    - Spatial Stream: Pretrained ConvNeXt-Tiny (768-dim)
    - Frequency Stream: Custom 4-Layer Spectrum CNN (256-dim)
    - Fusion Classifier: 1024-dim -> Dropout(0.3) -> Linear(1024, 256) -> ReLU -> Linear(256, 2)
      Classes: [0: Authentic, 1: AI-Generated / Deepfake]
    """

    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 2,
        spatial_dim: int = 768,
        spectral_dim: int = 256,
        dropout_rate: float = 0.3
    ):
        super().__init__()

        # Spatial Stream (ConvNeXt-Tiny backbone extracting 768-dim features)
        self.spatial_backbone = timm.create_model(
            'convnext_tiny',
            pretrained=pretrained,
            num_classes=0
        )

        # Frequency Stream (4-layer CNN extracting 256-dim spectral features)
        self.frequency_backbone = FrequencyStream(
            in_channels=1,
            out_features=spectral_dim
        )

        # Fusion Classifier (1024-dim combined vector)
        fusion_dim = spatial_dim + spectral_dim
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(
        self,
        rgb_tensor: torch.Tensor,
        freq_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for dual-stream inference.

        Args:
            rgb_tensor: Spatial RGB image tensor of shape (B, 3, 224, 224).
            freq_tensor: Frequency spectrum tensor of shape (B, 1, 224, 224).

        Returns:
            logits: Output classification logits of shape (B, 2).
        """
        spatial_features = self.spatial_backbone(rgb_tensor)       # (B, 768)
        spectral_features = self.frequency_backbone(freq_tensor)   # (B, 256)

        fused_features = torch.cat([spatial_features, spectral_features], dim=1)  # (B, 1024)
        logits = self.classifier(fused_features)                   # (B, 2)
        return logits
