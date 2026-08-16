"""
Model definition module for transfer learning using pretrained lightweight CNNs.
Supports full backbone freezing and controlled fine-tuning of top feature blocks.
"""

import torch
import torch.nn as nn
import torchvision.models as models


def create_model(
    num_classes: int,
    model_name: str = "efficientnet_b0",
    freeze_backbone: bool = True,
    fine_tune: bool = False,
    unfreeze_last_n_blocks: int = 2
) -> nn.Module:
    """
    Creates a pretrained CNN model and configures feature layer freezing / fine-tuning.

    Args:
        num_classes (int): Number of target food classes.
        model_name (str): Architecture name ('efficientnet_b0', 'mobilenet_v3_small', 'mobilenet_v3_large').
        freeze_backbone (bool): If True, freeze feature extractor parameters.
        fine_tune (bool): If True, unfreeze top N feature blocks for fine-tuning while keeping lower layers frozen.
        unfreeze_last_n_blocks (int): Number of top feature blocks to unfreeze during fine-tuning.

    Returns:
        nn.Module: PyTorch model configured for transfer learning or fine-tuning.
    """
    model_name = model_name.lower()

    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)

        # Handle freezing / fine-tuning
        if fine_tune:
            # Freeze lower blocks, unfreeze top N feature blocks
            total_blocks = len(model.features)
            freeze_limit = max(0, total_blocks - unfreeze_last_n_blocks)

            for i, block in enumerate(model.features):
                if i < freeze_limit:
                    for param in block.parameters():
                        param.requires_grad = False
                else:
                    for param in block.parameters():
                        param.requires_grad = True
        elif freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
        else:
            for param in model.features.parameters():
                param.requires_grad = True

        # Replace final classifier layer
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features=in_features, out_features=num_classes)
        )

    elif model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)

        if fine_tune:
            total_blocks = len(model.features)
            freeze_limit = max(0, total_blocks - unfreeze_last_n_blocks)
            for i, block in enumerate(model.features):
                if i < freeze_limit:
                    for param in block.parameters():
                        param.requires_grad = False
                else:
                    for param in block.parameters():
                        param.requires_grad = True
        elif freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False

        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features=in_features, out_features=num_classes)

    else:
        raise ValueError(f"Unsupported model architecture: '{model_name}'.")

    return model
