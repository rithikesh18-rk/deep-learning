"""
Dataset loading and data augmentation pipelines for food image classification.
"""

from pathlib import Path
from typing import Tuple, List, Dict
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image

import src.config as config


def get_transforms(image_size: int = config.IMAGE_SIZE) -> Dict[str, transforms.Compose]:
    """
    Returns image transformation pipelines for training and validation/testing.

    Args:
        image_size (int): Target image resolution width/height.

    Returns:
        Dict[str, transforms.Compose]: Dictionary with 'train' and 'val' transforms.
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),  # ~256 for 224
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN, std=config.NORMALIZE_STD),
    ])

    return {"train": train_transform, "val": val_transform}


def load_single_image(image_path: Path, image_size: int = config.IMAGE_SIZE) -> torch.Tensor:
    """
    Loads and preprocesses a single image for model inference.

    Args:
        image_path (Path): Path to the image file.
        image_size (int): Expected model input size.

    Returns:
        torch.Tensor: Preprocessed image tensor with shape (1, 3, H, W).
    """
    transform = get_transforms(image_size=image_size)["val"]
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor


def get_data_loaders(
    train_dir: Path = config.TRAIN_DIR,
    val_dir: Path = config.VAL_DIR,
    batch_size: int = config.BATCH_SIZE,
    image_size: int = config.IMAGE_SIZE,
    num_workers: int = config.NUM_WORKERS
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Loads training and validation datasets using ImageFolder and returns DataLoaders.

    Args:
        train_dir (Path): Directory containing training image subfolders.
        val_dir (Path): Directory containing validation image subfolders.
        batch_size (int): Batch size for DataLoaders.
        image_size (int): Image resolution for resizing.
        num_workers (int): Number of worker processes for data loading.

    Returns:
        Tuple[DataLoader, DataLoader, List[str]]: (train_loader, val_loader, class_names)
    """
    data_transforms = get_transforms(image_size=image_size)

    # Validate directory existence
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Dataset directories not found! Ensure {train_dir} and {val_dir} exist."
        )

    # Create PyTorch ImageFolder datasets
    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=data_transforms["train"])
    val_dataset = datasets.ImageFolder(root=str(val_dir), transform=data_transforms["val"])

    class_names = train_dataset.classes
    if not class_names:
        raise ValueError(
            f"No class folders found in {train_dir}. "
            "Please structure your dataset into subfolders (e.g. dataset/train/pizza/img1.jpg)."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, class_names
