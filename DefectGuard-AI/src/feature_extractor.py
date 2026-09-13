"""
Feature Extractor Module for DefectGuard (PatchCore).
Extracts multi-scale patch tokens from intermediate layers of WideResNet-50-2.
"""

from typing import Tuple, List, Optional, Union
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image


def isolate_foreground(
    image: Union[Image.Image, np.ndarray],
    min_bg_brightness: float = 25.0,
) -> Tuple[Image.Image, np.ndarray]:
    """
    Isolates the central bottle foreground from non-black backgrounds using OpenCV
    Otsu thresholding and contour extraction, forcing all background pixels to pure black [0, 0, 0].

    Args:
        image: PIL Image or RGB numpy array.
        min_bg_brightness: Mean corner pixel brightness threshold to decide if background is non-black.

    Returns:
        isolated_pil: PIL Image with non-bottle background forced to RGB [0, 0, 0].
        binary_mask: 2D uint8 numpy array [H, W] with 255 for foreground bottle, 0 for background.
    """
    if isinstance(image, Image.Image):
        img_rgb = np.array(image.convert("RGB"))
    else:
        img_rgb = np.array(image, copy=True)

    h, w, _ = img_rgb.shape

    # Sample corners (5% margin) to assess background brightness and uniformity
    margin_h = max(2, int(h * 0.05))
    margin_w = max(2, int(w * 0.05))
    corners = np.concatenate([
        img_rgb[:margin_h, :margin_w].reshape(-1, 3),
        img_rgb[:margin_h, -margin_w:].reshape(-1, 3),
        img_rgb[-margin_h:, :margin_w].reshape(-1, 3),
        img_rgb[-margin_h:, -margin_w:].reshape(-1, 3),
    ], axis=0)
    corner_mean = float(np.mean(corners))
    corner_std = float(np.std(corners))

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    center_h, center_w = h // 2, w // 2
    center_mean = float(np.mean(gray[center_h - margin_h:center_h + margin_h, center_w - margin_w:center_w + margin_w]))

    # Determine polarity: dark bottle on bright background vs bright bottle on dark background
    # MVTec 'bottle' is imaged on a pure white [255, 255, 255] background.
    if corner_mean > center_mean or corner_mean > 128:
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        bg_color = [255, 255, 255]
    else:
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bg_color = [0, 0, 0]

    # Morphological closing to bridge internal reflections and bottle specularities
    kernel_size = max(5, int(min(h, w) * 0.02))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find external contours and select the largest contour representing the bottle
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    binary_mask = np.zeros((h, w), dtype=np.uint8)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(binary_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    else:
        # Fallback to entire image if no contours detected
        binary_mask[:] = 255

    # Check if the input image is already on an authentic clean MVTec white background
    is_already_clean_mvtec = (corner_mean > 240.0 and corner_std < 10.0)

    isolated_rgb = img_rgb.copy()
    if not is_already_clean_mvtec:
        isolated_rgb[binary_mask == 0] = bg_color

    return Image.fromarray(isolated_rgb), binary_mask


def generate_patch_mask(
    binary_mask: np.ndarray,
    grid_size: Tuple[int, int] = (28, 28),
    threshold: float = 0.5,
    erode_iterations: int = 1,
) -> np.ndarray:
    """
    Downsamples a binary mask [H, W] to feature grid dimensions (e.g., [28, 28]),
    with optional erosion to prevent boundary transition artifacts.

    Args:
        binary_mask: 2D uint8 numpy array with values 0 and 255.
        grid_size: Target spatial grid (H_grid, W_grid), e.g. (28, 28).
        threshold: Overlap threshold between 0.0 and 1.0 (default: 0.5).
        erode_iterations: Iterations of 3x3 morphological erosion on the grid mask (default: 1).

    Returns:
        2D boolean array [H_grid, W_grid] indicating foreground patch tokens.
    """
    h_grid, w_grid = grid_size
    downsampled = cv2.resize(
        binary_mask.astype(np.float32) / 255.0,
        (w_grid, h_grid),
        interpolation=cv2.INTER_AREA,
    )
    patch_mask = downsampled >= threshold
    if erode_iterations > 0 and np.sum(patch_mask) > 50:
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(patch_mask.astype(np.uint8), kernel, iterations=erode_iterations) > 0
        if np.any(eroded):
            patch_mask = eroded

    if not np.any(patch_mask):
        patch_mask = np.ones((h_grid, w_grid), dtype=bool)
    return patch_mask



class PatchCoreFeatureExtractor(nn.Module):
    """
    Extracts patch features from intermediate layers (layer2 and layer3)
    of a pretrained WideResNet-50-2 backbone.
    """

    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained backbone
        if backbone_name == "wide_resnet50_2":
            weights = models.Wide_ResNet50_2_Weights.DEFAULT
            self.model = models.wide_resnet50_2(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Intermediate feature storage
        self.features = {}

        # Register forward hooks on layer2 and layer3
        self.hooks = []
        self._register_hooks()

        # Neighborhood aggregation pooling: 3x3 average pooling, stride 1, padding 1
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        # Standard preprocessing transform for input images
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _register_hooks(self):
        def _hook(layer_name):
            def forward_hook(module, input, output):
                self.features[layer_name] = output
            return forward_hook

        self.hooks.append(self.model.layer2.register_forward_hook(_hook("layer2")))
        self.hooks.append(self.model.layer3.register_forward_hook(_hook("layer3")))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocesses a PIL Image to a normalized tensor [1, 3, H, W]."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device)

    def preprocess_with_mask(
        self,
        image: Union[Image.Image, np.ndarray],
        grid_size: Tuple[int, int] = (28, 28),
    ) -> Tuple[torch.Tensor, np.ndarray, Image.Image, np.ndarray]:
        """
        Isolates foreground, converts background to pure black [0, 0, 0],
        computes the downsampled patch mask, and returns the normalized tensor.

        Returns:
            tensor: [1, 3, 224, 224] normalized tensor ready for model input.
            patch_mask: [H_grid, W_grid] boolean array for token masking.
            isolated_image: PIL Image with background forced to black.
            full_mask: [H, W] uint8 full-resolution binary mask.
        """
        isolated_image, full_mask = isolate_foreground(image)
        # Generate patch mask (resized to grid_size)
        patch_mask = generate_patch_mask(full_mask, grid_size=grid_size)
        tensor = self.preprocess_image(isolated_image)
        return tensor, patch_mask, isolated_image, full_mask


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Extracts multi-scale patch embeddings.

        Args:
            x: Input batch tensor of shape [B, 3, H, W].

        Returns:
            patch_tokens: Tensor of shape [B, H2*W2, 1536] (or flattened [B*H2*W2, 1536]).
            grid_size: Tuple (H2, W2) representing spatial grid dimensions.
        """
        self.features.clear()
        _ = self.model(x)

        # Features from layer2: [B, 512, H2, W2]
        feat_l2 = self.features["layer2"]
        # Features from layer3: [B, 1024, H3, W3]
        feat_l3 = self.features["layer3"]

        # Apply 3x3 adaptive/average pooling (stride 1, padding 1)
        feat_l2 = self.avg_pool(feat_l2)
        feat_l3 = self.avg_pool(feat_l3)

        # Spatial dimensions of layer2
        b, c2, h2, w2 = feat_l2.shape

        # Bilinearly upsample layer3 to match layer2 spatial dimensions
        feat_l3_up = F.interpolate(
            feat_l3,
            size=(h2, w2),
            mode="bilinear",
            align_corners=False,
        )

        # Concatenate channels: 512 + 1024 = 1536 channels
        concat_features = torch.cat([feat_l2, feat_l3_up], dim=1)  # [B, 1536, H2, W2]

        # Reshape to patch tokens: [B, H2, W2, 1536] -> [B, H2*W2, 1536]
        concat_features = concat_features.permute(0, 2, 3, 1).contiguous()
        patch_tokens = concat_features.view(b, h2 * w2, -1)  # [B, H2*W2, 1536]

        return patch_tokens, (h2, w2)


if __name__ == "__main__":
    extractor = PatchCoreFeatureExtractor()
    dummy_input = torch.randn(2, 3, 224, 224).to(extractor.device)
    tokens, grid = extractor(dummy_input)
    print(f"Extracted tokens shape: {tokens.shape}")
    print(f"Feature grid dimensions: {grid}")
    print(f"Channels: {tokens.shape[-1]} (expected 1536)")
