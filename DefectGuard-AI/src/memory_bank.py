"""
FAISS Memory Bank & Inference Engine for DefectGuard (PatchCore).
Stores nominal coreset patch features in a FAISS IndexFlatL2 index and provides
sub-millisecond similarity search, patch scoring, spatial anomaly map generation,
and image-level anomaly scoring.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union
import cv2
import faiss
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


class PatchCoreMemoryBank:
    """
    Wraps FAISS IndexFlatL2 for storing nominal patch features and executing
    PatchCore anomaly detection inference.
    """

    def __init__(
        self,
        feature_dim: int = 1536,
        index_path: Optional[Union[str, Path]] = None,
    ):
        """
        Args:
            feature_dim: Channel dimension of patch features (default: 1536).
            index_path: Optional path to an existing .index file to load immediately.
        """
        self.feature_dim = feature_dim
        self.index = faiss.IndexFlatL2(feature_dim)
        if index_path and Path(index_path).exists():
            self.load(index_path)

    def fit(self, features: Union[torch.Tensor, np.ndarray]):
        """
        Populates the FAISS memory bank with nominal patch feature vectors.

        Args:
            features: [N, D] array/tensor of nominal patch vectors.
        """
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()

        features = np.ascontiguousarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected features with shape [N, {self.feature_dim}], got {features.shape}"
            )

        print(f"[MemoryBank] Adding {features.shape[0]:,} vectors to FAISS IndexFlatL2...")
        self.index.reset()
        self.index.add(features)
        print(f"[MemoryBank] Index trained and populated. Total entries: {self.index.ntotal:,}")

    def save(self, output_path: Union[str, Path]):
        """Saves the FAISS index to disk."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        print(f"[MemoryBank] Index saved to {path} ({path.stat().st_size / 1e6:.2f} MB)")

    def load(self, input_path: Union[str, Path]):
        """Loads a FAISS index from disk."""
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"FAISS index file not found at {path}")
        self.index = faiss.read_index(str(path))
        self.feature_dim = self.index.d
        print(f"[MemoryBank] Loaded index from {path}. Entries: {self.index.ntotal:,}, Dim: {self.feature_dim}")

    def predict(
        self,
        patch_tokens: Union[torch.Tensor, np.ndarray],
        grid_size: Tuple[int, int],
        image_size: Tuple[int, int] = (224, 224),
        gaussian_sigma: float = 4.0,
        patch_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        full_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        Runs PatchCore anomaly inference on a batch or single image's patch tokens,
        optionally restricting nearest-neighbor search to foreground patches.

        Args:
            patch_tokens: [B, H_grid*W_grid, D] or [H_grid*W_grid, D] tensor/array.
            grid_size: Spatial grid dimensions (H_grid, W_grid) (e.g., (28, 28)).
            image_size: Target image dimensions (H, W) for output heatmap.
            gaussian_sigma: Standard deviation for Gaussian smoothing filter (default: 4.0).
            patch_mask: Optional boolean or binary mask [H_grid, W_grid] or [B, H_grid, W_grid]
                        indicating object foreground patches. When provided, FAISS nearest-neighbor
                        search is ONLY performed for foreground patches, ignoring surrounding
                        background completely.
            full_mask: Optional 2D uint8/bool full-resolution mask [H, W] to strictly zero out
                       background pixels in the final continuous anomaly map.

        Returns:
            Dictionary containing:
                - 'image_scores': List of float image-level anomaly scores.
                - 'anomaly_maps': List of 2D numpy arrays [H, W] smoothed pixel heatmaps.
                - 'patch_scores': List of 2D numpy arrays [H_grid, W_grid] raw patch distances.
        """
        if self.index.ntotal == 0:
            raise RuntimeError("Memory bank index is empty. Call fit() or load() before predicting.")

        if isinstance(patch_tokens, torch.Tensor):
            patch_tokens = patch_tokens.detach().cpu().numpy()

        h_grid, w_grid = grid_size
        patches_per_image = h_grid * w_grid

        # Ensure 3D shape: [B, patches_per_image, D]
        if patch_tokens.ndim == 2:
            patch_tokens = np.expand_dims(patch_tokens, axis=0)

        batch_size, n_patches, feat_dim = patch_tokens.shape
        if n_patches != patches_per_image:
            raise ValueError(
                f"Patch count mismatch: expected {patches_per_image} ({h_grid}x{w_grid}), got {n_patches}"
            )

        # Normalize patch_mask if provided
        norm_patch_mask = None
        if patch_mask is not None:
            if isinstance(patch_mask, torch.Tensor):
                norm_patch_mask = patch_mask.detach().cpu().numpy()
            else:
                norm_patch_mask = np.array(patch_mask)

            # Reshape to [B, patches_per_image]
            if norm_patch_mask.ndim == 2:
                if norm_patch_mask.shape == (h_grid, w_grid):
                    norm_patch_mask = np.broadcast_to(
                        (norm_patch_mask > 0).reshape(1, patches_per_image),
                        (batch_size, patches_per_image),
                    )
                elif norm_patch_mask.shape == (batch_size, patches_per_image):
                    norm_patch_mask = norm_patch_mask > 0
                else:
                    raise ValueError(f"Unexpected 2D patch_mask shape: {norm_patch_mask.shape}")
            elif norm_patch_mask.ndim == 3 and norm_patch_mask.shape == (batch_size, h_grid, w_grid):
                norm_patch_mask = (norm_patch_mask > 0).reshape(batch_size, patches_per_image)

        image_scores = []
        anomaly_maps = []
        raw_patch_grids = []
        target_h, target_w = image_size

        for b in range(batch_size):
            tokens_b = np.ascontiguousarray(patch_tokens[b], dtype=np.float32)
            l2_distances_b = np.zeros(patches_per_image, dtype=np.float32)

            if norm_patch_mask is not None:
                mask_b = norm_patch_mask[b]
                fg_indices = np.where(mask_b)[0]

                if len(fg_indices) > 0:
                    # Only compute FAISS nearest-neighbor distances for foreground patches
                    fg_queries = np.ascontiguousarray(tokens_b[fg_indices], dtype=np.float32)
                    fg_dist_sq, _ = self.index.search(fg_queries, k=1)
                    fg_l2 = np.sqrt(np.maximum(fg_dist_sq.flatten(), 0.0))
                    l2_distances_b[fg_indices] = fg_l2

                    # Compute image-level anomaly score strictly from foreground patches
                    k = min(5, len(fg_indices))
                    top_k_patches = np.sort(fg_l2)[-k:]
                    img_score = float(np.mean(top_k_patches))
                else:
                    img_score = 0.0
            else:
                # Unmasked: Full FAISS search across all patches
                dist_sq, _ = self.index.search(tokens_b, k=1)
                l2_distances_b = np.sqrt(np.maximum(dist_sq.flatten(), 0.0))
                top_k_patches = np.sort(l2_distances_b)[-5:]
                img_score = float(np.mean(top_k_patches))

            image_scores.append(img_score)

            patch_grid = l2_distances_b.reshape(h_grid, w_grid)
            raw_patch_grids.append(patch_grid)

            # Bilinear upsampling to original image size
            patch_tensor = torch.from_numpy(patch_grid).unsqueeze(0).unsqueeze(0).float()
            upsampled_map = F.interpolate(
                patch_tensor,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze().numpy()

            # Apply Gaussian blur smoothing (sigma = 4)
            smoothed_map = gaussian_filter(upsampled_map, sigma=gaussian_sigma)

            # If masked, zero out the upsampled anomaly map outside the object foreground
            if full_mask is not None:
                if isinstance(full_mask, torch.Tensor):
                    f_mask = full_mask.detach().cpu().numpy()
                else:
                    f_mask = np.array(full_mask)
                f_mask_resized = cv2.resize(
                    (f_mask > 0).astype(np.float32),
                    (target_w, target_h),
                    interpolation=cv2.INTER_NEAREST,
                )
                smoothed_map = smoothed_map * f_mask_resized
            elif norm_patch_mask is not None:
                mask_grid_b = norm_patch_mask[b].reshape(h_grid, w_grid).astype(np.float32)
                mask_tensor = torch.from_numpy(mask_grid_b).unsqueeze(0).unsqueeze(0)
                upsampled_mask = F.interpolate(
                    mask_tensor,
                    size=(target_h, target_w),
                    mode="nearest",
                ).squeeze().numpy()
                smoothed_map = smoothed_map * upsampled_mask

            anomaly_maps.append(smoothed_map)

        return {
            "image_scores": image_scores if batch_size > 1 else image_scores[0],
            "anomaly_maps": anomaly_maps if batch_size > 1 else anomaly_maps[0],
            "patch_scores": raw_patch_grids if batch_size > 1 else raw_patch_grids[0],
        }


if __name__ == "__main__":
    bank = PatchCoreMemoryBank(feature_dim=1536)
    dummy_nominal = np.random.randn(500, 1536).astype(np.float32)
    bank.fit(dummy_nominal)

    dummy_test = np.random.randn(1, 28 * 28, 1536).astype(np.float32)
    result = bank.predict(dummy_test, grid_size=(28, 28), image_size=(224, 224))
    print("Inference score:", result["image_scores"])
    print("Anomaly map shape:", result["anomaly_maps"].shape)
