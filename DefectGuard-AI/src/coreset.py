"""
Greedy Minimax Coreset Subsampling for DefectGuard (PatchCore).
Reduces nominal patch memory size by selecting a representative subset of features
using Johnson-Lindenstrauss random projection and Greedy K-Center selection.
"""

import math
from typing import Tuple, Optional, Union
import torch
import numpy as np


class GreedyCoresetSampler:
    """
    Greedy Minimax Coreset Subsampler.
    
    Reduces a large set of patch feature vectors down to a fraction (e.g., 10%)
    while preserving coverage of the nominal patch feature space.
    """

    def __init__(
        self,
        sampling_ratio: float = 0.1,
        projection_dim: int = 128,
        random_seed: int = 42,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            sampling_ratio: Fraction of features to retain (default: 0.1 for 10%).
            projection_dim: Johnson-Lindenstrauss projection dimension (default: 128).
            random_seed: Random seed for deterministic projection.
            device: Computation device (CPU or CUDA).
        """
        self.sampling_ratio = sampling_ratio
        self.projection_dim = projection_dim
        self.random_seed = random_seed
        self.device = device or torch.device("cpu")

    def _get_jl_projection_matrix(self, input_dim: int) -> torch.Tensor:
        """
        Generates a Johnson-Lindenstrauss random projection matrix R ~ N(0, 1/d).
        """
        torch.manual_seed(self.random_seed)
        # Scale by 1 / sqrt(projection_dim) to preserve expected Euclidean distances
        R = torch.randn(input_dim, self.projection_dim, dtype=torch.float32, device=self.device)
        R /= math.sqrt(self.projection_dim)
        return R

    def subsample(
        self,
        features: Union[torch.Tensor, np.ndarray],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes Johnson-Lindenstrauss projection and Greedy K-Center selection.

        Args:
            features: [N, D] tensor of uncompressed patch feature vectors (D=1536).

        Returns:
            coreset_features: [K, D] tensor of retained feature vectors (K = N * sampling_ratio).
            selected_indices: [K] 1D tensor of chosen indices in the original feature tensor.
        """
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)

        features = features.to(self.device, dtype=torch.float32)
        n_samples, feat_dim = features.shape
        n_coreset = max(1, int(n_samples * self.sampling_ratio))

        print(f"[Coreset] Total input patches: {n_samples:,} (dim={feat_dim})")
        print(f"[Coreset] Target coreset size ({self.sampling_ratio * 100:.1f}%): {n_coreset:,}")

        if n_coreset >= n_samples:
            print("[Coreset] Target size >= input count. Returning all features.")
            indices = torch.arange(n_samples, device=self.device)
            return features, indices

        # 1. Johnson-Lindenstrauss Random Projection (1536 -> 128)
        print(f"[Coreset] Projecting {feat_dim}-dim features down to {self.projection_dim}-dim...")
        proj_matrix = self._get_jl_projection_matrix(feat_dim)
        projected = torch.matmul(features, proj_matrix)  # [N, 128]

        # 2. Greedy Minimax (K-Center) Selection
        print("[Coreset] Starting Greedy Minimax (K-Center) selection...")
        # Precompute squared norms for ultra-fast BLAS distance updates:
        # ||x - c||^2 = ||x||^2 - 2 * (x . c) + ||c||^2
        sq_norms = torch.sum(projected ** 2, dim=1)  # [N]

        # Seed with initial center (index 0 or max norm)
        selected_indices = torch.zeros(n_coreset, dtype=torch.long, device=self.device)
        first_idx = 0
        selected_indices[0] = first_idx

        # Initial minimum distance to chosen center
        first_center = projected[first_idx]
        first_center_sq_norm = torch.sum(first_center ** 2)
        min_distances = sq_norms - 2.0 * torch.matmul(projected, first_center) + first_center_sq_norm
        min_distances = torch.clamp(min_distances, min=0.0)

        # Batch progress reporting interval
        report_step = max(1, n_coreset // 10)

        for step in range(1, n_coreset):
            # Select point with maximum distance from any already-selected center
            next_idx = torch.argmax(min_distances).item()
            selected_indices[step] = next_idx

            # Update minimum distances
            center = projected[next_idx]
            center_sq_norm = torch.sum(center ** 2)
            dists = sq_norms - 2.0 * torch.matmul(projected, center) + center_sq_norm
            dists = torch.clamp(dists, min=0.0)
            min_distances = torch.minimum(min_distances, dists)

            if (step + 1) % report_step == 0 or (step + 1) == n_coreset:
                pct = (step + 1) / n_coreset * 100
                print(f"[Coreset] Selection progress: {step + 1:,}/{n_coreset:,} ({pct:.1f}%)")

        coreset_features = features[selected_indices]
        print(f"[Coreset] Completed coreset selection: {coreset_features.shape[0]:,} vectors retained.")
        return coreset_features, selected_indices


if __name__ == "__main__":
    dummy_feats = torch.randn(1000, 1536)
    sampler = GreedyCoresetSampler(sampling_ratio=0.1, projection_dim=128)
    coreset, idxs = sampler.subsample(dummy_feats)
    print("Coreset shape:", coreset.shape)
    print("Indices count:", len(idxs))
