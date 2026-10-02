"""Opt-in CPU initialization workaround for unsupported bundled simple_knn SM 86.

This changes initial Gaussian scales and is an explicitly non-baseline study
experiment. The installed package and clean reference checkouts are untouched.
"""

import numpy as np
import torch
from scipy.spatial import cKDTree


def install() -> None:
    import gaussian_splatting.gaussian_model as gaussian_model

    def exact_three_neighbor_mean_squared(points: torch.Tensor) -> torch.Tensor:
        xyz = points.detach().cpu().numpy()
        if len(xyz) < 4:
            raise ValueError("CPU 3-NN initialization requires at least four points")
        distances, _ = cKDTree(xyz).query(xyz, k=4, workers=-1)
        mean_squared = np.square(distances[:, 1:4]).mean(axis=1).astype(np.float32)
        return torch.from_numpy(mean_squared).to(device=points.device)

    gaussian_model.distCUDA2 = exact_three_neighbor_mean_squared
    print(
        "STUDY EXPERIMENT: bundled simple_knn lacks SM 86; "
        "Gaussian initialization uses exact CPU 3-neighbor mean squared distance. "
        "Initial scales may differ from the baseline CUDA approximation.",
        flush=True,
    )
