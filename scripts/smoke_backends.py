"""One COLMAP-camera forward/backward pass with each PGSR backend.

Invoke separately per backend to release GPU memory between runs.
"""

import argparse
import gc
import json
import os
import random

import numpy as np
import torch

from pgsr.train import prepare_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="/mnt/e/PGSR/data/truck-320")
    parser.add_argument("--backend", choices=("gsplat", "gsplat-2dgs"), required=True)
    args = parser.parse_args()

    seed = 20260930
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if os.environ.get("PGSR_STUDY_CPU_KNN") == "1":
        from study_knn_fallback import install

        install()
    # 高层工厂按顺序加载相机/RGB、用 COLMAP 点初始化高斯、组合训练器。
    dataset, gaussians, trainer = prepare_training(
        sh_degree=3,
        source=args.source,
        device="cuda",
        mode="base",
        backend=args.backend,
        load_mask=False,
        load_depth=False,
    )
    camera = dataset[0]
    # 这里只测试一次已标定相机上的可微渲染；完整优化器更新由 train CLI 验证。
    out = gaussians(camera)
    loss = out["render"].mean()
    loss.backward()
    gradients = [p.grad for p in gaussians.parameters() if p.grad is not None]
    report = {
        "backend": args.backend,
        "seed": seed,
        "initialization": "exact_cpu_3nn_study" if os.environ.get("PGSR_STUDY_CPU_KNN") == "1" else "bundled_cuda_simple_knn",
        "cameras": len(dataset),
        "image_hw": [int(camera.image_height), int(camera.image_width)],
        "gaussians": int(gaussians.get_xyz.shape[0]),
        "outputs": {k: list(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)},
        "rgb_loss": float(loss.detach().cpu()),
        "parameter_tensors_with_grad": len(gradients),
        "nonzero_gradient_tensors": sum(int(bool(torch.count_nonzero(g).item())) for g in gradients),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1048576, 1),
    }
    print(json.dumps(report, indent=2), flush=True)
    del trainer, gaussians, dataset, camera, out, gradients
    gc.collect()


if __name__ == "__main__":
    main()
