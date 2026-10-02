"""从已保存的 PGSR PLY 导出少量 RGB、平面深度和法线验收样例。

只读取训练检查点，不启动训练；输出目录须为空，以免覆盖已有验收证据。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
from PIL import Image

from pgsr.render import prepare_rendering


def save_rgb(path: Path, tensor: torch.Tensor) -> None:
    pixels = tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    Image.fromarray(np.rint(pixels * 255).astype(np.uint8), "RGB").save(path)


def save_normal(path: Path, tensor: torch.Tensor, valid: np.ndarray) -> None:
    normal = tensor.detach().permute(1, 2, 0).cpu().numpy()
    length = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = np.divide(normal, np.maximum(length, 1e-8))
    color = np.rint(np.clip((normal + 1) * 127.5, 0, 255)).astype(np.uint8)
    color[~valid] = 0
    Image.fromarray(color, "RGB").save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/truck-320"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/truck-densify1500/point_cloud/iteration_1500/point_cloud.ply"))
    parser.add_argument("--output", type=Path, default=Path("outputs/truck-densify1500/acceptance/geometry"))
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 125, 250])
    parser.add_argument("--max-depth", type=float, default=10.0)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"Missing checkpoint: {args.checkpoint}")
    if args.max_depth <= 0:
        raise SystemExit("--max-depth must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"Refusing to overwrite nonempty output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        dataset, gaussians = prepare_rendering(
            sh_degree=3,
            source=str(args.source.resolve()),
            device="cuda",
            trainable_camera=False,
            load_ply=str(args.checkpoint.resolve()),
            load_mask=False,
            load_depth=False,
            backend="gsplat",
        )
        gaussians.render_depth_normal = True
        samples = []
        for index in args.indices:
            if not 0 <= index < len(dataset):
                raise SystemExit(f"Camera index {index} outside [0, {len(dataset)})")
            camera = dataset[index]
            out = gaussians(camera)
            prefix = f"{index:05d}"
            save_rgb(args.output / f"{prefix}_rgb.png", out["render"])
            save_rgb(args.output / f"{prefix}_gt.png", camera.ground_truth_image)
            depth = out["depth"].squeeze(0).detach().cpu().numpy()
            alpha = out["render_alphas"].squeeze(0).detach().cpu().numpy()
            valid = np.isfinite(depth) & (depth > 0.1) & (depth <= args.max_depth)
            tifffile.imwrite(args.output / f"{prefix}_depth.tiff", depth.astype(np.float32))
            safe_depth = np.where(np.isfinite(depth), depth, 0.0)
            depth_u8 = np.rint(np.clip(safe_depth / args.max_depth, 0, 1) * 255).astype(np.uint8)
            depth_color = cv2.cvtColor(cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
            depth_color[~valid] = 0
            Image.fromarray(depth_color, "RGB").save(args.output / f"{prefix}_depth.png")
            save_normal(args.output / f"{prefix}_normal.png", out["render_normals"], valid)
            save_normal(args.output / f"{prefix}_normal_from_depth.png", out["normals_from_depth"], valid)
            values = depth[valid]
            samples.append({
                "camera_index": index,
                "image_name": Path(camera.ground_truth_image_path).name,
                "image_hw": [int(camera.image_height), int(camera.image_width)],
                "valid_depth_pixels": int(valid.sum()),
                "valid_depth_fraction": float(valid.mean()),
                "finite_depth_fraction": float(np.isfinite(depth).mean()),
                "depth_p01_p50_p95": np.percentile(values, [1, 50, 95]).tolist() if values.size else None,
                "alpha_p50": float(np.median(alpha)),
            })
            del out
    torch.cuda.synchronize()
    manifest = {
        "purpose": "短训练 checkpoint 的渲染几何验收；非论文收敛质量评价",
        "command": sys.argv,
        "checkpoint": str(args.checkpoint.resolve()),
        "source": str(args.source.resolve()),
        "max_depth": args.max_depth,
        "samples": samples,
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "torch_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1048576, 1),
        "torch_peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 1048576, 1),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
