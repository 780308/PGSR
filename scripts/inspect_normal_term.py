"""Inspect the depth-normal loss ingredients from a saved PGSR checkpoint."""

import json

import torch

from pgsr.render import prepare_rendering
from pgsr.utils import get_img_grad_weight


def main() -> None:
    source = "/mnt/e/PGSR/data/truck-320"
    ply = "/mnt/e/PGSR/outputs/truck-geometry300/point_cloud/iteration_300/point_cloud.ply"
    dataset, model = prepare_rendering(
        sh_degree=3,
        source=source,
        device="cuda",
        load_ply=ply,
        load_mask=False,
        load_depth=False,
        backend="gsplat",
    )
    model.render_depth_normal = True
    for index in (0, 39, 100, 200):
        camera = dataset[index]
        with torch.no_grad():
            out = model(camera)
            normal = out["render_normals"]
            depth_normal = out["normals_from_depth"]
            weight = (1 - get_img_grad_weight(camera.ground_truth_image)).clamp(0, 1).detach() ** 2
            diff = (depth_normal - normal).abs().sum(0)
            term = 0.015 * (weight * diff).mean()
            report = {
                "image": camera.ground_truth_image_path,
                "render_normal_abs_mean": float(normal.abs().mean()),
                "depth_normal_abs_mean": float(depth_normal.abs().mean()),
                "normal_difference_mean": float(diff.mean()),
                "image_weight_mean": float(weight.mean()),
                "image_weight_nonzero": int(torch.count_nonzero(weight)),
                "weighted_normal_loss": float(term),
                "depth_valid_pixels": int(torch.count_nonzero(out["depth"] > 0)),
                "alpha_mean": float(out["render_alphas"].mean()),
            }
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
