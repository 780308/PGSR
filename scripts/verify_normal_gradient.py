"""Compare depth-normal loss/gradient at one fixed checkpoint and camera."""

import json

import torch

from pgsr.train import prepare_training
from pgsr.trainer.depth_normal_consistency import DepthNormalConsistencyTrainer


def main() -> None:
    dataset, model, trainer = prepare_training(
        sh_degree=3,
        source="/mnt/e/PGSR/data/truck-320",
        device="cuda",
        mode="base",
        load_ply="/mnt/e/PGSR/outputs/truck-geometry300/point_cloud/iteration_300/point_cloud.ply",
        load_mask=False,
        load_depth=False,
        configs={"depth_normal_consistency_from_iter": 100},
    )
    normal_wrapper = trainer
    while not isinstance(normal_wrapper, DepthNormalConsistencyTrainer):
        normal_wrapper = normal_wrapper.base_trainer
    trainer.curr_step = 101
    camera = dataset[39]
    out = model(camera)
    enabled = trainer.loss(out, camera)
    normal_wrapper.depth_normal_consistency_from_iter = 7000
    disabled = trainer.loss(out, camera)
    isolated = enabled - disabled
    grad = torch.autograd.grad(isolated, model.get_xyz, allow_unused=True)[0]
    print(json.dumps({
        "camera": camera.ground_truth_image_path,
        "step": trainer.curr_step,
        "enabled_loss": float(enabled.detach()),
        "disabled_loss": float(disabled.detach()),
        "isolated_normal_loss": float(isolated.detach()),
        "xyz_grad_norm": float(grad.norm().detach()) if grad is not None else None,
        "xyz_nonzero_gradient_elements": int(torch.count_nonzero(grad)) if grad is not None else 0,
    }, indent=2))


if __name__ == "__main__":
    main()
