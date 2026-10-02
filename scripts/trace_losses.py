"""Opt-in PGSR loss and gradient evidence; patch runtime classes, not upstream code.

Install this before running pgsr.train. It writes selected steps as JSON lines.
Each wrapper's loss includes inner losses; differences isolate each contribution.
"""

import json
import os
from pathlib import Path

import torch


def install(path: str) -> None:
    from gaussian_splatting.trainer import BaseTrainer
    from pgsr.trainer.depth_normal_consistency import DepthNormalConsistencyTrainer
    from pgsr.trainer.multi_view.trainer import MultiViewRegularizationTrainer
    from pgsr.trainer.reprojection import VirtualCameraReprojectionTrainer
    from pgsr.trainer.scale import PlanarScaleTrainer

    classes = (
        (BaseTrainer, "photometric"),
        (PlanarScaleTrainer, "planar_total"),
        (DepthNormalConsistencyTrainer, "normal_total"),
        (MultiViewRegularizationTrainer, "multiview_total"),
        (VirtualCameraReprojectionTrainer, "virtual_total"),
    )
    values = {}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    selected = {1, 20, 99, 100, 101, 200, 300, 500, 1000, 1500}

    def decorator(original, name):
        def wrapped(self, out, camera):
            if name == "photometric":
                values.clear()
            result = original(self, out, camera)
            # DepthNormalConsistencyTrainer uses in-place `loss += normal_loss`.
            # Preserve each intermediate scalar before an outer wrapper mutates it.
            values[name] = result.clone()
            if name != "virtual_total" or int(self.curr_step) not in selected:
                return result

            step = int(self.curr_step)
            chain = [n for _, n in classes]
            components = {}
            previous = None
            for key in chain:
                current = values.get(key)
                if current is None:
                    continue
                components[key] = current if previous is None else current - previous
                previous = current
            record = {
                "step": step,
                "camera": str(camera.ground_truth_image_path),
                "terms": {k: float(v.detach().cpu()) for k, v in components.items()},
                "terms_require_grad": {k: bool(v.requires_grad) for k, v in components.items()},
                "loss": float(result.detach().cpu()),
                "render_keys": sorted(out.keys()),
                "gaussians": int(self.model.get_xyz.shape[0]),
            }
            inner = getattr(self, "base_trainer", None)
            multi = inner if isinstance(inner, MultiViewRegularizationTrainer) else None
            if multi is not None:
                idx = multi.camera_indices.get(camera.ground_truth_image_path)
                record["cached_views"] = sum(c is not None for c in multi.camera_cache)
                record["neighbor_count"] = len(multi.nearest_indices[idx]) if idx is not None else None

            if os.environ.get("PGSR_TRACE_TERM_GRADS") == "1" and step in {101, 200, 300}:
                xyz = self.model.get_xyz
                record["xyz_grad_norms"] = {}
                for key, term in components.items():
                    if not term.requires_grad:
                        record["xyz_grad_norms"][key] = None
                        continue
                    grad = torch.autograd.grad(term, xyz, retain_graph=True, allow_unused=True)[0]
                    record["xyz_grad_norms"][key] = float(grad.norm().detach().cpu()) if grad is not None else None

            with target.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            values.clear()
            return result

        return wrapped

    for cls, name in classes:
        cls.loss = decorator(cls.loss, name)
