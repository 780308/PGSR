#!/usr/bin/env python3
"""Create a CPU-only visual inspection sheet and mesh metrics for a PGSR run.

The figure is a qualitative check for short, non-converged study runs. It is not
an evaluation metric and does not modify the inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
VIEW_SPECS = (
    ("Front · XY", (0, 1, 2)),
    ("Side · ZY", (2, 1, 0)),
    ("Top · XZ", (0, 2, 1)),
)


def _as_path(path: str | Path | None) -> Path | None:
    return Path(path).expanduser().resolve() if path else None


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _read_array(path: Path) -> np.ndarray:
    """Read PNG/JPEG/TIFF without importing a rendering or GPU package."""
    if path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import tifffile

            array = tifffile.imread(path)
        except ImportError:
            array = np.asarray(Image.open(path))
    else:
        with Image.open(path) as image:
            array = np.asarray(image)
    return np.asarray(array)


def _rgb_float(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    raw = _read_array(path)
    if raw.ndim == 3 and raw.shape[0] in (1, 3, 4) and raw.shape[-1] not in (1, 3, 4):
        raw = np.moveaxis(raw, 0, -1)
    if raw.ndim == 2:
        raw = np.repeat(raw[..., None], 3, axis=2)
    if raw.ndim != 3:
        raise ValueError(f"Expected an RGB image, got shape {raw.shape} from {path}")
    if raw.shape[-1] == 1:
        raw = np.repeat(raw, 3, axis=2)
    if raw.shape[-1] >= 4:
        raw = raw[..., :3]
    elif raw.shape[-1] != 3:
        raise ValueError(f"Expected 1, 3, or 4 channels, got shape {raw.shape} from {path}")

    if np.issubdtype(raw.dtype, np.integer):
        max_value = float(np.iinfo(raw.dtype).max)
        # Some 16-bit TIFF writers store 8-bit-range RGB in uint16 containers.
        if max_value > 255 and raw.size and int(np.nanmax(raw)) <= 255:
            max_value = 255.0
        rgb = raw.astype(np.float32) / max_value
    else:
        rgb = raw.astype(np.float32)
        finite = rgb[np.isfinite(rgb)]
        if finite.size and float(finite.max()) > 1.0:
            scale = 255.0 if float(finite.max()) <= 255.0 else float(finite.max())
            rgb /= scale
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb, {"path": str(path), "shape": list(raw.shape), "dtype": str(raw.dtype)}


def _scalar_image(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    raw = _read_array(path)
    if raw.ndim == 3 and raw.shape[0] == 1:
        raw = raw[0]
    if raw.ndim == 3 and raw.shape[-1] == 1:
        raw = raw[..., 0]
    if raw.ndim == 3:
        # Geometry exports can be RGB encoded; scalar depth TIFFs are normally HxW.
        raw = raw[..., 0]
    if raw.ndim != 2:
        raise ValueError(f"Expected a scalar image, got shape {raw.shape} from {path}")
    values = raw.astype(np.float32, copy=False)
    finite = np.isfinite(values)
    positive = finite & (values > 0)
    if positive.any():
        low, high = np.percentile(values[positive], [2, 98]).astype(float).tolist()
        if not math.isfinite(high) or high <= low:
            low = float(np.min(values[positive]))
            high = float(np.max(values[positive]))
    else:
        low = high = None
    return values, {
        "path": str(path),
        "shape": list(raw.shape),
        "dtype": str(raw.dtype),
        "finite_pixels": int(finite.sum()),
        "positive_finite_pixels": int(positive.sum()),
        "display_percentiles_2_98": [low, high],
    }


def _discover_images(root: Path, exclude: Path | None = None) -> list[Path]:
    if not root.is_dir():
        return []
    found = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if exclude is not None and _is_relative_to(path, exclude):
            continue
        found.append(path.resolve())
    return sorted(set(found), key=lambda p: str(p).casefold())


def _classify_images(paths: list[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {k: [] for k in ("render", "gt", "invdepth", "normal", "depth")}
    for path in paths:
        stem = path.stem.casefold()
        parent_names = [part.casefold() for part in path.parts[:-1]]
        if "invdepth" in stem or "invdepth" in path.parent.name.casefold():
            groups["invdepth"].append(path)
        elif "normal" in stem or "normal" in path.parent.name.casefold():
            groups["normal"].append(path)
        elif any(name in {"gt", "gts", "ground_truth", "ground-truth"} for name in parent_names):
            groups["gt"].append(path)
        elif "depth" in stem or "depth" in path.parent.name.casefold():
            groups["depth"].append(path)
        elif any(name in {"render", "renders", "prediction", "predictions"} for name in parent_names):
            groups["render"].append(path)
    return groups


def _image_key(path: Path, category: str) -> str:
    key = path.stem.casefold()
    for suffix in ("_invdepth", "_normal", "_depth"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
    if category == "invdepth":
        key = key.removesuffix("_invdepth")
    return key


def _matches_view(path: Path, view: str, category: str) -> bool:
    if path.stem.casefold() == view.casefold():
        return True
    if _image_key(path, category) == view.casefold():
        return True
    # Geometry exports are often named normal_00000.png or depth_00000.tiff.
    return re.search(rf"(^|\D){re.escape(view.casefold())}($|\D)", path.stem.casefold()) is not None


def _choose_image(
    paths: list[Path], view: str | None, category: str, explicit: Path | None = None,
    allow_single_fallback: bool = False,
) -> Path | None:
    if explicit is not None:
        return explicit if explicit.is_file() else None
    if not paths:
        return None
    if view is not None:
        matches = [p for p in paths if _matches_view(p, view, category)]
        if matches:
            return matches[0]
        # Numeric selectors are convenient when source files are named 00000, etc.
        if view.isdigit():
            matches = [p for p in paths if _image_key(p, category).isdigit() and int(_image_key(p, category)) == int(view)]
            if matches:
                return matches[0]
        if allow_single_fallback and len(paths) == 1:
            return paths[0]
        return None
    return paths[0]


def _rgb_statistics(render: np.ndarray | None, gt: np.ndarray | None) -> dict[str, Any] | None:
    if render is None or gt is None:
        return None
    if render.shape != gt.shape:
        return {"compatible_shapes": False, "render_shape": list(render.shape), "gt_shape": list(gt.shape)}
    diff = np.abs(render - gt)
    mae = float(diff.mean())
    rmse = float(np.sqrt(np.mean((render - gt) ** 2)))
    psnr = float("inf") if rmse == 0 else float(-20.0 * np.log10(rmse))
    return {"compatible_shapes": True, "mae_0_1": mae, "rmse_0_1": rmse, "psnr_db_selected_view": psnr}


def _mesh_components(mesh: Any, face_count: int, face_limit: int) -> tuple[int | None, str | None]:
    if face_count == 0:
        return 0, None
    if face_count > face_limit:
        return None, f"skipped above safety limit ({face_count} > {face_limit} faces)"
    try:
        # trimesh uses face-edge adjacency; this stays CPU-only and avoids Open3D's renderer.
        chunks = mesh.split(only_watertight=False)
        return len(chunks), None
    except Exception as exc:  # connectivity is supplementary; retain core metrics on failure
        return None, f"{type(exc).__name__}: {exc}"


def _load_mesh(path: Path, max_triangles: int, component_face_limit: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    entry: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    empty_views = [{"polygons": [], "colors": np.empty((0, 4), dtype=np.float32)} for _ in VIEW_SPECS]
    if not path.is_file():
        entry.update({"vertices": 0, "triangles": 0, "finite_vertices": 0, "all_vertex_coordinates_finite": False, "error": "mesh file not found"})
        return entry, empty_views

    try:
        import trimesh

        mesh = trimesh.load(path, force="mesh", process=False)
    except Exception as exc:
        entry.update({"vertices": 0, "triangles": 0, "finite_vertices": 0, "all_vertex_coordinates_finite": False, "error": f"{type(exc).__name__}: {exc}"})
        return entry, empty_views

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if faces.ndim != 2 or (faces.size and faces.shape[1] != 3):
        entry.update({"vertices": int(len(vertices)), "triangles": 0, "finite_vertices": int(np.isfinite(vertices).all(axis=1).sum()) if vertices.ndim == 2 else 0, "all_vertex_coordinates_finite": False, "error": f"expected triangle faces, got {faces.shape}"})
        return entry, empty_views
    finite_vertex_mask = np.isfinite(vertices).all(axis=1) if vertices.ndim == 2 else np.zeros(0, dtype=bool)
    finite_vertex_count = int(finite_vertex_mask.sum())
    in_range = np.ones(len(faces), dtype=bool)
    if len(faces):
        in_range = ((faces >= 0) & (faces < len(vertices))).all(axis=1)
    face_vertices_finite = np.zeros(len(faces), dtype=bool)
    valid_face_ids = np.flatnonzero(in_range)
    if valid_face_ids.size:
        face_vertices_finite[valid_face_ids] = finite_vertex_mask[faces[valid_face_ids]].all(axis=1)
    render_face_ids = np.flatnonzero(in_range & face_vertices_finite)

    entry.update({
        "vertices": int(len(vertices)),
        "triangles": int(len(faces)),
        "finite_vertices": finite_vertex_count,
        "finite_vertex_fraction": float(finite_vertex_count / len(vertices)) if len(vertices) else 0.0,
        "all_vertex_coordinates_finite": bool(finite_vertex_count == len(vertices)),
        "finite_coordinates": int(np.isfinite(vertices).sum()) if vertices.ndim == 2 else 0,
        "triangles_with_valid_indices": int(in_range.sum()),
        "triangles_with_finite_vertices": int((in_range & face_vertices_finite).sum()),
    })
    if finite_vertex_count:
        finite_vertices = vertices[finite_vertex_mask]
        bounds_min = finite_vertices.min(axis=0)
        bounds_max = finite_vertices.max(axis=0)
        entry["bounds_min"] = [float(v) for v in bounds_min]
        entry["bounds_max"] = [float(v) for v in bounds_max]
        entry["bounds_diagonal"] = float(np.linalg.norm(bounds_max - bounds_min))
    else:
        finite_vertices = np.empty((0, 3), dtype=np.float64)
    component_count, component_note = _mesh_components(mesh, len(faces), component_face_limit)
    entry["connected_components"] = component_count
    if component_note:
        entry["connected_components_note"] = component_note

    colors: np.ndarray | None = None
    try:
        if getattr(mesh.visual, "kind", None) == "vertex":
            vertex_colors = np.asarray(mesh.visual.vertex_colors)
            if vertex_colors.ndim == 2 and len(vertex_colors) == len(vertices) and vertex_colors.shape[1] >= 3:
                colors = vertex_colors[:, :4].astype(np.float32)
                if colors.size and colors.max() > 1.0:
                    colors /= 255.0
                if colors.shape[1] == 3:
                    colors = np.column_stack((colors, np.ones(len(colors), dtype=np.float32)))
    except Exception:
        colors = None

    if render_face_ids.size > max_triangles:
        # Evenly spaced face IDs give repeatable views while bounding plot memory.
        positions = np.linspace(0, len(render_face_ids) - 1, max_triangles, dtype=np.int64)
        render_face_ids = render_face_ids[positions]
    selected_faces = faces[render_face_ids].astype(np.int64, copy=False)
    if len(selected_faces):
        triangles = vertices[selected_faces].astype(np.float64, copy=False)
        edge_a = triangles[:, 1] - triangles[:, 0]
        edge_b = triangles[:, 2] - triangles[:, 0]
        normals = np.cross(edge_a, edge_b)
        normal_lengths = np.linalg.norm(normals, axis=1)
        normals /= np.maximum(normal_lengths[:, None], 1e-12)
        if colors is not None:
            base_colors = colors[selected_faces].mean(axis=1)
        else:
            base_colors = np.tile(np.array([[0.64, 0.72, 0.79, 1.0]], dtype=np.float32), (len(triangles), 1))
        light = np.array([0.38, 0.73, 0.57], dtype=np.float64)
        light /= np.linalg.norm(light)
        shade = 0.38 + 0.62 * np.abs(normals @ light)
        shaded_colors = base_colors.copy()
        shaded_colors[:, :3] *= shade[:, None]
        shaded_colors[:, :3] = np.clip(shaded_colors[:, :3], 0, 1)
        shaded_colors[:, 3] = np.clip(shaded_colors[:, 3], 0.25, 1.0)
    else:
        triangles = np.empty((0, 3, 3), dtype=np.float64)
        shaded_colors = np.empty((0, 4), dtype=np.float32)

    projected_views = []
    for _label, (horizontal, vertical, depth_axis) in VIEW_SPECS:
        if len(triangles):
            projected = triangles[:, :, (horizontal, vertical)]
            depth = triangles[:, :, depth_axis].mean(axis=1)
            order = np.argsort(depth, kind="stable")
            polygons = projected[order]
            view_colors = shaded_colors[order]
        else:
            polygons = np.empty((0, 3, 2), dtype=np.float64)
            view_colors = np.empty((0, 4), dtype=np.float32)
        projected_views.append({"polygons": polygons, "colors": view_colors})

    del mesh, vertices, faces, triangles, selected_faces
    return entry, projected_views


def _draw_rgb_panel(ax: Any, image: np.ndarray | None, title: str, missing: str) -> None:
    ax.set_title(title, fontsize=10, pad=7)
    ax.set_axis_off()
    if image is None:
        ax.set_facecolor("#eceff3")
        ax.text(0.5, 0.5, missing, ha="center", va="center", transform=ax.transAxes, color="#515a66", wrap=True)
    else:
        ax.imshow(image, interpolation="nearest")


def _draw_scalar_panel(ax: Any, image: np.ndarray | None, title: str, missing: str) -> None:
    ax.set_title(title, fontsize=10, pad=7)
    ax.set_axis_off()
    if image is None:
        ax.set_facecolor("#eceff3")
        ax.text(0.5, 0.5, missing, ha="center", va="center", transform=ax.transAxes, color="#515a66", wrap=True)
        return
    valid = np.isfinite(image) & (image > 0)
    if valid.any():
        low, high = np.percentile(image[valid], [2, 98])
        if high <= low:
            low, high = float(image[valid].min()), float(image[valid].max())
        if high <= low:
            low, high = low - 0.5, high + 0.5
        display = np.ma.masked_where(~valid, image)
        cmap = plt.get_cmap("turbo").copy()
        cmap.set_bad("#202833")
        ax.imshow(display, cmap=cmap, vmin=low, vmax=high, interpolation="nearest")
    else:
        ax.imshow(np.zeros_like(image), cmap="gray", interpolation="nearest")


def _draw_mesh_panel(ax: Any, view: dict[str, Any], mesh_entry: dict[str, Any], spec: tuple[str, tuple[int, int, int]]) -> None:
    view_label, (horizontal, vertical, _depth_axis) = spec
    ax.set_title(view_label, fontsize=9, pad=5)
    ax.set_facecolor("#f2f4f7")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    polygons = view["polygons"]
    if len(polygons):
        collection = PolyCollection(polygons, facecolors=view["colors"], edgecolors="none", linewidths=0, antialiaseds=False)
        ax.add_collection(collection)
        bounds_min = mesh_entry.get("bounds_min")
        bounds_max = mesh_entry.get("bounds_max")
        if bounds_min and bounds_max:
            xlo, xhi = bounds_min[horizontal], bounds_max[horizontal]
            ylo, yhi = bounds_min[vertical], bounds_max[vertical]
            span = max(xhi - xlo, yhi - ylo, 1e-6)
            margin = 0.04 * span
            ax.set_xlim(xlo - margin, xhi + margin)
            ax.set_ylim(ylo - margin, yhi + margin)
            ax.set_aspect("equal", adjustable="box")
    else:
        message = mesh_entry.get("error", "no finite triangle data")
        ax.text(0.5, 0.5, str(message), ha="center", va="center", transform=ax.transAxes, fontsize=8, wrap=True)
    if not mesh_entry.get("exists"):
        ax.text(0.5, 0.08, "MISSING", ha="center", va="bottom", transform=ax.transAxes, color="#b42318", fontweight="bold")


def _write_figure(
    png_path: Path,
    run_name: str,
    view_name: str,
    render_rgb: np.ndarray | None,
    gt_rgb: np.ndarray | None,
    invdepth: np.ndarray | None,
    normal_rgb: np.ndarray | None,
    extra_depth: np.ndarray | None,
    mesh_payloads: list[tuple[dict[str, Any], list[dict[str, Any]]]],
) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(15.5, 15.2), constrained_layout=True)
    fig.suptitle(
        f"{run_name} · selected view {view_name}\nShort run · non-converged · qualitative visual acceptance only",
        fontsize=14,
        fontweight="bold",
    )
    _draw_rgb_panel(axes[0, 0], render_rgb, "Rendered RGB", "Render image not found")
    _draw_rgb_panel(axes[0, 1], gt_rgb, "Ground-truth RGB", "GT image not found")
    if render_rgb is not None and gt_rgb is not None and render_rgb.shape == gt_rgb.shape:
        _draw_rgb_panel(axes[0, 2], np.abs(render_rgb - gt_rgb), "Absolute RGB difference", "")
    else:
        _draw_rgb_panel(axes[0, 2], None, "Absolute RGB difference", "Unavailable: RGB inputs missing or shapes differ")
    _draw_scalar_panel(axes[1, 0], invdepth, "Rendered inverse depth · normalized 2–98%", "Invdepth TIFF not found")
    _draw_rgb_panel(
        axes[1, 1], normal_rgb, "PGSR normal sample", "Normal not included\n(standard render saves RGB + invdepth only)"
    )
    _draw_scalar_panel(axes[1, 2], extra_depth, "Optional geometry depth sample", "No additional depth sample\n(invdepth heatmap is shown at left)")

    for row, (mesh_entry, views) in enumerate(mesh_payloads, start=2):
        mesh_name = Path(mesh_entry["path"]).name
        mesh_entry_title = f"{mesh_name} · {mesh_entry.get('vertices', 0):,} vertices · {mesh_entry.get('triangles', 0):,} triangles"
        axes[row, 0].text(-0.08, 1.18, mesh_entry_title, transform=axes[row, 0].transAxes, fontsize=10, fontweight="bold", ha="left", va="bottom")
        for column, spec in enumerate(VIEW_SPECS):
            _draw_mesh_panel(axes[row, column], views[column], mesh_entry, spec)
    fig.savefig(png_path, dpi=135, facecolor="white")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/truck-densify1500/ours_1500"), help="Render/mesh output directory (default: outputs/truck-densify1500/ours_1500).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Artifact directory; defaults to RUN_DIR/acceptance.")
    parser.add_argument("--view", default=None, help="Image stem such as 00000; defaults to the first sorted rendered frame.")
    parser.add_argument("--render-image", type=Path, default=None, help="Explicit rendered RGB image path.")
    parser.add_argument("--gt-image", type=Path, default=None, help="Explicit ground-truth RGB image path.")
    parser.add_argument("--invdepth-image", type=Path, default=None, help="Explicit rendered inverse-depth TIFF/image path.")
    parser.add_argument("--geometry-dir", type=Path, default=None, help="Optional directory with PGSR normal/depth sample PNG or TIFF images.")
    parser.add_argument("--fuse-mesh", type=Path, default=None, help="Override path to fuse.ply.")
    parser.add_argument("--post-mesh", type=Path, default=None, help="Override path to fuse_post.ply.")
    parser.add_argument("--max-mesh-triangles", type=int, default=20000, help="Maximum triangles per mesh used in the contact sheet (metrics still inspect all faces).")
    parser.add_argument("--component-face-limit", type=int, default=500000, help="Skip connected-component computation above this many faces.")
    parser.add_argument("--label", default="PGSR Truck iteration 1500", help="Display label; does not claim convergence or benchmark status.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.max_mesh_triangles < 1 or args.component_face_limit < 0:
        raise SystemExit("--max-mesh-triangles must be >= 1 and --component-face-limit must be >= 0")
    run_dir = _as_path(args.run_dir)
    assert run_dir is not None
    if not run_dir.exists():
        raise SystemExit(f"Run directory does not exist: {run_dir}")
    # Accept the model directory as shorthand when its ours_<iteration> child exists.
    if not any((run_dir / name).exists() for name in ("renders", "gt", "fuse.ply", "fuse_post.ply")):
        candidates = sorted(run_dir.glob("ours_*"))
        if len(candidates) == 1 and candidates[0].is_dir():
            run_dir = candidates[0].resolve()
    output_dir = _as_path(args.output_dir) if args.output_dir else run_dir / "acceptance"
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    discovered = _classify_images(_discover_images(run_dir, exclude=output_dir))
    geometry_dir = _as_path(args.geometry_dir)
    geometry = _classify_images(_discover_images(geometry_dir, exclude=output_dir)) if geometry_dir else {k: [] for k in ("normal", "depth")}
    render_path = _choose_image(discovered["render"], args.view, "render", _as_path(args.render_image))
    if args.view is None and render_path is not None:
        selected_view = _image_key(render_path, "render")
    else:
        selected_view = args.view or (_image_key(render_path, "render") if render_path else "unselected")
    gt_path = _choose_image(discovered["gt"], selected_view, "gt", _as_path(args.gt_image))
    invdepth_path = _choose_image(discovered["invdepth"], selected_view, "invdepth", _as_path(args.invdepth_image))
    normal_path = _choose_image(geometry["normal"], selected_view, "normal", allow_single_fallback=True)
    extra_depth_path = _choose_image(geometry["depth"], selected_view, "depth", allow_single_fallback=True)

    render_rgb = gt_rgb = normal_rgb = None
    invdepth = extra_depth = None
    image_metrics: dict[str, Any] = {}
    for key, path, loader in (
        ("render", render_path, _rgb_float),
        ("gt", gt_path, _rgb_float),
        ("invdepth", invdepth_path, _scalar_image),
        ("normal", normal_path, _rgb_float),
        ("depth", extra_depth_path, _scalar_image),
    ):
        if path is None:
            image_metrics[key] = {"available": False}
            continue
        try:
            loaded, details = loader(path)
            details["available"] = True
            image_metrics[key] = details
            if key == "render":
                render_rgb = loaded
            elif key == "gt":
                gt_rgb = loaded
            elif key == "invdepth":
                invdepth = loaded
            elif key == "normal":
                normal_rgb = loaded
            else:
                extra_depth = loaded
        except Exception as exc:
            image_metrics[key] = {"available": False, "path": str(path), "error": f"{type(exc).__name__}: {exc}"}

    fuse_path = _as_path(args.fuse_mesh) or (run_dir / "fuse.ply")
    post_path = _as_path(args.post_mesh) or (run_dir / "fuse_post.ply")
    mesh_payloads = [
        _load_mesh(fuse_path, args.max_mesh_triangles, args.component_face_limit),
        _load_mesh(post_path, args.max_mesh_triangles, args.component_face_limit),
    ]
    mesh_entries = [entry for entry, _views in mesh_payloads]
    png_path = output_dir / "visual_acceptance.png"
    json_path = output_dir / "visual_acceptance.json"
    _write_figure(
        png_path=png_path,
        run_name=args.label,
        view_name=selected_view,
        render_rgb=render_rgb,
        gt_rgb=gt_rgb,
        invdepth=invdepth,
        normal_rgb=normal_rgb,
        extra_depth=extra_depth,
        mesh_payloads=mesh_payloads,
    )
    rgb_stats = _rgb_statistics(render_rgb, gt_rgb)
    all_core_images = all(image_metrics.get(k, {}).get("available", False) for k in ("render", "gt", "invdepth"))
    both_meshes_valid = all(
        entry.get("exists")
        and entry.get("vertices", 0) > 0
        and entry.get("triangles_with_finite_vertices", 0) > 0
        and entry.get("all_vertex_coordinates_finite")
        for entry in mesh_entries
    )
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "purpose": "Qualitative visual inspection of a short, non-converged run; this is not a benchmark result.",
        "converged": False,
        "benchmark_result": False,
        "status": "ready" if all_core_images and both_meshes_valid else "incomplete",
        "run_directory": str(run_dir),
        "selected_view": selected_view,
        "artifacts": {"contact_sheet_png": str(png_path), "metrics_json": str(json_path)},
        "images": image_metrics,
        "rgb_comparison": rgb_stats,
        "normal_note": "PGSR standard rendering writes RGB and inverse-depth TIFF only; an optional normal sample is included only when found under --geometry-dir.",
        "meshes": {Path(fuse_path).name: mesh_entries[0], Path(post_path).name: mesh_entries[1]},
        "visualization": {
            "mesh_views": [name for name, _axes in VIEW_SPECS],
            "maximum_triangles_drawn_per_mesh": args.max_mesh_triangles,
            "connected_component_face_limit": args.component_face_limit,
            "projection": "CPU orthographic 2D painter projection with sampled faces; mesh metrics inspect full loaded arrays.",
        },
    }
    # Keep JSON standards-compliant if an exact-match RGB pair produces infinite PSNR.
    def clean_nonfinite(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: clean_nonfinite(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean_nonfinite(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    json_path.write_text(json.dumps(clean_nonfinite(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "contact_sheet_png": str(png_path), "metrics_json": str(json_path), "selected_view": selected_view}, ensure_ascii=False))
    return 0 if payload["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
