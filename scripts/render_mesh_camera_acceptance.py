#!/usr/bin/env python3
"""使用已注册的 COLMAP 相机对 PGSR 网格进行 CPU 射线投射，供视觉检查。

对每个指定的 COLMAP 图像，将针孔相机射线投射到导出的三角网格，利用
命中三角形的重心坐标插值 PLY 顶点颜色，并保存颜色图、命中掩码、相机 Z
深度和世界坐标命中点。这项检查用于定性观察网格与图像的对齐情况，不是基准评测。
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 在学习工作站上限制 Open3D 的 CPU 工作线程数量。
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "outputs/truck-densify1500/ours_1500"
DEFAULT_DATASET = ROOT / "data/truck-320"
DEFAULT_MESH = DEFAULT_RUN_DIR / "fuse_post.ply"
DEFAULT_OUTPUT = DEFAULT_RUN_DIR / "acceptance/mesh_camera"


def _load_colmap_reader() -> Any:
    """直接加载本地 COLMAP 二进制读取器，避免导入 scene/__init__.py。"""
    source = ROOT / "scene/colmap_loader.py"
    spec = importlib.util.spec_from_file_location("pgsr_colmap_loader", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load COLMAP reader: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_registered_cameras(dataset: Path) -> tuple[dict[str, Any], dict[int, Any]]:
    """读取 COLMAP 注册图像和相机参数，并按图像文件名建立索引。"""
    model_dir = dataset / "sparse/0"
    camera_bin = model_dir / "cameras.bin"
    image_bin = model_dir / "images.bin"
    if not camera_bin.is_file() or not image_bin.is_file():
        raise FileNotFoundError(f"Expected COLMAP binary model under {model_dir}")
    reader = _load_colmap_reader()
    cameras = reader.read_intrinsics_binary(str(camera_bin))
    images = reader.read_extrinsics_binary(str(image_bin))
    by_name = {image.name: image for image in images.values()}
    if len(by_name) != len(images):
        raise ValueError("COLMAP images.bin contains duplicate image names")
    return by_name, cameras


def _camera_intrinsics(camera: Any) -> tuple[float, float, float, float]:
    """返回无畸变针孔相机的 fx、fy、cx、cy；不支持的模型会明确报错。"""
    p = np.asarray(camera.params, dtype=np.float64)
    if camera.model == "PINHOLE" and p.size == 4:
        fx, fy, cx, cy = p
    elif camera.model == "SIMPLE_PINHOLE" and p.size == 3:
        fx = fy = p[0]
        cx, cy = p[1:]
    else:
        raise ValueError(
            f"Camera model {camera.model} is not an undistorted pinhole model; "
            "this script will not silently ignore lens distortion"
        )
    if not np.isfinite([fx, fy, cx, cy]).all() or fx <= 0 or fy <= 0:
        raise ValueError(f"Invalid pinhole intrinsics: {(fx, fy, cx, cy)}")
    return float(fx), float(fy), float(cx), float(cy)


def _load_run_mapping(run_dir: Path) -> dict[str, dict[str, Any]]:
    """从训练输出的 cameras.json 建立 COLMAP 图像到输出序号的映射。"""
    mapping_path = run_dir / "cameras.json"
    if not mapping_path.is_file():
        return {}
    rows = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list in {mapping_path}")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = row.get("img_name")
        if name:
            if name in result:
                raise ValueError(f"Duplicate img_name {name!r} in {mapping_path}")
            result[name] = row
    return result


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(path)


def _save_depth_visualization(path: Path, depth: np.ndarray, hit: np.ndarray) -> list[float] | None:
    """保存相机 Z 深度的灰度预览，并返回有效深度的 2%/98% 显示范围。"""
    values = depth[hit & np.isfinite(depth)]
    if values.size == 0:
        Image.fromarray(np.zeros(depth.shape, dtype=np.uint8), mode="L").save(path)
        return None
    low, high = np.percentile(values, [2.0, 98.0]).astype(float).tolist()
    if high <= low:
        low = float(values.min())
        high = float(values.max())
    if high <= low:
        scaled = np.zeros(depth.shape, dtype=np.uint8)
    else:
        scaled = np.zeros(depth.shape, dtype=np.float32)
        scaled[hit] = np.clip((depth[hit] - low) / (high - low), 0.0, 1.0)
        # 近处显示为亮色，远处显示为暗色；未命中像素保持黑色。
        scaled = np.round((1.0 - scaled) * 255.0).astype(np.uint8)
    scaled[~hit] = 0
    Image.fromarray(scaled, mode="L").save(path)
    return [float(low), float(high)]


def _image_metrics(a: np.ndarray | None, b: np.ndarray | None) -> dict[str, Any] | None:
    """计算两张 RGB 图像的 MAE、RMSE 和 PSNR；缺图或尺寸不同则保留说明。"""
    if a is None or b is None:
        return None
    if a.shape != b.shape:
        return {"compatible_shapes": False, "a_shape": list(a.shape), "b_shape": list(b.shape)}
    delta = (a.astype(np.float32) - b.astype(np.float32)) / 255.0
    rmse = float(np.sqrt(np.mean(delta * delta)))
    return {
        "compatible_shapes": True,
        "mae_0_1": float(np.mean(np.abs(delta))),
        "rmse_0_1": rmse,
        "psnr_db": "infinity" if rmse == 0 else float(-20.0 * math.log10(rmse)),
    }


def _rss_peak_mb() -> float:
    """读取当前进程的峰值常驻内存，并转换为 MiB。"""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux/WSL 返回 KiB；macOS 返回字节数。
    return float(peak / 1024.0 if sys.platform.startswith("linux") else peak / (1024.0 * 1024.0))


def _make_contact_sheet(
    path: Path,
    title: str,
    panels: list[tuple[str, np.ndarray | None]],
    width: int,
    height: int,
) -> None:
    """按输入顺序拼接图像面板；缺失面板明确标为 MISSING。"""
    pad = 10
    label_h = 28
    title_h = 32
    sheet = Image.new("RGB", (pad + len(panels) * (width + pad), title_h + height + label_h + 2 * pad), (25, 25, 25))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
        title_font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
        title_font = font
    draw.text((pad, 7), title, fill=(245, 245, 245), font=title_font)
    y0 = title_h
    for i, (label, array) in enumerate(panels):
        x = pad + i * (width + pad)
        draw.text((x, y0), label, fill=(235, 235, 235), font=font)
        y = y0 + label_h
        if array is None:
            draw.rectangle((x, y, x + width - 1, y + height - 1), fill=(64, 32, 32))
            draw.text((x + 8, y + height // 2 - 8), "MISSING", fill=(255, 210, 210), font=font)
        else:
            if array.shape[:2] != (height, width):
                panel = Image.fromarray(array).resize((width, height), Image.Resampling.BILINEAR)
            else:
                panel = Image.fromarray(array)
            if panel.mode != "RGB":
                panel = panel.convert("RGB")
            sheet.paste(panel, (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _make_rays(
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    R_world_to_camera: np.ndarray,
    t_world_to_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按 COLMAP 针孔模型生成射线原点、世界系单位方向和相机系方向。

    COLMAP 相机轴约定为 +x 向右、+y 向下、+z 向前；(u,v) 使用左上角
    为原点的图像坐标。输入内参应属于无畸变针孔模型。
    """
    # COLMAP 相机轴为 +x 向右、+y 向下、+z 向前；(u,v) 使用左上角为原点的无畸变图像坐标。
    vv, uu = np.mgrid[0:height, 0:width]
    dirs_camera = np.stack(((uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)), axis=-1).astype(np.float32)
    norms = np.linalg.norm(dirs_camera, axis=-1, keepdims=True)
    dirs_camera /= norms
    # 世界到相机变换为 x_cam = R x_world + t；按行存储时，方向向量用 d_cam @ R 转回世界系。
    dirs_world = dirs_camera.reshape(-1, 3) @ R_world_to_camera.astype(np.float32)
    dirs_world /= np.linalg.norm(dirs_world, axis=1, keepdims=True)
    center_world = -R_world_to_camera.T @ t_world_to_camera
    origins = np.broadcast_to(center_world.astype(np.float32), dirs_world.shape).copy()
    return origins, dirs_world, dirs_camera.reshape(-1, 3)


def _cast_view(
    scene: Any,
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_colors: np.ndarray,
    camera: Any,
    image: Any,
) -> dict[str, np.ndarray | float | int]:
    """从一个 COLMAP 视角投射网格，并生成颜色、掩码、深度和命中点。

    位姿按 x_camera = R_world_to_camera @ x_world + t 解释。Open3D 返回的
    primitive_uvs 表示三角形第 1、2 个顶点的重心权重，第 0 个顶点权重为
    1-u-v；命中后的网格颜色由这三个顶点色加权插值得到。
    """
    fx, fy, cx, cy = _camera_intrinsics(camera)
    width, height = int(camera.width), int(camera.height)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid camera dimensions: {width}x{height}")
    R = np.asarray(image.qvec2rotmat(), dtype=np.float64)
    t = np.asarray(image.tvec, dtype=np.float64)
    if R.shape != (3, 3) or t.shape != (3,) or not np.isfinite(R).all() or not np.isfinite(t).all():
        raise ValueError("Non-finite or malformed COLMAP world-to-camera pose")
    if not np.allclose(R @ R.T, np.eye(3), atol=2e-5) or np.linalg.det(R) < 0.999:
        raise ValueError("COLMAP rotation is not a proper world-to-camera rotation")
    origins, dirs_world, dirs_camera_unit = _make_rays(width, height, fx, fy, cx, cy, R, t)
    rays = np.empty((len(origins), 6), dtype=np.float32)
    rays[:, :3] = origins
    rays[:, 3:] = dirs_world
    hit_result = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
    primitive_ids = hit_result["primitive_ids"].numpy().reshape(-1).astype(np.int64, copy=False)
    t_hit = hit_result["t_hit"].numpy().reshape(-1).astype(np.float32, copy=False)
    uv = hit_result["primitive_uvs"].numpy().reshape(-1, 2).astype(np.float32, copy=False)
    hit = np.isfinite(t_hit) & (t_hit >= 0.0) & (primitive_ids >= 0) & (primitive_ids < len(faces))

    color = np.zeros((height * width, 3), dtype=np.float32)
    hit_ids = np.flatnonzero(hit)
    if hit_ids.size:
        tri = faces[primitive_ids[hit_ids]].astype(np.int64, copy=False)
        u = uv[hit_ids, 0]
        v = uv[hit_ids, 1]
        w0 = np.clip(1.0 - u - v, 0.0, 1.0)
        # Open3D primitive_uvs 是三角形第 1、2 个顶点的重心权重；第 0 个顶点权重为 1-u-v。
        w1 = np.clip(u, 0.0, 1.0)
        w2 = np.clip(v, 0.0, 1.0)
        weight_sum = w0 + w1 + w2
        if np.any(weight_sum <= 0):
            raise RuntimeError("Open3D returned invalid barycentric coordinates")
        bary = np.stack((w0, w1, w2), axis=1) / weight_sum[:, None]
        color[hit_ids] = np.einsum("ni,nij->nj", bary, vertex_colors[tri], optimize=True)

    # 世界系射线方向已归一化，因此 Open3D 的 t_hit 是沿射线的欧氏距离。
    # 乘以相机系单位射线的 z 分量后，得到相机坐标系中的 Z 深度。
    ray_z = dirs_camera_unit[:, 2]
    depth_z = np.full(height * width, np.nan, dtype=np.float32)
    depth_z[hit] = t_hit[hit] * ray_z[hit]
    world_points = np.full((height * width, 3), np.nan, dtype=np.float32)
    world_points[hit] = origins[hit] + dirs_world[hit] * t_hit[hit, None]

    # 数值一致性检查用于发现误把位姿当作相机到世界变换、或轴向/符号写错的情况：
    # 将命中点变换到相机系后，其 Z 值必须与保存的相机 Z 深度一致。
    points_cam = world_points[hit].astype(np.float64) @ R.T + t
    if points_cam.size:
        max_z_error = float(np.max(np.abs(points_cam[:, 2] - depth_z[hit])))
        if max_z_error > 2e-3:
            raise RuntimeError(f"Camera-depth transform inconsistency: max error {max_z_error:.6g}")
    else:
        max_z_error = 0.0

    return {
        "color": np.clip(np.round(color.reshape(height, width, 3) * 255.0), 0, 255).astype(np.uint8),
        "hit": hit.reshape(height, width),
        "depth_camera_z": depth_z.reshape(height, width),
        "world_points": world_points.reshape(height, width, 3),
        "hit_count": int(hit.sum()),
        "pixel_count": int(height * width),
        "depth_transform_max_abs_error": max_z_error,
        "world_to_camera_rotation": R,
        "world_to_camera_translation": t,
        "camera_center_world": -R.T @ t,
        "intrinsics": [fx, fy, cx, cy],
    }


def main() -> None:
    """读取网格与 COLMAP 模型，投射请求视角并写出图像、映射表和 JSON 摘要。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image", dest="images", action="append", default=None,
                        help="registered COLMAP image name; repeat to render several views (default: 000001.jpg)")
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    run_dir = args.run_dir.resolve()
    mesh_path = args.mesh.resolve()
    output = args.output.resolve()
    requested = args.images or ["000001.jpg"]
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh does not exist: {mesh_path}")
    output.mkdir(parents=True, exist_ok=True)

    image_by_name, camera_by_id = _read_registered_cameras(dataset)
    unknown = [name for name in requested if name not in image_by_name]
    if unknown:
        raise KeyError(f"Requested images are not registered in COLMAP: {unknown}")
    run_mapping = _load_run_mapping(run_dir)

    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    colors = np.asarray(mesh.vertex_colors, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError(f"Mesh has no valid vertices: {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError(f"Mesh has no valid triangle faces: {faces.shape}")
    if colors.shape != vertices.shape:
        raise ValueError(
            f"Expected one RGB vertex color per mesh vertex; got {colors.shape} for {vertices.shape}. "
            "Refusing to invent a mesh color"
        )
    if not np.isfinite(vertices).all() or not np.isfinite(colors).all():
        raise ValueError("Mesh has non-finite vertex coordinates or colors")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Mesh contains triangle indices outside the vertex array")
    if not len(mesh.vertex_colors):
        raise ValueError("PLY contains no vertex colors; refusing to fabricate a vertex-color render")

    print(f"Loading CPU raycasting scene: {len(vertices):,} vertices, {len(faces):,} triangles", flush=True)
    tensor_mesh = o3d.t.geometry.TriangleMesh()
    tensor_mesh.vertex.positions = o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32)
    tensor_mesh.triangle.indices = o3d.core.Tensor(faces.astype(np.uint32, copy=False), dtype=o3d.core.Dtype.UInt32)
    scene = o3d.t.geometry.RaycastingScene(nthreads=2)
    geometry_id = scene.add_triangles(tensor_mesh)
    if geometry_id < 0:
        raise RuntimeError("Open3D did not add the mesh to RaycastingScene")

    records: list[dict[str, Any]] = []
    for image_name in requested:
        colmap_image = image_by_name[image_name]
        camera = camera_by_id[colmap_image.camera_id]
        frame = _cast_view(scene, vertices, faces, colors, camera, colmap_image)
        width, height = int(camera.width), int(camera.height)
        frame_dir = output / Path(image_name).stem
        frame_dir.mkdir(parents=True, exist_ok=True)
        source_path = dataset / "images" / image_name
        if not source_path.is_file():
            raise FileNotFoundError(f"COLMAP source image is missing: {source_path}")
        source_rgb = _read_rgb(source_path)
        if source_rgb.shape[:2] != (height, width):
            raise ValueError(f"Source image {source_path} has shape {source_rgb.shape}, expected {(height, width)}")

        mapping = run_mapping.get(image_name)
        render_path: Path | None = None
        gt_path: Path | None = None
        output_index: int | None = None
        if mapping is not None:
            output_index = int(mapping["id"])
            render_path = run_dir / "renders" / f"{output_index:05d}.png"
            gt_path = run_dir / "gt" / f"{output_index:05d}.png"
        render_rgb = _read_rgb(render_path) if render_path is not None and render_path.is_file() else None
        output_gt_rgb = _read_rgb(gt_path) if gt_path is not None and gt_path.is_file() else None

        mesh_rgb = np.asarray(frame["color"], dtype=np.uint8)
        hit = np.asarray(frame["hit"], dtype=bool)
        depth = np.asarray(frame["depth_camera_z"], dtype=np.float32)
        world_points = np.asarray(frame["world_points"], dtype=np.float32)
        if mesh_rgb.shape != (height, width, 3) or hit.shape != (height, width):
            raise RuntimeError("Raycast output resolution does not match COLMAP intrinsics")

        mesh_png = frame_dir / "mesh_vertex_color.png"
        mask_png = frame_dir / "hit_mask.png"
        depth_png = frame_dir / "depth_camera_z.png"
        depth_npy = frame_dir / "depth_camera_z.npy"
        points_npy = frame_dir / "world_hit_points.npy"
        _save_rgb(mesh_png, mesh_rgb)
        Image.fromarray(hit.astype(np.uint8) * 255, mode="L").save(mask_png)
        depth_percentiles = _save_depth_visualization(depth_png, depth, hit)
        np.save(depth_npy, depth)
        np.save(points_npy, world_points)

        raw_gt_metrics = _image_metrics(source_rgb, output_gt_rgb)
        render_gt_metrics = _image_metrics(render_rgb, output_gt_rgb)
        mesh_gt_metrics = _image_metrics(mesh_rgb, output_gt_rgb)
        mesh_source_metrics = _image_metrics(mesh_rgb, source_rgb)
        hit_mesh_gt_metrics = None
        if output_gt_rgb is not None and output_gt_rgb.shape == mesh_rgb.shape and hit.any():
            delta = (mesh_rgb[hit].astype(np.float32) - output_gt_rgb[hit].astype(np.float32)) / 255.0
            hit_rmse = float(np.sqrt(np.mean(delta * delta)))
            hit_mesh_gt_metrics = {
                "pixels": int(hit.sum()),
                "mae_0_1_on_hit": float(np.mean(np.abs(delta))),
                "rmse_0_1_on_hit": hit_rmse,
                "psnr_db_on_hit": float("inf") if hit_rmse == 0 else float(-20.0 * math.log10(hit_rmse)),
            }

        contact_path = frame_dir / "contact_sheet.png"
        _make_contact_sheet(
            contact_path,
            f"Mesh camera raycast · {image_name} · vertex RGB / COLMAP PINHOLE",
            [
                ("Mesh vertex color", mesh_rgb),
                ("Ray hit mask", np.repeat(hit[..., None].astype(np.uint8) * 255, 3, axis=2)),
                ("Camera-Z depth", np.repeat(np.asarray(Image.open(depth_png).convert("L"))[..., None], 3, axis=2)),
                ("PGSR rendered RGB", render_rgb),
                ("Run GT", output_gt_rgb),
                ("COLMAP source JPEG", source_rgb),
            ],
            width=width,
            height=height,
        )

        hit_count = int(frame["hit_count"])
        record = {
            "colmap_image": image_name,
            "colmap_image_id": int(colmap_image.id),
            "colmap_camera_id": int(camera.id),
            "camera_model": camera.model,
            "resolution": [width, height],
            "K": [[float(frame["intrinsics"][0]), 0.0, float(frame["intrinsics"][2])],
                  [0.0, float(frame["intrinsics"][1]), float(frame["intrinsics"][3])],
                  [0.0, 0.0, 1.0]],
            "world_to_camera_R": np.asarray(frame["world_to_camera_rotation"]).tolist(),
            "world_to_camera_t": np.asarray(frame["world_to_camera_translation"]).tolist(),
            "camera_center_world": np.asarray(frame["camera_center_world"]).tolist(),
            "source_jpeg": str(source_path),
            "run_camera_mapping_found": mapping is not None,
            "run_output_index": output_index,
            "render_rgb": str(render_path) if render_path is not None else None,
            "render_rgb_available": render_rgb is not None,
            "run_gt": str(gt_path) if gt_path is not None else None,
            "run_gt_available": output_gt_rgb is not None,
            "source_jpeg_vs_run_gt": raw_gt_metrics,
            "render_rgb_vs_run_gt": render_gt_metrics,
            "mesh_color_vs_run_gt_all_pixels": mesh_gt_metrics,
            "mesh_color_vs_source_jpeg_all_pixels": mesh_source_metrics,
            "mesh_color_vs_run_gt_on_hit_pixels": hit_mesh_gt_metrics,
            "mesh_vertices": int(len(vertices)),
            "mesh_triangles": int(len(faces)),
            "vertex_color_min_rgb": colors.min(axis=0).astype(float).tolist(),
            "vertex_color_max_rgb": colors.max(axis=0).astype(float).tolist(),
            "hit_pixels": hit_count,
            "hit_fraction": hit_count / int(frame["pixel_count"]),
            "depth_camera_z_min_max": [float(np.nanmin(depth)), float(np.nanmax(depth))] if hit.any() else None,
            "depth_visualization_percentiles_2_98": depth_percentiles,
            "depth_transform_max_abs_error": float(frame["depth_transform_max_abs_error"]),
            "artifacts": {
                "mesh_vertex_color_png": str(mesh_png),
                "hit_mask_png": str(mask_png),
                "depth_camera_z_npy": str(depth_npy),
                "depth_camera_z_png": str(depth_png),
                "world_hit_points_npy": str(points_npy),
                "contact_sheet_png": str(contact_path),
            },
        }
        records.append(record)
        print(
            f"{image_name}: {hit_count}/{int(frame['pixel_count'])} pixels hit "
            f"({record['hit_fraction']:.1%}); mapping index={output_index}; "
            f"outputs at {frame_dir}",
            flush=True,
        )

    summary: dict[str, Any] = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "CPU qualitative mesh-to-camera projection and artifact linkage; not a benchmark result.",
        "mesh": str(mesh_path),
        "dataset": str(dataset),
        "run_directory": str(run_dir),
        "projection": {
            "source_pose": "COLMAP images.bin qvec/tvec, x_camera = R_world_to_camera @ x_world + t",
            "intrinsics": "COLMAP PINHOLE or SIMPLE_PINHOLE K; distorted camera models are rejected",
            "camera_axes": "+x right, +y down, +z forward",
            "raycast": "Open3D RaycastingScene on CPU; unit-length world rays",
            "vertex_color": "interpolate PLY vertex RGB with Open3D primitive_uvs barycentric weights",
            "depth": "camera-frame Z in mesh world units; misses are NaN in NPY",
            "world_points": "hit locations in COLMAP world coordinates; misses are NaN",
        },
        "mesh_vertices": int(len(vertices)),
        "mesh_triangles": int(len(faces)),
        "process_peak_rss_mb": _rss_peak_mb(),
        "views": records,
        "mapping_csv": str(output / "camera_mapping.csv"),
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    fieldnames = [
        "colmap_image", "colmap_image_id", "colmap_camera_id", "camera_model", "width", "height",
        "source_jpeg", "run_output_index", "render_rgb", "run_gt", "run_camera_mapping_found",
        "hit_pixels", "hit_fraction", "mesh_vertex_color_png", "hit_mask_png", "depth_camera_z_npy",
        "depth_camera_z_png", "world_hit_points_npy", "contact_sheet_png",
        "source_jpeg_vs_run_gt_mae_0_1", "render_rgb_vs_run_gt_mae_0_1",
        "mesh_color_vs_run_gt_mae_0_1", "mesh_color_vs_run_gt_on_hit_mae_0_1",
    ]
    with (output / "camera_mapping.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            artifacts = rec["artifacts"]
            writer.writerow({
                "colmap_image": rec["colmap_image"],
                "colmap_image_id": rec["colmap_image_id"],
                "colmap_camera_id": rec["colmap_camera_id"],
                "camera_model": rec["camera_model"],
                "width": rec["resolution"][0],
                "height": rec["resolution"][1],
                "source_jpeg": rec["source_jpeg"],
                "run_output_index": rec["run_output_index"],
                "render_rgb": rec["render_rgb"],
                "run_gt": rec["run_gt"],
                "run_camera_mapping_found": rec["run_camera_mapping_found"],
                "hit_pixels": rec["hit_pixels"],
                "hit_fraction": rec["hit_fraction"],
                "mesh_vertex_color_png": artifacts["mesh_vertex_color_png"],
                "hit_mask_png": artifacts["hit_mask_png"],
                "depth_camera_z_npy": artifacts["depth_camera_z_npy"],
                "depth_camera_z_png": artifacts["depth_camera_z_png"],
                "world_hit_points_npy": artifacts["world_hit_points_npy"],
                "contact_sheet_png": artifacts["contact_sheet_png"],
                "source_jpeg_vs_run_gt_mae_0_1": _metric_value(rec["source_jpeg_vs_run_gt"], "mae_0_1"),
                "render_rgb_vs_run_gt_mae_0_1": _metric_value(rec["render_rgb_vs_run_gt"], "mae_0_1"),
                "mesh_color_vs_run_gt_mae_0_1": _metric_value(rec["mesh_color_vs_run_gt_all_pixels"], "mae_0_1"),
                "mesh_color_vs_run_gt_on_hit_mae_0_1": _metric_value(rec["mesh_color_vs_run_gt_on_hit_pixels"], "mae_0_1_on_hit"),
            })
    print(f"Wrote {summary_path}", flush=True)
    print(f"Peak RSS: {summary['process_peak_rss_mb']:.1f} MiB", flush=True)


def _metric_value(metrics: Any, key: str) -> float | str | None:
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(key)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


if __name__ == "__main__":
    main()
