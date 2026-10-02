#!/usr/bin/env python3
"""Download and prepare the official 3DGS Truck scene for local study.

The script keeps the upstream archive, extracts only ``tandt/truck/`` into
``data/truck-original``, and creates a 320-pixel-long-edge derivative with a
matching COLMAP model in ``data/truck-320``. It needs Pillow and NumPy; the
COLMAP reader/writer is reused from ``scripts/preprocess/read_write_model.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


ARCHIVE_URL = (
    "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/"
    "datasets/input/tandt_db.zip"
)
EXPECTED_ARCHIVE_BYTES = 682_628_995
ARCHIVE_SCENE_PREFIX = "tandt/truck/"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
# The COLMAP reader lives inside the preserved upstream tree. Avoid creating
# bytecode artifacts there when this study helper imports it.
sys.dont_write_bytecode = True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_archive(path: Path, download: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size == EXPECTED_ARCHIVE_BYTES:
        return
    if not download:
        found = path.stat().st_size if path.exists() else 0
        raise RuntimeError(
            f"Archive is missing or incomplete ({found} bytes). Run without "
            "--skip-download to download/resume it."
        )
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is required for resumable archive downloads")
    print(f"Downloading/resuming {ARCHIVE_URL} -> {path}", flush=True)
    subprocess.run(
        [
            curl,
            "--http1.1",
            "-L",
            "--fail",
            "--retry",
            "8",
            "--retry-all-errors",
            "--retry-delay",
            "2",
            "-C",
            "-",
            ARCHIVE_URL,
            "-o",
            str(path),
        ],
        check=True,
    )
    actual = path.stat().st_size
    if actual != EXPECTED_ARCHIVE_BYTES:
        raise RuntimeError(
            f"Archive size mismatch: expected {EXPECTED_ARCHIVE_BYTES}, got {actual}"
        )


def scene_members(archive: zipfile.ZipFile) -> list[tuple[zipfile.ZipInfo, str]]:
    selected: list[tuple[zipfile.ZipInfo, str]] = []
    for info in archive.infolist():
        name = info.filename.replace("\\", "/")
        if info.is_dir() or not name.startswith(ARCHIVE_SCENE_PREFIX):
            continue
        relative = name[len(ARCHIVE_SCENE_PREFIX) :]
        rel_path = PurePosixPath(relative)
        if not relative or rel_path.is_absolute() or ".." in rel_path.parts:
            raise RuntimeError(f"Unsafe archive member path: {info.filename}")
        selected.append((info, relative))
    if not selected:
        raise RuntimeError(f"No {ARCHIVE_SCENE_PREFIX} members found in archive")
    required = {"sparse/0/cameras.bin", "sparse/0/images.bin", "sparse/0/points3D.bin"}
    names = {relative for _, relative in selected}
    missing = required - names
    if missing:
        raise RuntimeError(f"Truck archive is missing COLMAP files: {sorted(missing)}")
    if not any(PurePosixPath(name).parts[0] == "images" for name in names):
        raise RuntimeError("Truck archive has no images/ directory contents")
    return selected


def extract_original(
    archive: zipfile.ZipFile,
    members: list[tuple[zipfile.ZipInfo, str]],
    destination: Path,
) -> None:
    if destination.exists():
        expected = {relative for _, relative in members}
        actual = {
            p.relative_to(destination).as_posix()
            for p in destination.rglob("*")
            if p.is_file()
        }
        if actual == expected and all(
            (destination / rel).stat().st_size == info.file_size
            for info, rel in members
        ):
            print(f"Original scene already extracted and complete: {destination}")
            return
        raise FileExistsError(
            f"{destination} exists but does not match the archive contents; "
            "move it aside manually before preparing again."
        )

    staging = destination.with_name(destination.name + ".building")
    if staging.exists():
        raise FileExistsError(
            f"Interrupted extraction staging directory exists: {staging}; "
            "inspect it before removing or moving it."
        )
    staging.mkdir(parents=True)
    try:
        for info, relative in members:
            output = staging.joinpath(*PurePosixPath(relative).parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, output.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            if output.stat().st_size != info.file_size:
                raise RuntimeError(f"Extracted size mismatch for {relative}")
        staging.rename(destination)
    except BaseException:
        # Only this script's staging path is touched on failure.
        shutil.rmtree(staging, ignore_errors=True)
        raise


def import_colmap_reader(root: Path):
    try:
        import numpy as np
        from PIL import Image as PILImage
    except ImportError as exc:
        raise RuntimeError("Install Pillow and NumPy in the selected Python environment") from exc
    sys.path.insert(0, str(root / "scripts" / "preprocess"))
    try:
        import read_write_model as colmap
    except ImportError as exc:
        raise RuntimeError("Could not import scripts/preprocess/read_write_model.py") from exc
    return np, PILImage, colmap


def resize_dimensions(width: int, height: int, max_edge: int) -> tuple[int, int]:
    longest = max(width, height)
    if longest <= max_edge:
        return width, height
    scale = max_edge / longest
    return max(1, round(width * scale)), max(1, round(height * scale))


def read_model(colmap, sparse: Path):
    required = [sparse / f"{stem}.bin" for stem in ("cameras", "images", "points3D")]
    if all(path.is_file() for path in required):
        return colmap.read_model(str(sparse), ext=".bin"), ".bin"
    required = [sparse / f"{stem}.txt" for stem in ("cameras", "images", "points3D")]
    if all(path.is_file() for path in required):
        return colmap.read_model(str(sparse), ext=".txt"), ".txt"
    raise RuntimeError(f"No complete COLMAP binary or text model in {sparse}")


def image_files(images_dir: Path) -> dict[str, Path]:
    return {
        p.name: p
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    }


def validate_tracks(images: dict, points: dict) -> int:
    track_count = 0
    for point in points.values():
        for image_id, point2d_index in zip(point.image_ids, point.point2D_idxs):
            image_id = int(image_id)
            point2d_index = int(point2d_index)
            image = images.get(image_id)
            if image is None:
                raise RuntimeError(f"3D point {point.id} references missing image {image_id}")
            if point2d_index < 0 or point2d_index >= len(image.point3D_ids):
                raise RuntimeError(
                    f"3D point {point.id} references invalid observation index "
                    f"{point2d_index} in image {image.name}"
                )
            if int(image.point3D_ids[point2d_index]) != int(point.id):
                raise RuntimeError(
                    f"Track mismatch for point {point.id}, image {image.name}, "
                    f"observation {point2d_index}"
                )
            track_count += 1
    return track_count


def project(camera, xyz, np):
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz[2] <= 0:
        return None
    if camera.model == "SIMPLE_PINHOLE":
        focal, cx, cy = map(float, camera.params)
        return np.array([focal * xyz[0] / xyz[2] + cx, focal * xyz[1] / xyz[2] + cy])
    if camera.model == "PINHOLE":
        fx, fy, cx, cy = map(float, camera.params)
        return np.array([fx * xyz[0] / xyz[2] + cx, fy * xyz[1] / xyz[2] + cy])
    raise RuntimeError(f"Projection validation supports PINHOLE models; found {camera.model}")


def reprojection_statistics(cameras, images, points, np) -> dict:
    """Summarize COLMAP feature-to-point residuals in the model's pixel frame."""
    from read_write_model import qvec2rotmat

    norms = []
    for image in images.values():
        camera = cameras[image.camera_id]
        rotation = qvec2rotmat(image.qvec)
        for observed, point_id in zip(image.xys, image.point3D_ids):
            point_id = int(point_id)
            if point_id < 0:
                continue
            xyz_camera = rotation @ points[point_id].xyz + image.tvec
            predicted = project(camera, xyz_camera, np)
            if predicted is None:
                continue
            residual = np.asarray(observed, dtype=np.float64) - predicted
            if np.all(np.isfinite(residual)):
                norms.append(float(np.linalg.norm(residual)))
    if not norms:
        return {"observation_count": 0, "mean_px": None, "median_px": None, "p95_px": None, "max_px": None}
    values = np.asarray(norms, dtype=np.float64)
    return {
        "observation_count": int(values.size),
        "mean_px": float(values.mean()),
        "median_px": float(np.median(values)),
        "p95_px": float(np.percentile(values, 95)),
        "max_px": float(values.max()),
    }


def build_scaled_scene(
    original: Path,
    destination: Path,
    max_edge: int,
    np,
    PILImage,
    colmap,
) -> dict:
    if destination.exists():
        raise FileExistsError(
            f"Scaled destination already exists: {destination}; refusing to overwrite it."
        )
    staging = destination.with_name(destination.name + ".building")
    if staging.exists():
        raise FileExistsError(
            f"Interrupted derivative staging directory exists: {staging}; "
            "inspect it before removing or moving it."
        )
    original_images = image_files(original / "images")
    source_model, source_ext = read_model(colmap, original / "sparse" / "0")
    cameras, images, points = source_model
    registered_names = {image.name for image in images.values()}
    if registered_names != set(original_images):
        missing_files = sorted(registered_names - set(original_images))
        unregistered_files = sorted(set(original_images) - registered_names)
        raise RuntimeError(
            "Image files and COLMAP image names differ. "
            f"Missing files={missing_files[:10]}, "
            f"files absent from model={unregistered_files[:10]}"
        )
    if any(cam.model not in {"SIMPLE_PINHOLE", "PINHOLE"} for cam in cameras.values()):
        models = sorted({cam.model for cam in cameras.values()})
        raise RuntimeError(f"Expected only SIMPLE_PINHOLE/PINHOLE cameras; found {models}")
    tracks_before = validate_tracks(images, points)
    source_reprojection = reprojection_statistics(cameras, images, points, np)
    if source_reprojection["observation_count"] == 0:
        raise RuntimeError("No valid observations available for source projection validation")

    try:
        output_images = staging / "images"
        output_sparse = staging / "sparse" / "0"
        output_images.mkdir(parents=True)
        output_sparse.mkdir(parents=True)

        dimensions: dict[str, tuple[int, int]] = {}
        source_dimensions: dict[str, tuple[int, int]] = {}
        camera_output_dimensions: dict[int, tuple[int, int]] = {}
        image_records = []
        for name, source_path in sorted(original_images.items()):
            with PILImage.open(source_path) as source:
                source.load()
                width, height = source.size
                new_width, new_height = resize_dimensions(width, height, max_edge)
                dimensions[name] = (new_width, new_height)
                source_dimensions[name] = (width, height)
                output_path = output_images / name
                if (new_width, new_height) == source.size:
                    shutil.copy2(source_path, output_path)
                else:
                    resized = source.resize(
                        (new_width, new_height), resample=PILImage.Resampling.LANCZOS
                    )
                    save_options = {}
                    if source.format == "JPEG":
                        save_options = {"quality": 95, "subsampling": 0, "optimize": True}
                    resized.save(output_path, format=source.format, **save_options)
                image_records.append(
                    {
                        "name": name,
                        "original_width": width,
                        "original_height": height,
                        "output_width": new_width,
                        "output_height": new_height,
                        "original_bytes": source_path.stat().st_size,
                        "output_bytes": output_path.stat().st_size,
                    }
                )

        for image in images.values():
            output_size = dimensions[image.name]
            prior_size = camera_output_dimensions.setdefault(image.camera_id, output_size)
            if prior_size != output_size:
                raise RuntimeError(
                    f"Camera {image.camera_id} is shared by differently sized images; "
                    "one COLMAP camera cannot describe this derivative without splitting IDs."
                )

        output_cameras = {}
        camera_scales = {}
        camera_records = []
        for camera_id, camera in cameras.items():
            # This T&T archive stores half-size JPEGs while the COLMAP model
            # and its feature coordinates are in the full-size camera frame.
            # Keep camera and raster dimensions aligned in the derivative,
            # scaling intrinsics and observations from the model frame.
            new_width, new_height = camera_output_dimensions.get(
                camera_id, resize_dimensions(camera.width, camera.height, max_edge)
            )
            sx = new_width / camera.width
            sy = new_height / camera.height
            params = [float(value) for value in camera.params]
            output_model = camera.model
            if camera.model == "SIMPLE_PINHOLE":
                focal, cx, cy = params
                if math.isclose(sx, sy, rel_tol=0.0, abs_tol=1e-12):
                    output_params = [focal * sx, cx * sx, cy * sy]
                else:
                    # Independent rounding of integer image dimensions can
                    # make sx and sy differ. PINHOLE exactly represents the
                    # anisotropically scaled camera.
                    output_model = "PINHOLE"
                    output_params = [focal * sx, focal * sy, cx * sx, cy * sy]
            else:
                fx, fy, cx, cy = params
                output_params = [fx * sx, fy * sy, cx * sx, cy * sy]
            output_cameras[camera_id] = camera._replace(
                model=output_model,
                width=new_width,
                height=new_height,
                params=np.asarray(output_params, dtype=np.float64),
            )
            camera_scales[camera_id] = (sx, sy)
            camera_records.append(
                {
                    "camera_id": int(camera_id),
                    "input_model": camera.model,
                    "output_model": output_model,
                    "input_width": int(camera.width),
                    "input_height": int(camera.height),
                    "output_width": new_width,
                    "output_height": new_height,
                    "scale_x": sx,
                    "scale_y": sy,
                    "input_params": params,
                    "output_params": output_params,
                }
            )

        output_model_images = {}
        for image_id, image in images.items():
            camera = cameras[image.camera_id]
            source_path = original_images[image.name]
            with PILImage.open(source_path) as source:
                output_size = dimensions[image.name]
                if output_size != (output_cameras[image.camera_id].width, output_cameras[image.camera_id].height):
                    raise RuntimeError(
                        f"Resized raster size {output_size} for {image.name} does not "
                        f"match camera {image.camera_id}'s output dimensions "
                        f"{(output_cameras[image.camera_id].width, output_cameras[image.camera_id].height)}"
                    )
            sx, sy = camera_scales[image.camera_id]
            scaled_xys = np.asarray(image.xys, dtype=np.float64).copy()
            if scaled_xys.size:
                scaled_xys[:, 0] *= sx
                scaled_xys[:, 1] *= sy
            output_model_images[image_id] = image._replace(xys=scaled_xys)

        # Verify the transformed projection algebraically across all model
        # observations with 3D point associations.
        from read_write_model import qvec2rotmat

        checked_projections = 0
        max_projection_transform_error = 0.0
        for image_id, image in images.items():
            camera = cameras[image.camera_id]
            scaled_camera = output_cameras[image.camera_id]
            sx, sy = camera_scales[image.camera_id]
            rotation = qvec2rotmat(image.qvec)
            for xy, point_id in zip(image.xys, image.point3D_ids):
                point_id = int(point_id)
                if point_id < 0:
                    continue
                xyz_camera = rotation @ points[point_id].xyz + image.tvec
                projected_before = project(camera, xyz_camera, np)
                projected_after = project(scaled_camera, xyz_camera, np)
                if projected_before is None or projected_after is None:
                    continue
                expected_after = projected_before * np.array([sx, sy])
                error = float(np.max(np.abs(projected_after - expected_after)))
                max_projection_transform_error = max(max_projection_transform_error, error)
                checked_projections += 1

        if checked_projections == 0:
            raise RuntimeError("No valid 3D observations available for projection validation")
        if max_projection_transform_error > 1e-8:
            raise RuntimeError(
                "Camera projection changed beyond numerical tolerance: "
                f"max error={max_projection_transform_error} px"
            )

        output_reprojection = reprojection_statistics(
            output_cameras, output_model_images, points, np
        )

        colmap.write_model(output_cameras, output_model_images, points, str(output_sparse), ext=".bin")
        project_ini = original / "sparse" / "0" / "project.ini"
        if project_ini.is_file():
            shutil.copy2(project_ini, output_sparse / "project.ini")

        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    output_names = set(image_files(destination / "images"))
    if output_names != set(original_images):
        raise RuntimeError("Image names changed during derivative creation")
    reloaded, _ = read_model(colmap, destination / "sparse" / "0")
    scaled_cameras, scaled_images, scaled_points = reloaded
    tracks_after = validate_tracks(scaled_images, scaled_points)
    if set(scaled_images) != set(images) or set(scaled_cameras) != set(cameras):
        raise RuntimeError("COLMAP camera/image IDs changed in the derivative")
    if set(scaled_points) != set(points):
        raise RuntimeError("COLMAP 3D point IDs changed in the derivative")
    for image_id, source_image in images.items():
        result_image = scaled_images[image_id]
        if (
            source_image.name != result_image.name
            or source_image.camera_id != result_image.camera_id
            or not np.array_equal(source_image.qvec, result_image.qvec)
            or not np.array_equal(source_image.tvec, result_image.tvec)
            or not np.array_equal(source_image.point3D_ids, result_image.point3D_ids)
        ):
            raise RuntimeError(f"Pose or feature-track identity changed for image {source_image.name}")
    for point_id, source_point in points.items():
        result_point = scaled_points[point_id]
        if (
            not np.array_equal(source_point.xyz, result_point.xyz)
            or not np.array_equal(source_point.rgb, result_point.rgb)
            or not np.array_equal(source_point.error, result_point.error)
            or not np.array_equal(source_point.image_ids, result_point.image_ids)
            or not np.array_equal(source_point.point2D_idxs, result_point.point2D_idxs)
        ):
            raise RuntimeError(f"3D point or track changed for point {point_id}")

    # Confirm the rewritten model's camera projection still transforms exactly.
    reloaded_error = 0.0
    for image_id, image in images.items():
        original_camera = cameras[image.camera_id]
        result_camera = scaled_cameras[image.camera_id]
        sx, sy = camera_scales[image.camera_id]
        rotation = qvec2rotmat(image.qvec)
        for point_id in image.point3D_ids:
            point_id = int(point_id)
            if point_id < 0:
                continue
            xyz_camera = rotation @ points[point_id].xyz + image.tvec
            before = project(original_camera, xyz_camera, np)
            after = project(result_camera, xyz_camera, np)
            if before is not None and after is not None:
                reloaded_error = max(
                    reloaded_error,
                    float(np.max(np.abs(after - before * np.array([sx, sy])))),
                )
    if reloaded_error > 1e-8:
        raise RuntimeError(f"Reloaded COLMAP projection validation failed: {reloaded_error} px")
    reloaded_reprojection = reprojection_statistics(
        scaled_cameras, scaled_images, scaled_points, np
    )

    return {
        "source_model_format": source_ext,
        "output_model_format": ".bin",
        "camera_count": len(cameras),
        "image_count": len(images),
        "point3D_count": len(points),
        "point_track_observations": tracks_before,
        "point_track_observations_after": tracks_after,
        "projection_observations_checked": checked_projections,
        "max_projection_transform_error_px": max_projection_transform_error,
        "max_reloaded_projection_transform_error_px": reloaded_error,
        "source_reprojection_residuals": source_reprojection,
        "scaled_reprojection_residuals": output_reprojection,
        "reloaded_scaled_reprojection_residuals": reloaded_reprojection,
        "source_camera_dimensions_match_source_jpegs": all(
            (camera.width, camera.height) == source_dimensions[image.name]
            for image in images.values()
            for camera in [cameras[image.camera_id]]
        ),
        "note_on_source_camera_frame": (
            "When camera metadata dimensions differ from stored raster dimensions, "
            "features and intrinsics are scaled from the COLMAP camera coordinate "
            "frame directly to the resized raster; registered poses and 3D tracks remain fixed."
        ),
        "exact_image_name_set_preserved": True,
        "exact_poses_and_camera_ids_preserved": True,
        "exact_3d_points_and_tracks_preserved": True,
        "images": image_records,
        "cameras": camera_records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--archive", type=Path, default=root / "data" / "tandt_db.zip")
    parser.add_argument("--original-dir", type=Path, default=root / "data" / "truck-original")
    parser.add_argument("--scaled-dir", type=Path, default=root / "data" / "truck-320")
    parser.add_argument("--manifest", type=Path, default=root / "data" / "truck-preparation-manifest.json")
    parser.add_argument("--target-long-edge", type=int, default=320)
    parser.add_argument("--skip-download", action="store_true", help="require an already-complete archive")
    args = parser.parse_args()
    if args.target_long_edge <= 0:
        parser.error("--target-long-edge must be positive")

    args.archive = args.archive.resolve()
    args.original_dir = args.original_dir.resolve()
    args.scaled_dir = args.scaled_dir.resolve()
    args.manifest = args.manifest.resolve()
    ensure_archive(args.archive, download=not args.skip_download)
    archive_size = args.archive.stat().st_size
    archive_hash = sha256_file(args.archive)
    np, PILImage, colmap = import_colmap_reader(root)

    with zipfile.ZipFile(args.archive, "r") as archive:
        members = scene_members(archive)
        extract_original(archive, members, args.original_dir)
        archive_scene_bytes = sum(info.file_size for info, _ in members)
        archive_scene_compressed_bytes = sum(info.compress_size for info, _ in members)

    result = build_scaled_scene(
        args.original_dir,
        args.scaled_dir,
        args.target_long_edge,
        np,
        PILImage,
        colmap,
    )
    source_model = read_model(colmap, args.original_dir / "sparse" / "0")[0]
    source_cameras, source_images, source_points = source_model
    original_file_records = []
    for name, path in sorted(image_files(args.original_dir / "images").items()):
        with PILImage.open(path) as image:
            original_file_records.append(
                {
                    "name": name,
                    "width": image.width,
                    "height": image.height,
                    "bytes": path.stat().st_size,
                }
            )

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "archive": {
            "source_url": ARCHIVE_URL,
            "path": str(args.archive),
            "expected_size_bytes": EXPECTED_ARCHIVE_BYTES,
            "size_bytes": archive_size,
            "sha256": archive_hash,
            "scene_prefix_extracted": ARCHIVE_SCENE_PREFIX,
            "scene_member_count": len(members),
            "scene_uncompressed_bytes": archive_scene_bytes,
            "scene_compressed_bytes": archive_scene_compressed_bytes,
            "other_scenes_extracted": False,
        },
        "scene": {
            "original_path": str(args.original_dir),
            "scaled_path": str(args.scaled_dir),
            "target_long_edge_px": args.target_long_edge,
            "source_image_count": len(original_file_records),
            "camera_count": len(source_cameras),
            "registered_image_count": len(source_images),
            "point3D_count": len(source_points),
            "original_images": original_file_records,
        },
        "validation": result,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "archive_size_bytes": archive_size,
        "archive_sha256": archive_hash,
        "truck_archive_members": len(members),
        "source_image_count": len(original_file_records),
        "cameras": len(source_cameras),
        "registered_images": len(source_images),
        "points3D": len(source_points),
        "scaled_scene": str(args.scaled_dir),
        "manifest": str(args.manifest),
        "max_projection_transform_error_px": result["max_projection_transform_error_px"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
