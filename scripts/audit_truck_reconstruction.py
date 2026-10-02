"""复算 Truck 短训练的训练视角画质与 TSDF 网格拓扑统计。"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import open3d as o3d


QUANTILES = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]


def image_metrics(csv_path: Path) -> dict:
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"图像指标 CSV 为空：{csv_path}")
    result = {"view_count": len(rows), "metrics": {}}
    for name in ("psnr", "ssim", "lpips"):
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} 存在非有限值")
        result["metrics"][name] = {
            "mean": float(values.mean()),
            "std_population": float(values.std()),
            "quantiles_linear_0_10_25_50_75_90_100": np.quantile(
                values, QUANTILES
            ).tolist(),
        }
        if name == "psnr":
            result["metrics"][name]["below_20_db"] = int((values < 20).sum())
    return result


def mesh_topology(path: Path) -> dict:
    mesh = o3d.io.read_triangle_mesh(str(path))
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError(f"网格为空：{path}")
    if not np.isfinite(vertices).all():
        raise ValueError(f"顶点存在非有限坐标：{path}")

    _, component_faces, component_areas = mesh.cluster_connected_triangles()
    sizes = np.asarray(component_faces, dtype=np.int64)
    areas = np.asarray(component_areas, dtype=np.float64)
    ordered = np.sort(sizes)[::-1]

    # 无向边引用一次表示开放边界；引用超过两次表示非流形边。
    edges = np.sort(
        np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])),
        axis=1,
    )
    _, edge_counts = np.unique(edges, axis=0, return_counts=True)
    del edges
    boundary_count = int((edge_counts == 1).sum())

    tri_areas = 0.5 * np.linalg.norm(
        np.cross(
            vertices[faces[:, 1]] - vertices[faces[:, 0]],
            vertices[faces[:, 2]] - vertices[faces[:, 0]],
        ),
        axis=1,
    )
    return {
        "file": str(path),
        "vertices": int(len(vertices)),
        "triangles": int(len(faces)),
        "edge_connected_components": int(len(sizes)),
        "component_faces_median": float(np.median(sizes)),
        "component_faces_p90": float(np.quantile(sizes, 0.9)),
        "component_faces_p99": float(np.quantile(sizes, 0.99)),
        "largest_component_faces": int(ordered[0]),
        "largest_component_face_fraction": float(ordered[0] / len(faces)),
        "second_component_faces": int(ordered[1]) if len(ordered) > 1 else 0,
        "components_under_100_faces": int((sizes < 100).sum()),
        "faces_in_components_under_100": int(sizes[sizes < 100].sum()),
        "total_area_scene_units_squared": float(areas.sum()),
        "boundary_edges": boundary_count,
        "unique_edges": int(len(edge_counts)),
        "boundary_edge_fraction": float(boundary_count / len(edge_counts)),
        "nonmanifold_edges": int((edge_counts > 2).sum()),
        "zero_area_triangles": int((tri_areas == 0).sum()),
        "bounds_min": vertices.min(axis=0).tolist(),
        "bounds_max": vertices.max(axis=0).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-csv", type=Path, required=True)
    parser.add_argument("--mesh-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "method": "Open3D 共边三角面连通片；NumPy 默认线性分位数；边界边为引用一次的无向边",
        "image_metrics": image_metrics(args.quality_csv),
        "meshes": {
            name: mesh_topology(args.mesh_dir / name)
            for name in ("fuse.ply", "fuse_post.ply")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写入 {args.output}")


if __name__ == "__main__":
    main()
