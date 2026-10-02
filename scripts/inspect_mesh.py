"""Check PGSR TSDF meshes after extraction, without interpreting mesh quality."""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = {}
    for name in ("fuse.ply", "fuse_post.ply"):
        path = args.directory / name
        mesh = o3d.io.read_triangle_mesh(str(path)) if path.is_file() else None
        vertices = np.asarray(mesh.vertices) if mesh is not None else np.empty((0, 3))
        triangles = np.asarray(mesh.triangles) if mesh is not None else np.empty((0, 3), dtype=int)
        entry = {
            "path": str(path),
            "exists": path.is_file(),
            "vertices": len(vertices),
            "triangles": len(triangles),
            "finite_vertices": bool(np.isfinite(vertices).all()),
        }
        if len(vertices):
            entry["bounds_min"] = vertices.min(axis=0).tolist()
            entry["bounds_max"] = vertices.max(axis=0).tolist()
        report[name] = entry
    print(json.dumps(report, indent=2))
    if not any(r["vertices"] > 0 and r["triangles"] > 0 and r["finite_vertices"] for r in report.values()):
        raise SystemExit("No nonempty finite triangle mesh found")


if __name__ == "__main__":
    main()
