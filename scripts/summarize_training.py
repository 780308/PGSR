"""Summarize completed local PGSR training without loading a GPU model.

The tqdm log reports the Gaussian count after each optimizer/densifier step.
The JSONL loss trace is captured inside the loss call, before that step's
possible densification. Checking both clarifies the iteration-1500 count.
"""

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "outputs/logs/densify1500.log"
TRACE = ROOT / "outputs/logs/densify1500.jsonl"
PLY = ROOT / "outputs/truck-densify1500/point_cloud/iteration_1500/point_cloud.ply"


def ply_vertex_count(path: Path) -> int:
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError("Checkpoint is not a PLY file")
        for line in stream:
            if line.startswith(b"element vertex "):
                return int(line.split()[2])
            if line.strip() == b"end_header":
                break
    raise ValueError("PLY header lacks a vertex count")


def main() -> None:
    text = LOG.read_text(encoding="utf-8", errors="replace")
    counts = {}
    for step, count in re.findall(r"(\d+)/1500[^\r\n]*?n=(\d+)", text):
        counts[int(step)] = int(count)
    traces = [json.loads(line) for line in TRACE.read_text(encoding="utf-8").splitlines()]
    report = {
        "checkpoint": str(PLY),
        "checkpoint_bytes": PLY.stat().st_size,
        "checkpoint_gaussians": ply_vertex_count(PLY),
        "initial_gaussians": traces[0]["gaussians"],
        "final_logged_gaussians": counts.get(1500),
        "selected_after_step_counts": {str(step): counts.get(step) for step in (100, 300, 500, 600, 800, 1000, 1001, 1200, 1499, 1500)},
        "selected_loss_snapshots": [
            {"step": row["step"], "pre_densification_gaussians": row["gaussians"], "terms": row["terms"]}
            for row in traces if row["step"] in {100, 101, 300, 500, 1000, 1500}
        ],
        "torch_peak_allocated_mib": float(re.search(r"torch_peak_allocated_mib=([0-9.]+)", text).group(1)),
        "torch_peak_reserved_mib": float(re.search(r"torch_peak_reserved_mib=([0-9.]+)", text).group(1)),
    }
    print(json.dumps(report, indent=2))
    if report["checkpoint_gaussians"] != report["final_logged_gaussians"]:
        raise SystemExit("Checkpoint count differs from the final training log")


if __name__ == "__main__":
    main()
