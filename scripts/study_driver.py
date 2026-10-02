"""Run an installed PGSR CLI with a recorded seed and package provenance.

Example: python scripts/study_driver.py pgsr.train -s data/truck-320 -d outputs/base -i 20 ...
"""

import importlib.metadata
import os
import random
import runpy
import sys

import numpy as np
import torch


def main() -> None:
    if len(sys.argv) < 2 or not sys.argv[1].startswith("pgsr."):
        raise SystemExit("usage: study_driver.py pgsr.train|pgsr.render|pgsr.mesh [arguments]")
    module = sys.argv[1]
    if module not in {"pgsr.train", "pgsr.render", "pgsr.mesh"}:
        raise SystemExit(f"unsupported module: {module}")
    seed = int(os.environ.get("PGSR_STUDY_SEED", "20260930"))
    # 训练 CLI 本身没有 seed 参数；在 runpy 启动 CLI 前固定 Python/NumPy/Torch。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"seed={seed} torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
    for package in ("pgsr", "gaussian-splatting", "gsplat", "open3d"):
        print(f"{package}={importlib.metadata.version(package)}", flush=True)
    print("module=" + module + " args=" + repr(sys.argv[2:]), flush=True)
    if module == "pgsr.train" and os.environ.get("PGSR_STUDY_CPU_KNN") == "1":
        # 只替换点云初始化使用的旧 simple_knn 扩展。训练库与渲染后端不改动。
        from study_knn_fallback import install

        install()
    if module == "pgsr.train" and os.environ.get("PGSR_TRACE_LOG"):
        # 包装损失方法以记录各层增量；必须在导入并运行 pgsr.train 前安装。
        from trace_losses import install

        install(os.environ["PGSR_TRACE_LOG"])
        print("loss_trace=" + os.environ["PGSR_TRACE_LOG"], flush=True)
    sys.argv = [module, *sys.argv[2:]]
    try:
        runpy.run_module(module, run_name="__main__")
    finally:
        if torch.cuda.is_available():
            print(
                "torch_peak_allocated_mib="
                f"{torch.cuda.max_memory_allocated() / 1048576:.1f} "
                "torch_peak_reserved_mib="
                f"{torch.cuda.max_memory_reserved() / 1048576:.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
