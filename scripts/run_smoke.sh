#!/usr/bin/env bash
set -euo pipefail
cd /mnt/e/PGSR
python_bin=/mnt/e/PGSR/.envs/pgsr-1.0.0/bin/python
scene=/mnt/e/PGSR/data/truck-320
mkdir -p outputs/logs
export TORCH_EXTENSIONS_DIR=/mnt/e/PGSR/.cache/torch_extensions
export XDG_CACHE_HOME=/mnt/e/PGSR/.cache/xdg
export TRITON_CACHE_DIR=/mnt/e/PGSR/.cache/triton
export CUDA_CACHE_PATH=/mnt/e/PGSR/.cache/cuda
export TMPDIR=/mnt/e/PGSR/.cache/tmp
export PGSR_STUDY_CPU_KNN=1
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader
for backend in gsplat gsplat-2dgs; do
  "$python_bin" scripts/smoke_backends.py --source "$scene" --backend "$backend" \
    2>&1 | tee "outputs/logs/smoke-${backend}.log"
done
