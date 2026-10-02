#!/usr/bin/env bash
# Run from /mnt/e/PGSR within WSL. Each stage is separate and logs its exact command.
set -euo pipefail

stage="${1:-}"
scene="/mnt/e/PGSR/data/truck-320"
python_bin="/mnt/e/PGSR/.envs/pgsr-1.0.0/bin/python"
outputs="/mnt/e/PGSR/outputs"
mkdir -p "$outputs/logs"

if [[ ! -x "$python_bin" || ! -d "$scene/sparse/0" ]]; then
  echo "Environment or COLMAP scene is missing" >&2
  exit 2
fi

nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader
df -h /mnt/e
export PGSR_STUDY_SEED=20260930
export TORCH_EXTENSIONS_DIR=/mnt/e/PGSR/.cache/torch_extensions
export XDG_CACHE_HOME=/mnt/e/PGSR/.cache/xdg
export TRITON_CACHE_DIR=/mnt/e/PGSR/.cache/triton
export CUDA_CACHE_PATH=/mnt/e/PGSR/.cache/cuda
export TMPDIR=/mnt/e/PGSR/.cache/tmp
export PGSR_STUDY_CPU_KNN=1

run() {
  local log="$1"; shift
  printf 'command:' | tee "$log"
  printf ' %q' "$@" | tee -a "$log"
  printf '\n' | tee -a "$log"
  "$@" 2>&1 | tee -a "$log"
}

assert_fresh_training_output() {
  local destination="$1"
  # gaussian_splatting.train.training 会先删除 destination/point_cloud；
  # 拒绝复用已有检查点的目录，以保留学习实验和可追溯性。
  if [[ -e "$destination/point_cloud" || -L "$destination/point_cloud" ]]; then
    echo "Refusing to overwrite existing checkpoints: $destination/point_cloud" >&2
    exit 2
  fi
}

case "$stage" in
  base)
    # base 保留 PGSR 的平面/法线等训练包装器，但不执行高斯增密；20 步用于确认
    # RGB -> loss -> backward -> optimizer -> checkpoint 的最短闭环。
    assert_fresh_training_output "$outputs/truck-base20"
    run "$outputs/logs/base20.log" "$python_bin" scripts/study_driver.py pgsr.train \
      -s "$scene" -d "$outputs/truck-base20" -i 20 --save_iterations 20 \
      --mode base --backend gsplat --no_image_mask --no_depth_data
    ;;
  geometry)
    # 原默认几何损失约在 7000 步才启用。这里只在学习实验中提前到 100 步，
    # 使 300 步短跑能观察几何项及其梯度。邻居缓存每 100 步更新一次。
    export PGSR_TRACE_LOG="$outputs/logs/geometry300.jsonl"
    export PGSR_TRACE_TERM_GRADS=1
    assert_fresh_training_output "$outputs/truck-geometry300"
    run "$outputs/logs/geometry300.log" "$python_bin" scripts/study_driver.py pgsr.train \
      -s "$scene" -d "$outputs/truck-geometry300" -i 300 --save_iterations 300 \
      --mode base --backend gsplat --no_image_mask --no_depth_data \
      -o depth_normal_consistency_from_iter=100 \
      -o virtual_camera_reprojection_from_iter=100 \
      -o multi_view_regularize_from_iter=100 \
      -o neighbor_view_update_interval=100
    ;;
  densify)
    # densify 模式在相同损失链外增加 split/clone/prune/trim/opacity-reset 日程。
    # 训练上限 1500 步，显式保存末步；不把该非收敛结果当论文基准。
    export PGSR_TRACE_LOG="$outputs/logs/densify1500.jsonl"
    assert_fresh_training_output "$outputs/truck-densify1500"
    run "$outputs/logs/densify1500.log" "$python_bin" scripts/study_driver.py pgsr.train \
      -s "$scene" -d "$outputs/truck-densify1500" -i 1500 --save_iterations 1500 \
      --mode densify --backend gsplat --no_image_mask --no_depth_data \
      -o depth_normal_consistency_from_iter=100 \
      -o virtual_camera_reprojection_from_iter=100 \
      -o multi_view_regularize_from_iter=100 \
      -o neighbor_view_update_interval=100
    ;;
  render)
    # 从训练输出的 point_cloud.ply 重载模型并调用库提供的渲染入口。
    run "$outputs/logs/render1500.log" "$python_bin" scripts/study_driver.py pgsr.render \
      -s "$scene" -d "$outputs/truck-densify1500" -i 1500 \
      --backend gsplat --no_image_mask
    ;;
  mesh)
    # 深度图融合为 TSDF 网格；mesh_res 是底层 extract_mesh 接受的参数。
    run "$outputs/logs/mesh1500.log" "$python_bin" scripts/study_driver.py pgsr.mesh \
      -s "$scene" -d "$outputs/truck-densify1500" -i 1500 \
      --backend gsplat --no_image_mask -o max_depth=10.0 -o mesh_res=256
    ;;
  *)
    echo "usage: $0 base|geometry|densify|render|mesh" >&2
    exit 2
    ;;
esac
