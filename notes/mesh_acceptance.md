# Truck-320 短训练的渲染与 TSDF 网格验收

## 结论与边界

2026-10-01 在 WSL2 的 `/mnt/e/PGSR/.envs/pgsr-1.0.0` 中，从 S04 的 iteration 1500 PLY 重载模型，完成全部 251 个相机视角的渲染，并调用已安装的 `pgsr==1.0.0` 和 `gaussian-splatting==2.8.4` 完成 TSDF 融合。原始和后处理 PLY 都能重新读取，具有非零三角面且所有顶点坐标有限，故 **RGB/COLMAP → 短训练 Gaussian → checkpoint → 渲染深度 → TSDF mesh 的执行链路通过**。但**物体级 Truck 网格质量不通过**：全局 mesh 严重碎片化，无法作为完整、易辨识的车体。少数相机投影中能拼出车体轮廓，只证明局部对齐，不证明三维连贯。量化诊断见[Truck 网格诊断](truck_reconstruction_diagnosis.md)。

S04 的 Gaussian 初始尺度使用了明确标记的 `PGSR_STUDY_CPU_KNN=1` 学习实验绕过方式，原因是随包的 `simple_knn` 内核仅有 SM75 版本，而本机 GPU 为 SM86。这可能改变训练结果。S05/S06 从已保存的 PLY 重载，不再调用这个初始化绕过函数；结果仍继承 S04 的模型状态。根检出提交为 `de24f1a38b350387e8d8fe381b2cd70c1ae946e7`，重构库发布源码提交为 `497f926782753558c2f0d5d19c0234088c52a058`。场景来源、SHA-256 和相机缩放检验见 `notes/data/truck-320.md`。

## 环境补充与实际命令

首次导入 Open3D 0.19.0 时缺少 `libgomp.so.1`。只在上述隔离 Conda 环境中安装 `libgomp=15.2.0=h4751f2c_8`，未改驱动、系统 CUDA 或全局 Python。安装前的 `conda --dry-run` 显示只新增 437 KB 的包；安装后 `import open3d` 返回 0.19.0，`pip check` 返回 `No broken requirements found`。
补装后的完整版本快照为 `notes/environment_pip_freeze_20261001.txt` 与 `notes/environment_conda_explicit_20261001.txt`。

```bash
CONDA_PKGS_DIRS=/mnt/e/PGSR/.cache/conda/pkgs \
TMPDIR=/mnt/e/PGSR/.cache/tmp \
/home/dongxianghong/miniconda3/bin/conda install -y \
  -p /mnt/e/PGSR/.envs/pgsr-1.0.0 --override-channels \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
  --no-deps 'libgomp=15.2.0=h4751f2c_8'
```

```bash
bash scripts/run_local_study.sh render
.envs/pgsr-1.0.0/bin/python scripts/export_geometry_samples.py
bash scripts/run_local_study.sh mesh
.envs/pgsr-1.0.0/bin/python scripts/inspect_mesh.py outputs/truck-densify1500/ours_1500
.envs/pgsr-1.0.0/bin/python scripts/visualize_acceptance.py \
  --run-dir outputs/truck-densify1500/ours_1500 \
  --geometry-dir outputs/truck-densify1500/acceptance/geometry
.envs/pgsr-1.0.0/bin/python scripts/render_mesh_camera_acceptance.py \
  --image 000001.jpg --image 000126.jpg --image 000251.jpg
```

`run_local_study.sh` 在每个 GPU 阶段启动前运行 `nvidia-smi` 和 `df`，并固定随机种子 `20260930`。完整展开命令与运行输出见 `outputs/logs/render1500.log`、`outputs/logs/geometry_samples1500.log`、`outputs/logs/mesh1500.log`。S06 开始时 GPU 空闲约 2275 MiB；E: 有约 31 GiB 可用。`pgsr.mesh` 使用 `--backend gsplat --no_image_mask -o max_depth=10.0 -o mesh_res=256`，没有下载额外评测数据。

## 从 PLY 到网格的实际数据通路

入口 `pgsr/mesh.py` 用 `pgsr.render.prepare_rendering()` 加载 `point_cloud/iteration_1500/point_cloud.ply` 和 Truck-320 的 COLMAP 相机。然后由依赖包 `gaussian_splatting/mesh.py:extract_mesh()` 对 251 个视角重新调用 `gaussians(camera)`，读取 `out["render"]` 与 PGSR 平面交点深度 `out["depth"]`。这一步不把 S05 已写出的逆深度 TIFF 当作融合输入。提取函数将 `active_sh_degree` 置为 0，故用于 TSDF 顶点颜色的渲染与 S05 的正常 RGB 渲染不能假设完全一致。

每帧只保留 `0.1 < depth <= 10.0` 的像素，将无效深度置零；Open3D 使用相机内参 `K`、`camera.world_view_transform.T` 的 world-to-camera 外参及 `depth_scale=1` 创建 RGBD 并积分。该场景的 `scene_extent` 约为 5.8434，因此 `depth_trunc=min(10, 2×scene_extent)=10`，体素边长 `10/256=0.0390625`，SDF 截断距离为 `5×0.0390625=0.1953125`（均为 COLMAP 场景单位）。TSDF 零等值面输出 `fuse.ply`；按连通三角片阈值过滤并去掉无引用/退化面后输出 `fuse_post.ply`。有关 PGSR 平面深度与普通期望深度的区别，见 `notes/source_map.md`。

独立导出的第 0、125、250 个相机样例尺寸均为 `178×320`；在 `0.1 < depth <= 10` 内的有效像素占比为 0.6271、0.7266、0.6876，有效像素中位深度为 3.071、3.337、3.938。每帧深度、法线、RGB、GT 文件和统计在 `outputs/truck-densify1500/acceptance/geometry/`。这是三帧抽样，不是所有视角的有效深度总体分布。

以 `pgsr.mesh` 和共享库 `extract_mesh()` 为模块边界，可按工作区的八个学习问题核对：

| 问题 | 此模块中的答案与证据 |
| --- | --- |
| 输入是什么 | 已保存的 Gaussian PLY、Truck-320 RGB/COLMAP 相机和 `max_depth=10, mesh_res=256`；命令见 S06 日志。 |
| 输出是什么 | 每视角渲染的 RGB/平面深度进入 Open3D TSDF；磁盘输出 `fuse.ply` 和 `fuse_post.ply`，数量见下表。 |
| 形状、坐标和单位 | 每帧 RGB 为 `[3,178,320]`，深度为 `[1,178,320]`，`K` 为 `[3,3]`，world-to-camera 为 `[4,4]`；深度是相机 Z，尺度沿用无米制保证的 COLMAP 场景单位。 |
| 对应论文概念 | PGSR 第 IV-A 节的平面交点深度作为多视图表面融合的几何输入；TSDF 将这些深度统一到世界坐标并提取零等值面。 |
| 哪些来自复用库 | `pgsr.mesh` 负责重载与调用；逐相机 TSDF 积分、连通片过滤来自 `gaussian_splatting/mesh.py:extract_mesh()` 与 Open3D，非 PGSR 独有创新。 |
| 反向传播改动什么 | 提取函数使用 `@torch.no_grad()`，本阶段没有 loss、`backward()` 或 optimizer；Gaussian 参数保持 checkpoint 状态。训练阶段深度梯度路径见 `notes/source_map.md`。 |
| 增密/裁剪/reset 改动什么 | 本阶段都不触发；这些已在 S04 的训练模式中作用于 Gaussian 数量，PLY 只提供最终状态。`fuse_post.ply` 的连通片过滤是三角网格后处理，不是 Gaussian 增密或裁剪。 |
| 如何推翻错误理解 | 若误以为融合直接读取 S05 的逆深度 TIFF，可在保留同一 PLY 与相机的独立输出目录中不放 TIFF，调用 `pgsr.mesh` 并观察是否仍能融合；源码也显示它直接调用 `gaussians(camera)`。此独立对照尚未执行，不能写成实测。 |

## 数值和可视化验收

| 产物 | 顶点数 | 三角面数 | 顶点坐标 | 包围盒最小值 | 包围盒最大值 |
| --- | ---: | ---: | --- | --- | --- |
| `fuse.ply` | 1,348,676 | 1,688,137 | 全部有限 | `[-11.465,-5.684,-11.191]` | `[13.301,8.613,11.426]` |
| `fuse_post.ply` | 455,965 | 819,553 | 全部有限 | `[-10.527,-5.684,-8.496]` | `[13.301,7.715,10.527]` |

两份文件分别约 56 MiB 和 22 MiB；`inspect_mesh.py` 的非空检查通过。S06 日志中 TSDF 积分进度为 251/251，PyTorch 已分配/已保留显存峰值为 426.6/602.0 MiB。首张渲染图能辨认 Truck，和 GT 的单帧 RGB MAE 为 0.0477、PSNR 为 22.68 dB；这些只说明该短跑所选视角的图像拟合情况，不能当作论文基准。联系表 `ours_1500/acceptance/visual_acceptance.png` 同时展示 RGB、GT、差异、逆深度、法线、平面深度，以及两版 mesh 的三个全局投影视图。其 JSON 状态为 `ready`，含义仅是核心文件可读取且网格坐标与索引有效；全局网格图仍有碎片和大范围背景结构。

为直接检查网格与相机的关系，`scripts/render_mesh_camera_acceptance.py` 从 COLMAP `images.bin` 读取 world-to-camera 位姿及 PINHOLE 内参，CPU 上以 Open3D `RaycastingScene` 向 `fuse_post.ply` 发射每像素射线，按命中三角面的重心坐标插值 PLY 顶点颜色，并保存相机 Z 深度。输入 JPEG 与运行目录 GT 像素完全一致；命中点变回相机坐标的最大 Z 误差在首帧约 `2.1e-6` 场景单位，这两项检查帮助排除图片映射和外参方向错误。PNG 中黑色表示射线未命中网格；它不是材质或真实黑色表面。

| COLMAP 图像 | 网格命中像素比例 | 命中区域网格色对 GT 的 MAE |
| --- | ---: | ---: |
| `000001.jpg` | 56.5% | 0.0600 |
| `000126.jpg` | 55.1% | 0.0828 |
| `000251.jpg` | 37.1% | 0.0911 |

相机对齐联系表位于 `ours_1500/acceptance/mesh_camera/{000001,000126,000251}/contact_sheet.png`，统计和相机映射见同目录 `summary.json`、`camera_mapping.csv`。三帧中 Truck 主体位置、外形与 GT 总体对齐，尤其首帧与末帧能辨出车轮和车厢；地面与背景有大面积缺失，最后一帧的总像素命中率最低。表中 MAE 仅在命中区域计算，不等于全图误差，也不能当论文指标。部分命中表面也有破碎和模糊。当前**只验收提取执行链路**；物体级连通、完整、易辨识的 mesh 未达到要求，不能将局部车体轮廓称作重建质量验收通过。全量连通片和边界统计见[Truck 网格诊断](truck_reconstruction_diagnosis.md)。
首帧从融合后网格再投射得到的相机 Z 深度最大约 12.60，可能超过融合时每张源相机的 `max_depth=10`：网格汇集了别的源相机视角可见的表面，故在当前检查相机下距离可以更远。这不是将当前相机超出 10 的深度直接送入 TSDF 的证据。

下一步如需提高表面质量，可比较更长训练、场景裁剪和深度过滤，并逐一记录对 mesh 覆盖及精度的影响；本轮不报告 DTU/T&T 评测指标。

本地交互查看页面为 `outputs/truck-densify1500/interactive_viewer.html`；其构建方式、减面边界和离线使用说明见 `notes/interactive_viewer.md`。
