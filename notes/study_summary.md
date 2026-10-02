# PGSR Python 库 Truck-320 本地学习实验总结

2026-10-02 追加的 [S07 碎片网格诊断](truck_reconstruction_diagnosis.md) 已用完整 PLY 证实严重碎片化。以下 S01–S06 的“执行链路通过、网格非空”只代表流程验收；**Truck 物体级 mesh 质量不通过**，不能把局部相机视角能辨出车体理解为完整重建。

实验日期：2026-09-30 至 2026-10-02（Asia/Shanghai；10-02 对法线梯度做了固定 checkpoint 重检）。本页是这轮实验的入口；精确日志、张量表、论文—代码对应关系和网格验收分别保留在 [`experiment_log.md`](experiment_log.md)、[`tensor_trace.md`](tensor_trace.md)、[`source_map.md`](source_map.md) 和 [`mesh_acceptance.md`](mesh_acceptance.md)，适合 Git 的紧凑证据见 [`artifact_manifest.md`](artifact_manifest.md)。下文的“观察”均指已经运行且有日志或产物支持；源码解释与后续设想另行标明。

## 目标、结论和范围

本轮用已标定的 Tanks and Temples Truck 图像，在本机走通 `pgsr==1.0.0` 的相机加载、Gaussian 初始化、两种后端前向与反向、短训练、几何损失、增密、PLY 保存与重载、全视角渲染、平面深度、TSDF 网格和可视化检查。它是理解实现的受控实验。**已观察到完整执行链路和可读取的非空网格；1500 步结果尚未收敛，不能作为论文指标或官方 PGSR 复现。**

本次从已有 RGB、COLMAP 相机和稀疏点开始，没有重新做特征匹配、位姿估计或稀疏重建。训练未输入真值深度或 LiDAR，命令显式使用 `--no_depth_data --no_image_mask`。`pgsr.mesh` 用模型渲染的平面深度融合；后续的网格投影只是验收，不向训练提供监督。

| 范围 | 已观察的结果 | 尚不能据此认定 |
| --- | --- | --- |
| 前向/反向 | `gsplat` 和 `gsplat-2dgs` 各完成一帧 Truck RGB 前向和 RGB 均值反向；`gsplat` 完成实际训练 | 两后端或官方 CUDA 的逐像素、逐梯度等价 |
| 训练 | `base` 20/300 步与 `densify` 1500 步；几何项有非零损失/梯度，Gaussian 数量变化并保存 PLY | 收敛、每一种增删点机制的独立贡献 |
| 推理/网格 | PLY 重载，251 帧渲染，251 帧深度 TSDF 融合；`fuse.ply` 和 `fuse_post.ply` 均非空且坐标有限 | 网格水密、完整、达到论文重建精度 |

## 可复核的环境与数据

根工作区实验前提交为 `de24f1a38b350387e8d8fe381b2cd70c1ae946e7`；官方参考克隆 `upstream/PGSR-official` 在该提交，重构库参考克隆 `upstream/PGSR-python` 在 `497f926782753558c2f0d5d19c0234088c52a058`。训练调用隔离环境中安装的 PyPI `pgsr==1.0.0`，而非直接运行两个参考克隆；[`verify_package_source.py`](../scripts/verify_package_source.py) 核对了安装包与发布提交的 31 个 Python 文件。两份上游参考目录保持干净；学习插桩放在工作区脚本中。重构包和官方实现的行为差异见 [`library_audit.md`](library_audit.md)。

环境是 WSL2 Ubuntu 24.04.1、`/mnt/e/PGSR/.envs/pgsr-1.0.0`、Python 3.10.21、PyTorch `2.4.1+cu121`、`gaussian-splatting==2.8.4`、`gsplat==1.5.3+pt24cu121`、Open3D 0.19.0；设备为 RTX 3050 Ti Laptop GPU（4 GiB，驱动 546.30）。随机种子 `20260930` 由 [`study_driver.py`](../scripts/study_driver.py) 在启动 `pgsr.*` 入口前设给 Python、NumPy 与 Torch。安装细节与固定依赖快照见 [`environment_setup.md`](environment_setup.md)、[`environment_pip_freeze_20261001.txt`](environment_pip_freeze_20261001.txt) 和 [`environment_conda_explicit_20261001.txt`](environment_conda_explicit_20261001.txt)。首次 Conda 安装因 E: 盘不区分大小写导致 `ncurses` 符号链接冲突；为新环境及缓存目录启用大小写敏感后安装成功。网格阶段另在隔离环境中安装 `libgomp`，解决 Open3D 导入时缺失 `libgomp.so.1`；没有改动全局 Python/CUDA。

数据来自 INRIA 的 `tandt_db.zip`（682,628,995 字节；SHA-256 `816e62f22a161abbfe841d2a6b10cdf036e297c9fa289b3bfeee9c6ec526d7e1`），仅提取 Truck。原始 251 张 JPG 为 979×546，而 COLMAP 元数据使用 1957×1091。[`prepare_truck_dataset.py`](../scripts/prepare_truck_dataset.py) 生成 `data/truck-320/` 的 320×178 图像，并同比缩放 `PINHOLE` 内参和二维观测；保留图像名、位姿、三维点和 track。派生相机内参约为 `(fx,fy,cx,cy)=(190.2103,188.6507,160,89)`，有 251 个注册视图、136,029 个稀疏三维点。投影变换的最大误差 `1.14e-13` 像素；派生观测重投影误差中位数/p95 为 `0.109/0.360` 像素。可重新核对 [`truck-320.md`](data/truck-320.md)、Git 内的[数据准备清单](evidence/truck-preparation-manifest.json)及本机原件 `data/truck-preparation-manifest.json`。COLMAP 世界尺度没有米制保证，本页所有深度和网格坐标均按**场景单位**理解。

## 实际调用链和张量

[`run_local_study.sh`](../scripts/run_local_study.sh) 在每个 GPU 阶段前记录 `nvidia-smi` 和磁盘空间；`study_driver.py` 记录种子、包版本、CLI 参数与 PyTorch 峰值显存。训练入口 `pgsr/train.py:prepare_training` 先经共享库 `gaussian_splatting.prepare.prepare_dataset` 读取 COLMAP 相机/图像，再经 `pgsr.prepare.prepare_gaussians` 用 `points3D.bin` 初始化 136,029 个 Gaussian，最后经 `pgsr.prepare.prepare_trainer` 组合损失层。外层 `gaussian_splatting.train.training` 逐步采样相机，并调用共享 `trainer/abc.py:AbstractTrainer.step`：渲染、构造损失、`backward()`、可选增密钩子、Adam 更新和清梯度。指定步数由共享训练入口保存 PLY。详细调用点、输出形状和坐标系在 [`tensor_trace.md`](tensor_trace.md) 与 [`source_map.md`](source_map.md)。

```text
Truck RGB + sparse/0/{cameras,images,points3D}.bin
  → 251 个 Camera + 136029 个 Gaussian
  → 单视图 model(camera)：RGB / alpha / 平面法线 / 平面距离 / 平面深度
  → 光度 + 平面尺度 + 按步启用的几何损失
  → backward → densify 模式的增删点钩子 → Adam 更新 → PLY
  → pgsr.render 重新加载 PLY，输出 251 帧 RGB/逆深度
  → pgsr.mesh 再次逐相机渲染 PGSR 平面深度，Open3D TSDF → 两份 mesh
```

相机 RGB 是 `[3,178,320]`；内参 `K` 是 `[3,3]`，世界到相机矩阵供 rasterizer 使用。六组可训练参数包括 `_xyz:[N,3]`、`_features_dc:[N,1,3]`、`_features_rest:[N,15,3]`、`_opacity:[N,1]`、`_scaling:[N,3]` 和 `_rotation:[N,4]`；这组六项的形状来自参数容器源码，冒烟测试没有逐项保存运行时形状。S01 实测 `out["render"]:[3,178,320]`，`out["render_normals"]:[3,178,320]`，以及 `out["rendered_distance"]`、`out["depth"]`、`out["invdepth"]`、`out["render_alphas"]` 各 `[1,178,320]`。`depth` 是模型预测的相机 Z 深度，非输入真值深度。

PGSR 的主要几何改动是先以最短尺度轴定义局部平面法线，再用 alpha/透射率共同权重合成平面法线 `N` 与距离 `D`，最后让像素射线 `r=K⁻¹[u,v,1]ᵀ` 与合成平面求交。此实现的符号约定写作 `z=-D/(N·r+ε)`，见 `pgsr/gaussian_model.py:plane_params`、`pgsr/models/gsplat.py:render_plane`。它区别于直接合成 Gaussian 中心深度，理想单一平面时共同 alpha 权重可在比值中约去；多层遮挡或退化分母仍可能出错。论文含义、官方 CUDA 对应和梯度路径见 [`source_map.md`](source_map.md)。

底层光度项是 `0.8·L1+0.2·(1−SSIM)`；PGSR 的平面尺度项惩罚可见 Gaussian 的最短尺度，默认权重 100。深度—法线一致性、多视图重投影/patch 光度、虚拟相机重投影默认约 7000 步才启用，本轮 S03/S04 **明确把阈值改到 100**，使短跑能观察几何通路。损失从渲染 RGB、法线、距离和深度回传至 Gaussian；增密、裁剪、透明度重置则按独立日程改参数容器数量或状态。

## 已完成的实验与证据

| ID、日期 | 目的与配置 | 已观察结果及原始证据 |
| --- | --- | --- |
| S01，09-30 | `bash scripts/run_smoke.sh`；`gsplat` 与 `gsplat-2dgs` 各一帧前向、RGB 均值反向 | 两后端均成功。251 视图可加载，初始 N=136,029；六组参数有 `.grad`，其中非零组数分别为 4/5；各进程 PyTorch 已分配峰值 337.6 MiB。`outputs/logs/smoke-gsplat.log`、`smoke-gsplat-2dgs.log`。均值目标仅用于冒烟测试。 |
| S02，09-30 | `bash scripts/run_local_study.sh base`；默认损失日程，20 步 | 保存 `outputs/truck-base20/point_cloud/iteration_20/point_cloud.ply`；N 保持 136,029；显存峰值已分配/已保留 378/482 MiB。`outputs/logs/base20.log`。几何启用门槛未到。 |
| S03，09-30；10-02 重检 | `bash scripts/run_local_study.sh geometry`；`base` 300 步，三类几何起点及邻视图更新间隔设为 100 | 相机缓存达 251；第 300 步 multi-view/virtual 项对 `_xyz` 的梯度范数为 `2.01e-5/0.00620`；显存峰值 608.2/712.0 MiB。`outputs/logs/geometry300.log`、`geometry300.jsonl`。第一次插桩的 normal 列因原地 `+=` 张量别名错误为零，**不能**用于判断法线项未生效。10-02 对固定 geometry300 PLY 的 `000040.jpg` 在 step 101 重新开/关法线项，得到独立损失贡献 `0.017462491989135742`、`_xyz` 梯度范数 `0.004385402891784906`、236,166 个非零梯度元素；原始 JSON 见 `outputs/logs/normal_gradient_recheck_20261002.log`，计算见 [`verify_normal_gradient.py`](../scripts/verify_normal_gradient.py)。 |
| S04，09-30 | `bash scripts/run_local_study.sh densify`；相同提前启用的几何配置，1500 步 | N 从 136,029 增至 423,230；第 500/1000/1500 步后分别为 139,984/254,494/423,230。`outputs/truck-densify1500/point_cloud/iteration_1500/point_cloud.ply` 为 104,962,571 字节，PLY header 有 423,230 个顶点；显存峰值 1284.7/1660.0 MiB。证据：`outputs/logs/densify1500.log`、`densify1500.jsonl`、`outputs/truck-densify1500/training-summary.json`。 |
| S05，10-01 | `bash scripts/run_local_study.sh render`；重载 S04 PLY，再导出 0/125/250 号相机的几何样例 | `pgsr.render` 完成 251/251 帧，`ours_1500/` 内有 251 张 RGB、251 张 GT、251 张逆深度 TIFF 和逐帧 `quality.csv`；显存峰值 426.5/602.0 MiB。三帧平面深度在 `0.1<z≤10` 的有效像素比例为 62.71%/72.66%/68.76%。证据：`outputs/logs/render1500.log`、`geometry_samples1500.log`、`outputs/truck-densify1500/acceptance/geometry/manifest.json`。 |
| S06，10-01 | `bash scripts/run_local_study.sh mesh`；`max_depth=10.0, mesh_res=256` | 再次逐相机渲染深度并融合 251/251 帧；`fuse.ply` 为 1,348,676 顶点/1,688,137 面，`fuse_post.ply` 为 455,965 顶点/819,553 面；所有顶点坐标有限。显存峰值 426.6/602.0 MiB。证据：`outputs/logs/mesh1500.log`、`ours_1500/acceptance/visual_acceptance.json` 与 [`mesh_acceptance.md`](mesh_acceptance.md)。 |
| V01，10-01 | `./.envs/pgsr-1.0.0/bin/python scripts/build_interactive_viewer.py` | 生成 `outputs/truck-densify1500/interactive_viewer.html`；Chrome 本地 `file:///` 加载、页面就绪标志与截图已检查。页面可浏览 251 对 RGB/GT、三帧几何样例和两份减面网格，见 [`interactive_viewer.md`](interactive_viewer.md)。 |

推送后可先核对 Git 内的原始日志：[S01 `gsplat`](evidence/logs/smoke-gsplat.log) 与 [`gsplat-2dgs`](evidence/logs/smoke-gsplat-2dgs.log)、[S02](evidence/logs/base20.log)、[S03](evidence/logs/geometry300.log)及[法线梯度重检](evidence/logs/normal_gradient_recheck_20261002.log)、[S04](evidence/logs/densify1500.log)、[S05 渲染](evidence/logs/render1500.log)与[几何抽样](evidence/logs/geometry_samples1500.log)、[S06](evidence/logs/mesh1500.log)。[训练摘要](evidence/training-summary.json)、[三帧深度清单](evidence/geometry-manifest.json)和[网格验收数据](evidence/visual-acceptance.json)也在证据包中；哈希和原件关系见[证据清单](artifact_manifest.md)。S03 首次 [`geometry300.jsonl`](evidence/logs/geometry300.jsonl) 的 normal 列有已知插桩错误，应使用 10-02 重检日志。训练阶段的精确展开命令见对应 `.log` 首行，也可读 [`run_local_study.sh`](../scripts/run_local_study.sh)。例如 S04 实际命令为：

```bash
/mnt/e/PGSR/.envs/pgsr-1.0.0/bin/python scripts/study_driver.py pgsr.train \
  -s /mnt/e/PGSR/data/truck-320 -d /mnt/e/PGSR/outputs/truck-densify1500 \
  -i 1500 --save_iterations 1500 --mode densify --backend gsplat \
  --no_image_mask --no_depth_data \
  -o depth_normal_consistency_from_iter=100 \
  -o virtual_camera_reprojection_from_iter=100 \
  -o multi_view_regularize_from_iter=100 \
  -o neighbor_view_update_interval=100
```

S04 step 1500 对**该步抽中的一台相机**，修正后的 trace 记录光度 `0.077965`、平面尺度 `0.140923`、深度法线 `0.012151`、多视图 `0.035034`、虚拟相机 `0.001092`。这些是当前损失贡献，不是数据集平均值或评测指标。该条 trace 的 Gaussian 数 `391,324` 是当步增密前；最终 PLY 的 `423,230` 是当步增密后保存，两个数字并不冲突。第 1000 步日志显示 `Multi-view trim` 扫描，但没有分别统计 split/clone/prune/trim 数量；默认 opacity reset 从 3000 步开始，本轮没有触发。PLY 只包含 Gaussian 参数，不包含 Adam 动量和全部调度状态，不能视为无损续训点。

## 本机 workaround 与对结论的影响

首次 S01 未打补丁时，依赖包的 `simple_knn.distCUDA2` 在 Gaussian 初始化阶段报 `This program was not compiled for SM 86`。只读检查发现所装 wheel 的扩展包含 SM75 标记，本机 RTX 3050 Ti 是 SM86。成功的 S01–S04 均显式设置 `PGSR_STUDY_CPU_KNN=1`：[`study_knn_fallback.py`](../scripts/study_knn_fallback.py) 在进程内临时把初始化用的 `distCUDA2` 换成 CPU `cKDTree` 精确三近邻均方距离，返回同设备的 float32 张量；渲染后端、训练器安装文件和参考克隆未改。该替换可能改变初始 Gaussian 尺度以及后续训练结果，因此所有 checkpoint、S05/S06 mesh 都继承“学习实验”的限定。要衡量与原实现的差距，仍需有 SM86 可用的 `simple-knn` 后做同条件对照。详细故障证据见 [`environment_setup.md`](environment_setup.md)。

## 从 checkpoint 到渲染与网格：实际做了什么

S05 的实际 CLI 是 `pgsr.render -s /mnt/e/PGSR/data/truck-320 -d /mnt/e/PGSR/outputs/truck-densify1500 -i 1500 --backend gsplat --no_image_mask`，通过 `study_driver.py` 执行。它从 `point_cloud/iteration_1500/point_cloud.ply` 重建模型和相机，对 251 个注册视图渲染 RGB 和逆深度。独立的 [`export_geometry_samples.py`](../scripts/export_geometry_samples.py) 又在相机索引 0、125、250 导出 PGSR 平面深度、法线和 RGB；三帧有效深度中位数分别为 `3.071/3.337/3.938` 场景单位。此抽样不足以代表 251 帧的深度分布。

S06 的准确命令是：

```bash
/mnt/e/PGSR/.envs/pgsr-1.0.0/bin/python scripts/study_driver.py pgsr.mesh \
  -s /mnt/e/PGSR/data/truck-320 -d /mnt/e/PGSR/outputs/truck-densify1500 \
  -i 1500 --backend gsplat --no_image_mask \
  -o max_depth=10.0 -o mesh_res=256
```

源码路径是 `pgsr/mesh.py` → `pgsr.render.prepare_rendering()` → 共享库 `gaussian_splatting/mesh.py:extract_mesh()`。融合函数**重新**逐相机执行 `gaussians(camera)` 并读 `out["depth"]` 与渲染 RGB，不把 S05 已写出的逆深度 TIFF 当作融合输入。有效深度要求 `0.1<z≤10`；Open3D 用相机内参、世界到相机外参和 `depth_scale=1` 积分。Truck 的 `scene_extent≈5.8434`，实际深度截断为 10，体素边长 `10/256=0.0390625`、SDF 截断 `0.1953125` 场景单位；提取零等值面写 `fuse.ply`，连通片后处理写 `fuse_post.ply`。融合前共享库将活动 SH 阶数设为 0，因而用于网格颜色的 RGB 与 S05 正常渲染不必逐像素相同。源码边界和官方实现的 TSDF 参数差异见 [`source_map.md`](source_map.md)。

CPU 网格结构检查证实两份 PLY 的面索引有效、顶点有限。以 COLMAP 位姿将 `fuse_post.ply` 投回 `000001.jpg`、`000126.jpg`、`000251.jpg`，网格命中比例分别为 **56.5%、55.1%、37.1%**；命中区域网格颜色对 GT 的 MAE 为 `0.0600/0.0828/0.0911`，范围 `[0,1]`。三个视角可辨认 Truck 主体，但地面、背景和部分车体缺失，存在破碎片；最后一帧覆盖较低。相机投影检查及图像联系表在 `outputs/truck-densify1500/ours_1500/acceptance/mesh_camera/`；[`mesh_acceptance.md`](mesh_acceptance.md) 说明了命中率和误差的分母。联系表/离线 viewer 使用减面预览；结构统计读的是完整 PLY。**非空和可投影仅验证流程、坐标关联与部分可见结构，不验证重建质量。**

## 与官方代码的关系、局限和下一问

源码审计确认重构库保留最短轴局部平面、法线/距离合成与射线交点深度、平面尺度、深度法线和多视图几何的核心思路。它同时改变了 rasterizer、邻居选择、重投影采样与损失缩放、增密/裁剪日程，并默认组装虚拟相机约束；官方版默认关闭该约束。`pgsr` 还依赖共享库的训练入口，同一输出目录重训会删除旧 `point_cloud`，所以 [`run_local_study.sh`](../scripts/run_local_study.sh) 已增加拒绝覆盖检查。当前没有同条件跑官方 CUDA 实现，不能声称两者输出等价。逐项代码证据见 [`library_audit.md`](library_audit.md)。

当前可回答“库的核心 PGSR 数据通路能否在这套环境运行”：可以，包含几何梯度、短训练、PLY 重载、全视角渲染和 TSDF 融合。仍待用独立短实验回答的问题是：①在 SM86 兼容的原始 KNN 初始化下，初始尺度和结果改变多少；②各 split/clone/prune/trim mask 的数量及对 423,230 净点数的贡献；③同相机、同 Gaussian 参数下官方 CUDA 与重构 `gsplat` 的 RGB、平面深度和梯度误差；④增加训练步数或改变深度过滤/场景裁剪后，固定三视角的 mesh 覆盖和误差是否改善。它们是后续实验建议，**本轮没有运行**。
