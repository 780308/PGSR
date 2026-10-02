# PGSR 本地学习实验记录

本文档区分已经实际观察的运行与尚待执行的工作，覆盖 2026-09-30 至 2026-10-02（Asia/Shanghai）。全程结论与阅读导航见 `notes/study_summary.md`。

## 启动审计——已观察

- 目的：在安装或运行 PGSR 之前，确认硬件、存储空间和源码来源。
- 根检出目录：`E:\\PGSR`，提交 `de24f1a38b350387e8d8fe381b2cd70c1ae946e7`；初始 `git status --short` 仅包含原先就存在的未跟踪项 `.codex/`、`.vscode/`、`AGENTS.md`。
- 参考克隆：`upstream/PGSR-official` 以 detached HEAD 检出在 `de24f1a38b350387e8d8fe381b2cd70c1ae946e7`；`upstream/PGSR-python` 以 detached HEAD 检出在 `497f926782753558c2f0d5d19c0234088c52a058`；可编辑目录 `worktrees/PGSR-python-study` 的 `study/local-trace` 分支从后一个提交开始。两个参考克隆均未被编辑。
- 本地环境：WSL2 Ubuntu 24.04.1；NVIDIA GeForce RTX 3050 Ti Laptop GPU，4096 MiB 显存，驱动 546.30。检查 `nvidia-smi` 时，已使用 1534 MiB，空闲 2442 MiB。这是空闲状态基线，不是实验的显存峰值。
- 初次检查时的可用磁盘空间：E: 59,935,379,456 字节；C: 22,785,200,128 字节（`Get-PSDrive`）。
- 结果：在 `notes/source_map.md` 中形成静态源码映射；创建了 `scripts/run_local_study.sh`、`scripts/study_driver.py` 和 `scripts/smoke_backends.py`。语法检查结果：`bash -n` 和 `python3 -m py_compile` 均成功。此时尚未运行 PGSR 训练或网格提取。

## 运行记录格式

每次运行需记录：实验 ID 和日期；目的/假设；仓库和提交；环境名称及 `pip freeze`；数据集 SHA-256 和场景；完整命令；迭代范围；配置覆盖项和随机种子；GPU 及已分配/已保留显存峰值；观察到的张量、损失和梯度信息；变化前后的 Gaussian 数量；checkpoint/render/mesh 产物；结论；下一个问题。失败也应作为观察结果记录，不得为迎合预期结论而改写。

计划实验 ID：

| ID | 测试 | 预期产物 | 状态 |
| --- | --- | --- | --- |
| S01 | 两个后端，单相机前向/反向传播 | `outputs/logs/smoke-*.log` | 已完成，并明确标注使用 CPU 3-NN 初始化 |
| S02 | `base`，20 步 | `outputs/truck-base20/` | 已完成 |
| S03 | 几何调度研究，300 步 | `outputs/truck-geometry300/` | 已完成；第一次法线跟踪存在插桩错误，随后已独立验证 |
| S03-R | 固定 checkpoint 的深度法线梯度重检 | `outputs/logs/normal_gradient_recheck_20261002.log` | 2026-10-02 已重检，损失与梯度非零，原始 JSON 已保存 |
| S04 | `densify`，1500 步 | `outputs/truck-densify1500/` | 已完成；S05 随后验证 PLY 可重载 |
| S05 | checkpoint 重新加载与渲染 | `outputs/truck-densify1500/ours_1500/` | 已完成：成功加载 iteration 1500 PLY 并渲染 251 帧；另抽查 3 个视角的平面深度 |
| S06 | TSDF 融合与网格检查 | `fuse.ply`, `fuse_post.ply` | 已完成执行与结构检查；视觉上仍有碎片，非质量收敛 |
| V01 | 离线交互 HTML 查看器 | `outputs/truck-densify1500/interactive_viewer.html` | 已生成，并在本机 Chrome `file:///` 加载与截图验证 |

## E01——环境创建失败

- 日期：2026-09-30；目的：在 E: 盘创建 WSL Python 3.10 PGSR 环境，并将所有缓存置于 E: 盘。
- 完整命令：`cd /mnt/e/PGSR && bash scripts/setup_env.sh`。
- 环境：WSL2 Ubuntu 24.04.1，Miniconda 25.11.1，驱动 546.30，RTX 3050 Ti 4096 MiB。数据集：无。迭代：无。GPU 计算：无。
- 观察：Conda 在进入 pip 阶段之前失败；链接 `ncurses-6.6-hfaaeb4e_0` 时，在 `share/terminfo/n/ncr260vt300wpp` 处出现 `OSError: [Errno 40] Too many levels of symbolic links`。E: 盘中的环境前缀目录不完整，其中没有可用的 Python。完整日志：`.cache/setup_env_20260930_193336.log`。
- 诊断：E: 盘工作区目录不区分大小写。该 Conda 包同时包含小写 `n/` 和大写 `N/` 的 terminfo 路径；二者在此 DrvFs 挂载点发生冲突，并形成自引用符号链接。`fsutil file queryCaseSensitiveInfo E:\\PGSR\\.cache` 返回大小写敏感未启用。在 `.cache/case_probe` 下新建的测试目录已启用大小写敏感，WSL 能够在其中创建 inode 不同的 `n` 和 `N` 目录。该测试没有安装软件包，也没有改动不完整的环境。
- 结论：固定版本的 PGSR 导入、两个后端、训练和网格提取均尚未验证。存储规划和软件包固定版本见 `notes/environment_setup.md`。下一个问题：如何在保留失败环境前缀以供检查的同时，安全地新建启用大小写敏感的 E: 盘 Conda 缓存和环境前缀。

## D01——Truck 解压及 320 像素派生数据集，已观察

- 目的：准备一个紧凑且已配准的 RGB 场景，同时不破坏 COLMAP 几何关系。仓库：根目录提交 `de24f1a38b350387e8d8fe381b2cd70c1ae946e7`。环境：现有的 Inkscape Python 3.12，包含 Pillow 和 NumPy；未安装 PGSR 包。GPU：未使用。
- 命令：`python scripts/prepare_truck_dataset.py`（使用 `C:\\Program Files\\Inkscape\\bin\\python.exe` 执行）；源文件为 `notes/data/truck-320.md` 中记录的 INRIA `tandt_db.zip` URL。
- 归档文件：682,628,995 字节；SHA-256 `816e62f22a161abbfe841d2a6b10cdf036e297c9fa289b3bfeee9c6ec526d7e1`。
- 观察到的场景：251 张已配准图像；一个 PINHOLE 相机条目；136,029 个 COLMAP 点；1,083,072 条 track 观测。原始 JPG 栅格尺寸为 979x546，而 COLMAP 元数据尺寸为 1957x1091。派生数据尺寸为 320x178，内参和二维观测均从 COLMAP 坐标系按比例缩放。投影变换最大误差为 `1.14e-13` 像素；缩放后特征点到三维点的重投影误差 median/p95 分别为 0.109/0.360 像素。相机位姿、点 ID、tracks 和图像名称均保持不变。
- 输出：`data/tandt_db.zip`、`data/truck-original/`、`data/truck-320/`、`data/truck-preparation-manifest.json`。详细说明和文件清单见 `notes/data/truck-320.md`。
- 结论：该数据集内部一致，可用于 PGSR 加载 smoke test。下一个问题：在现有 GPU 显存范围内，已安装的 `gaussian_splatting` loader 是否能够载入全部 251 个相机视角？

## S01–S04——已观察的本地训练链路

- 日期：2026-09-30（Asia/Shanghai）；目的：依次跟踪一次 render/backward、默认短训练、启用后的几何损失以及 densification。包来源：已安装的 PyPI `pgsr==1.0.0`；用于代码映射的发布源码提交为 `497f926782753558c2f0d5d19c0234088c52a058`；共享依赖为 `gaussian-splatting==2.8.4`。环境：`/mnt/e/PGSR/.envs/pgsr-1.0.0`，Python 3.10.21，torch 2.4.1+cu121，gsplat 1.5.3+pt24cu121。完整软件包版本见 `.cache/setup_env_20260930_195227.log` 的 `pip freeze` 部分。场景：Truck-320；manifest 和 SHA-256 见 D01。随机种子：20260930。
- 命令：`bash scripts/run_smoke.sh`；`bash scripts/run_local_study.sh base`；`bash scripts/run_local_study.sh geometry`；`bash scripts/run_local_study.sh densify`。展开后的完整 CLI 参数和日志见 `outputs/logs/`。
- 基线二进制失败：首次在未修补状态下运行 Truck smoke test 时，`gaussian_splatting.gaussian_model.create_from_pcd` 失败，原因是随包提供的 `simple_knn.distCUDA2` 只包含 SM75 kernel，而这块 RTX 3050 Ti 的架构是 SM86。成功完成的 S01–S04 明确使用 `PGSR_STUDY_CPU_KNN=1`，以精确的 CPU 三近邻平均平方距离替代 Gaussian *初始*尺度的计算。这可能改变输出；这些运行属于学习实验，不是 PGSR 基线复现。已安装的软件包文件和干净的 upstream 检出目录均未被编辑。
- S01：`gsplat` 和 `gsplat-2dgs` 均完成了一次真实 Truck 相机视角的前向渲染和 RGB 反向传播。数据集长度为 251；图像尺寸为 3x178x320；初始 Gaussian 数量为 136,029；有 6 个参数张量取得梯度，其中非零梯度张量数量分别为 4 和 5；两个进程各自测得的 PyTorch 已分配显存峰值均为 337.6 MiB。详细张量表见 `notes/tensor_trace.md`。
- S02：`base` 运行 20 步，并保存 `outputs/truck-base20/point_cloud/iteration_20/point_cloud.ply`；Gaussian 数量保持为 136,029；已分配/已保留显存峰值为 378/482 MiB。目的在于跟踪代码路径，不用于判断收敛性。
- S03：`base` 运行 300 步，将 depth-normal、virtual-camera 和 multi-view 的启用阈值提前到 100，并设置 `neighbor_view_update_interval=100`；保存 iteration 300。相机缓存达到 251 个视角。在第 300 步的跟踪结果中，multi-view 和 virtual-camera 损失分量产生的非零 `xyz` 梯度范数分别为 `2.01e-5` 和 `0.00620`。第一次跟踪所用的辅助函数因原地加法别名问题，将 depth-normal 错误记录为零；随后使用固定 checkpoint 进行启用/禁用对比，测得第 101 步实际的 depth-normal 损失贡献为 `0.01746249`，`xyz` 梯度范数为 `0.00438540`。该辅助函数已在 S04 之前修复。已分配/已保留显存峰值为 608.2/712.0 MiB。
- S04：`densify` 使用相同的提前几何调度和 `--save_iterations 1500` 运行 1500 步；tqdm 显示的运行时间为 40m38s。Gaussian 数量在第 500 步后由 136,029 变为 139,984，第 1000 步后变为 254,494，第 1500 步后变为 423,230。第 1500 步的损失跟踪仍报告 391,324 个 Gaussian，因为该跟踪发生在该步 densification 之前。最终 PLY header 包含 423,230 个顶点，文件大小为 104,962,571 字节，与该步之后的日志一致。已分配/已保留显存峰值为 1284.7/1660.0 MiB。见 `outputs/truck-densify1500/training-summary.json` 和 `outputs/logs/densify1500.jsonl`。
- S04 第 1500 步的观察值：photometric 0.07797，planar scale 0.14092，depth-normal 0.01215，multi-view 0.03503，virtual-camera 0.00109；这些是一个采样相机视角对应的损失贡献，不是 benchmark 指标。完整命令和路径证据见上述文件。
- 结论：在已明确标注初始化 workaround 的条件下，已在本地观察到如下链路：Truck RGB/COLMAP → Gaussian 初始化 → 两个后端的渲染张量 → 梯度 → PGSR 几何损失 → optimizer → densification → checkpoint。日志显示多视图 trim 扫描被调用，但尚未分别记录 trim/prune 的实际移除数量；opacity reset 默认从第 3000 步起，本次 1500 步没有达到门槛。Gaussian 数量的净变化本身不能确定每一种增删点的数量。S05/S06 随后验证了该学习检查点可重载、全相机渲染及非空 TSDF 网格。若后续要评估不含 CPU KNN workaround 的初始化路径，需为 SM86 重新编译或替换随包提供且仅支持 SM75 的 `simple-knn`。

## S03-R——固定 checkpoint 的深度法线梯度重检，已观察

- 日期：2026-10-02（Asia/Shanghai）；目的：针对 S03 首次 trace 的 `normal_total=0` 插桩错误，用固定模型和相机重新比较启用/关闭深度法线项时的损失及 `_xyz` 梯度。没有重新训练或改变 checkpoint。
- 仓库与环境：根仓库整理前提交 `de24f1a38b350387e8d8fe381b2cd70c1ae946e7`；已安装的 `pgsr==1.0.0`、发布源码对应 `497f926782753558c2f0d5d19c0234088c52a058`；环境 `/mnt/e/PGSR/.envs/pgsr-1.0.0`。数据是 D01 的 Truck-320；脚本从 `outputs/truck-geometry300/point_cloud/iteration_300/point_cloud.ply` 加载 S03 模型，固定 `dataset[39]` 对应 `000040.jpg`，把训练器 `curr_step` 设为 101。此 PLY 继承训练时的 CPU 3-NN workaround。运行前 `nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader` 返回设备显存总量/已用/空闲 `4096/1597/2379 MiB`。
- 实际在 PowerShell 中执行的命令（退出码 0）：

```powershell
wsl -e bash -lc 'cd /mnt/e/PGSR && export XDG_CACHE_HOME=/mnt/e/PGSR/.cache/xdg TORCH_EXTENSIONS_DIR=/mnt/e/PGSR/.cache/torch_extensions TRITON_CACHE_DIR=/mnt/e/PGSR/.cache/triton TMPDIR=/mnt/e/PGSR/.cache/tmp && /mnt/e/PGSR/.envs/pgsr-1.0.0/bin/python scripts/verify_normal_gradient.py 2>&1 | tee outputs/logs/normal_gradient_recheck_20261002.log; exit ${PIPESTATUS[0]}'
```

- 原始 JSON：`outputs/logs/normal_gradient_recheck_20261002.log`。启用/禁用损失分别为 `2.6583237648010254/2.6408612728118896`；独立深度法线损失 `0.017462491989135742`，该项对 `_xyz` 的梯度范数 `0.004385402891784906`，非零梯度元素数 `236166`。计算脚本为 `scripts/verify_normal_gradient.py`，它在同一次前向输出上切换 `depth_normal_consistency_from_iter` 并调用 `torch.autograd.grad`。
- 结论：固定参数和相机下，深度法线项确实进入损失并对位置参数回传非零梯度；S03 首次 trace 的零值是插桩错误。重检没有验证官方 CUDA 梯度逐元素等价，也没有提供新的收敛或 mesh 质量证据。下一问：若需数值等价，应在同一 Gaussian/相机上比较两种实现的逐像素深度和对应梯度。

## S05——iteration 1500 checkpoint 重载、全视角渲染与深度抽样，已观察

- 日期：2026-10-01（Asia/Shanghai）；目的：验证 S04 保存的 Gaussian PLY 能否经 `pgsr.render` 重载，并检查全体训练相机渲染是否完成；在网格提取前，抽查 3 个相机视角的 PGSR 平面深度，为 `max_depth` 配置提供依据。此实验只验证短程 checkpoint 的可读性和渲染/深度输出，不评价收敛或论文指标。
- 仓库与环境：重构包 `pgsr==1.0.0`（发布源码 `497f926782753558c2f0d5d19c0234088c52a058`），共享依赖 `gaussian-splatting==2.8.4`；环境 `/mnt/e/PGSR/.envs/pgsr-1.0.0`，Python 3.10.21、torch 2.4.1+cu121、CUDA 12.1、gsplat 1.5.3+pt24cu121、Open3D 0.19.0。设备为 RTX 3050 Ti Laptop GPU，4 GiB 显存。随机种子 20260930。输入数据为 D01 的 Truck-320（251 个相机，320x178）；被加载的 PLY 来自 S04，因此继承 S04 所用的 `PGSR_STUDY_CPU_KNN=1` 初始化 workaround，结果仍属于学习实验而非基线复现。环境完整包版本见 `.cache/setup_env_20260930_195227.log` 的 `pip freeze` 部分。
- 完整渲染命令：`/mnt/e/PGSR/.envs/pgsr-1.0.0/bin/python scripts/study_driver.py pgsr.render -s /mnt/e/PGSR/data/truck-320 -d /mnt/e/PGSR/outputs/truck-densify1500 -i 1500 --backend gsplat --no_image_mask`。日志：`outputs/logs/render1500.log`。程序从 `outputs/truck-densify1500/point_cloud/iteration_1500/point_cloud.ply` 读取 Gaussian，写入 `outputs/truck-densify1500/ours_1500/`。tqdm 从 0/251 完成到 251/251，耗时约 7 分 34 秒；目录中有 251 张 RGB render、251 张 GT、251 张逆深度 TIFF，另有 251 行的 `quality.csv`。PyTorch 已分配/已保留显存峰值为 426.5/602.0 MiB。
- 几何样例由 `scripts/export_geometry_samples.py` 导出，使用脚本默认相机索引 0、125、250 和 `max_depth=10.0`，结果见 `outputs/truck-densify1500/acceptance/geometry/manifest.json`。该脚本对 PGSR 平面深度按 `finite & (depth > 0.1) & (depth <= 10)` 计为有效像素；三个视角的有效像素占比分别为 62.71%、72.66%、68.76%，有限深度占比均为 100%。有效像素深度的 p01/p50/p95（场景坐标尺度，非米制）分别为：相机 0 `2.023/3.071/8.057`，相机 125 `1.956/3.337/7.788`，相机 250 `1.535/3.938/6.262`。样例导出耗时 10.94 秒，已分配/已保留显存峰值为 415.7/602.0 MiB。完整图像样例写在同目录；脚本的准确有效条件见 `scripts/export_geometry_samples.py`。
- 观察与限制：`pgsr.render` 成功重载 PLY 并完成全部 251 个相机的渲染；这是 checkpoint 可读取和前向渲染可运行的证据。深度分布只来自 0、125、250 三帧，且统计分位数仅针对 `0.1 < depth <= 10` 的有效像素，不能当成全部 251 帧的总体分布；100% finite 也不表示每个像素都通过有效深度范围。`quality.csv` 中逐帧图像指标服务于本次短程检查，不是收敛或 benchmark 结果。TSDF 的执行结果单独记录在 S06。

## S06——iteration 1500 checkpoint 的 TSDF 融合与结构验收，已观察

- 日期：2026-10-01（Asia/Shanghai）；目的：验证从已保存的短训练 Gaussian 再渲染 PGSR 平面深度并提取可读取、非空三角网格。仓库、提交、环境、数据来源和种子与 S05 相同；新增隔离环境依赖 `libgomp=15.2.0=h4751f2c_8` 以解决 Open3D 0.19.0 导入时找不到 `libgomp.so.1` 的错误。安装后 `import open3d` 和 `pip check` 成功。无训练迭代；加载 iteration 1500 PLY。运行前 GPU 空闲约 2275 MiB，E: 约 31 GiB 可用。
- 完整 mesh 命令：`/mnt/e/PGSR/.envs/pgsr-1.0.0/bin/python scripts/study_driver.py pgsr.mesh -s /mnt/e/PGSR/data/truck-320 -d /mnt/e/PGSR/outputs/truck-densify1500 -i 1500 --backend gsplat --no_image_mask -o max_depth=10.0 -o mesh_res=256`；也可用 `bash scripts/run_local_study.sh mesh`。完整日志：`outputs/logs/mesh1500.log`。Open3D 逐帧积分 251/251，进度条报告融合段约 8 秒；PyTorch 已分配/已保留显存峰值 426.6/602.0 MiB。默认 `min_depth=0.1`，该场景 `scene_extent≈5.8434`，所以深度截断 10、体素边长 0.0390625、SDF 截断 0.1953125（COLMAP 场景单位）。
- 检查命令：`.envs/pgsr-1.0.0/bin/python scripts/inspect_mesh.py outputs/truck-densify1500/ours_1500`。原始 `fuse.ply` 有 1,348,676 顶点、1,688,137 三角面；后处理 `fuse_post.ply` 有 455,965 顶点、819,553 三角面。两者所有顶点有限，包围盒分别为 `[-11.465,-5.684,-11.191]` 至 `[13.301,8.613,11.426]`，以及 `[-10.527,-5.684,-8.496]` 至 `[13.301,7.715,10.527]`。网格路径见 `outputs/truck-densify1500/ours_1500/`。
- 可视化命令：`.envs/pgsr-1.0.0/bin/python scripts/visualize_acceptance.py --run-dir outputs/truck-densify1500/ours_1500 --geometry-dir outputs/truck-densify1500/acceptance/geometry`。产物为 `ours_1500/acceptance/visual_acceptance.png` 和 `visual_acceptance.json`。JSON 的 `status=ready` 只代表文件齐备及有限非空网格；选中首帧的 RGB MAE=0.04771、PSNR=22.68 dB，非论文评测。联系表显示 RGB 中 Truck 清晰可辨，平面深度和法线有对应结构；全局网格视图有明显碎片与大范围背景结构，不能判定车体表面完整。`mesh_acceptance.md` 记录全部界限和数据通路。
- 相机对齐检查命令：`.envs/pgsr-1.0.0/bin/python scripts/render_mesh_camera_acceptance.py --image 000001.jpg --image 000126.jpg --image 000251.jpg`。脚本以 CPU 对 `fuse_post.ply` 做 Open3D 三角面射线投射，按 COLMAP 位姿与内参对齐源 JPEG、渲染 RGB、GT 和网格顶点色；输出三张联系表以及命中 mask、相机 Z 深度、世界命中点、`summary.json` 和 `camera_mapping.csv`。三帧网格射线命中率为 56.5%、55.1%、37.1%；命中区域网格色对 GT 的 MAE 为 0.0600、0.0828、0.0911。源 JPEG 与运行目录 GT 像素完全一致；脚本峰值 RSS 约 616 MiB。三视角都能看出车体与相机图像总体对齐，但地面、背景和部分车体区域存在缺失，第三视角覆盖最低。完整路径与限制见 `notes/mesh_acceptance.md`。
- 结论：S06 的执行链路、非空 mesh 结构及车体可辨识的定性验收通过；视觉质量仍有显著缺陷，这与 1500 步学习短跑的非收敛定位相符，但不能仅凭训练步数断定具体缺陷成因。下一问题：若要提高重建质量，分别试验更长训练、场景裁剪和深度过滤，并用同一组三视角检查覆盖与误差。

## V01——离线交互 HTML 查看器，已观察

- 日期：2026-10-01（Asia/Shanghai）；目的：让 251 个注册相机视角的 RGB/GT 对照、三个深度/法线样例及两版 mesh 能在本地浏览器中交互查看。输入为 S05/S06 的 `outputs/truck-densify1500/ours_1500/`、`acceptance/geometry/` 和 `cameras.json`，继承 S04 的短训练及 CPU 3-NN 初始化限制。根检出提交 `de24f1a38b350387e8d8fe381b2cd70c1ae946e7`；环境为 `/mnt/e/PGSR/.envs/pgsr-1.0.0`。本次只用 CPU，无训练迭代或 GPU 显存峰值。
- 命令：`./.envs/pgsr-1.0.0/bin/python scripts/build_interactive_viewer.py`；后续为加入默认 Truck 局部视图，使用 `./.envs/pgsr-1.0.0/bin/python scripts/build_interactive_viewer.py --overwrite` 重建同一生成文件。输出 `outputs/truck-densify1500/interactive_viewer.html` 约 6.7 MB；内嵌已安装 Plotly JavaScript 和减面网格数据，图像使用相对路径按需加载。构建约 50 秒，峰值 RSS 约 1.4 GiB。
- 网格预览：`fuse.ply` 从 1,688,137 面减至 17,999 面；`fuse_post.ply` 从 819,553 面减至 18,000 面，保留顶点色。默认 Truck 局部坐标视口为 x=[-3.5,4]、y=[-1,2.5]、z=[-1.5,3]；用户可切换全场景。该选择只改变页面显示范围，不改 PLY 或嵌入数组。
- 验证：构建器检查 251 对 RGB/GT、三组几何样例和两个 PLY 文件均存在，两个内嵌数组通过 Plotly `Mesh3d` schema 校验；页面无外部资源引用。Windows Chrome 以 `file:///E:/PGSR/outputs/truck-densify1500/interactive_viewer.html` 无头加载，DOM 的 `body[data-viewer-ready]` 为 `true`，非 `error`；最终截图 `outputs/truck-densify1500/interactive_viewer_final_smoke.png` 可见 RGB/GT、深度样例和默认局部 3D 网格。浏览器与页面不需要联网或 GPU 训练。截图是初始化检查，不代表已人工逐帧查看 251 张图。
- 结论：本机离线交互页面可打开，图像相对路径及 3D 网格初始化通过。查看方式、数据来源和限制见 `notes/interactive_viewer.md`。下一问题：若要更精细地评价表面形状，直接用完整 PLY 在专门网格工具中检查，或以更多训练步数比较相同视角下的覆盖。
