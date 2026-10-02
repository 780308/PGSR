# PGSR 本地实验产物与 Git 证据清单

本清单记录 2026-10-02（Asia/Shanghai）整理的 Truck-320 短训练实验产物。整理开始前，根仓库 `HEAD` 为 `de24f1a38b350387e8d8fe381b2cd70c1ae946e7`。`notes/evidence/` 保存适合 Git 的紧凑证据副本；`outputs/` 和 `data/` 中的原件均未移动或删除。

本次实验是本地学习运行，不是论文基准复现：训练只进行 1,500 步，Gaussian 初始尺度使用了 `PGSR_STUDY_CPU_KNN=1` 的 CPU 三近邻替代路径。下面的数值只能证明训练、重载、渲染、PGSR 平面深度、TSDF 融合和验收链路实际执行过，不能解释为收敛质量或 PGSR 官方指标。

## 纳入 Git 的紧凑证据包

初始证据包共 21 个文件、3,454,593 bytes（约 3.29 MiB），低于 5 MiB 目标。下表及后面的日志表所列初始文件均从对应原件逐字节复制，并已用 SHA-256 核对副本与原件一致。2026-10-02 追加了 1 份由已保留原件重新统计得到的 CPU 审计 JSON，见下一节；它不是原有文件的逐字节副本。

| Git 路径 | 原始路径 | 内容 | bytes | SHA-256 |
| --- | --- | --- | ---: | --- |
| `notes/evidence/truck-preparation-manifest.json` | `data/truck-preparation-manifest.json` | Truck ZIP、解压成员、图像缩放与 COLMAP 一致性检查 | 94,540 | `613050b9d4870210feeaec3a4fb659645a4bc79ddca12ebeb4751065f2c83333` |
| `notes/evidence/training-summary.json` | `outputs/truck-densify1500/training-summary.json` | 训练损失快照、Gaussian 数量和显存峰值 | 2,531 | `d43041890e0b9d1ea732213fe0fa06f92a1dbbaaba88a1651a3b3e691bed14d2` |
| `notes/evidence/geometry-manifest.json` | `outputs/truck-densify1500/acceptance/geometry/manifest.json` | 3 个相机的平面深度覆盖率与分位数 | 1,654 | `bef6c4a2f9460f6dabd35726227e064dd64171f7e9ca2873ae32aa9a0be3889e` |
| `notes/evidence/render-quality.csv` | `outputs/truck-densify1500/ours_1500/quality.csv` | 251 个训练相机的逐帧渲染质量记录 | 15,870 | `44b44d6bcd91947390c7f258108bb9f2df4cd5dcad0bbf3f956c51072ad27356` |
| `notes/evidence/visual-acceptance.json` | `outputs/truck-densify1500/ours_1500/acceptance/visual_acceptance.json` | RGB/GT 抽样指标和两份完整 mesh 的结构检查 | 4,291 | `ea1a3368bf6ab65c1bf7b4cf655e121ba9fb43503f4c01d70c0b7e253a537676` |
| `notes/evidence/mesh-camera/summary.json` | `outputs/truck-densify1500/ours_1500/acceptance/mesh_camera/summary.json` | 3 个相机的外参、射线命中率、深度和颜色误差 | 12,553 | `c3e39a5102a28a5f3385fb101f62539bc02418097d66d8d4dd2abd7d9f2cdfd0` |
| `notes/evidence/mesh-camera/camera_mapping.csv` | `outputs/truck-densify1500/ours_1500/acceptance/mesh_camera/camera_mapping.csv` | COLMAP 图像、渲染序号与联系表映射 | 3,052 | `43d2c226895de8a46de7e68e82465944bb520bb99bc4171031a9c07c1bf37f4a` |
| `notes/evidence/mesh-camera/000001_contact_sheet.png` | `outputs/truck-densify1500/ours_1500/acceptance/mesh_camera/000001/contact_sheet.png` | 相机 `000001.jpg` 的 mesh—图像对齐联系表 | 330,427 | `2844a3eca9325352bf4d9579a994f679a038c708ced0fd6db762f9a886613c29` |
| `notes/evidence/mesh-camera/000126_contact_sheet.png` | `outputs/truck-densify1500/ours_1500/acceptance/mesh_camera/000126/contact_sheet.png` | 相机 `000126.jpg` 的 mesh—图像对齐联系表 | 323,914 | `b626862842333ceb34fac082431eea4871dab07f9a109203725d6b7f716d1ec3` |
| `notes/evidence/mesh-camera/000251_contact_sheet.png` | `outputs/truck-densify1500/ours_1500/acceptance/mesh_camera/000251/contact_sheet.png` | 相机 `000251.jpg` 的 mesh—图像对齐联系表 | 304,338 | `24145e5d4b864fca23394216461e90826ee2067b59236e6d342bfd65c3cd9f3f` |

三张联系表是学习验收用的可视化抽样，含 INRIA `tandt_db.zip` 中 Truck 场景经本项目缩放后的原始/GT 帧，以及本地渲染、命中 mask、mesh 顶点色投影和相机 Z 深度。上游数据来源为 `https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip`；将这些抽样放入 Git 不代表完整 Truck 图像集或完整数据包已纳入仓库。

这些副本保留了生成时写入的绝对 WSL 路径；路径用于追溯原始运行位置，换机阅读 JSON 时不应把它们当作当前有效路径。

### S07 追加：训练视角质量与全量网格拓扑审计

2026-10-02 为检验“HTML 看起来有车、全局 mesh 却碎裂”这一观察，在原有 `render-quality.csv`、`fuse.ply`、`fuse_post.ply` 上执行 CPU 只读统计。新证据 `notes/evidence/truck-reconstruction-audit.json` 为 **3,317 bytes**，SHA-256 为 `8d906a9072e2736c275071b7d0abe7f8f358887481c70e35a062a8497bcff10f`。其输入 PLY 的 SHA-256 已在本清单“大体积产物”表中列出；环境为 WSL Conda `pgsr-1.0.0`、Open3D 0.19.0，运行没有使用 GPU，也没有重新训练。复算脚本为 `scripts/audit_truck_reconstruction.py`，解释见 `notes/truck_reconstruction_diagnosis.md`。

```bash
.envs/pgsr-1.0.0/bin/python scripts/audit_truck_reconstruction.py \
  --quality-csv notes/evidence/render-quality.csv \
  --mesh-dir outputs/truck-densify1500/ours_1500 \
  --output notes/evidence/truck-reconstruction-audit.json
```

新增统计与初始证据包合计 22 个文件、3,457,910 bytes；上述输入和命令可在本机原件仍在时复算。S07 的实测结论是 251 个训练视角平均 PSNR `19.54145 dB`，原始网格共边连通片 `130,180` 个、最大连通片占 `13.61%` 三角面；它不包含 vanilla 3DGS 或留出视角对照。

### 纳入 Git 的原始运行日志

`notes/evidence/logs/` 共保存 11 份原始日志、2,361,423 bytes。`.log` 文件保留实际命令、版本、运行进度和终端输出；`.jsonl` 文件保留训练插桩的逐步机器记录。

| Git 路径 | 内容 | bytes | SHA-256 |
| --- | --- | ---: | --- |
| `notes/evidence/logs/smoke-gsplat.log` | `gsplat` 后端单视角前向/反向 smoke test | 1,064 | `f03879cf90b966fadb5f4fe69e62f6961570073ae5b7f2d954d30ce345e738a8` |
| `notes/evidence/logs/smoke-gsplat-2dgs.log` | `gsplat-2dgs` 后端单视角前向/反向 smoke test | 1,215 | `63887fe022f3952a434e19271c29a02c4fd879fc255108407a22eca608f1cb85` |
| `notes/evidence/logs/base20.log` | `base` 模式 20 步训练及 checkpoint 保存 | 1,784 | `49195ac05e7280e85c30b16cbf56c47f4b29f509058efae343e38bf697914dfe` |
| `notes/evidence/logs/geometry300.log` | 提前启用几何日程的 300 步训练终端日志 | 292,867 | `a4fdd20050c7ac47649c27a9cc3483f28f3ee7fdadcff74efc8d8456c3c52a23` |
| `notes/evidence/logs/geometry300.jsonl` | 300 步实验的稀疏损失/梯度插桩记录；normal 字段无效，见下文 | 5,116 | `23d427ea93dcc826e3763c1d0fddeed4d01347374bb120ce813927adaf85b939` |
| `notes/evidence/logs/densify1500.log` | `densify` 模式 1,500 步完整终端日志 | 1,979,355 | `ecfc033779a67f2ae5566dd75f85fdc6719cca0f3a32dc9f5520881a077703ce` |
| `notes/evidence/logs/densify1500.jsonl` | 1,500 步实验的稀疏损失和 Gaussian 数量记录 | 6,677 | `25b0335e5f8e5638946067288e84ffad0a8830c8d9c02cc06db05597e6e164f2` |
| `notes/evidence/logs/render1500.log` | iteration 1500 checkpoint 重载及 251 视角渲染 | 66,159 | `69a11efa1e2f34549ad26390fc0e82a45c57ce9305640bd411f076027bf99e1c` |
| `notes/evidence/logs/geometry_samples1500.log` | 三相机平面深度抽样的原始 JSON 输出 | 1,654 | `bef6c4a2f9460f6dabd35726227e064dd64171f7e9ca2873ae32aa9a0be3889e` |
| `notes/evidence/logs/mesh1500.log` | 251 视角平面深度的 TSDF 融合终端日志 | 5,245 | `cee2743dc697b036d44962d23f452a793357678275470e6a87cb11cd9c77058c` |
| `notes/evidence/logs/normal_gradient_recheck_20261002.log` | step 101 的 depth-normal 损失与 `xyz` 梯度隔离复核 JSON | 287 | `21d9ff6de927cb87a11455ede3e096eccd58d0b525e2ee2f83b41ca33cbdc7e4` |

`geometry300.jsonl` 中的 `terms.normal_total` 和 `xyz_grad_norms.normal_total` 受当时追踪辅助函数的原地加法别名错误影响，被错误记录为零，**不能用于判断 depth-normal 损失或梯度是否存在**。该文件的 normal 字段已知无效；独立修复后的证据以 `normal_gradient_recheck_20261002.log` 为准。不要用旧 JSONL 的零值覆盖独立复核结论。

## 仅在本机保存的大体积产物

下列文件继续保存在表中的仓库相对路径，但不纳入 Git。PLY 和 HTML 是可由已提交脚本重建的生成物，体积远大于紧凑证据包；Truck ZIP 是上游数据集原包，因体积和来源约束而避免在本仓库二次分发完整数据包。使用数据集时仍需遵守上游来源的许可和使用条款。

| 本地路径 | 作用 | bytes | SHA-256 | 不进 Git 的主要原因 |
| --- | --- | ---: | --- | --- |
| `outputs/truck-densify1500/point_cloud/iteration_1500/point_cloud.ply` | iteration 1500 Gaussian checkpoint，423,230 个 Gaussian | 104,962,571 | `fe71c73d903581f62799c7b6e59d2f92b72919e6ead1abc54252b218789dfe83` | 大体积、可重新训练生成；且继承 Truck 数据与本次 CPU KNN 实验条件 |
| `outputs/truck-densify1500/ours_1500/fuse.ply` | 251 视角 PGSR 平面深度的原始 TSDF 网格 | 58,360,304 | `271440cd4c27c9d3cc2b835ab1c13216c173aca61e590cfaabae514399904599` | 大体积生成物，可由 checkpoint 和相机重新融合 |
| `outputs/truck-densify1500/ours_1500/fuse_post.ply` | 连通片过滤后的 TSDF 网格 | 22,965,513 | `e737ad697e5a1b9cc16a28b43588514eec454b2f7d34bf5cae682cf08bfaa4bf` | 大体积生成物，可与原始网格一同重建 |
| `data/tandt_db.zip` | INRIA 发布的 Tanks and Temples 数据包，实验只使用其中 Truck 场景 | 682,628,995 | `816e62f22a161abbfe841d2a6b10cdf036e297c9fa289b3bfeee9c6ec526d7e1` | 上游数据包体积大；避免完整数据包二次分发，许可与使用条款由上游来源约束 |
| `outputs/truck-densify1500/interactive_viewer.html` | 内嵌 Plotly JavaScript 和两份减面网格的离线查看器 | 6,702,521 | `fd299c2085a488e0fba4a0111885edc38223a7009a92bcfd29aaf38dec44aad3` | 单文件生成物超过证据包总量，且依赖未提交的本地渲染和 mesh 目录 |

哈希核验可在 PowerShell 中运行：

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath <path>
```

精确重跑使用相同命令和环境仍可能因 GPU 算子、软件版本或训练随机过程产生不同字节；上表 SHA-256 用于确认当前本地文件身份，不承诺重建后的文件逐字节相同。

## 再生成路径与来源记录

### Gaussian checkpoint 与训练汇总

- 训练入口：`bash scripts/run_local_study.sh densify`
- 实际展开命令、包版本、随机种子、CPU KNN 实验声明与完整训练输出：`outputs/logs/densify1500.log`
- 逐步损失追踪：`outputs/logs/densify1500.jsonl`
- 紧凑训练汇总生成器：`scripts/summarize_training.py`
- 已提交的结果摘要：`notes/evidence/training-summary.json`

`run_local_study.sh densify` 当前将输出目录固定为 `outputs/truck-densify1500`，并会拒绝覆盖已有训练目录。若要保留当前产物并并行重建，应以 `outputs/logs/densify1500.log` 首行的展开命令为模板，同时把 `-d`、`PGSR_TRACE_LOG` 和终端日志改到新的实验 ID；不要删除当前 checkpoint 来换取重跑空间。

### Depth-normal 梯度隔离复核

2026-10-02 的独立复核运行成功完成。运行前 `nvidia-smi` 显示 RTX 3050 Ti Laptop GPU 总显存 4,096 MiB、已用 1,597 MiB、空闲 2,379 MiB。脚本在 step 101、相机 `000040.jpg` 上测得：

- 启用 depth-normal 项时总损失：`2.6583237648010254`
- 禁用 depth-normal 项时总损失：`2.6408612728118896`
- 隔离出的 depth-normal 损失：`0.017462491989135742`
- 该隔离项对 Gaussian `xyz` 的梯度范数：`0.004385402891784906`
- `xyz` 非零梯度元素数：`236166`

这组结果说明该固定 checkpoint、固定相机和固定 step 下，depth-normal 项非零且能把非零梯度传到 Gaussian 位置参数。它是一次局部机制复核，不是收敛性或跨视角统计。

复核命令如下；`exit ${PIPESTATUS[0]}` 保证 Python 即使经过 `tee` 也以自身退出码决定命令成功或失败。

```powershell
wsl -e bash -lc 'cd /mnt/e/PGSR && export XDG_CACHE_HOME=/mnt/e/PGSR/.cache/xdg TORCH_EXTENSIONS_DIR=/mnt/e/PGSR/.cache/torch_extensions TRITON_CACHE_DIR=/mnt/e/PGSR/.cache/triton TMPDIR=/mnt/e/PGSR/.cache/tmp && /mnt/e/PGSR/.envs/pgsr-1.0.0/bin/python scripts/verify_normal_gradient.py 2>&1 | tee outputs/logs/normal_gradient_recheck_20261002.log; exit ${PIPESTATUS[0]}'
```

- 原始输出：`outputs/logs/normal_gradient_recheck_20261002.log`
- Git 证据副本：`notes/evidence/logs/normal_gradient_recheck_20261002.log`
- 复核脚本：`scripts/verify_normal_gradient.py`

### 渲染、几何抽样与 TSDF 网格

- checkpoint 重载与 251 视角渲染：`bash scripts/run_local_study.sh render`
- 渲染源日志：`outputs/logs/render1500.log`
- 三视角平面深度抽样：`.envs/pgsr-1.0.0/bin/python scripts/export_geometry_samples.py`
- 几何抽样源日志：`outputs/logs/geometry_samples1500.log`
- TSDF 融合与两份 mesh：`bash scripts/run_local_study.sh mesh`
- mesh 源日志：`outputs/logs/mesh1500.log`
- 网格结构与可视验收：`.envs/pgsr-1.0.0/bin/python scripts/inspect_mesh.py outputs/truck-densify1500/ours_1500`，随后运行 `.envs/pgsr-1.0.0/bin/python scripts/visualize_acceptance.py --run-dir outputs/truck-densify1500/ours_1500 --geometry-dir outputs/truck-densify1500/acceptance/geometry`
- 三相机射线投影联系表：`.envs/pgsr-1.0.0/bin/python scripts/render_mesh_camera_acceptance.py --image 000001.jpg --image 000126.jpg --image 000251.jpg`
- 验收解释与边界：`notes/mesh_acceptance.md`

`mesh` 阶段的实际参数由脚本固定为 `max_depth=10.0`、`mesh_res=256`，并从 Gaussian PLY 逐相机重新渲染平面深度后做 Open3D TSDF 融合；它不直接读取已导出的逆深度 TIFF。

### Truck 原始 ZIP 与 320 像素派生数据

- 上游来源：`https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip`
- 下载、断点续传、解压、COLMAP 元数据缩放与一致性检查：`python scripts/prepare_truck_dataset.py`
- 来源、下载过程、相机缩放公式与验证记录：`notes/data/truck-320.md`
- 本地机器可读准备清单：`data/truck-preparation-manifest.json`
- Git 内机器可读副本：`notes/evidence/truck-preparation-manifest.json`

数据准备过程没有保留独立终端日志；上述中文记录和机器清单是现存的来源证据。脚本默认复用校验通过的完整 ZIP，并拒绝覆盖已有派生目录。

### 离线交互查看器

- 构建：`.envs/pgsr-1.0.0/bin/python scripts/build_interactive_viewer.py`
- 明确重建已有文件：`.envs/pgsr-1.0.0/bin/python scripts/build_interactive_viewer.py --overwrite`
- 构建输入：`outputs/truck-densify1500/cameras.json`、`outputs/truck-densify1500/ours_1500/`、`outputs/truck-densify1500/acceptance/geometry/` 和两份完整 mesh
- 使用方式、减面规模、Chrome `file:///` 验证和限制：`notes/interactive_viewer.md`
- 本机验证截图：`outputs/truck-densify1500/interactive_viewer_final_smoke.png`

查看器构建没有保留独立终端日志；生成命令嵌入 HTML 元数据，并已写入 `notes/interactive_viewer.md`。查看器只用于本地交互检查，移动 HTML 时还必须保留其引用的相对目录结构。
