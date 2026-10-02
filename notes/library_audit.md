# `pgsr==1.0.0` 对官方 PGSR 的整合审计

审计日期：2026-10-01。比较对象为官方 `upstream/PGSR-official@de24f1a38b350387e8d8fe381b2cd70c1ae946e7`、重构源码 `upstream/PGSR-python@497f926782753558c2f0d5d19c0234088c52a058`、本机安装的 `pgsr==1.0.0` 与 `gaussian-splatting==2.8.4`。`scripts/verify_package_source.py` 已验证安装的 PGSR 与发布提交的 31 个 Python 文件对应。以下结论来自源码对照；Truck-320 的 S01–S06 已验证重构包自身的前向、反向、几何项、增密、PLY 重载、渲染与 TSDF 执行链路，**没有**在相同条件下运行官方 CUDA 版作逐像素、逐梯度或指标对照。全程总结见 `notes/study_summary.md`。

## 结论与适用范围

`pgsr` 将官方方法的核心思路正确拆分成数据集/模型/训练器：最短尺度轴定义平面法线，渲染法线与平面距离后求射线交点深度，并用单视图和多视图几何约束优化 Gaussian。这是适合学习调用链的实现。它**不是官方实现的逐项等价移植**：渲染后端、邻居选择、损失权重细节、增密/裁剪日程、可选能力与输出安全性均有差异。它的默认训练结果不能直接标为官方 PGSR 复现；若需对照论文指标，必须独立做同数据、同配置和相同评测路径的实验。

| 审计面 | 官方实现 | 重构库 | 判断 |
| --- | --- | --- | --- |
| 平面方向与参数 | `scene/gaussian_model.py:145-158` 选最短轴，朝相机翻转；`gaussian_renderer/__init__.py:132-139` 建相机系法线和距离 | `pgsr/gaussian_model.py:38-63` 对应计算 | 核心公式对齐；数值等价尚未测 |
| 无偏深度 | 官方 rasterizer `submodules/diff-plane-rasterization/cuda_rasterizer/forward.cu:373-404` alpha 累加平面参数并作射线求交 | `pgsr/models/gsplat.py:13-45,78-116` 以 gsplat 的附加 4 通道合成，`render_plane` 求交 | 思路和公式对齐；不同 CUDA rasterizer 的排序、抗锯齿及数值细节仍需对照 |
| 平面尺度与深度法线 | `train.py:177-196`，最小尺度权重、法线损失及第 7000 步附近门槛 | `pgsr/trainer/scale.py:12-26`、`depth_normal_consistency.py:12-49` | 主要公式和默认权重对应；S03 已验证重构版法线项的梯度 |
| 数据/后端 | 官方场景支持 COLMAP 与 Blender，并使用自定义 `diff-plane-rasterization` | 依赖 `gaussian_splatting.prepare` 的 COLMAP 入口；提供 `gsplat`、`gsplat-2dgs` | 是接口和后端重构，不应把 `gsplat-2dgs` 视为官方 rasterizer 的同义替换 |

## 会改变行为或结果的差异

1. **高：同目录重新训练会删除旧检查点。** `pgsr/train.py:56-59` 调用共享 `gaussian_splatting.train.training`，后者在 `train.py:42-43` 先执行 `shutil.rmtree(destination/point_cloud, ignore_errors=True)`。再次用相同 `-d` 直接启动库入口，会删掉该目录先前的所有 PLY。这是依赖训练入口的实际副作用，与算法等价性无关，却直接影响实验可追溯性。已在本地 `scripts/run_local_study.sh` 对 `base`、`geometry`、`densify` 增加保护：若目标已有 `point_cloud`，脚本在调用库入口前拒绝运行；直接调用库入口时仍须使用新输出目录。
2. **高：增密和剪枝并非官方规则。** 官方 `scene/gaussian_model.py:415-506` 有普通梯度与绝对梯度扩展、总点数上限 600 万、绝对梯度分裂数量上限、clone 扰动等路径；依赖 `gaussian_splatting/trainer/densifier/densifier.py:16-23,50-129` 主要使用普通屏幕梯度，默认无总点数上限，clone 为原位复制。`percent_dense` 官方默认 `0.001`（`arguments/__init__.py:121`），依赖默认 `0.01`；opacity 学习率官方 `0.05`、共享 `BaseTrainer` 默认 `0.025`。官方 `train.py:360-370` 的增删点在 `iteration>500` 且 `<15000`、每 100 步触发，屏幕尺寸剪枝阈值 20 在 3000 步后才用；共享实现默认从第 500 步增密、1000 步起使用屏幕半径 20 剪枝，并在第 15000 步仍可运行。S04 的净点数增长不代表这些机制与官方相同。
3. **高：多视图 trimming 的“被观察”语义不同。** 官方 CUDA 在 `forward.cu:381-384` 按透射率条件累计 `out_observe`，`train.py:373-384` 据此 trim；重构 `pgsr/trainer/trim/trainer.py:34-47` 在每个相机上用 `radii>0` 的 `visibility_filter` 计数。进入视锥并有屏幕半径不等于通过官方透射率条件，因此移除集合可不同。本轮日志证明扫描调用，未记录两种 mask 的对照。
4. **中：邻视图选择改变。** 官方 `scene/__init__.py:82-114` 按相机位置/朝向先选邻居；重构 `pgsr/trainer/multi_view/trainer.py:114-180` 缓存模型深度、用重投影可见点数动态排名。训练早期深度较差时，邻居集合和正则启用情况可改变。
5. **中：多视图损失有额外缩放和采样变化。** 官方 `train.py:269-271,326-330` 对有效像素/patch 求均值；重构 `pgsr/trainer/multi_view/reprojction/abc.py:20-35`、`geometric.py:35`、`photometric.py:171` 还乘有效重投影比例，低重叠时损失更弱。官方邻视图深度采样见 `scene/gaussian_model.py:524-545` 的双线性 `grid_sample`；重构 `pgsr/utils/reproj.py:84-111` 用最近邻，并过滤深度范围，因而像素对应集及梯度不会逐项相等。
6. **中：虚拟相机约束的默认行为改变。** 官方 `arguments/__init__.py:128-161` 默认不启用虚拟相机；重构 `pgsr/trainer/combinations.py:22-31` 无条件组装 `VirtualCameraReprojectionTrainerWrapper`，到 `pgsr/trainer/reprojection.py:204-224` 的默认第 7000 步起点后便可能增加一项。其采样方式也不是官方训练脚本的逐句复制。短跑 S03 为学习目的主动提前门槛，不能用来说明官方默认也会触发。
7. **中：输入/可选功能范围变窄。** 官方支持 Blender 合成场景、白背景开关和可选外观/曝光补偿；重构当前准备入口主要按 COLMAP 建场景，默认相机黑背景（共享 `camera.py:build_camera`）。论文的曝光补偿是贡献之一，但这一版的 PGSR 训练器未移植官方可选 appearance model。官方 `_knn_f` 参数未在渲染或损失中使用，可将其缺失视为格式/实现差异，而非已证实的核心损失缺失。
8. **中：示例参数可能静默无效。** 重构 README 的 `-o opacity_cull_threshold=0.05`（`README.md:116`）与实际 `gaussian_splatting/trainer/densifier/pruner.py` 的 `prune_opacity_threshold` 不同；链尾 `NoopDensifier` 接受多余 `**configs`，不一定报错。学习实验必须在训练器实例中核对覆盖值，不能仅凭命令行字符串判断生效。
9. **中：安装依赖没有固定可复现组合。** 重构 `pyproject.toml` 仅要求 `gaussian-splatting>=2.8.4`，对 `gsplat`、`torch`、`numpy` 等未设精确版本。未来解析到更新依赖时，以上训练/渲染行为可能改变。本机观察只适用于 `notes/environment_setup.md` 记录的版本组合，复现实验时应保留环境锁定清单。
10. **低：mesh 融合颜色的球谐阶数不同。** 共享 `gaussian_splatting/mesh.py:19` 在融合前把 `active_sh_degree` 设为 0，只用 SH DC 渲染用于 TSDF 的 RGB；官方 `scene/gaussian_model.py:314` 加载 PLY 后将活动阶数设为最大阶，`render.py` 用该模型渲染。几何融合仍取平面深度，但 mesh 顶点颜色可能不同。

上述“高/中”表示对复现或数据保护的影响，不表示已测得网格质量下降。两个 backend、四种 mode 的存在也不代表四种组合均经过本机训练验证：目前有两个 backend 的单视角前向/反向，`gsplat` 后端的 `base` 与 `densify` Truck 短训练；相机优化模式和官方逐项对照未执行。S05/S06 已从 `densify` 的 iteration 1500 PLY 重载，完成 251 帧渲染与 TSDF 融合；可读取的非空网格和视觉缺陷见 `notes/mesh_acceptance.md`，不能当作收敛质量对照。

## 核对方法与下一步

已运行的最小证据：`outputs/logs/smoke-gsplat.log`、`smoke-gsplat-2dgs.log`、`base20.log`、`geometry300.log`、`densify1500.log`、`render1500.log`、`mesh1500.log`，以及 `outputs/truck-densify1500/training-summary.json` 和 `ours_1500/acceptance/visual_acceptance.json`。详情见 `notes/study_summary.md`、`notes/tensor_trace.md` 与 `notes/experiment_log.md`。本机训练使用标记过的 CPU 三近邻初始化；它又引入一个相对官方 CUDA 初始化的数值差异。

若需要判定“与官方输出等价”，应另建隔离实验：统一数据、相机列表、随机数、背景、学习率和增密日程；先关闭所有几何/增密，只对同一相机比较平面参数、alpha、RGB、深度及梯度；再逐项打开损失与增密，比较有效像素、邻居集合、各新增/移除 mask。当前审计只能得出**核心算法思想已被整合、默认训练行为不等价**的结论。
