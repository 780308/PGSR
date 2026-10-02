# Truck-320 的张量与训练数据通路

本记录以 2026-09-30 的本机训练张量实验为主，并补充 2026-10-01 的 checkpoint 重载、渲染、网格结果和 2026-10-02 的固定 checkpoint 法线梯度重检。环境：`/mnt/e/PGSR/.envs/pgsr-1.0.0`，`pgsr==1.0.0`、`gaussian-splatting==2.8.4`、`gsplat==1.5.3+pt24cu121`、PyTorch 2.4.1+cu121。源码依据：官方 `PGSR@de24f1a`、重构包 `PGSR-python@497f926` 和已安装的 `gaussian_splatting`。下文将**观察**（有命令与日志）和**代码推断**（尚未单独插桩）分开。命令见 `scripts/run_smoke.sh`、`scripts/run_local_study.sh`；全程总结见 `notes/study_summary.md`，逐次记录见 `notes/experiment_log.md`。

本轮从**已有 COLMAP 标定的 RGB**开始，未执行特征匹配、位姿估计或稀疏重建。训练停在 1500 步增密；随后已从该步 PLY 重载并渲染 251 帧，再以重新渲染的 PGSR 平面深度融合 TSDF mesh。后两项的观察和质量边界见 `notes/mesh_acceptance.md`。

```text
Truck JPG + sparse/0/{cameras,images,points3D}.bin
  ├─ prepare_dataset → 251 个 Camera（RGB、K、外参、相机中心）
  └─ colmap_init → 136029 个 Gaussian 的 6 组可训练参数
prepare_trainer(mode=base/densify) → RGB/平面/几何损失包装链
training → 抽取相机 → model(camera) → RGB、alpha、法线、距离、平面深度
         → loss → backward → 增密前钩子（densify 模式）→ Adam.step
         → 指定迭代保存 point_cloud.ply + cameras.json
         → 重载 PLY 后渲染 RGB/逆深度；再次渲染平面深度后融合 TSDF mesh
```

## 1. RGB 与相机

`pgsr/train.py:prepare_training` 依次调用 `gaussian_splatting.prepare.prepare_dataset`、`pgsr.prepare.prepare_gaussians`、`pgsr.prepare.prepare_trainer`。未给 `--load_camera` 时，`prepare_dataset` 选 `ColmapCameraDataset`；`gaussian_splatting/dataset/colmap/dataset.py:read_colmap_cameras` 读取 `images.bin` 和 `cameras.bin`，`camera.py:build_camera` 再读取 JPG、建立相机张量。本轮命令显式使用 `--no_image_mask --no_depth_data`，因此没有 mask 或真值深度输入；后文的 `out["depth"]` 全是模型渲染结果。

| 数据 | Truck-320 值与形状 | 坐标、单位与用途 | 证据 |
| --- | --- | --- | --- |
| 注册视图/RGB | 251 张；单视图 `ground_truth_image:[3,178,320]`，float32、RGB、范围 `[0,1]` | `[C,H,W]`；监督 `out["render"]` | `outputs/logs/smoke-gsplat.log`；`gaussian_splatting/utils/general.py:read_image` |
| 内参 | `K:[3,3]`；派生 `(fx,fy,cx,cy)=(190.21027745,188.65069852,160,89)` | 焦距/主点单位为派生图像的像素；用于 rasterization 和像素射线 | `notes/data/truck-320.md`；`camera.py:build_camera` |
| 外参与中心 | `R:[3,3]`、`T:[3]`、`world_view_transform:[4,4]`、`camera_center:[3]` | COLMAP 世界系到相机系；`world_view_transform` 以转置布局存储，`gsplat` 前向传入 `.T` 后的 `viewmats:[1,4,4]` | `camera.py:build_camera`；`pgsr/models/gsplat.py:render`；形状为代码推断 |
| 数据驻留 | `ColmapCameraDataset` 构造相机列表并 `.to(device)` | 当前共享库把所有视图的图像放到指定设备，而非每步从磁盘延迟载入 | `dataset/colmap/dataset.py:ColmapCameraDataset`，代码推断 |

原始 JPG 是 979×546，但原始 COLMAP 标定为 1957×1091。派生脚本把图像变为 320×178，并同步缩放内参与二维观测，保留位姿、三维点、点轨迹；详细投影误差见 `notes/data/truck-320.md`。本场景派生主点恰在 `(W/2,H/2)`，与共享库 `getK` 用图像中心重建主点的做法一致；对偏心主点数据不能据此假定仍精确。

## 2. COLMAP 点与 Gaussian 状态

`pgsr/prepare.py:prepare_gaussians` 按 backend 选择 PGSR Gaussian 子类；无 `--load_ply` 时由 `gaussian_splatting.dataset.colmap.colmap_init` 读取 `points3D.bin` 的 XYZ/RGB，调用 `GaussianModel.create_from_pcd`。XYZ 保留 COLMAP 世界系和场景单位，RGB 除以 255 后编码为 SH 的 DC 系数。**观察：**初始共有 `N=136029` 个 Gaussian。`sh_degree=3` 为每个颜色通道预留 16 个 SH 系数；`SHLifter` 从 0 阶开始逐步开放更高阶，不改变存储形状。

| 可训练张量 | 初始形状 | 激活后的意义及变化 |
| --- | --- | --- |
| `_xyz` | `[N,3]` | 世界系中心；Adam 更新，增密/裁剪改变数量 |
| `_features_dc` | `[N,1,3]` | 0 阶 SH 颜色；Adam 更新、增密复制 |
| `_features_rest` | `[N,15,3]` | 其余 SH 颜色；启用相应阶数后参与前向 |
| `_opacity` | `[N,1]` | logit，经 `sigmoid` 为 alpha；可由优化、裁剪和重置流程处理 |
| `_scaling` | `[N,3]` | log scale，经 `exp` 为三轴长度；最短轴定义 PGSR 平面法线 |
| `_rotation` | `[N,4]` | 四元数，前向归一化；决定 Gaussian 及平面方向 |

形状从 `gaussian_splatting/gaussian_model.py:create_from_pcd` 推断；S01 实测的是初始数量、渲染输出形状和有梯度的参数组数，并未逐项保存六组参数的运行时形状。`gsplat-2dgs` 子类把第三个尺度设为极小值，法线取旋转矩阵第三轴；`gsplat` 后端从三个当前尺度中选最短轴。两套参数化不能预设逐像素结果相等。

**本机初始化限制：**依赖自带的 `simple_knn.distCUDA2` 二进制只有 SM75，RTX 3050 Ti（SM86）会报错。S01–S04 显式用 `PGSR_STUDY_CPU_KNN=1`，由 `scripts/study_knn_fallback.py` 的 CPU 精确三近邻均方距离求初始尺度。它可能改变数值，因此是标记过的学习实验，并非未修改的 PGSR 基线；已安装的训练器和 rasterizer 文件未改动。

## 3. 一次前向：Gaussian 到 RGB 与平面深度

`GaussianModel.forward(camera)` 取激活后的中心、opacity、尺度、旋转、SH，交给 `pgsr/models/gsplat.py:GsplatPGSRGaussianModel.render`。SH 与视线方向生成每个 Gaussian 的 RGB。`pgsr/gaussian_model.py:plane_params` 从最短尺度轴取世界法线、翻转使其朝向相机，再变换为相机系 `(normal_x,normal_y,normal_z,distance)`，形状 `[N,4]`。模型将 RGB 与这四个值拼为 7 通道，由 `gsplat.rasterization` alpha 合成。调用虽使用 `render_mode="RGB+ED"`，但 PGSR 的 `out["depth"]` **并非**附带的普通期望深度 ED 通道。

对像素 `(x,y)`，射线 `r=((x-cx)/fx,(y-cy)/fy,1)`；令 alpha 合成的平面法线/距离为 `n̄,d̄`，`pgsr/models/gsplat.py:render_plane` 计算 `z=-d̄/(n̄·r+1e-8)`。这是以相机 z 为参数的射线交点深度，长度单位随 COLMAP 场景单位；它不是欧氏射线长度或真值深度。比值在单一平面的理想情况下可约掉共同 alpha 权重；多层混合、低 alpha 和退化分母仍有影响。与论文的“无偏深度”解释见 `notes/source_map.md`。

| `out` 键 | S01 `gsplat` 实测形状 | 含义与消费者 |
| --- | --- | --- |
| `render` | `[3,178,320]` | 经相机后处理并截到 `[0,1]` 的 RGB；基础 L1+SSIM 损失 |
| `render_alphas` | `[1,178,320]` | 累计 alpha；给深度导出法线加权，不是二值可见掩码 |
| `render_normals` | `[3,178,320]` | alpha 合成的相机系平面法线；用于深度法线一致性 |
| `rendered_distance` | `[1,178,320]` | alpha 合成的相机到平面距离；与法线/射线求交 |
| `depth`、`invdepth` | 各 `[1,178,320]` | 平面交点 z 深度及安全倒数；用于重投影、邻居缓存、后续 mesh 输入 |
| `normals_from_depth` | `[3,178,320]` | 从相邻深度重建法线，再乘 `render_alphas.detach()`；深度路径仍可求导 |
| `radii`、`visibility_filter` | `[N]`、`[105653,1]` | 屏幕半径、该视图可见 Gaussian 索引；尺度惩罚和增密统计 |
| `means2d` | `[1,N,2]` | 保留梯度的屏幕投影；`get_viewspace_grad` 供增密判据使用 |

S01 在独立进程检查 `gsplat-2dgs`：主 RGB/平面图形状相同，可见索引 `[105996,1]`；额外暴露 `gradient_2dgs:[1,N,2]`，以及 `render_distort`、`render_median` 各 `[1,178,320,1]`。这两个额外输出来自 2DGS rasterizer；本轮未证明 PGSR 损失使用它们。两个 backend 都以 `out["render"].mean().backward()` 完成真实 Truck 相机的前向/反向，但该均值只是烟雾测试目标，不是训练损失。见 `outputs/logs/smoke-gsplat.log`、`smoke-gsplat-2dgs.log`。

## 4. 一次训练迭代：损失、反向与优化器

`pgsr.train` 把数据集、Gaussian 和训练器传给 `gaussian_splatting.train.training`。循环按 epoch 打乱视图索引；`trainer.step(dataset[idx])` 在 `gaussian_splatting/trainer/abc.py:AbstractTrainer.step` 中执行：

1. `update_learning_rate()` 先增加 `curr_step`，更新 XYZ 的指数衰减学习率；`preprocess` 设置背景。
2. `out=model(camera)` 可微渲染 RGB 与平面几何图；PGSR 加入平面与几何，依赖包提供 Gaussian 容器与训练框架。
3. `loss(out,camera)` 先算光度项，再由 PGSR 包装层加平面尺度、深度法线、多视图、虚拟相机项；门槛和有效像素决定某一项是否实际非零。
4. `loss.backward()` 经渲染器反传至 Gaussian 参数。S01 的 RGB 均值测试有六组 `.grad`；`gsplat` 四组、`gsplat-2dgs` 五组非零。S03 的几何项独立测试另外观察到 `_xyz` 非零梯度。
5. `densify` 模式的 `before_optim_hook` 在 backward 后用屏幕梯度、可见性、半径等生成新增/移除/替换指令，调整模型参数及 Adam 状态；随后 `optimizer.step()` 与 `zero_grad(set_to_none=True)`。新建参数在本轮 backward 后才进入容器，下一轮才有自身梯度（代码推断）。
6. 外层 `training()` 计算仅用于日志的 PSNR，在指定步数保存 PLY 和 `cameras.json`。PLY 只有 Gaussian 参数，没有 Adam 动量、调度器状态或邻居缓存，不能视为完整训练恢复点。

光度项在 `gaussian_splatting/trainer/base.py:BaseTrainer.loss` 中为 `L_photo=0.8·L1(render,RGB)+0.2·(1−SSIM(render,RGB))`（默认 `lambda_dssim=0.2`）。`pgsr/trainer/scale.py:PlanarScaleTrainer.loss` 对可见 Gaussian 的最小激活尺度取均值并乘默认权重 100，促使椭球变薄。深度法线默认 `curr_step>7000` 才加入；多视图和虚拟相机默认从 7000 开始。多视图层每次见到相机先缓存降采样且已 `detach` 的深度，按间隔筛邻居，再对选中的邻视图重新可微渲染；虚拟相机层根据当前可见深度生成新视角并计算重投影误差。对应代码：`pgsr/trainer/depth_normal_consistency.py`、`multi_view/trainer.py`、`reprojection.py`。

| 运行 | 配置与观察 | 实际证明的范围 |
| --- | --- | --- |
| S02 `base` 20 步 | 默认几何门槛未到；`N=136029`；保存 iteration 20；PyTorch 峰值 378/482 MiB（已分配/已保留） | 光度+平面、反向/优化器/保存闭环；不证明几何项已触发 |
| S03 `base` 300 步 | 三个几何起点改到 100，邻居更新间隔 100；缓存覆盖 251 视图；step 300 多视图、虚拟相机的 `_xyz` 梯度范数分别为 `2.01e-5`、`0.00620` | 几何计算进入反向；不代表收敛 |
| S03 固定 checkpoint 开/关测试；10-02 重检 | 固定 `geometry300` PLY 和 `000040.jpg`，在 step 101 仅切换深度法线项；损失差 `0.017462491989135742`，其 `_xyz` 梯度范数 `0.004385402891784906`，非零元素 236,166；原始 JSON 为 `outputs/logs/normal_gradient_recheck_20261002.log` | 验证深度法线参与梯度；旧 `geometry300.jsonl` normal 列因 `+=` 别名错误不可作证据 |
| S04 `densify` 1500 步 | 仍用提前启用的几何配置；`N:136029→423230`；峰值 1284.7/1660.0 MiB；PLY header 423230 顶点 | 证明数量变化和末步保存；训练 PSNR 不是基准结果 |

## 5. 增密、检查点与可证伪实验

`base` 模式只有损失包装层；`densify` 模式经 `pgsr/trainer/combinations.py` 接入共享库的 split/clone、opacity prune、PGSR 的 `MultiViewTrimmer` 和共享库的 `OpacityResetter`。共享 `SplitCloneDensifier` 默认第 500 步起每 100 步判断屏幕梯度与尺度；`OpacityPruner` 默认第 1000 步起每 100 步判断 opacity/大小；PGSR trimmer 默认每 1000 步对所有视图扫描并合并低可见次数移除掩码。S04 日志确有 `Multi-view trim` 进度条，但**没有分别记录每类新增/移除的点数**。opacity reset 默认第 3000 步才开始，因此 1500 步内按代码日程不会触发，不能声称本轮验证了它。

S04 记录第 500 步后 139,984，第 1000 步后 254,494，第 1500 步后 423,230 个 Gaussian。step 1500 的损失 trace 中 391,324 是该步增密**之前**的数量；PLY header 中 423,230 是 `before_optim_hook` 后保存的数量。详见 `outputs/truck-densify1500/training-summary.json` 和 `outputs/truck-densify1500/point_cloud/iteration_1500/point_cloud.ply`。

可用短检查证伪错误解释：固定相机和参数，只切换深度法线项并比较损失及 `_xyz` 梯度（**已做**）；固定像素的 `n̄,d̄,K` 手算交点并比对 `out["depth"]`（待做）；记录 `DensificationInstruct` 的各类新增/移除/替换 mask 数量，解释净增量（待做）；加载 step 1500 PLY 并完成全视角渲染、三帧深度抽样、TSDF 融合（**已做**，见 `notes/mesh_acceptance.md`）。本轮没有保存训练结束前与重载后同一相机的逐像素 RGB/深度对照，因此不能据此声称重载完全数值无损。
