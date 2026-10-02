# Truck-320 短训练：RGB 质量推断与碎片网格诊断

实验 ID：S07（CPU 只读审计）。

日期：2026-10-02。分析对象是 `pgsr==1.0.0` 的 `gsplat` 后端、`densify` 模式，在 251 张 `320×178` 的 Truck 已标定图像上训练 1500 步后保存的模型。这里的目标是解释一次学习实验，**不是**论文基准评测。训练与提取的完整命令分别保存在 [`densify1500.log`](evidence/logs/densify1500.log) 和 [`mesh1500.log`](evidence/logs/mesh1500.log) 的首行；根项目提交为 `de24f1a38b350387e8d8fe381b2cd70c1ae946e7`，重构库发布源码为 `497f926782753558c2f0d5d19c0234088c52a058`。

## 先给判断

视觉渲染可以接受，这与本次网格失败并不矛盾：RGB 拟合允许不同深度与不连续表面产生近似颜色，TSDF 却要求各视角的深度在同一三维表面上相互吻合。当前证据只支持 **RGB/训练/提取执行链路通过，物体级 Truck mesh 质量不通过**。后处理网格从某些训练相机看能拼出局部车体，但全局并不是完整、易辨识的 Truck。

本轮没有 vanilla 3DGS、官方 PGSR 或留出视角的同条件对照。对 3DGS 的优劣只能给方向性推断，不能填入虚构的 PSNR、SSIM、LPIPS 或 mesh 指标。全量统计及复算脚本见 [`truck-reconstruction-audit.json`](evidence/truck-reconstruction-audit.json) 与 [`audit_truck_reconstruction.py`](../scripts/audit_truck_reconstruction.py)。

## 1. 与 vanilla 3DGS 相比，能推断到什么程度？

| 比较对象 | 本次 PGSR 实测 | 同条件 vanilla 3DGS 的谨慎推断 |
| --- | --- | --- |
| 251 个**训练视角** RGB | 平均 PSNR `19.54 dB`、SSIM `0.7171`、LPIPS `0.1778`；PSNR 中位 `19.51 dB`，`165/251` 帧低于 `20 dB`。这些值来自 [`render-quality.csv`](evidence/render-quality.csv)，是逐帧指标的算术均值，不是留出视角性能。 | 只有颜色目标的 vanilla 3DGS，在相同 1500 步、相同数据和可比训练设置下，训练视角 RGB **可能相近或略好**：它不必同时满足平面正则和提前施加的跨视角约束。但优化轨迹、增密和初始化会影响结论，不能断言一定赢或给出差值。 |
| 新视角视觉质量 | 未测；离线 HTML 和目前的渲染指标都不足以代表新视角。 | 无可靠胜负推断。PGSR 的几何约束若充分训练，可能改善结构与视角一致性；1500 步实验不能验证这一收益。 |
| 深度与 mesh | 本轮 PGSR 的平面深度经 TSDF 得到非空但严重碎片化的网格；没有真值 mesh、前景 mask 或几何精度指标。 | vanilla 3DGS 本身不直接输出可比的物体级 mesh。若用它的深度走**同一相机、同一过滤与 TSDF 管线**，PGSR 的方法设计有几何优势的理由，但此处既不能证明优势，也不能断言 3DGS 会更差。 |

这个比较应把“同条件”落实为同一 Truck-320/COLMAP 划分、训练与留出相机、分辨率、1500 步和随机种子、可比优化/增密预算与统一图像指标。算法专属损失必须如实保留并记录；比较 mesh 时还需相同的物体 ROI、有效深度/遮挡筛选、TSDF 参数和后处理。当前 PGSR 的几何损失阈值人为由默认约 7000 步提前到 100 步，且使用 `PGSR_STUDY_CPU_KNN=1` 初始化替代，因此它不是官方默认日程的结果。

## 2. 为什么画面可接受，网格却布满碎片？

**实测严重程度。** 对原始 `fuse.ply` 的全部 1,688,137 个三角面用 Open3D `cluster_connected_triangles()` 按共边关系分簇，得到 **130,180** 个连通片；每簇面数中位数仅 **4**，129,615 个不足 100 面的片共占 **44.27%** 的三角面。最大一片仅 229,742 面，占 **13.61%**。原始网格的边界边为 752,307/2,908,359 条独立边（**25.87%**）。这不是全局截图抽样造成的错觉。后处理 `fuse_post.ply` 仍有 50 片、819,553 面、108,267 条边界边，且三帧完整网格射线命中率只有 **56.5% / 55.1% / 37.1%**；这些命中率是整幅图的覆盖率，不能冒充车体表面召回率。面积与尺寸均为 COLMAP 场景单位，不能写成平方米或米。细目见统计 JSON、[`mesh-camera/summary.json`](evidence/mesh-camera/summary.json) 和[相机联系图](evidence/mesh-camera/000001_contact_sheet.png)。

按证据强弱，原因分为以下几层：

1. **训练状态尚早，是几何不稳定的高优先级解释。** 初始 136,029 个 Gaussian 在 1500 步后增至 423,230 个（约 3.11 倍）；第 1500 步增密前为 391,324，新增约 31,906 个点后便保存，新增点尚未经过后续优化。[`training-summary.json`](evidence/training-summary.json) 记录了数量与损失轨迹。1500 步仅为常见 30k 日程的 5%，RGB 能形成外观并不意味着沿各相机射线的同一表面已经稳定。Gaussian 数量增长本身不能证明新增点一定是漂浮点。
2. **监督并非缺席，却是在不成熟几何上提前施加。** 本轮命令把法线/深度、多视图和虚拟相机项的启用阈值均设为 100，邻居更新间隔设为 100；它不是“1500 步前完全没有几何损失”。第 101 步的独立法线项在 [`normal_gradient_recheck_20261002.log`](evidence/logs/normal_gradient_recheck_20261002.log) 中产生非零 `xyz` 梯度；第 1500 步单相机 trace 有 photometric `0.07797`、planar `0.14092`、normal `0.01215`、multi-view `0.03503`、virtual `0.00109`。这些标量不能直接比较梯度主导性，也不能证明各相机的平面深度已一致。早启用项可能反馈早期错误深度，仍需消融验证。
3. **融合把大量未经可靠性筛选的全场景深度送入 TSDF。** `gaussian_splatting/mesh.py:29–39` 逐相机重渲染 `out["depth"]`，只检查 `0.1 < depth ≤ 10` 与可选图像 mask；本次显式 `--no_image_mask`，也没有深度真值、LiDAR、alpha、掠射角或跨视图一致性筛选。251 张图含车体、地面、树木和其他背景，提取没有 Truck 前景 ROI。三帧抽样中有效深度像素占 62.71%、72.66%、68.76%，但“范围内有效”不等于“几何正确”。有限但互相冲突的深度容易在融合后形成漂浮面、错层和孔洞。
4. **本次体素和后处理会改变细节及可见碎片。** `max_depth=10, mesh_res=256`，共享 `mesh.py:20–25` 得到体素边长 `0.0390625`、SDF 截断距离 `0.1953125`（COLMAP 场景单位）；这可能合并或切断细薄结构，却不能单独解释 13 万个原始连通片。`mesh.py:18,63–69` 默认保留面数最大的 50 个连通片，故 `fuse_post.ply` 恰有 50 片是参数结果，**不是**训练自然生成 50 个物体部件。只改成 1 片可改善显示杂乱，但最大一片也可能是地面或背景，不能凭此制造完整 Truck。

PGSR 深度具体由混合后的平面参数与相机射线求交，`pgsr/models/gsplat.py:13–45` 中形如 `z = -D/(N·r+1e-8)`；它不是 LiDAR 测距或真值深度。颜色可以由若干 Gaussian 混合得到合理像素，而这些 Gaussian 的平面、遮挡和深度未必对应同一处物理车壳。靠近掠射角或多层透明表面时，即使深度数值有限，也可能跨视角不一致。这里是源码支持的失效机制，尚未通过本次逐像素深度对照定量确认。

原始 `fuse.ply` 包围盒约从 `[-11.465,-5.684,-11.191]` 至 `[13.301,8.613,11.426]`；后处理仍横跨大范围背景。三个相机的局部投影基本对应照片位置，使“整个失败由相机外参方向颠倒造成”不太可能，但目前没有逐视角深度一致性或物体 mask 检查，不能完全排除局部标定与深度误差。全场景投影截图可能显示车体轮廓，不能取代物体级连通性、覆盖率和几何精度验收。

## 3. 重构 `pgsr` 库是不是原因？

**目前没有证据证明它的核心 Nonbiased depth rendering 公式写错，也不能排除其实现差异影响质量。** 源码审计见[整合审计](library_audit.md)：`pgsr/gaussian_model.py:38–63` 取 Gaussian 最短尺度轴构造局部平面；`pgsr/models/gsplat.py:13–45,78–116` 将平面参数随 alpha 合成，再由射线与平面求交得到深度。方向与官方 `gaussian_renderer/__init__.py:132–165`、`submodules/diff-plane-rasterization/cuda_rasterizer/forward.cu:373–404` 对齐；这不等于两个 rasterizer 逐像素数值等价。本机已实际完成训练、梯度与 mesh 提取，但没有和官方实现逐像素或同条件训练对照。

更直接的差异在**提取路径**。重构包的 `pgsr.mesh` 调用共享 `gaussian_splatting/mesh.py:extract_mesh()`，本次未做掠射角/深度一致性过滤，默认保留 50 连通片；官方 T&T 路径 `scripts/run_tnt.py:21` 启用 `--use_depth_filter --num_cluster 1`，其 `scripts/render_tnt.py:120–127` 剔除法线与射线夹角超过 80° 的深度，另可用给定空间边界裁剪。官方普通 `render.py` 的过滤也可能关闭，因此比较的是具体的 T&T 提取命令，不是“所有官方运行都自动过滤”。这些差异直接改变哪些表面进入或留在网格，足以使输出观感不同；没有相同 PLY 的提取消融，无法量化各自贡献。

因此，对这次目标的直接评价是：重构库适合分模块学习和走通调用链，但默认 mesh 提取策略偏宽松，不能把它当作即用即得的 Truck 物体网格质量保证。这是具体的参数/功能差异，不等同于已经查实 Nonbiased depth 核心实现有错误。

训练层还有**可能**改变深度质量的非等价整合：共享增密/裁剪日程和阈值、`radii>0` 的 trimming 可见性定义、动态选择多视图邻居、有效重投影比例额外缩放及最近邻深度采样，均与官方实现不同。精确调用与差异见 [`library_audit.md`](library_audit.md)。本次 CPU 3-NN 初始化替代也可能改变初始尺度。它们是需要 A/B 检验的机制，不应统称为已证实的“库代码质量问题”。相反，mesh 阶段将 `active_sh_degree` 置 0 主要改变颜色，不直接改平面深度；默认 opacity reset 从 3000 步开始，本轮未触发；融合重新渲染深度，不会误读先前保存的逆深度 TIFF。

## 最小判别实验

优先固定**同一个 1500 步 checkpoint**，先做不训练的提取消融：分别记录原始 TSDF、`n_cluster_to_keep=1/50`、可信深度筛选、Truck ROI、体素分辨率变化后的车体覆盖与连通片，而不只看美观截图。这样能分开“深度本身互相矛盾”和“全场景融合/后处理留了碎片”。下一层对照再固定数据、种子和评测，比较 vanilla 3DGS、当前提前几何项 PGSR 与默认日程 PGSR；评估训练及留出视角 RGB、跨视角平面深度一致性，以及 Truck ROI 网格的连通、覆盖和（若有真值时）精度。没有 mask/真值时，只报告可观测的拓扑与相机投影结果。

本次 CPU 审计命令（无 GPU 训练）：

```bash
.envs/pgsr-1.0.0/bin/python scripts/audit_truck_reconstruction.py \
  --quality-csv notes/evidence/render-quality.csv \
  --mesh-dir outputs/truck-densify1500/ours_1500 \
  --output notes/evidence/truck-reconstruction-audit.json
```

复算输入是 S05 的逐视角 CSV 和 S06 的两个 PLY；环境为已锁定的 WSL Conda `pgsr-1.0.0`，GPU 不参与。结论是**当前物体级 mesh 不合格，但根因份额仍待上述消融确定**。
