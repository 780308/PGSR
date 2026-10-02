# PGSR 本地学习笔记入口

本工作区从已标定的 Truck-320 RGB/COLMAP 数据开始，使用隔离环境中的 `pgsr==1.0.0` 完成了短程训练、checkpoint 重载、251 帧渲染和 TSDF 网格提取。**先读[完整实验总结](study_summary.md)**：它说明实际运行的步骤、可复核证据和“链路已运行”与“重建质量已收敛”的边界。

| 需要回答的问题 | 文档或原始证据 |
| --- | --- |
| 实验做了什么、命令和结论是什么？ | [完整实验总结](study_summary.md)、[逐次实验记录](experiment_log.md)、`../outputs/logs/` |
| 数据集怎样保持 COLMAP 一致？ | [Truck-320 数据准备](data/truck-320.md)、`../data/truck-preparation-manifest.json` |
| 一张图怎样进入 Gaussian、渲染、loss 和反向？ | [张量与数据通路](tensor_trace.md)、[论文—代码地图](source_map.md) |
| `pgsr` 与官方实现有哪些重要差异？ | [整合审计](library_audit.md) |
| 环境和 CPU 3-NN workaround 怎样影响实验？ | [环境记录](environment_setup.md)、[pip 版本快照](environment_pip_freeze_20261001.txt)、[Conda 版本快照](environment_conda_explicit_20261001.txt) |
| PLY、渲染深度如何生成网格，实际质量如何？ | [渲染与网格验收](mesh_acceptance.md)、[可视化验收用法](visualization_usage.md)、[离线查看器用法](interactive_viewer.md) |
| Git 中保存了哪些可复核的紧凑证据？ | [实验产物与证据清单](artifact_manifest.md)、`evidence/` |

优先核对 Git 内的[证据清单](artifact_manifest.md)及其 SHA-256，再看[数据准备清单](evidence/truck-preparation-manifest.json)、[训练摘要](evidence/training-summary.json)、[三帧平面深度](evidence/geometry-manifest.json)、[法线梯度重检原始日志](evidence/logs/normal_gradient_recheck_20261002.log)、[网格结构验收](evidence/visual-acceptance.json)与[相机投影验收](evidence/mesh-camera/summary.json)。`evidence/logs/` 还保存 S01–S06 的原始日志与损失 trace；其中 `geometry300.jsonl` 的法线列已知由插桩错误污染，应以重检日志为准。`outputs/` 保留原件；训练/渲染/网格由 `../scripts/run_local_study.sh` 启动，实际展开命令见 Git 内对应训练日志首行。

本地输出是学习实验产物。1500 步模型未被判定收敛；`pgsr` 与官方代码没有做同条件逐像素或基准评测对照。所有文档中的损失快照、显存峰值和三视角图像误差，只能按其记录的步骤、视角和设备解释。
