# PGSR 短程输出可视化验收

`scripts/visualize_acceptance.py` 在 CPU 上读取已经生成的渲染图像与 TSDF 网格，输出一张联系表和一份 JSON 结构指标。默认输入目录为 `outputs/truck-densify1500/ours_1500/`；脚本不会启动 PGSR、加载 checkpoint、访问 GPU 或改写输入。

```bash
.envs/pgsr-1.0.0/bin/python scripts/visualize_acceptance.py \
  --run-dir outputs/truck-densify1500/ours_1500 \
  --geometry-dir outputs/truck-densify1500/acceptance/geometry
```

脚本按文件名选择一组 `renders/*.png`、`gt/*.png` 和 `renders/*_invdepth.tiff`。默认取排序后的第一帧；可用 `--view 00042` 指定帧。若另行导出了少量 PGSR 法线和深度样例，可放在 `--geometry-dir` 下，文件名含 `normal` 或 `depth`；文件名带帧号时会优先匹配所选视图。也可用 `--render-image`、`--gt-image`、`--invdepth-image` 指向特定文件。

结果写入 `ours_1500/acceptance/visual_acceptance.png` 和 `visual_acceptance.json`。PNG 展示渲染/GT/绝对差异、按有效值 2–98 百分位归一化的逆深度、可选法线/深度样例，以及 `fuse.ply` 和 `fuse_post.ply` 各自的正面、侧面、俯视三角面投影。JSON 记录选中图像路径与 RGB 差异统计、顶点/三角形数、有限坐标、包围盒、可计算时的连通分量数，并标出缺项。

图中固定标注“短程、未收敛、仅定性验收”；图表中的 RGB 误差针对所选帧，不是 benchmark 指标。网格三视图用 CPU 正交投影绘制，并限制每个网格用于绘图的三角面数量；JSON 的顶点和三角形统计针对整个输入网格。超过组件计算安全上限的网格会保留其他指标，并在 JSON 中注明未计算连通分量。

JSON 的 `status=ready` 仅表示 RGB、GT、逆深度文件可读取，且原始和后处理网格都有有限顶点与非零有效三角面。它不检查网格是否封闭、是否流形、尺度是否正确，也不代表 Truck 主体在网格中完整。应结合相机视角下的网格投影和三维检查作定性判断。

要把后处理网格直接投到注册相机的视角，可运行：

```bash
.envs/pgsr-1.0.0/bin/python scripts/render_mesh_camera_acceptance.py \
  --image 000001.jpg --image 000126.jpg --image 000251.jpg
```

它从 COLMAP 模型读取位姿和内参，CPU 射线投射 `fuse_post.ply`，在 `ours_1500/acceptance/mesh_camera/` 保存逐视角网格顶点色、命中掩码、相机 Z 深度、联系表和 JSON/CSV 映射。联系表中黑色表示没有命中网格。`summary.json` 的命中率以整幅图为分母，网格色与 GT 的 `on_hit` 误差只在命中区域计算；两者不能混用，也不构成论文 benchmark 指标。当前三帧的实际观察和局限见 `notes/mesh_acceptance.md`。
