# PGSR Truck 交互式本地查看器

交付文件为 `outputs/truck-densify1500/interactive_viewer.html`。在 Windows 中直接打开 `E:\PGSR\outputs\truck-densify1500\interactive_viewer.html`，或在 WSL 中用浏览器打开对应的 `file:///E:/PGSR/outputs/truck-densify1500/interactive_viewer.html`。页面的 Plotly JavaScript 和两份减面网格数据已嵌入 HTML，不需要联网、Python 服务或 GPU。251 张渲染 RGB 与 GT，以及三个深度/法线样例仍通过相对路径从同一工作区读取；移动 HTML 时必须连同 `ours_1500/` 和 `acceptance/geometry/` 保持原有相对目录结构。

页面提供 251 个已注册视角的滑块和相机名选择，可并排或叠加比较 PGSR 渲染 RGB 与 GT。深度和法线选择器只覆盖事先导出的视角 0、125、250，并提供原始深度 TIFF 链接；其它 248 帧没有法线样例，不应把缺项当作零法线。三维区可鼠标旋转、滚轮缩放、切换 `fuse.ply` 与 `fuse_post.ply`，并可重置相机视角。完整 PLY 通过相对链接保留。

3D 浏览器预览使用 Open3D CPU 二次误差减面：原始融合网格从 1,688,137 面减到 17,999 面；后处理网格从 819,553 面减到 18,000 面。默认“Truck 局部”将坐标轴显示范围设为 x=[-3.5,4]、y=[-1,2.5]、z=[-1.5,3]，便于看清车体；可以切换“全场景”查看其它结构。局部范围仅裁切浏览视口，不删除网格数据。页面显示的是保留顶点色的简化版，不是用于结构验收的完整 PLY。相机对齐的定性结论仍以 `notes/mesh_acceptance.md` 和 `ours_1500/acceptance/mesh_camera/` 下的三张联系表为准；该 1500 步模型尚未收敛，不能把页面截图用作论文指标。

构建命令如下。生成器默认拒绝覆盖现有 HTML；只有明确要重建时才传 `--overwrite`。生成器只在 CPU 上运行，首次构建约 50 秒，峰值 RSS 约 1.4 GiB。

```bash
./.envs/pgsr-1.0.0/bin/python scripts/build_interactive_viewer.py
# 已有 HTML 需重建时：
./.envs/pgsr-1.0.0/bin/python scripts/build_interactive_viewer.py --overwrite
```

验证范围：生成器检查 251 对图像、三个几何样例和两个完整网格文件都存在；Plotly `Mesh3d` 校验内嵌顶点、三角面索引与顶点色。Windows Chrome 以本地 `file:///` 打开页面后，`body[data-viewer-ready]` 到达 `true`，无头截图中 RGB/GT 和网格均可见。此验证证明页面初始化与本机相对文件加载成功，不证明每个浏览器或所有 251 张图的人工视觉质量都逐一检查过。
