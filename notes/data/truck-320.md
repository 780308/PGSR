# Tanks and Temples Truck 本地学习数据

Truck 场景已准备在 `data/truck-original/` 和 `data/truck-320/` 中，用于短时本地检查。上游压缩包仍保留在 `data/tandt_db.zip`；解压目录中仅包含 Truck 场景。

## 来源与压缩包

- 来源：[INRIA 3D Gaussian Splatting `tandt_db.zip`](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip)
- 下载压缩包大小：682,628,995 bytes；SHA-256：`816e62f22a161abbfe841d2a6b10cdf036e297c9fa289b3bfeee9c6ec526d7e1`
- ZIP 场景前缀：`tandt/truck/`；共提取 255 个 Truck 文件成员（251 个 JPG、3 个 COLMAP 二进制模型文件及 `project.ini`）。其他场景仅保留在压缩包中。
- 首次传输在 412 MB 处停滞。随后使用 `curl -C -` 从第 432,238,592 byte 续传，并在达到预期 byte 数后完成。
- 原始解压场景：187,282,181 bytes。320 像素派生数据：87,689,476 bytes。除 651 MiB 压缩包外，两者合计约占 263 MiB。

准备完成后，机器 E: 盘有 53.12 GiB 可用空间，高于工作区要求保留的 30 GiB 本地空间。数据集和派生文件均写入 E:\\PGSR；没有启动训练运行。

## 源格式与坐标系结论

存储的 251 张图像均为 979×546 JPG。唯一的 COLMAP 相机模型为 `PINHOLE`，ID 1，元数据尺寸为 1957×1091，参数为 `(fx, fy, cx, cy) = (1163.25472803, 1156.28040499, 978.5, 545.5)`。已注册图像中的 2D 观测大致覆盖完整的 COLMAP 画幅（x 约为 −1 至 1962，y 约为 −7 至 1097），而非存储的半尺寸光栅。在 COLMAP 画幅中，这些观测相对于 3D 轨迹的重投影残差中位数为 0.668 px，第 95 百分位数为 2.204 px。这证实源模型的观测值与内参使用较大的坐标系，尽管压缩包中的 JPG 光栅尺寸约为其一半。

因此，派生数据会将 COLMAP 坐标直接映射到缩放后的光栅尺寸。这与上游加载器的分辨率处理方式一致：`scene/dataset_readers.py::readColmapCameras` 根据相机元数据推导视场角，而 `utils/camera_utils.py::loadCam` 选择所需的光栅分辨率。对于 Truck，320 像素输出光栅为 320×178，输出相机也为 320×178。参数缩放比例为 `sx = 320/1957 = 0.1635155851` 和 `sy = 178/1091 = 0.1631530706`；得到的相机参数为 `(190.21027745, 188.65069852, 160, 89)`。每个 2D 特征坐标都使用与相机内参相同的 x/y 比例进行缩放。

## 派生数据内容与验证

`data/truck-320/` 包含全部 251 个原始文件名对应的 320×178 缩放 JPG，以及 `sparse/0/{cameras.bin,images.bin,points3D.bin,project.ini}`。模型包含 1 台相机、251 张已注册图像、136,029 个 3D 点和 1,083,072 个点轨迹观测。脚本检查了每张输入图像都在模型中有对应项、每条 3D 轨迹都能反向指向相应的 2D 特征，并且每个输出图像名称都与来源一致。

输出保留了全部图像 ID、名称、相机 ID、位姿、3D 点 ID/坐标/颜色以及点轨迹。它只缩放相机尺寸/内参和 2D 特征坐标。每个变换后相机投影与缩放到输出画幅的相应源投影之间，最大差异为 `1.14e-13 px`；重新加载的二进制模型也得到相同的上界。特征到点的重投影残差按相机画幅比例，从中位数 0.668 px / p95 2.204 px 缩放到中位数 0.109 px / p95 0.360 px，符合预期。

机器可读的文件清单与测量结果位于 `data/truck-preparation-manifest.json`。

## 复现方法

在包含 Pillow 和 NumPy 的 Python 环境中，从仓库根目录运行：

```powershell
python scripts/prepare_truck_dataset.py
```

脚本会在需要时使用 curl 下载压缩包或继续未完成的下载，核对精确的预期压缩包大小，记录其 SHA-256，只提取 `tandt/truck/`，缩放每张图像，重写 COLMAP 二进制模型，验证位姿/轨迹/投影，并写入清单文件。默认参数指向上述路径。脚本会复用完整的原始解压目录，并拒绝覆盖已有派生目录。如需创建另一份派生数据，请传入新的 `--scaled-dir` 与 `--manifest` 路径。成功的数据准备使用 `C:\Program Files\Inkscape\bin\python.exe`（Python 3.12、Pillow 12.0.0、NumPy 2.3.5）；没有更改该解释器的软件包。
