# 环境配置记录

日期：2026-09-30（Asia/Shanghai）

## 结果

所需的 Python 环境已安装在 `/mnt/e/PGSR/.envs/pgsr-1.0.0`。为 E: 盘中新建的软件包缓存目录和环境前缀启用 NTFS 大小写敏感属性后，Conda 和 pip 安装成功。`pip check` 报告 `No broken requirements found.`。Torch、torchvision、gsplat、PGSR、gaussian-splatting、Open3D、OpenCV 和 NumPy 的导入及来源检查均成功。对于采用 CUDA 12.1 构建的 Torch，RTX 3050 Ti 上的 `torch.cuda.is_available()` 返回 `True`。

此次环境配置步骤没有运行光栅化或基于数据集的训练冒烟测试；这些检查由根代理负责。

## 运行时冒烟测试跟进：simple-knn 架构与 C: 盘空间

首次基于数据集的冒烟测试在 Gaussian 初始化阶段停止于 `gaussian_splatting.simple_knn._C.distCUDA2`，尚未进入光栅化，错误为 `This program was not compiled for SM 86`。已安装的 GPU 报告计算能力为 8.6（RTX 3050 Ti Laptop GPU，驱动 546.30）。

对已安装扩展的只读检查发现：

- 二进制文件：`.envs/pgsr-1.0.0/lib/python3.10/site-packages/gaussian_splatting/simple_knn/_C.cpython-310-x86_64-linux-gnu.so`（11,902,313 bytes）。
- ELF 的 `.nv_fatbin` 节位于偏移量 `0xa8ba8`，大小为 `0xc3f60`。扫描该节只找到 `sm_75` 架构标记；没有发现 `sm_86`、`compute_XX` 或 PTX `.target` 标记。这与 SM86 GPU 上的运行时失败一致。
- 该扩展随 `gaussian_splatting 2.8.4` 的 CPython 3.10 manylinux wheel 分发。上游 `submodules/simple-knn/setup.py` 调用 `CUDAExtension` 时没有显式指定架构标志或 `TORCH_CUDA_ARCH_LIST`；wheel 中也没有记录最初的构建参数。由于 `cuobjdump` 不可用，架构证据来自嵌入 fatbin 的字符串，而非 CUDA 的二进制检查工具。

软件包安装后，C: 盘可用空间为 20,237,066,240 bytes，后来测得为 17,884,954,624 bytes，减少了 2,352,111,616 bytes。一次只读宿主机检查发现 `C:\pagefile.sys` 大小为 19,097,554,944 bytes（18,212 MiB），修改时间为 20:18。WMI 报告该文件处于活动状态，当前使用量为 2,615 MiB，峰值使用量为 3,378 MiB。时间戳与可用空间下降的时间相符，因此页面文件分配或调整大小是目前最可能的解释；但由于没有记录早先的页面文件长度，无法证明这一原因。当前注册表列出的是 E: 盘页面文件设置，而 WMI 仍将 C: 盘页面文件列为存在且活动。Ubuntu 的 `ext4.vhdx` 当前占用 5,974,786,048 bytes；没有可供对比的早期大小快照。C: 盘当前仍低于配置脚本设置的 20,000,000,000-byte 预检阈值。

## 文件系统失败与恢复

第一次 Conda 创建尝试在事务验证阶段失败，错误为：

```text
OSError: [Errno 40] Too many levels of symbolic links:
/mnt/e/PGSR/.cache/conda/pkgs/ncurses-6.6-hfaaeb4e_0/share/terminfo/n/ncr260vt300wpp
```

这是由不区分大小写的 NTFS/DrvFs 路径冲突造成的。缓存软件包中的 `info/paths.json` 将 `share/terminfo/n/ncr260vt300wpp` 标为普通硬链接，并将 `share/terminfo/N/NCR260VT300WPP` 标为指向该文件的软链接。在 E: 盘默认挂载方式下，`N` 与 `n` 路径组件会被视为同一路径。`ncurses 6.6 hfaaeb4e_0` 软件包来自已配置的清华 `pkgs/main` 镜像；对官方 defaults 元数据的查询返回了同一构建，因此仅更换镜像无法解决此问题。

失败产物没有删除，而是在工作区内移动并保留为：

- `.envs/pgsr-1.0.0.failed-20260930`
- `.cache/conda/pkgs.failed-20260930`

重试时新建了 `.envs/pgsr-1.0.0` 和 `.cache/conda/pkgs` 目录，然后使用 `fsutil file setCaseSensitiveInfo` 启用大小写敏感属性。配置脚本会在 Conda 运行前，验证两个路径下嵌套的 `share/terminfo/N` 与 `n` 目录具有不同 inode。该检查已通过。对 conda-forge `ncurses 6.5 h2d0b736_3` 的只读检查发现，其 terminfo 路径使用不同的十六进制目录（大写 NCR 对应 `4e`，小写 ncr 对应 `6e`），这也是绕开该特定冲突的另一种可能方法；成功的重试仍使用已配置的 defaults channel，并采用大小写敏感目录。

## 宿主机与源码状态

- 工作区：`/mnt/e/PGSR`
- WSL：Ubuntu 24.04.1 LTS，WSL2 内核 `6.6.114.1-microsoft-standard-WSL2`
- Conda：位于 `/home/dongxianghong/miniconda3` 的 Miniconda 25.11.1
- 根仓库 commit：`de24f1a38b350387e8d8fe381b2cd70c1ae946e7`
- 重构包 commit：`497f926782753558c2f0d5d19c0234088c52a058`
- GPU：NVIDIA GeForce RTX 3050 Ti Laptop GPU，4096 MiB，驱动 546.30
- 创建 Conda 环境前的初始可用空间：E: 58,977,894,400 bytes；C: 20,256,579,584 bytes。
- 软件包安装后：E: 46,101,106,688 bytes；C: 20,237,066,240 bytes。
- 后续仅运行缓存钩子预检后的最新读数：E: 46,101,118,976 bytes；C: 17,884,954,624 bytes。E: 盘仍高于 30 GiB；C: 盘现已低于 20,000,000,000-byte 阈值，因此仅钩子命令在写入激活钩子前停止。当时没有正在运行的软件包安装。

## 已安装的软件包版本

| 软件包 | 已安装版本 |
|---|---|
| Python | 3.10.21 |
| PyTorch | `2.4.1+cu121` |
| torchvision | `0.19.1+cu121` |
| gsplat | `1.5.3+pt24cu121` |
| gaussian-splatting | `2.8.4` |
| pgsr | `1.0.0`，包含 `[mesh]` extra |
| NumPy | `1.26.4` |
| Open3D | `0.19.0` |
| opencv-python | `4.10.0.84` |
| trimesh | `5.1.0` |

首先使用 `--no-deps` 安装了官方 gsplat CPython 3.10 wheel。`pgsr[mesh]` 在 `upstream/PGSR-python/pyproject.toml` 中声明了未指定版本的 `gsplat` 依赖；随后，受约束的 pip 解析器报告 `gsplat 1.5.3+pt24cu121` 与 `torch 2.4.1+cu121` 已满足要求，没有替换预先安装的 gsplat wheel。`pip freeze` 记录的已安装 wheel SHA256 为 `0493bab68ed5fc71f4ce8bfc2be03b584d8a41a06a6d9362e09a795340f8c488`。

导入检查将软件包文件定位到 `/mnt/e/PGSR/.envs/pgsr-1.0.0/lib/python3.10/site-packages`。Torch 报告其 CUDA 构建版本为 12.1，CUDA 可用性为 `True`，设备为 `NVIDIA GeForce RTX 3050 Ti Laptop GPU`。

## 缓存路径与环境激活

安装期间的缓存被定向到 E: 盘：

- Conda 软件包：`/mnt/e/PGSR/.cache/conda/pkgs`（大小写敏感）
- pip：`/mnt/e/PGSR/.cache/pip`
- 临时文件：`/mnt/e/PGSR/.cache/tmp`
- XDG：`/mnt/e/PGSR/.cache/xdg`
- torch 扩展：`/mnt/e/PGSR/.cache/torch_extensions`
- Triton：`/mnt/e/PGSR/.cache/triton`
- CUDA：`/mnt/e/PGSR/.cache/cuda`
- Matplotlib：`/mnt/e/PGSR/.cache/matplotlib`
- Torch 模型缓存：`/mnt/e/PGSR/.cache/torch`
- OpenCV：`/mnt/e/PGSR/.cache/opencv`

`scripts/setup_env.sh` 会在全新安装时写入 Conda `activate.d` 钩子，并为已有前缀提供 `--cache-hook-only` 模式。安装完成后曾尝试运行后一种模式，但由于 C: 盘空间降至阈值以下而停止；因此当前环境**尚未**包含该钩子。待 C: 盘可用空间恢复到 20,000,000,000 bytes 以上后，运行 `bash scripts/setup_env.sh --cache-hook-only` 即可创建钩子。常规软件包安装本身已在 C: 盘越过该限制前完成。

## 复现方法与日志

成功的重试使用：

```bash
cd /mnt/e/PGSR
bash scripts/setup_env.sh
```

- 第一次失败尝试的日志：`.cache/setup_env_20260930_193336.log`
- 成功安装与 pip freeze 日志：`.cache/setup_env_20260930_195227.log`
- 缓存钩子预检日志：`.cache/setup_env_20260930_201845.log`（在修改环境之前止于 C: 盘阈值检查）
- 所需固定版本与约束：`scripts/setup_env_constraints.txt`
- 可重复执行的配置脚本与大小写敏感检查：`scripts/setup_env.sh`

完整的成功日志包含准确的 Conda 方案、pip 安装记录、`pip check`、`pip freeze` 和安装后可用空间检查。
