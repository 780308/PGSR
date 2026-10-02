# 重构版 PGSR 学习注释

`pgsr-python-study-comments.patch` 保存 `worktrees/PGSR-python-study/pgsr/` 相对干净发布源码 `497f926782753558c2f0d5d19c0234088c52a058` 的中文注释和文档字符串。它覆盖训练入口、工厂、Gaussian 平面参数、渲染、损失组合、多视图缓存、裁剪及 mesh 入口；没有把整个上游仓库复制进根仓库。

此补丁用于阅读与断点学习，不是改动已安装的 `pgsr==1.0.0`，也不是官方 PGSR 基线的一部分。在独立检出中可先运行 `git apply --check`，再按需应用：

```bash
git apply --check /mnt/e/PGSR/notes/patches/pgsr-python-study-comments.patch
git apply /mnt/e/PGSR/notes/patches/pgsr-python-study-comments.patch
```

已在干净的 `upstream/PGSR-python` 上执行只读 `git apply --check`，补丁可以应用；没有修改参考克隆。实际训练调用的是隔离环境中的已安装包，实验注入仅由根目录 `scripts/` 中明确标记的脚本完成。
