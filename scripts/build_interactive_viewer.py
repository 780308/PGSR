#!/usr/bin/env python3
"""为 Truck densify1500 短训练结果生成离线可打开的交互查看页。"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import plotly


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/truck-densify1500/interactive_viewer.html"
DEFAULT_RUN_DIR = ROOT / "outputs/truck-densify1500/ours_1500"
DEFAULT_CAMERAS = ROOT / "outputs/truck-densify1500/cameras.json"
DEFAULT_SAMPLES = ROOT / "outputs/truck-densify1500/acceptance/geometry"
DEFAULT_PLOTLY_JS = Path(plotly.__file__).parent / "package_data/plotly.min.js"
ACCEPTANCE_INDICES = (0, 125, 250)


def relative_url(path: Path, base: Path) -> str:
    """返回相对 HTML 所在目录的 URL 编码路径，并确认目标存在。"""
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"缺少页面所需文件：{path}")
    relative = Path(os.path.relpath(resolved, base.resolve())).as_posix()
    return quote(relative, safe="/-._~")


def load_cameras(path: Path) -> list[dict[str, Any]]:
    cameras = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cameras, list) or len(cameras) != 251:
        raise ValueError(f"预期 cameras.json 包含 251 台相机，实际为 {len(cameras)}")
    for index, camera in enumerate(cameras):
        if camera.get("id") != index or not camera.get("img_name"):
            raise ValueError(f"相机索引 {index} 缺少匹配的 id/img_name：{camera}")
    return cameras


def simplify_mesh(path: Path, target_faces: int, fallback_color: tuple[int, int, int]) -> dict[str, Any]:
    """CPU 读取并二次误差减面，返回可直接交给 Plotly Mesh3d 的数组。"""
    source = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
    if source.is_empty() or not source.has_triangles():
        raise ValueError(f"无法读取非空三角网格：{path}")
    source_vertices = len(source.vertices)
    source_faces = len(source.triangles)
    if source_faces > target_faces:
        reduced = source.simplify_quadric_decimation(target_number_of_triangles=target_faces)
    else:
        reduced = source
    reduced.remove_degenerate_triangles()
    reduced.remove_duplicated_triangles()
    reduced.remove_unreferenced_vertices()

    vertices = np.asarray(reduced.vertices, dtype=np.float32)
    triangles = np.asarray(reduced.triangles, dtype=np.int32)
    if not vertices.size or not triangles.size or not np.isfinite(vertices).all():
        raise ValueError(f"减面后网格为空或含非有限顶点：{path}")
    if triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError(f"网格三角索引越界：{path}")

    has_colors = reduced.has_vertex_colors() and len(reduced.vertex_colors) == len(vertices)
    if has_colors:
        color_values = np.asarray(reduced.vertex_colors, dtype=np.float32)
        colors = np.round(np.clip(color_values, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        colors = np.tile(np.asarray(fallback_color, dtype=np.uint8), (len(vertices), 1))

    # Plotly 的 vertexcolor 接受 CSS 颜色字符串；坐标四舍五入以控制 HTML 体积。
    vertex_colors = [f"rgb({r},{g},{b})" for r, g, b in colors.tolist()]
    mesh_data = {
        "x": np.round(vertices[:, 0], 5).tolist(),
        "y": np.round(vertices[:, 1], 5).tolist(),
        "z": np.round(vertices[:, 2], 5).tolist(),
        "i": triangles[:, 0].tolist(),
        "j": triangles[:, 1].tolist(),
        "k": triangles[:, 2].tolist(),
        "vertexcolor": vertex_colors,
    }
    # 在构建阶段用 Plotly Python schema 先校验这些数据可以作为 Mesh3d 输入。
    trace = go.Mesh3d(
        x=mesh_data["x"],
        y=mesh_data["y"],
        z=mesh_data["z"],
        i=mesh_data["i"],
        j=mesh_data["j"],
        k=mesh_data["k"],
        vertexcolor=mesh_data["vertexcolor"],
        flatshading=True,
    )
    if trace.type != "mesh3d":
        raise ValueError(f"Plotly 未识别 Mesh3d 数据：{path}")
    go.Figure(data=[trace]).to_json(validate=True)
    return {
        "data": mesh_data,
        "source_vertices": source_vertices,
        "source_faces": source_faces,
        "display_vertices": len(vertices),
        "display_faces": len(triangles),
        "source_has_vertex_colors": bool(source.has_vertex_colors()),
        "display_has_vertex_colors": bool(has_colors),
    }


def build_html(
    *,
    output: Path,
    run_dir: Path,
    cameras_path: Path,
    samples_dir: Path,
    plotly_js_path: Path,
    target_faces: int,
    build_command: str,
) -> tuple[str, dict[str, Any]]:
    cameras = load_cameras(cameras_path)
    images: list[dict[str, Any]] = []
    for index, camera in enumerate(cameras):
        image_id = f"{index:05d}"
        render_path = run_dir / "renders" / f"{image_id}.png"
        gt_path = run_dir / "gt" / f"{image_id}.png"
        images.append(
            {
                "index": index,
                "name": camera["img_name"],
                "render": relative_url(render_path, output.parent),
                "gt": relative_url(gt_path, output.parent),
            }
        )

    sample_entries: list[dict[str, Any]] = []
    for index in ACCEPTANCE_INDICES:
        stem = f"{index:05d}"
        camera = cameras[index]
        sample_entries.append(
            {
                "index": index,
                "name": camera["img_name"],
                "render": relative_url(run_dir / "renders" / f"{stem}.png", output.parent),
                "depth": relative_url(samples_dir / f"{stem}_depth.png", output.parent),
                "depth_tiff": relative_url(samples_dir / f"{stem}_depth.tiff", output.parent),
                "normal": relative_url(samples_dir / f"{stem}_normal.png", output.parent),
                "normal_from_depth": relative_url(samples_dir / f"{stem}_normal_from_depth.png", output.parent),
                "gt": relative_url(samples_dir / f"{stem}_gt.png", output.parent),
            }
        )

    mesh_paths = {
        "fuse": run_dir / "fuse.ply",
        "post": run_dir / "fuse_post.ply",
    }
    mesh_payload: dict[str, Any] = {
        "fuse": simplify_mesh(mesh_paths["fuse"], target_faces, (110, 184, 240)),
        "post": simplify_mesh(mesh_paths["post"], target_faces, (246, 176, 107)),
    }
    for key in ("fuse", "post"):
        mesh_payload[key]["source_path"] = relative_url(mesh_paths[key], output.parent)

    page_meta = {
        "build_command": build_command,
        "repository_commit": "de24f1a38b350387e8d8fe381b2cd70c1ae946e7",
        "run_dir": str(run_dir.resolve()),
        "camera_manifest": str(cameras_path.resolve()),
        "geometry_samples": str(samples_dir.resolve()),
        "plotly_js": str(plotly_js_path.resolve()),
        "simplification": {
            key: {
                "source_path": str(mesh_paths[key].resolve()),
                "source_vertices": mesh_payload[key]["source_vertices"],
                "source_triangles": mesh_payload[key]["source_faces"],
                "display_vertices": mesh_payload[key]["display_vertices"],
                "display_triangles": mesh_payload[key]["display_faces"],
                "vertex_colors_preserved": mesh_payload[key]["display_has_vertex_colors"],
            }
            for key in ("fuse", "post")
        },
        "camera_count": len(cameras),
        "depth_normal_sample_indices": list(ACCEPTANCE_INDICES),
    }

    # 防止 HTML 内联脚本被素材中的结束标签片段提前关闭；把内嵌 vendor 中
    # 的 URL 斜杠写成 JS 等价转义，避免离线页面含可被误认成外网依赖的 URL。
    plotly_js = plotly_js_path.read_text(encoding="utf-8")
    plotly_js = re.sub(r"</script", r"<\/script", plotly_js, flags=re.IGNORECASE)
    plotly_js = re.sub(r"(?i)(https?)://", lambda match: match.group(1) + r":\/\/", plotly_js)
    plotly_script = "<script>\n" + plotly_js + "\n</script>"

    image_json = json.dumps(images, ensure_ascii=False, separators=(",", ":"))
    sample_json = json.dumps(sample_entries, ensure_ascii=False, separators=(",", ":"))
    mesh_json = json.dumps(mesh_payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    meta_json = json.dumps(page_meta, ensure_ascii=False, indent=2)
    full_mesh_fuse = html.escape(relative_url(mesh_paths["fuse"], output.parent), quote=True)
    full_mesh_post = html.escape(relative_url(mesh_paths["post"], output.parent), quote=True)
    command_text = html.escape(build_command)

    document = f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>PGSR Truck 短训练交互查看器</title>
  <style>
    :root {{ color-scheme: light; --ink:#172033; --muted:#667085; --line:#d9e0ea; --paper:#f4f6fa; --card:#fff; --blue:#2368d1; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--paper); color:var(--ink); font:15px/1.5 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif; }}
    main {{ max-width:1440px; margin:0 auto; padding:26px 28px 44px; }}
    h1 {{ margin:0 0 5px; font-size:28px; line-height:1.25; }}
    h2 {{ margin:0 0 14px; font-size:20px; }}
    h3 {{ margin:0 0 8px; font-size:16px; }}
    p {{ margin:8px 0; }}
    .subtitle,.muted {{ color:var(--muted); }}
    .notice {{ margin:18px 0; padding:14px 17px; border:1px solid #f0cb82; border-left:5px solid #e2a62d; border-radius:10px; background:#fff8e8; }}
    .card {{ margin:17px 0; padding:20px; background:var(--card); border:1px solid var(--line); border-radius:14px; box-shadow:0 2px 9px #15294b0b; }}
    .controls {{ display:grid; grid-template-columns:minmax(230px,1fr) minmax(260px,2fr) minmax(180px,1fr); gap:16px; align-items:end; }}
    label {{ display:block; margin-bottom:5px; font-weight:650; }}
    select,input[type=range],button {{ width:100%; }}
    select,button {{ min-height:40px; border:1px solid #c6cfdd; border-radius:8px; background:#fff; color:var(--ink); padding:7px 10px; font:inherit; }}
    input[type=range] {{ accent-color:var(--blue); }}
    .range-row {{ display:grid; grid-template-columns:1fr auto; gap:12px; align-items:center; }}
    .range-row output {{ min-width:55px; text-align:right; font-variant-numeric:tabular-nums; font-weight:700; }}
    .mode-row {{ display:flex; flex-wrap:wrap; gap:9px 18px; margin:13px 0 3px; align-items:center; }}
    .mode-row label {{ display:flex; align-items:center; gap:6px; margin:0; font-weight:500; }}
    .mode-row input {{ accent-color:var(--blue); }}
    .opacity-control {{ max-width:300px; display:flex; align-items:center; gap:10px; margin-left:auto; }}
    .opacity-control[hidden] {{ display:none; }}
    .split {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
    figure {{ margin:0; min-width:0; }}
    figure img {{ display:block; width:100%; height:auto; max-height:70vh; object-fit:contain; border-radius:7px; background:#e9edf4; }}
    figcaption {{ margin:4px 2px; color:var(--muted); font-size:13px; }}
    .overlay {{ position:relative; width:min(100%,1000px); margin:0 auto; line-height:0; background:#e9edf4; }}
    .overlay img {{ display:block; width:100%; height:auto; max-height:70vh; object-fit:contain; }}
    .overlay img+img {{ position:absolute; inset:0; width:100%; height:100%; object-fit:contain; }}
    .single {{ width:min(100%,1000px); margin:0 auto; }}
    .status {{ min-height:1.5em; margin:9px 0 0; color:#a23a2d; }}
    .sample-controls {{ display:grid; grid-template-columns:repeat(2,minmax(220px,1fr)); gap:14px; margin-bottom:14px; }}
    .sample-image {{ width:min(100%,1000px); margin:0 auto; }}
    .caption-line {{ display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:8px; }}
    a {{ color:#1558b0; }}
    .links {{ display:flex; flex-wrap:wrap; gap:12px 22px; }}
    .mesh-toolbar {{ display:grid; grid-template-columns:repeat(2,minmax(200px,1fr)) auto; gap:12px; align-items:end; margin-bottom:8px; }}
    .mesh-toolbar button {{ width:auto; min-width:130px; cursor:pointer; }}
    #meshPlot {{ height:min(70vh,680px); min-height:420px; }}
    .pill {{ display:inline-block; padding:2px 9px; border-radius:999px; background:#edf3ff; color:#1a56a8; font-size:13px; font-weight:650; }}
    details {{ margin-top:13px; }}
    pre {{ white-space:pre-wrap; overflow-wrap:anywhere; padding:12px; background:#f5f7fa; border-radius:8px; font-size:12px; }}
    @media (max-width:760px) {{ main {{ padding:18px 12px 30px; }} .controls,.sample-controls,.mesh-toolbar {{ grid-template-columns:1fr; }} .split {{ grid-template-columns:1fr; }} .opacity-control {{ margin-left:0; }} #meshPlot {{ min-height:360px; }} }}
  </style>
</head>
<body>
{plotly_script}
<main>
  <header>
    <h1>PGSR Truck 短训练交互查看器</h1>
    <p class="subtitle">251 个相机视角的 RGB / GT 对照、三组几何验收样例，以及可旋转的简化网格。</p>
  </header>
  <aside class="notice"><strong>实验范围：</strong>这是 1,500 次迭代的短训练检查结果，尚未收敛，不能作为论文基准结果解读。下方交互网格是从完整 PLY 经 CPU 二次误差减面后嵌入的展示版；完整 PLY 可从页面链接下载或在本地查看。</aside>

  <section class="card" aria-labelledby="image-title">
    <h2 id="image-title">渲染与真值图像</h2>
    <div class="controls">
      <div><label for="cameraSelect">相机名称</label><select id="cameraSelect" aria-label="选择相机图像名称"></select></div>
      <div><label for="cameraSlider">相机序号</label><div class="range-row"><input id="cameraSlider" type="range" min="0" max="250" step="1" value="125"><output id="cameraIndex">125 / 250</output></div></div>
      <div><label for="compareMode">比较方式</label><select id="compareMode"><option value="split">并排</option><option value="overlay">叠加</option><option value="render">只看渲染</option><option value="gt">只看 GT</option></select></div>
    </div>
    <div class="mode-row"><span class="muted">当前相机：<strong id="cameraName"></strong></span><label class="opacity-control" id="opacityControl" hidden>GT 叠加不透明度 <input id="opacitySlider" type="range" min="0" max="100" value="50"><output id="opacityValue">50%</output></label></div>
    <div id="compareSplit" class="split"><figure><img id="renderImage" alt="PGSR 渲染 RGB"><figcaption>PGSR 渲染 RGB</figcaption></figure><figure><img id="gtImage" alt="真实图像 GT"><figcaption>真实图像 GT</figcaption></figure></div>
    <div id="compareOverlay" class="overlay" hidden><img id="overlayRender" alt="PGSR 渲染 RGB"><img id="overlayGt" alt="叠加的真实图像 GT"></div>
    <div id="compareSingle" class="single" hidden><figure><img id="singleImage" alt="选中的图像"><figcaption id="singleCaption"></figcaption></figure></div>
    <p id="imageStatus" class="status" role="status"></p>
  </section>

  <section class="card" aria-labelledby="geometry-title">
    <div class="caption-line"><h2 id="geometry-title">深度与法线验收样例</h2><span class="pill">仅相机 0 / 125 / 250 导出</span></div>
    <p class="muted">每个验收视角可切换 RGB、深度预览、PGSR 渲染法线，以及由深度计算的法线。深度 PNG 是便于观察的归一化预览；原始深度 TIFF 可单独打开。其他 248 个视角没有导出法线图。</p>
    <div class="sample-controls"><div><label for="sampleSelect">验收相机</label><select id="sampleSelect"></select></div><div><label for="sampleKind">显示内容</label><select id="sampleKind"><option value="render">PGSR 渲染 RGB</option><option value="depth">深度预览 PNG</option><option value="normal">PGSR 渲染法线</option><option value="normal_from_depth">由深度计算的法线</option><option value="gt">真实图像 GT</option></select></div></div>
    <figure class="sample-image"><img id="sampleImage" alt="几何验收样例"><figcaption id="sampleCaption"></figcaption></figure>
    <p id="depthTiffLink" class="muted"></p>
  </section>

  <section class="card" aria-labelledby="mesh-title">
    <div class="caption-line"><div><h2 id="mesh-title">融合网格交互预览</h2><span id="meshCount" class="pill"></span></div><span class="muted">拖动旋转 · 滚轮缩放 · 双击重置</span></div>
    <p class="muted">Plotly Mesh3d 使用嵌入 HTML 的顶点、三角索引和 PLY 顶点颜色数据。两个版本均经过 CPU 二次误差减面；展示面数和完整源网格信息见下方构建记录。</p>
    <div class="mesh-toolbar"><div><label for="meshSelect">网格版本</label><select id="meshSelect"><option value="post">fuse_post.ply（后处理）</option><option value="fuse">fuse.ply（原始融合）</option></select></div><div><label for="meshExtent">显示范围</label><select id="meshExtent"><option value="truck" selected>Truck 局部</option><option value="full">全场景</option></select></div><button id="resetCamera" type="button">重置 3D 视角</button></div>
    <p class="muted">Truck 局部范围约为世界坐标 x=[-3.5, 4]、y=[-1, 2.5]、z=[-1.5, 3]。这是为了浏览而裁切坐标显示范围，不会删改 PLY 或嵌入的网格数据；选择“全场景”可查看完整范围。</p>
    <div id="meshPlot" aria-label="交互三角网格预览"></div>
    <p id="meshStatus" class="status" role="status"></p>
    <div class="links"><a href="{full_mesh_fuse}" download>完整原始融合 fuse.ply</a><a href="{full_mesh_post}" download>完整后处理 fuse_post.ply</a></div>
  </section>

  <section class="card" aria-labelledby="provenance-title">
    <h2 id="provenance-title">构建记录与数据来源</h2>
    <p>构建命令：<code>{command_text}</code></p>
    <p class="muted">相机清单、图像、验收深度/法线、Plotly JS 和两个源 PLY 的路径及减面前后计数保存在此页。Plotly JavaScript 已内嵌；图像沿用相对本地路径，无需网络服务。</p>
    <details><summary>查看完整构建元数据</summary><pre id="metadata"></pre></details>
  </section>
</main>
<script>
document.body.dataset.viewerReady = 'loading';
window.addEventListener('error', event => {{
  document.body.dataset.viewerReady = 'error';
  const status = document.getElementById('meshStatus');
  if (status) status.textContent = `查看器脚本错误：${{event.message || '未知错误'}}`;
}});
const IMAGES = {image_json};
const SAMPLES = {sample_json};
const MESHES = {mesh_json};
const META = {meta_json};
document.getElementById('metadata').textContent = JSON.stringify(META, null, 2);

const cameraSelect = document.getElementById('cameraSelect');
const cameraSlider = document.getElementById('cameraSlider');
const cameraIndex = document.getElementById('cameraIndex');
const cameraName = document.getElementById('cameraName');
const mode = document.getElementById('compareMode');
const splitView = document.getElementById('compareSplit');
const overlayView = document.getElementById('compareOverlay');
const singleView = document.getElementById('compareSingle');
const opacityControl = document.getElementById('opacityControl');
const opacitySlider = document.getElementById('opacitySlider');
const opacityValue = document.getElementById('opacityValue');
const imageStatus = document.getElementById('imageStatus');

for (const image of IMAGES) {{
  const option = document.createElement('option');
  option.value = String(image.index);
  option.textContent = `${{image.name}} · 视角 ${{image.index}}`;
  cameraSelect.appendChild(option);
}}

function renderImages() {{
  const image = IMAGES[Number(cameraSlider.value)];
  cameraSelect.value = String(image.index);
  cameraIndex.value = `${{image.index}} / 250`;
  cameraName.textContent = `${{image.name}}（视角 ${{image.index}}）`;
  document.getElementById('renderImage').src = image.render;
  document.getElementById('gtImage').src = image.gt;
  document.getElementById('overlayRender').src = image.render;
  document.getElementById('overlayGt').src = image.gt;
  document.getElementById('singleImage').src = mode.value === 'gt' ? image.gt : image.render;
  document.getElementById('singleCaption').textContent = mode.value === 'gt' ? '真实图像 GT' : 'PGSR 渲染 RGB';
  splitView.hidden = mode.value !== 'split';
  overlayView.hidden = mode.value !== 'overlay';
  singleView.hidden = mode.value !== 'render' && mode.value !== 'gt';
  opacityControl.hidden = mode.value !== 'overlay';
  document.getElementById('overlayGt').style.opacity = Number(opacitySlider.value) / 100;
  imageStatus.textContent = '';
}}
cameraSlider.addEventListener('input', renderImages);
cameraSelect.addEventListener('change', () => {{ cameraSlider.value = cameraSelect.value; renderImages(); }});
mode.addEventListener('change', renderImages);
opacitySlider.addEventListener('input', () => {{
  opacityValue.value = `${{opacitySlider.value}}%`;
  document.getElementById('overlayGt').style.opacity = Number(opacitySlider.value) / 100;
}});
for (const imageId of ['renderImage','gtImage','overlayRender','overlayGt','singleImage']) {{
  document.getElementById(imageId).addEventListener('error', () => {{ imageStatus.textContent = '图像加载失败。请将页面保留在输出目录中，并确认相对路径所指 PNG 文件仍在本地。'; }});
}}
renderImages();

const sampleSelect = document.getElementById('sampleSelect');
const sampleKind = document.getElementById('sampleKind');
for (const sample of SAMPLES) {{
  const option = document.createElement('option');
  option.value = String(sample.index);
  option.textContent = `视角 ${{sample.index}} · ${{sample.name}}`;
  sampleSelect.appendChild(option);
}}
function renderSample() {{
  const sample = SAMPLES.find(row => row.index === Number(sampleSelect.value));
  const kind = sampleKind.value;
  const labels = {{render:'PGSR 渲染 RGB', depth:'深度预览 PNG', normal:'PGSR 渲染法线', normal_from_depth:'由深度计算的法线', gt:'真实图像 GT'}};
  document.getElementById('sampleImage').src = sample[kind];
  document.getElementById('sampleImage').alt = `${{labels[kind]}}，相机 ${{sample.name}}`;
  document.getElementById('sampleCaption').textContent = `${{labels[kind]}} · 视角 ${{sample.index}} · ${{sample.name}}`;
  const depthTiffLink = document.getElementById('depthTiffLink');
  const depthTiffAnchor = document.createElement('a');
  depthTiffAnchor.href = sample.depth_tiff;
  depthTiffAnchor.textContent = decodeURIComponent(sample.depth_tiff.split('/').pop());
  depthTiffLink.replaceChildren(document.createTextNode('该相机的原始深度 TIFF：'), depthTiffAnchor);
}}
sampleSelect.addEventListener('change', renderSample);
sampleKind.addEventListener('change', renderSample);
renderSample();

const meshSelect = document.getElementById('meshSelect');
const meshExtent = document.getElementById('meshExtent');
const meshPlot = document.getElementById('meshPlot');
const LOCAL_RANGES = {{x:[-3.5,4],y:[-1,2.5],z:[-1.5,3]}};
const defaultEye = {{x:1.6,y:1.6,z:1.4}};
function getFullSceneRanges() {{
  const bounds = {{x:[Infinity,-Infinity],y:[Infinity,-Infinity],z:[Infinity,-Infinity]}};
  for (const version of Object.values(MESHES)) {{
    const mesh = version.data;
    for (const axis of ['x','y','z']) {{
      for (const value of mesh[axis]) {{
        if (value < bounds[axis][0]) bounds[axis][0] = value;
        if (value > bounds[axis][1]) bounds[axis][1] = value;
      }}
    }}
  }}
  for (const axis of ['x','y','z']) {{
    const padding = Math.max((bounds[axis][1] - bounds[axis][0]) * 0.02, 0.02);
    bounds[axis] = [bounds[axis][0] - padding, bounds[axis][1] + padding];
  }}
  return bounds;
}}
function defaultCamera() {{
  return {{eye:defaultEye,up:{{x:0,y:0,z:1}},center:{{x:0,y:0,z:0}}}};
}}
function meshTrace(key, name) {{
  const mesh = MESHES[key].data;
  return {{type:'mesh3d',name:name,x:mesh.x,y:mesh.y,z:mesh.z,i:mesh.i,j:mesh.j,k:mesh.k,vertexcolor:mesh.vertexcolor,flatshading:true,hoverinfo:'skip',showscale:false}};
}}
function showMesh() {{
  const key = meshSelect.value;
  const detail = MESHES[key];
  const ranges = meshExtent.value === 'truck' ? LOCAL_RANGES : getFullSceneRanges();
  const name = key === 'post' ? 'fuse_post.ply 后处理减面版' : 'fuse.ply 原始融合减面版';
  document.getElementById('meshCount').textContent = `${{detail.display_vertices.toLocaleString()}} 顶点 · ${{detail.display_faces.toLocaleString()}} 面`;
  const axis = (title,range) => ({{title,range,autorange:false,showbackground:true,backgroundcolor:'#f3f5f8'}});
  const camera = defaultCamera();
  const layout = {{margin:{{l:0,r:0,t:10,b:0}},paper_bgcolor:'#fff',plot_bgcolor:'#fff',showlegend:false,scene:{{aspectmode:'data',camera,xaxis:axis('X',ranges.x),yaxis:axis('Y',ranges.y),zaxis:axis('Z',ranges.z)}}}};
  Plotly.react(meshPlot,[meshTrace(key,name)],layout,{{responsive:true,displaylogo:false,scrollZoom:true}})
    .then(() => {{
      document.getElementById('meshStatus').textContent = '';
      document.body.dataset.viewerReady = 'true';
    }})
    .catch(error => {{
      document.body.dataset.viewerReady = 'error';
      document.getElementById('meshStatus').textContent = `3D 网格加载失败：${{error.message || error}}`;
    }});
}}
meshSelect.addEventListener('change', showMesh);
meshExtent.addEventListener('change', showMesh);
document.getElementById('resetCamera').addEventListener('click', () => {{
  Plotly.relayout(meshPlot, {{'scene.camera':defaultCamera()}});
}});
showMesh();
</script>
</body>
</html>
'''
    return document, page_meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="HTML 输出路径（默认 outputs/truck-densify1500/interactive_viewer.html）")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="包含 renders/、gt/ 和融合 PLY 的 ours_1500 目录")
    parser.add_argument("--cameras", type=Path, default=DEFAULT_CAMERAS, help="训练输出 cameras.json")
    parser.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES, help="acceptance/geometry 样例目录")
    parser.add_argument("--plotly-js", type=Path, default=DEFAULT_PLOTLY_JS, help="本地 Plotly package_data/plotly.min.js")
    parser.add_argument("--target-faces", type=int, default=18_000, help="每份网格的目标三角面数（CPU 二次误差减面）")
    parser.add_argument("--overwrite", action="store_true", help="明确允许覆盖已存在的 HTML 输出")
    args = parser.parse_args()
    if args.target_faces < 100:
        parser.error("--target-faces 必须至少为 100")
    output = args.output if args.output.is_absolute() else (Path.cwd() / args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"输出已存在，为避免覆盖用户文件已停止：{output}\n如确需重建，请显式传入 --overwrite。")
    if not args.plotly_js.is_file():
        raise FileNotFoundError(f"找不到本地 Plotly JavaScript：{args.plotly_js}")
    output.parent.mkdir(parents=True, exist_ok=True)
    invocation = [sys.executable, "scripts/build_interactive_viewer.py", *sys.argv[1:]]
    command = shlex.join(invocation)
    document, metadata = build_html(
        output=output,
        run_dir=args.run_dir.resolve(),
        cameras_path=args.cameras.resolve(),
        samples_dir=args.samples_dir.resolve(),
        plotly_js_path=args.plotly_js.resolve(),
        target_faces=args.target_faces,
        build_command=command,
    )
    # 构建完成后再创建输出，确保输入缺失或减面失败不会留下半成品。
    output.write_text(document, encoding="utf-8")
    print(f"HTML: {output}")
    print(f"HTML bytes: {output.stat().st_size:,}")
    print(f"Plotly.js bytes: {args.plotly_js.stat().st_size:,}")
    for key, item in metadata["simplification"].items():
        print(
            f"{key}: {item['source_vertices']:,} vertices / {item['source_triangles']:,} triangles"
            f" -> {item['display_vertices']:,} vertices / {item['display_triangles']:,} triangles"
        )


if __name__ == "__main__":
    main()
