"""Scene-macro summaries and dependency-light PNG/Markdown reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

METRICS = [
    "positive_similarity",
    "negative_similarity",
    "similarity_gap",
    "retrieval_voxel_hit",
    "retrieval_hit_distance",
    "retrieval_world_error_mean",
    "retrieval_world_error_median",
]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    )
    temporary.replace(path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows, keys):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    result = []
    for group, values in sorted(groups.items()):
        record = dict(zip(keys, group))
        record["contributing_rows"] = len(values)
        record["valid_rows"] = sum(v.get("status", "ok") == "ok" for v in values)
        for metric in METRICS:
            a = np.array(
                [v[metric] for v in values if v.get(metric) is not None], float
            )
            record[metric] = float(a.mean()) if len(a) else None
            record[metric + "_std"] = float(a.std()) if len(a) else None
            record[metric + "_count"] = len(a)
        record["status"] = (
            "ok" if record["positive_similarity"] is not None else "no_valid_pairs"
        )
        result.append(record)
    return result


def pca_preview(features, images, path):
    """One shared PCA basis across sampled frames, not independent per-frame PCA."""
    f = np.asarray(features, dtype=np.float32)  # T,C,H,W
    x = f.transpose(0, 2, 3, 1).reshape(-1, f.shape[1])
    x = x - x.mean(0)
    # Thin randomized range finder bounds CPU cost for many channels.
    rng = np.random.default_rng(42)
    projection = rng.normal(size=(x.shape[1], min(12, x.shape[1]))).astype(np.float32)
    q, _ = np.linalg.qr(x @ projection)
    _, _, vt = np.linalg.svd(q.T @ x, full_matrices=False)
    rgb = x @ vt[:3].T
    lo, hi = np.percentile(rgb, [1, 99], axis=0)
    rgb = np.clip((rgb - lo) / np.maximum(hi - lo, 1e-8), 0, 1)
    rgb = (rgb.reshape(f.shape[0], f.shape[2], f.shape[3], 3) * 255).astype(np.uint8)
    count = min(4, len(images))
    panel = Image.new("RGB", (320 * count, 400), "white")
    for i in range(count):
        panel.paste(images[i].resize((320, 200)), (320 * i, 0))
        panel.paste(
            Image.fromarray(rgb[i]).resize((320, 200), Image.Resampling.NEAREST),
            (320 * i, 200),
        )
    panel.save(path)


def matching_preview(
    fi,
    fj,
    images,
    geometry,
    path,
    hit_distance=0.1,
    *,
    query_features=None,
    similarity_kind="cosine",
):
    """Unfiltered feature top-1 predictions, colored by the configured distance."""
    from .metrics import normalize

    ni, nj = geometry["ids_i"], geometry["ids_j"]
    if not len(ni) or not len(nj):
        return
    query = fi if query_features is None else query_features
    project = normalize if similarity_kind == "cosine" else np.asarray
    a = project(query.reshape(-1, query.shape[-1])[ni])
    b = project(fj.reshape(-1, fj.shape[-1])[nj])
    same = (geometry["vox_i"][:, None] == geometry["vox_j"][None]).all(-1)
    eligible = np.flatnonzero(same.any(1))
    if not len(eligible):
        return
    selected = eligible[
        np.linspace(0, len(eligible) - 1, min(20, len(eligible)), dtype=int)
    ]
    panel = Image.new("RGB", (960, 320), "white")
    panel.paste(images[0].resize((480, 300)), (0, 20))
    panel.paste(images[1].resize((480, 300)), (480, 20))
    draw = ImageDraw.Draw(panel)
    draw.text(
        (4, 4),
        f"Feature top-1; green <= {hit_distance:g}m. Display ties use first index.",
        fill="black",
    )
    h, w = fi.shape[:2]
    for i in selected:
        j = int((a[i] @ b.T).argmax())
        sy, sx = divmod(int(ni[i]), w)
        ty, tx = divmod(int(nj[j]), w)
        distance = np.linalg.norm(geometry["xyz_i"][i] - geometry["xyz_j"][j])
        color = "lime" if distance <= hit_distance else "red"
        start = ((sx + 0.5) * 480 / w, 20 + (sy + 0.5) * 300 / h)
        end = (480 + (tx + 0.5) * 480 / w, 20 + (ty + 0.5) * 300 / h)
        draw.line((start, end), fill=color, width=1)
    panel.save(path)


def generate(result_dir, report_dir):
    result_dir, report_dir = Path(result_dir), Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for file in sorted((result_dir / "pairs").glob("*.json")):
        rows.extend(json.loads(file.read_text()))
    write_csv(result_dir / "pair_metrics.csv", rows)
    scenes = aggregate(rows, ["scene", "timestep", "block"])
    write_csv(result_dir / "scene_metrics.csv", scenes)
    summary = aggregate(scenes, ["timestep", "block"])
    write_json(result_dir / "summary.json", summary)
    write_csv(result_dir / "summary.csv", summary)
    config_path = result_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    hit_cm = config.get("hit_distance", 0.1) * 100
    lines = [
        "# ScanNet / Wan 空间特征探索",
        "",
        "相邻采样帧比较；帧对等权汇总到场景，再对有效场景等权汇总。",
        "这是五场景探索口径，不是 VEGA 附录的全视角对统计。",
        "标准差描述跨场景波动，不是置信区间；各指标有效场景数可能不同。",
        "缺少共享体素的帧对不填零；检索只评价目标视角存在同体素的查询，搜索全部有效目标 token。",
        "同体素与异体素分数使用体素表征；并列 top-1 使用分数化命中率和期望误差。",
        "",
        f"| timestep | block（从0编号） | 有效场景 | 同体素相似度 | 异体素相似度 | 差值 | {hit_cm:g}cm命中率 | 三维误差(m) |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def fmt(x):
        return "—" if x is None else f"{x:.4f}"

    for s in summary:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(s["timestep"]),
                    str(s["block"]),
                    str(s["positive_similarity_count"]),
                ]
                + [
                    fmt(s[m])
                    for m in [
                        "positive_similarity",
                        "negative_similarity",
                        "similarity_gap",
                        "retrieval_hit_distance",
                        "retrieval_world_error_mean",
                    ]
                ]
            )
            + " |"
        )
    lines += [
        "",
        "逐场景指标见 scene_metrics.csv；帧对覆盖与无效原因见 pair_metrics.csv。",
        "运行配置、数据身份、实际噪声和几何检查记录保存在实验目录。",
        "",
    ]
    (report_dir / "report.md").write_text("\n".join(lines))
    if summary:
        canvas = Image.new("RGB", (900, 70 + 45 * len(summary)), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (15, 12),
            "Scene-macro spatial probe: blue=positive, orange=negative; axis [-1,1]",
            fill="black",
        )
        for i, row in enumerate(summary):
            y = 55 + 45 * i
            draw.text((10, y), f"k={row['timestep']} B{row['block']:02d}", fill="black")
            for metric, color, offset in [
                ("positive_similarity", "steelblue", 0),
                ("negative_similarity", "orange", 13),
            ]:
                value = row[metric]
                if value is not None:
                    x = 200 + 600 * (max(-1, min(1, value)) + 1) / 2
                    draw.line((500, y + offset, x, y + offset), fill=color, width=7)
        canvas.save(report_dir / "layer_comparison.png")
    return summary
