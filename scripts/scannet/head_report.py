"""Head rankings with explicit voxel hits and separate QK/KK score scales."""

from __future__ import annotations

import json
from pathlib import Path

from .report import aggregate, write_csv, write_json


def generate_heads(result_dir, report_dir):
    result_dir, report_dir = Path(result_dir), Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        row
        for file in sorted((result_dir / "pairs").glob("*.json"))
        for row in json.loads(file.read_text())
    ]
    scenes = aggregate(rows, ["scene", "timestep", "block", "head", "mode"])
    summary = aggregate(scenes, ["timestep", "block", "head", "mode"])
    write_csv(result_dir / "pair_metrics.csv", rows)
    write_csv(result_dir / "scene_metrics.csv", scenes)
    for mode in ("qk", "kk"):
        for timestep in sorted({r["timestep"] for r in summary}):
            selected = [
                r for r in summary if r["mode"] == mode and r["timestep"] == timestep
            ]
            for metric, reverse in [
                ("positive_similarity", True),
                ("similarity_gap", True),
                ("retrieval_voxel_hit", True),
                ("retrieval_hit_distance", True),
                ("retrieval_world_error_mean", False),
            ]:
                eligible = sorted(
                    [r for r in selected if r[metric] is not None],
                    key=lambda r: r[metric],
                    reverse=reverse,
                )
                for rank, row in enumerate(eligible, 1):
                    row[metric + "_rank"] = rank
    write_json(result_dir / "summary.json", summary)
    write_csv(result_dir / "summary.csv", summary)
    config = json.loads((result_dir / "config.json").read_text())
    threshold = config["hit_distance"] * 100
    lines = [
        "# ScanNet / Wan attention head 空间扫描",
        "",
        "扫描 attn1 中实际使用的 Q/K：归一化和 RoPE 之后；block/head 均从0编号。",
        "QK = Q·K / sqrt(128)，KK = cosine(K,K)。二者原始分数不可直接比较；几何命中率和误差可比较。",
        "双向分别计算 Q_i→K_j 与 Q_j→K_i，不能把前向 QK 矩阵转置当作反向。",
        "沿用逐帧 VAE、相邻采样帧、有效深度及14×14池化口径。QK是在池化Q/K上计算的跨帧分数，不是原生帧内softmax注意力图。",
        "先帧对等权，再有效场景等权；无共享体素的帧对记缺失。未去除RoPE，结果含图像位置编码影响。",
        "同体素得分：先在每个视角内平均该体素的原始特征，再计算分数；体素命中：在全部有效目标token中检索后检查体素归属。",
        "体素命中与距离阈值命中分别统计；两者不等价。并列最高分的量化结果使用分数化命中率。",
        "",
    ]
    for mode in ("qk", "kk"):
        lines += [
            f"## {mode.upper()}",
            "",
            f"| k | block | head | 有效场景 | 同体素分数 | 异体素分数 | 差值 | 体素命中率 | {threshold:g}cm命中率 | 三维误差(m) |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        selected = sorted(
            [r for r in summary if r["mode"] == mode],
            key=lambda r: (r["timestep"], -(r["retrieval_voxel_hit"] or 0)),
        )
        for row in selected:
            values = [
                str(row[k])
                for k in ("timestep", "block", "head", "positive_similarity_count")
            ]
            values += [
                "—" if row[k] is None else f"{row[k]:.5f}"
                for k in (
                    "positive_similarity",
                    "negative_similarity",
                    "similarity_gap",
                    "retrieval_voxel_hit",
                    "retrieval_hit_distance",
                    "retrieval_world_error_mean",
                )
            ]
            lines.append("| " + " | ".join(values) + " |")
        lines += [""]
    lines += [
        "每个head保存共享PCA（K特征），每种方式保存最好/最差帧对匹配图。不同候选的best/worst可能不是同一帧对。",
        "完整配置、逐场景/帧对分数和有效样本数见实验目录；这是五场景、单噪声下的探索性结果。",
        "",
    ]
    (report_dir / "report.md").write_text("\n".join(lines))
    return summary
