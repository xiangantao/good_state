"""Reports for predefined channel counts, keeping seed and scene variation apart."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .report import METRICS, aggregate, write_csv, write_json


def summarize(rows):
    scenes = aggregate(rows, ["scene", "method", "dimensions", "repeat"])
    repeats = aggregate(scenes, ["method", "dimensions", "repeat"])
    summary = []
    for method, dimensions in sorted({(r["method"], r["dimensions"]) for r in repeats}):
        runs = [r for r in repeats if (r["method"], r["dimensions"]) == (method, dimensions)]
        observations = [r for r in scenes if (r["method"], r["dimensions"]) == (method, dimensions)]
        scene_means = aggregate(observations, ["scene"])
        row = {"method": method, "dimensions": dimensions, "repeats": len(runs), "scenes": len(scene_means)}
        for metric in METRICS:
            values = [r[metric] for r in runs if r[metric] is not None]
            scene_values = [r[metric] for r in scene_means if r[metric] is not None]
            row[metric] = float(np.mean(values)) if values else None
            row[metric + "_seed_std"] = float(np.std(values)) if values else None
            row[metric + "_scene_std"] = float(np.std(scene_values)) if scene_values else None
            row[metric + "_scene_count"] = len(scene_values)
        summary.append(row)
    return scenes, repeats, summary


def generate(out, report, rows):
    out, report = Path(out), Path(report)
    report.mkdir(parents=True, exist_ok=True)
    config = json.loads((out / "config.json").read_text())
    scenes, repeats, summary = summarize(rows)
    write_csv(out / "pair_metrics.csv", rows)
    write_csv(out / "scene_metrics.csv", scenes)
    write_csv(out / "repeat_metrics.csv", repeats)
    write_csv(out / "summary.csv", summary)
    write_json(out / "summary.json", summary)
    baseline = next(r for r in summary if r["method"] == "original")
    by_key = {(r["method"], r["dimensions"]): r for r in summary}
    lines = [
        f"# B{config['block']} hidden channel scan", "",
        "Five-scene exploration. Fit channel masks and PCA on four scenes, score the fifth.",
        "All listed dimensions were fixed before scoring. A winning dimension selected from this table needs new-scene validation.",
        "LOSO measures the selection procedure; the five fold masks are not one fixed deployment mask.",
        "", "## Protocol", "",
        "Cached pooled block outputs only; no model inference. Original geometry and evaluate_pair are unchanged.",
        "Pair macro -> scene macro. Missing pairs are excluded, not zero-filled. Bidirectional top-1 searches all valid targets.",
        "Cosine is normalized again after channel selection. Ties within 1e-6 receive fractional credit.",
        "Deletion ranking uses training voxel-hit loss first, best-correct minus best-incorrect cosine margin second, then channel index.",
        "Once: rank all original channels once. Iterative: rerank the current subset at each dimension in the schedule.",
        "Combination scores are recomputed on training scenes after every pruning step; no held-out score controls pruning.",
        "PCA uses scene-equal valid-token covariance, training-only centering, and no whitening. Centered-full separates centering from compression.",
        "Random subsets use five fixed permutations, nested across dimensions; no data-dependent selection.",
        "Seed standard deviation is across five scene-macro random runs; scene variation is separate in summary.csv. Neither is a confidence interval.",
        "", "## Held-out results", "",
        "| Method | Dimensions | Voxel hit (%) | 10cm hit (%) | Mean error (m) | Similarity gap | Voxel change (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    ordered = [baseline, next(r for r in summary if r["method"] == "centered_full")]
    ordered += [by_key[method, k] for k in config["dimensions"] for method in ("ablation_once", "ablation_iterative", "pca", "random")]
    for row in ordered:
        voxel = f"{row['retrieval_voxel_hit'] * 100:.3f}"
        if row["method"] == "random":
            voxel += f" +/- {row['retrieval_voxel_hit_seed_std'] * 100:.3f}"
        lines.append(
            f"| {row['method']} | {row['dimensions']} | {voxel} | "
            f"{row['retrieval_hit_distance'] * 100:.3f} | {row['retrieval_world_error_mean']:.4f} | "
            f"{row['similarity_gap']:.4f} | {(row['retrieval_voxel_hit'] - baseline['retrieval_voxel_hit']) * 100:+.3f} |"
        )
    lines += ["", "## Per-scene pruning results", "",
              "| Scene | Original voxel (%) | Iterative 512 | Iterative 256 | Iterative 128 |",
              "|---|---:|---:|---:|---:|"]
    for name in sorted({r["scene"] for r in scenes}):
        values = { (r["method"], r["dimensions"]): r for r in scenes if r["scene"] == name }
        cells = [name, f"{values['original', baseline['dimensions']]['retrieval_voxel_hit'] * 100:.3f}"]
        for k in (512, 256, 128):
            row = values.get(("ablation_iterative", k))
            cells.append(f"{row['retrieval_voxel_hit'] * 100:.3f}" if row else "n/a")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Training versus held-out pruning", "",
              "| Method | Dimensions | Training voxel (%) | Held-out voxel (%) |",
              "|---|---:|---:|---:|"]
    traces = []
    for path in sorted((out / "folds").glob("*/training_trace.json")):
        traces.extend(json.loads(path.read_text()))
    for k in config["dimensions"]:
        for method in ("ablation_once", "ablation_iterative"):
            values = [r["training_scores"]["retrieval_voxel_hit"] for r in traces if r["method"] == method and r["dimensions"] == k]
            lines.append(f"| {method} | {k} | {np.mean(values) * 100:.3f} | {by_key[method, k]['retrieval_voxel_hit'] * 100:.3f} |")
    lines += ["", "Training scores use float64 accelerated retrieval and overlap across folds; they are diagnostics, not additional independent observations.",
              "", "![Dimension curves](dimension_curves.png)", "", "## Artifacts", "",
              f"Results: `{out}`", "",
              "`baseline_check.json`: every archived pair checked before selection.",
              "`folds/<scene>/split.json`, `masks.json`, `pca.npz`: fit provenance and reusable transforms.",
              "`initial_ranking.npz`, `ranking_from_*.npz`, `training_trace.json`: deletion effects and combination checks.",
              "All original metrics, including pair medians and coverage, are in the CSV files.", ""]
    (report / "report.md").write_text("\n".join(lines))
    try:
        import matplotlib
    except ModuleNotFoundError:
        plot_with_pillow(summary, config, report / "dimension_curves.png")
        return summary

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout="constrained")
    for ax, metric, label, scale in zip(
        axes, ("retrieval_voxel_hit", "retrieval_hit_distance", "retrieval_world_error_mean"),
        ("Voxel hit (%)", "10cm hit (%)", "Mean 3D error (m)"), (100, 100, 1),
    ):
        for method, title, color in (
            ("ablation_iterative", "Iterative pruning", "#178251"),
            ("ablation_once", "One-shot pruning", "#c4454e"),
            ("pca", "PCA (centered)", "#177ea3"),
            ("random", "Random (5 seeds)", "#777777"),
        ):
            selected = sorted((r for r in summary if r["method"] == method), key=lambda r: r["dimensions"])
            x = [r["dimensions"] for r in selected]
            y = np.array([r[metric] for r in selected]) * scale
            ax.plot(x, y, "o-", label=title, color=color, markersize=4)
            if method == "random":
                sd = np.array([r[metric + "_seed_std"] for r in selected]) * scale
                ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.15)
        ax.axhline(baseline[metric] * scale, color="black", ls="--", label="Original 1536")
        ax.axhline(by_key["centered_full", baseline["dimensions"]][metric] * scale, color="#b17d18", ls=":", label="Centered 1536")
        ax.set(xlabel="Retained dimensions", ylabel=label, xscale="log")
        ax.set_xticks(sorted(config["dimensions"]), labels=sorted(config["dimensions"]))
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(alpha=0.18)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"B{config['block']} hidden: five-scene leave-one-out evaluation")
    fig.savefig(report / "dimension_curves.png", dpi=160)
    plt.close(fig)
    return summary


def plot_with_pillow(summary, config, path):
    """Use the repository's existing Pillow plotting dependency when offline."""
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.new("RGB", (1500, 620), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
        title_font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        font = title_font = ImageFont.load_default()
    methods = [
        ("ablation_iterative", "Iterative pruning", "#178251"),
        ("ablation_once", "One-shot pruning", "#c4454e"),
        ("pca", "PCA (centered)", "#177ea3"),
        ("random", "Random (5 seeds)", "#777777"),
    ]
    baseline = next(r for r in summary if r["method"] == "original")
    centered = next(r for r in summary if r["method"] == "centered_full")
    draw.text((30, 18), f"B{config['block']} hidden: five-scene leave-one-out evaluation", fill="black", font=title_font)
    dims = sorted(config["dimensions"])
    for panel, (metric, label, scale) in enumerate([
        ("retrieval_voxel_hit", "Voxel hit (%)", 100),
        ("retrieval_hit_distance", "10cm hit (%)", 100),
        ("retrieval_world_error_mean", "Mean 3D error (m)", 1),
    ]):
        left, right, top, bottom = panel * 500 + 72, panel * 500 + 475, 110, 440
        values = [r[metric] * scale for r in summary]
        padding = max((max(values) - min(values)) * .12, .005)
        low, high = min(values) - padding, max(values) + padding

        def point(k, value):
            fraction = (np.log(k) - np.log(dims[0])) / max(np.log(dims[-1]) - np.log(dims[0]), 1e-12)
            return int(left + fraction * (right - left)), int(bottom - (value - low) / (high - low) * (bottom - top))

        draw.text((left, 73), label, fill="black", font=title_font)
        for value in np.linspace(low, high, 6):
            y = point(dims[0], value)[1]
            draw.line((left, y, right, y), fill="#e0e0e0")
            draw.text((left - 62, y - 9), f"{value:.1f}" if scale == 100 else f"{value:.3f}", fill="#444444", font=font)
        for k in dims:
            x = point(k, low)[0]
            draw.line((x, bottom, x, bottom + 5), fill="#444444")
            draw.text((x - 21, bottom + 12), str(k), fill="#444444", font=font)
        draw.line((left, top, left, bottom, right, bottom), fill="#444444", width=2)
        draw.text((left + 95, bottom + 42), "Retained dimensions", fill="#444444", font=font)
        for row, color, stride in ((baseline, "black", 12), (centered, "#b17d18", 5)):
            y = point(dims[0], row[metric] * scale)[1]
            for x in range(left, right, stride):
                draw.line((x, y, min(x + stride // 2, right), y), fill=color, width=2)
        for method, _, color in methods:
            selected = sorted((r for r in summary if r["method"] == method), key=lambda r: r["dimensions"])
            points = [point(r["dimensions"], r[metric] * scale) for r in selected]
            if len(points) > 1:
                draw.line(points, fill=color, width=3)
            for row, (x, y) in zip(selected, points):
                if method == "random":
                    delta = row[metric + "_seed_std"] * scale
                    y1 = point(row["dimensions"], row[metric] * scale - delta)[1]
                    y2 = point(row["dimensions"], row[metric] * scale + delta)[1]
                    draw.line((x, y1, x, y2), fill=color)
                    draw.line((x - 4, y1, x + 4, y1), fill=color)
                    draw.line((x - 4, y2, x + 4, y2), fill=color)
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
    legend = methods + [("original", "Original 1536", "black"), ("centered", "Centered 1536", "#b17d18")]
    for index, (_, label, color) in enumerate(legend):
        x, y = 50 + (index % 3) * 480, 540 + (index // 3) * 35
        draw.line((x, y + 9, x + 30, y + 9), fill=color, width=3)
        draw.text((x + 40, y), label, fill="#333333", font=font)
    canvas.save(path)
