"""Compare video-conditioned block/head descriptors under fixed 3D metrics."""

from __future__ import annotations

import json
from pathlib import Path

from .report import aggregate, write_csv, write_json


def summarize_video(rows, sampled):
    contexts = {
        (scene["scene"], frame): chunk["index"]
        for scene in sampled for chunk in scene["video_chunks"] for frame in chunk["frame_ids"]
    }
    expanded = []
    for row in rows:
        ci = contexts[row["scene"], row["frame_i"]]
        cj = contexts[row["scene"], row["frame_j"]]
        record = {**row, "chunk_i": ci, "chunk_j": cj, "same_chunk": ci == cj}
        expanded.append({**record, "scope": "all_pairs"})
        if ci == cj:
            expanded.append({**record, "scope": "within_chunk"})
    keys = ["scope", "feature", "timestep", "block", "head", "mode"]
    scenes = aggregate(expanded, ["scene", *keys])
    return expanded, scenes, aggregate(scenes, keys)


def generate_video(result_dir, report_dir):
    result_dir, report_dir = Path(result_dir), Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((result_dir / "config.json").read_text())
    sampled = json.loads((result_dir / "sampled_frames.json").read_text())
    rows = []
    for directory, defaults in (
        ("pairs", {"feature": "hidden", "head": -1, "mode": "cosine"}),
        ("head_pairs", {"feature": "head"}),
    ):
        for path in sorted((result_dir / directory).glob("*.json")):
            rows.extend({**row, **defaults} for row in json.loads(path.read_text()))
    expanded, scenes, summary = summarize_video(rows, sampled)
    write_csv(result_dir / "pair_metrics.csv", expanded)
    write_csv(result_dir / "scene_metrics.csv", scenes)
    write_csv(result_dir / "summary.csv", summary)
    write_json(result_dir / "summary.json", summary)
    write_json(report_dir / "summary.json", summary)
    noise = config["requested_noise"][0]
    lines = [
        "# ScanNet / HEFT Multi-Frame 3D Probe", "",
        f"Run: `{result_dir.name}`. Chunk size: {config['chunk_size']}; last-frame tail padding.",
        f"WanPipeline: {noise['num_inference_steps']} scheduled steps, capture index {noise.get('capture_step', noise['requested_timestep'])}, "
        f"actual timestep {noise['actual_timestep']:g}, sigma {noise['sigma']:.9f}.",
        f"Noise mode: {noise.get('noise_mode', 'heft')}; shift {noise['shift']:g}. "
        "One active denoising step; features are captured before its scheduler update.",
        "Video VAE encoding and joint spatiotemporal Transformer attention use the HEFT video path.",
        "Block outputs and conditional post-normalization/post-RoPE Q/K are pooled per frame.",
        "Frames are uniformly sampled from each scene before chunking; original frame gaps and timestamps are retained.",
        "This aligns video encoding, not sampling density or downstream tracking/semantic evaluation.",
        "Within-chunk pairs share input context. All-pairs also includes chunk boundaries; padded frames are never scored.",
        "Geometry, all-target retrieval, fractional ties, and pair-macro then scene-macro aggregation are unchanged.",
        "Head QK is scaled dot product; KK and block hidden use cosine. RoPE is retained.",
        "Channel masks fitted on the previous single-frame features have not been validated on these descriptors.", "",
    ]
    for scope in ("within_chunk", "all_pairs"):
        lines += [f"## {scope}", "", "| Feature | Block | Head | Mode | Scenes | Voxel hit (%) | 10cm hit (%) | Error (m) |", "|---|---:|---:|---|---:|---:|---:|---:|"]
        selected = sorted(
            (r for r in summary if r["scope"] == scope),
            key=lambda r: (r["feature"] != "hidden", -(r["retrieval_voxel_hit"] or 0)),
        )
        for row in selected:
            values = [f"{row[m] * scale:.4f}" if row[m] is not None else "NA" for m, scale in (
                ("retrieval_voxel_hit", 100), ("retrieval_hit_distance", 100),
                ("retrieval_world_error_mean", 1),
            )]
            lines.append(f"| {row['feature']} | {row['block']} | {row['head']} | {row['mode']} | "
                         f"{row['positive_similarity_count']} | " + " | ".join(values) + " |")
        lines.append("")
    lines += ["Full metrics: [summary.json](summary.json).", "",
              "Raw pair/scene CSVs, configuration, chunk identities and forward-shape audits are in the experiment directory.",
              "Five-scene results are exploratory. Legacy-noise mode matches the old noise point while retaining video VAE sampling and precision.", ""]
    (report_dir / "report.md").write_text("\n".join(lines))
    return summary
