"""Fit one fixed channel mask on every calibration scene after LOSO selection."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .channel_report import summarize
from .channel_scan import (
    cached_ablation,
    check_baseline,
    file_identity,
    load_cached_scenes,
    log,
    read_json,
    score_scene,
)
from .channel_selection import (
    ABLATION_METRICS,
    AblationScene,
    rank_channels,
    scene_retrieval,
)
from .report import write_csv, write_json
from .scan import atomic_npz, digest, source_identity


def refit_schedule(selection, dimensions, input_dimensions):
    schedule = selection["dimensions"]
    if (
        not schedule
        or schedule != sorted(set(schedule), reverse=True)
        or min(schedule) < 2
        or max(schedule) >= input_dimensions
    ):
        raise ValueError("Invalid source pruning schedule")
    if dimensions not in schedule:
        raise ValueError("Choose a dimension evaluated by the source selection run")
    return schedule[: schedule.index(dimensions) + 1]


def fit_fixed_mask(prepared, schedule, out, args, protocol):
    """Use equal scene weights at every stage, then freeze the final indices."""
    names = sorted(prepared)
    current = np.arange(prepared[names[0]].features.shape[-1])
    traces = []
    for dimensions in schedule:
        previous_count = len(current)
        log(f"Fit on all {len(names)} scenes: {previous_count} -> {dimensions}")
        stats = [
            cached_ablation(
                out / "ablation" / f"{name}_from_{previous_count}.npz",
                prepared[name],
                current,
                args,
                protocol,
            )
            for name in names
        ]
        ranked, full, deleted, importance = rank_channels(current, stats)
        atomic_npz(
            out / f"ranking_from_{previous_count}.npz",
            channels=current,
            ranked_channels=ranked,
            full=full,
            deleted=deleted,
            importance=importance,
            training_scenes=np.array(names),
            metrics=np.array(ABLATION_METRICS),
        )
        current = np.sort(ranked[:dimensions])
        scores = {
            name: dict(
                zip(
                    ABLATION_METRICS,
                    scene_retrieval(
                        prepared[name], current, protocol["hit_distance"]
                    ).tolist(),
                )
            )
            for name in names
        }
        traces.append(
            {
                "from_dimensions": previous_count,
                "dimensions": dimensions,
                "calibration_scenes": names,
                "channels": current.tolist(),
                "calibration_scores": scores,
            }
        )
        write_json(out / "training_trace.json", traces)
    return current


def export_mask(out, report, config, channels, rows):
    report.mkdir(parents=True, exist_ok=True)
    masks = {f"ablation_iterative_{len(channels)}": channels.tolist()}
    path = report / "masks.json"
    if path.exists() and read_json(path) != masks:
        raise ValueError(
            "Fixed mask already exists with different indices; use a new report root"
        )
    write_json(out / "masks.json", masks)
    scene_rows, _, summary = summarize(rows)
    for directory in (out, report):
        write_json(directory / "config.json", config)
        write_json(directory / "masks.json", masks)
        write_json(directory / "calibration_summary.json", summary)
        write_csv(directory / "calibration_scene_metrics.csv", scene_rows)
    write_json(report / "training_trace.json", read_json(out / "training_trace.json"))
    write_json(report / "baseline_check.json", read_json(out / "baseline_check.json"))
    write_csv(out / "calibration_pair_metrics.csv", rows)
    audit = {
        "mask": file_identity(path),
        "channel_count": len(channels),
        "unique_channels": len(np.unique(channels)),
        "calibration_scenes": config["calibration_scenes"],
        "same_mask_for_every_scene": True,
        "requires_held_out": False,
        "evaluation_scope": "calibration only; no independent fixed-mask test",
        "model_forwards": 0,
        "results_directory": str(out.resolve()),
    }
    write_json(report / "audit.json", audit)
    return summary


def parser():
    workspace = Path(__file__).resolve().parents[3]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--selection-run",
        type=Path,
        default=workspace / "eval/scannet_channels/73f4e810824b58cb",
    )
    p.add_argument("--cache-root", type=Path, default=workspace / "cache/scannet")
    p.add_argument(
        "--output-root", type=Path, default=workspace / "eval/scannet_channels"
    )
    p.add_argument(
        "--report-root", type=Path, default=workspace / "heft/reports/scannet_channels"
    )
    p.add_argument("--dimensions", type=int, default=256)
    p.add_argument("--device", default="cpu")
    p.add_argument("--channel-batch", type=int, default=64)
    p.add_argument("--threads", type=int, default=2)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.threads < 1 or args.channel_batch < 1:
        raise ValueError("Thread and batch counts must be positive")
    source_path = args.selection_run / "config.json"
    selection = read_json(source_path)
    if read_json(args.selection_run / "status.json").get("status") != "complete":
        raise ValueError(
            "Source selection must be complete before fitting a fixed mask"
        )
    core_code = source_identity(["channel_selection.py", "geometry.py", "metrics.py"])
    if any(selection["code"].get(name) != value for name, value in core_code.items()):
        raise ValueError("Selection or metric code changed since the source experiment")
    args.baseline_run = Path(selection["baseline_run"])
    args.block, args.timestep = selection["block"], selection["timestep"]
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.04, args.device)
    started = time.time()
    protocol, scenes, provenance = load_cached_scenes(args)
    if protocol != selection["protocol"] or provenance != selection["provenance"]:
        raise ValueError(
            "Calibration features or protocol changed since the source experiment"
        )
    input_dimensions = next(iter(scenes.values()))["features"].shape[-1]
    schedule = refit_schedule(selection, args.dimensions, input_dimensions)
    config = {
        "schema": "heft.scannet_fixed_channels",
        "schema_version": 1,
        "source_selection": file_identity(source_path),
        "block": args.block,
        "timestep": args.timestep,
        "noise": next(
            n
            for n in protocol["requested_noise"]
            if n["requested_timestep"] == args.timestep
        ),
        "input_dimensions": input_dimensions,
        "dimensions": args.dimensions,
        "schedule": schedule,
        "calibration_scenes": sorted(scenes),
        "selection": selection["selection"],
        "aggregation": "pair macro, then equal weight per calibration scene",
        "evaluation_scope": "calibration only; LOSO scores describe the selection procedure",
        "device": args.device,
        "channel_batch": args.channel_batch,
        "threads": args.threads,
        "protocol": protocol,
        "provenance": provenance,
        "code": source_identity(
            [
                "channel_refit.py",
                "channel_scan.py",
                "channel_selection.py",
                "channel_report.py",
                "geometry.py",
                "metrics.py",
                "report.py",
                "scan.py",
            ]
        ),
        "versions": {"numpy": np.__version__, "torch": torch.__version__},
    }
    out = args.output_root / digest(config)
    report = args.report_root / args.selection_run.name / f"global_{args.dimensions}"
    write_json(out / "config.json", config)
    write_json(out / "status.json", {"status": "running", "started": started})
    log(f"Output: {out}")
    try:
        rows = check_baseline(scenes, protocol, out)
        prepared = {
            name: AblationScene.from_arrays(
                name, scene["features"], scene["pairs"], scene["valid"], args.device
            )
            for name, scene in scenes.items()
        }
        channels = fit_fixed_mask(prepared, schedule, out, args, protocol)
        for name, scene in scenes.items():
            rows.extend(
                score_scene(
                    name,
                    scene,
                    scene["features"][..., channels],
                    protocol,
                    "fixed_mask",
                )
            )
        summary = export_mask(out, report, config, channels, rows)
        for row in summary:
            log(
                f"Calibration {row['method']} ({row['dimensions']}): "
                f"voxel={row['retrieval_voxel_hit']:.6f}, "
                f"10cm={row['retrieval_hit_distance']:.6f}, "
                f"error={row['retrieval_world_error_mean']:.6f} m"
            )
        status = {
            "status": "complete",
            "seconds": time.time() - started,
            "pair_rows": len(rows),
        }
        if args.device.startswith("cuda"):
            status["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(
                args.device
            )
        write_json(out / "status.json", status)
        log(f"Fixed mask: {report / 'masks.json'}")
    except Exception as error:
        write_json(
            out / "status.json",
            {
                "status": "failed",
                "error": repr(error),
                "seconds": time.time() - started,
            },
        )
        raise


if __name__ == "__main__":
    main()
