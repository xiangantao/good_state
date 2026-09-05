"""Extract Wan Q/K features for the OVIS validation set."""

from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import replace
from pathlib import Path

from ovis_data import OvisDataset, OvisFrameArchive

from heft import (
    WAN_2_1,
    ExtractionConfig,
    ExtractionTask,
    FeatureExtractionPool,
    TailFramePolicy,
)
from heft.attn_hook import CaptureSpec, FeatureKind


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_indices(value: str, *, upper_bound: int) -> tuple[int, ...] | None:
    if value.strip().lower() == "all":
        return None
    indices: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, stop_text = item.split("-", 1)
            start, stop = int(start_text), int(stop_text)
            if stop < start:
                raise argparse.ArgumentTypeError(f"invalid descending range: {item}")
            indices.update(range(start, stop + 1))
        else:
            indices.add(int(item))
    if not indices:
        raise argparse.ArgumentTypeError("index selection must not be empty")
    invalid = sorted(index for index in indices if not 0 <= index < upper_bound)
    if invalid:
        raise argparse.ArgumentTypeError(
            f"indices {invalid} are outside the valid range 0..{upper_bound - 1}"
        )
    return tuple(sorted(indices))


def _read_model_shape(model_path: Path) -> tuple[int, int]:
    config_path = model_path / "transformer" / "config.json"
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)
    return int(config["num_layers"]), int(config["num_attention_heads"])


def _selection_tag(
    layers: tuple[int, ...],
    heads: tuple[int, ...] | None,
    *,
    num_layers: int,
) -> str:
    layer_tag = (
        "all"
        if layers == tuple(range(num_layers))
        else "-".join(f"{layer:03d}" for layer in layers)
    )
    head_tag = "all" if heads is None else "-".join(f"{head:02d}" for head in heads)
    return f"layers_{layer_tag}_heads_{head_tag}"


def _comma_values(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    return values or None


def build_parser() -> argparse.ArgumentParser:
    workspace = _workspace_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=workspace / "annotations_valid_withgt.json.tos-download",
    )
    parser.add_argument(
        "--frames-zip",
        type=Path,
        default=workspace / "valid.zip.tos-download",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=workspace / "models" / "Wan2.1-T2V-1.3B-Diffusers",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=workspace / "cache" / "hf",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="default: <workspace>/features/ovis_wan/<capture selection>",
    )
    parser.add_argument(
        "--layers",
        default="all",
        help="comma/range selection such as 15,18-20, or all (default: all)",
    )
    parser.add_argument(
        "--heads",
        default="all",
        help="comma/range selection such as 0-11, or all (default: all)",
    )
    parser.add_argument(
        "--gpu-ids",
        default=os.getenv("HEFT_GPU_IDS", "2,3"),
        help="comma-separated CUDA device ids",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=int(os.getenv("HEFT_EXTRACT_WORKERS_PER_GPU", "3")),
        help="persistent model processes per GPU (default: 3)",
    )
    parser.add_argument(
        "--video-batch-size",
        type=int,
        default=int(os.getenv("HEFT_OVIS_VIDEO_BATCH_SIZE", "12")),
        help="decode and submit this many videos together (default: 12)",
    )
    parser.add_argument("--videos", default=None, help="comma-separated names or ids")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--max-pending-features", type=int, default=96)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate paths and print the selected workload without loading the model",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    num_layers, num_heads = _read_model_shape(args.model_path)
    parsed_layers = _parse_indices(args.layers, upper_bound=num_layers)
    layers = tuple(range(num_layers)) if parsed_layers is None else parsed_layers
    heads = _parse_indices(args.heads, upper_bound=num_heads)
    gpu_ids = tuple(int(value) for value in args.gpu_ids.split(",") if value.strip())
    if not gpu_ids:
        raise SystemExit("--gpu-ids must contain at least one device")

    dataset = OvisDataset(args.annotations)
    videos = dataset.select(_comma_values(args.videos))
    if args.max_videos is not None:
        if args.max_videos <= 0:
            raise SystemExit("--max-videos must be positive")
        videos = videos[: args.max_videos]
    if not videos:
        raise SystemExit("no OVIS videos selected")

    output_root = args.output_root or (
        _workspace_root()
        / "features"
        / "ovis_wan"
        / _selection_tag(layers, heads, num_layers=num_layers)
    )
    if args.workers_per_gpu <= 0:
        raise SystemExit("--workers-per-gpu must be positive")
    if args.video_batch_size <= 0:
        raise SystemExit("--video-batch-size must be positive")

    pending = tuple(
        video
        for video in videos
        if args.overwrite or not (output_root / video.name / "manifest.json").is_file()
    )
    print(
        json.dumps(
            {
                "annotations": str(args.annotations),
                "frames_zip": str(args.frames_zip),
                "model_path": str(args.model_path),
                "output_root": str(output_root),
                "layers": list(layers),
                "heads": "all" if heads is None else list(heads),
                "gpu_ids": list(gpu_ids),
                "workers_per_gpu": args.workers_per_gpu,
                "total_gpu_workers": len(gpu_ids) * args.workers_per_gpu,
                "video_batch_size": args.video_batch_size,
                "max_pending_features": args.max_pending_features,
                "selected_videos": len(videos),
                "pending_videos": len(pending),
                "selected_frames": sum(video.length for video in videos),
                "tail_policy": "pad",
            },
            indent=2,
        )
    )
    if args.dry_run or not pending:
        return

    model = replace(WAN_2_1, model_id=str(args.model_path))
    capture = CaptureSpec(
        step=model.start_step,
        layers=layers,
        heads=heads,
        features=(FeatureKind.QUERY, FeatureKind.KEY),
    )
    config = ExtractionConfig(
        capture=capture,
        chunk_size=args.chunk_size,
        tail_policy=TailFramePolicy.PAD,
        max_pending_features=args.max_pending_features,
        overwrite=args.overwrite,
    )
    output_root.mkdir(parents=True, exist_ok=True)

    with (
        FeatureExtractionPool(
            model=model,
            config=config,
            gpu_ids=gpu_ids,
            workers_per_gpu=args.workers_per_gpu,
            cache_dir=args.cache_dir,
        ) as extractor,
        OvisFrameArchive(args.frames_zip) as archive,
    ):
        for batch_start in range(0, len(pending), args.video_batch_size):
            batch_videos = pending[batch_start : batch_start + args.video_batch_size]
            tasks: list[ExtractionTask] = []
            for offset, video in enumerate(batch_videos):
                index = batch_start + offset + 1
                print(
                    f"[{index}/{len(pending)}] decode {video.name}: "
                    f"{video.length} frames",
                    flush=True,
                )
                tasks.append(
                    ExtractionTask(
                        name=video.name,
                        input_video=archive.read_video(video, size=model.resolution),
                        output_dir=output_root / video.name,
                    )
                )

            results = extractor.extract(tasks)
            for offset, result in enumerate(results):
                index = batch_start + offset + 1
                print(
                    f"[{index}/{len(pending)}] wrote {result.task_name}: "
                    f"{len(result.chunks)} chunks -> {result.output_dir}",
                    flush=True,
                )
            del tasks, results
            gc.collect()


if __name__ == "__main__":
    main()
