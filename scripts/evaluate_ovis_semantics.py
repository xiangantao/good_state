"""Rank captured attention heads using OVIS instance/category semantic masks."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
from ovis_data import OvisDataset, OvisVideo
from pycocotools import mask as mask_utils
from safetensors import safe_open
from torch import Tensor

from heft import FeatureVideo, FrameRange


@dataclass(frozen=True, slots=True)
class SemanticSample:
    """One source patch, target frame, and mutually exclusive P/I/G/O regions."""

    track_id: int
    source_frame: int
    target_frame: int
    source_token: int
    position: np.ndarray
    instance: np.ndarray
    category: np.ndarray
    other: np.ndarray


@dataclass(slots=True)
class MetricAccumulator:
    sums: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add(self, metrics: Mapping[str, float]) -> None:
        for name, value in metrics.items():
            if math.isfinite(value):
                self.sums[name] += float(value)
                self.counts[name] += 1

    def means(self) -> dict[str, float]:
        return {
            name: self.sums[name] / count
            for name, count in self.counts.items()
            if count
        }


@dataclass(slots=True)
class HeadMetricAccumulator:
    """Accumulate one track's metrics for all selected heads together."""

    heads: tuple[int, ...]
    sums: dict[str, np.ndarray] = field(default_factory=dict)
    counts: dict[str, np.ndarray] = field(default_factory=dict)

    def add(self, metrics: Mapping[str, np.ndarray]) -> None:
        for name, values in metrics.items():
            values = np.asarray(values, dtype=np.float64)
            if values.shape != (len(self.heads),):
                raise ValueError(
                    f"metric {name} has shape {values.shape}; "
                    f"expected {(len(self.heads),)}"
                )
            finite = np.isfinite(values)
            sums = self.sums.setdefault(
                name, np.zeros(len(self.heads), dtype=np.float64)
            )
            counts = self.counts.setdefault(
                name, np.zeros(len(self.heads), dtype=np.int64)
            )
            sums[finite] += values[finite]
            counts[finite] += 1

    def means(self) -> dict[str, np.ndarray]:
        means: dict[str, np.ndarray] = {}
        for name, sums in self.sums.items():
            counts = self.counts[name]
            values = np.full(len(self.heads), np.nan, dtype=np.float64)
            np.divide(sums, counts, out=values, where=counts > 0)
            means[name] = values
        return means


@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    frame_gap: int
    frame_stride: int
    queries_per_instance: int
    query_min_coverage: float
    region_min_coverage: float
    position_radius: float
    temperature: float
    epsilon: float
    seed: int
    query_batch_size: int


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _comma_values(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    return values or None


def _parse_optional_indices(value: str | None) -> tuple[int, ...] | None:
    if value is None or value.strip().lower() == "all":
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
                raise SystemExit(f"invalid descending range: {item}")
            indices.update(range(start, stop + 1))
        else:
            indices.add(int(item))
    return tuple(sorted(indices)) or None


def _decode_grid_mask(
    segmentation: Mapping[str, Any] | None,
    *,
    feature_size: tuple[int, int],
    min_coverage: float,
) -> np.ndarray | None:
    if segmentation is None:
        return None
    decoded = mask_utils.decode(cast(Any, dict(segmentation)))
    if decoded.ndim == 3:
        decoded = decoded.any(axis=2)
    feature_height, feature_width = feature_size
    coverage = cv2.resize(
        decoded.astype(np.float32),
        (feature_width, feature_height),
        interpolation=cv2.INTER_AREA,
    )
    return coverage >= min_coverage


def _segmentation(
    annotation: Mapping[str, Any], frame: int
) -> Mapping[str, Any] | None:
    segmentations = annotation["segmentations"]
    if frame < 0 or frame >= len(segmentations):
        return None
    value = segmentations[frame]
    return value if isinstance(value, Mapping) else None


def _position_disk(
    y: int,
    x: int,
    *,
    feature_size: tuple[int, int],
    radius: float,
) -> np.ndarray:
    height, width = feature_size
    grid_y, grid_x = np.ogrid[:height, :width]
    return (grid_y - y) ** 2 + (grid_x - x) ** 2 <= radius**2


def _build_samples(
    video: OvisVideo,
    chunk: FrameRange,
    *,
    feature_size: tuple[int, int],
    settings: EvaluationSettings,
) -> tuple[SemanticSample, ...]:
    if chunk.feature_frames != chunk.stop - chunk.start:
        raise ValueError(
            f"{video.name} chunk {chunk.index}: semantic evaluation requires one "
            "feature frame per source frame"
        )
    if chunk.stop - chunk.start <= settings.frame_gap:
        return ()

    region_cache: dict[tuple[int, int], np.ndarray | None] = {}
    query_cache: dict[tuple[int, int], np.ndarray | None] = {}

    def mask_for(
        annotation: Mapping[str, Any], frame: int, *, query: bool = False
    ) -> np.ndarray | None:
        key = (int(annotation["id"]), frame)
        cache = query_cache if query else region_cache
        if key not in cache:
            cache[key] = _decode_grid_mask(
                _segmentation(annotation, frame),
                feature_size=feature_size,
                min_coverage=(
                    settings.query_min_coverage
                    if query
                    else settings.region_min_coverage
                ),
            )
        return cache[key]

    samples: list[SemanticSample] = []
    _height, width = feature_size
    for source_frame in range(
        chunk.start,
        chunk.stop - settings.frame_gap,
        settings.frame_stride,
    ):
        target_frame = source_frame + settings.frame_gap
        target_visible = [
            annotation
            for annotation in video.annotations
            if _segmentation(annotation, target_frame) is not None
        ]
        for annotation in target_visible:
            source_mask = mask_for(annotation, source_frame, query=True)
            target_instance = mask_for(annotation, target_frame)
            if source_mask is None or target_instance is None:
                continue
            candidates = np.flatnonzero(source_mask.reshape(-1))
            if candidates.size == 0:
                continue

            track_id = int(annotation["id"])
            category_id = int(annotation["category_id"])
            seed = (
                settings.seed
                + video.id * 1_000_003
                + track_id * 10_007
                + source_frame * 101
                + target_frame
            )
            rng = np.random.default_rng(seed)
            count = min(settings.queries_per_instance, int(candidates.size))
            selected = rng.choice(candidates, size=count, replace=False)

            category_mask = np.zeros(feature_size, dtype=bool)
            for other_annotation in target_visible:
                if (
                    int(other_annotation["id"]) == track_id
                    or int(other_annotation["category_id"]) != category_id
                ):
                    continue
                other_mask = mask_for(other_annotation, target_frame)
                if other_mask is not None:
                    category_mask |= other_mask

            for source_token in selected.tolist():
                y, x = divmod(int(source_token), width)
                position = _position_disk(
                    y,
                    x,
                    feature_size=feature_size,
                    radius=settings.position_radius,
                )
                instance = target_instance & ~position
                category = category_mask & ~position & ~instance
                other = ~(position | instance | category)
                if not instance.any() or not other.any():
                    continue
                samples.append(
                    SemanticSample(
                        track_id=track_id,
                        source_frame=source_frame - chunk.start,
                        target_frame=target_frame - chunk.start,
                        source_token=int(source_token),
                        position=position.reshape(-1),
                        instance=instance.reshape(-1),
                        category=category.reshape(-1),
                        other=other.reshape(-1),
                    )
                )
    return tuple(samples)


def _tensor_heads(file: Any) -> tuple[int, ...]:
    heads = []
    keys = file.keys()
    for key in keys:
        prefix = "head_"
        if not key.startswith(prefix) or not key[len(prefix) :].isdigit():
            raise ValueError(f"invalid feature tensor key: {key}")
        heads.append(int(key[len(prefix) :]))
    return tuple(sorted(heads))


def _score_regions(
    z: np.ndarray,
    sample: SemanticSample,
    *,
    temperature: float,
    epsilon: float,
) -> dict[str, float]:
    shifted = z / temperature
    shifted -= shifted.max()
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()

    other_z = z[sample.other]
    other_probability = probabilities[sample.other]
    other_density = float(other_probability.sum() / sample.other.sum())
    other_mean = float(other_z.mean())
    threshold = float(np.quantile(other_z, 0.95))
    argmax = int(z.argmax())

    metrics: dict[str, float] = {
        "other_mean_z": other_mean,
        "other_density": other_density,
    }
    regions = (
        ("position", sample.position),
        ("instance", sample.instance),
        ("category", sample.category),
    )
    for name, region in regions:
        area = int(region.sum())
        if area == 0:
            continue
        mass = float(probabilities[region].sum())
        density = mass / area
        mean_z = float(z[region].mean())
        metrics[f"{name}_mass"] = mass
        metrics[f"{name}_density"] = density
        metrics[f"{name}_lift"] = math.log(
            (density + epsilon) / (other_density + epsilon)
        )
        metrics[f"{name}_delta_z"] = mean_z - other_mean
        metrics[f"{name}_coverage"] = float((z[region] > threshold).mean())
        metrics[f"top1_{name}"] = float(region[argmax])
    return metrics


def _score_regions_by_head(
    z: np.ndarray,
    sample: SemanticSample,
    *,
    temperature: float,
    epsilon: float,
) -> dict[str, np.ndarray]:
    """Vectorized equivalent of ``_score_regions`` for ``[head, token]`` maps."""

    if z.ndim != 2:
        raise ValueError(f"expected [head, token] similarities, got {z.shape}")
    shifted = z / temperature
    shifted -= shifted.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    other_z = z[:, sample.other]
    other_probability = probabilities[:, sample.other]
    other_density = other_probability.sum(axis=1) / sample.other.sum()
    other_mean = other_z.mean(axis=1)
    threshold = np.quantile(other_z, 0.95, axis=1)
    argmax = z.argmax(axis=1)

    metrics: dict[str, np.ndarray] = {
        "other_mean_z": other_mean,
        "other_density": other_density,
    }
    regions = (
        ("position", sample.position),
        ("instance", sample.instance),
        ("category", sample.category),
    )
    for name, region in regions:
        area = int(region.sum())
        if area == 0:
            continue
        mass = probabilities[:, region].sum(axis=1)
        density = mass / area
        mean_z = z[:, region].mean(axis=1)
        metrics[f"{name}_mass"] = mass
        metrics[f"{name}_density"] = density
        metrics[f"{name}_lift"] = np.log(
            (density + epsilon) / (other_density + epsilon)
        )
        metrics[f"{name}_delta_z"] = mean_z - other_mean
        metrics[f"{name}_coverage"] = (z[:, region] > threshold[:, np.newaxis]).mean(
            axis=1
        )
        metrics[f"top1_{name}"] = region[argmax].astype(np.float64)
    return metrics


def _feature_path(
    feature_video: FeatureVideo,
    chunk: FrameRange,
    layer: int,
    kind: str,
) -> Path:
    return (
        feature_video.root
        / f"chunk_{chunk.index:03d}"
        / f"step_{feature_video.step:03d}"
        / f"layer_{layer:03d}"
        / f"{kind}.safetensors"
    )


def _similarity_maps(
    query: Tensor,
    key: Tensor,
    samples: Sequence[SemanticSample],
    *,
    feature_size: tuple[int, int],
    device: torch.device,
    batch_size: int,
) -> Iterable[tuple[Sequence[SemanticSample], np.ndarray, np.ndarray]]:
    if query.ndim != 3 or key.ndim != 3:
        raise ValueError(
            f"expected [head, token, channel] Q/K, got {query.shape}, {key.shape}"
        )
    if query.shape != key.shape:
        raise ValueError(f"Q/K shape mismatch: {query.shape} != {key.shape}")
    spatial_tokens = feature_size[0] * feature_size[1]
    frames, remainder = divmod(query.shape[1], spatial_tokens)
    if remainder:
        raise ValueError("feature token count is not divisible by the spatial grid")
    channels = query.shape[2]
    num_heads = query.shape[0]

    with torch.inference_mode():
        query_device = query.to(device=device)
        key_device = key.to(device=device)
        key_video = (
            key_device.reshape(num_heads, frames, spatial_tokens, channels)
            .permute(1, 0, 2, 3)
            .contiguous()
        )

        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            source_indices = torch.tensor(
                [
                    sample.source_frame * spatial_tokens + sample.source_token
                    for sample in batch
                ],
                dtype=torch.long,
                device=device,
            )
            target_frames = torch.tensor(
                [sample.target_frame for sample in batch],
                dtype=torch.long,
                device=device,
            )
            source_query = (
                query_device.index_select(1, source_indices)
                .permute(1, 0, 2)
                .contiguous()
                .float()
            )
            source_key = (
                key_device.index_select(1, source_indices)
                .permute(1, 0, 2)
                .contiguous()
                .float()
            )
            target_key = key_video.index_select(0, target_frames).float()

            flat_target = target_key.reshape(-1, spatial_tokens, channels)
            flat_query = source_query.reshape(-1, channels, 1)
            flat_source_key = source_key.reshape(-1, channels, 1)
            qk = torch.bmm(flat_target, flat_query).reshape(
                len(batch), num_heads, spatial_tokens
            )
            qk.mul_(channels**-0.5)
            kk = torch.bmm(flat_target, flat_source_key).reshape(
                len(batch), num_heads, spatial_tokens
            )
            kk.div_(torch.linalg.vector_norm(target_key, dim=3).clamp_min_(1e-12))
            kk.div_(
                torch.linalg.vector_norm(source_key, dim=2)
                .clamp_min_(1e-12)
                .unsqueeze(2)
            )
            yield batch, qk.cpu().numpy(), kk.cpu().numpy()


def _evaluate_video(
    video: OvisVideo,
    feature_video: FeatureVideo,
    *,
    layers: tuple[int, ...],
    heads: tuple[int, ...] | None,
    device: torch.device,
    settings: EvaluationSettings,
) -> tuple[
    dict[tuple[int, int, str], dict[str, float]],
    dict[tuple[int, int, str], int],
]:
    track_metrics: dict[tuple[int, str, int], HeadMetricAccumulator] = {}
    sample_counts: dict[tuple[int, int, str], int] = defaultdict(int)

    if device.type == "cuda":
        torch.cuda.set_device(device)
        stream_context = torch.cuda.stream(torch.cuda.Stream(device=device))
    else:
        stream_context = nullcontext()

    with stream_context:
        for chunk in feature_video.chunks:
            samples = _build_samples(
                video,
                chunk,
                feature_size=feature_video.feature_size,
                settings=settings,
            )
            if not samples:
                continue
            for layer in layers:
                query_path = _feature_path(feature_video, chunk, layer, "query")
                key_path = _feature_path(feature_video, chunk, layer, "key")
                with (
                    safe_open(query_path, framework="pt", device="cpu") as query_file,
                    safe_open(key_path, framework="pt", device="cpu") as key_file,
                ):
                    available = tuple(
                        sorted(
                            set(_tensor_heads(query_file))
                            & set(_tensor_heads(key_file))
                        )
                    )
                    selected_heads = available if heads is None else heads
                    missing = [head for head in selected_heads if head not in available]
                    if missing:
                        raise ValueError(
                            f"heads {missing} are missing from {query_path}"
                        )
                    query = torch.cat(
                        [
                            query_file.get_tensor(f"head_{head:03d}")
                            for head in selected_heads
                        ],
                        dim=0,
                    )
                    key = torch.cat(
                        [
                            key_file.get_tensor(f"head_{head:03d}")
                            for head in selected_heads
                        ],
                        dim=0,
                    )

                for batch, qk_maps, kk_maps in _similarity_maps(
                    query,
                    key,
                    samples,
                    feature_size=feature_video.feature_size,
                    device=device,
                    batch_size=settings.query_batch_size,
                ):
                    for sample_index, sample in enumerate(batch):
                        for mode, similarities in (
                            ("qk", qk_maps[sample_index]),
                            ("kk", kk_maps[sample_index]),
                        ):
                            key_tuple = (layer, mode, sample.track_id)
                            accumulator = track_metrics.get(key_tuple)
                            if accumulator is None:
                                accumulator = HeadMetricAccumulator(selected_heads)
                                track_metrics[key_tuple] = accumulator
                            elif accumulator.heads != selected_heads:
                                raise ValueError(
                                    f"inconsistent heads for layer {layer}: "
                                    f"{accumulator.heads} != {selected_heads}"
                                )
                            accumulator.add(
                                _score_regions_by_head(
                                    similarities,
                                    sample,
                                    temperature=settings.temperature,
                                    epsilon=settings.epsilon,
                                )
                            )

                for head in selected_heads:
                    sample_counts[(layer, head, "qk")] += len(samples)
                    sample_counts[(layer, head, "kk")] += len(samples)

    grouped_tracks: dict[tuple[int, int, str], list[dict[str, float]]] = defaultdict(
        list
    )
    for (layer, mode, _track_id), accumulator in track_metrics.items():
        metrics = accumulator.means()
        for head_index, head in enumerate(accumulator.heads):
            grouped_tracks[(layer, head, mode)].append(
                {
                    name: float(values[head_index])
                    for name, values in metrics.items()
                    if math.isfinite(float(values[head_index]))
                }
            )

    video_metrics: dict[tuple[int, int, str], dict[str, float]] = {}
    for candidate, track_values in grouped_tracks.items():
        metric_names = sorted({name for values in track_values for name in values})
        video_metrics[candidate] = {
            name: float(
                np.mean([values[name] for values in track_values if name in values])
            )
            for name in metric_names
        }
    return video_metrics, dict(sample_counts)


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "videos": int(array.size),
    }


def _top20_fractions(
    per_video: Mapping[str, Mapping[tuple[int, int, str], Mapping[str, float]]],
) -> dict[tuple[int, int, str, str], float]:
    hits: dict[tuple[int, int, str, str], int] = defaultdict(int)
    totals: dict[tuple[int, int, str, str], int] = defaultdict(int)
    for candidate_values in per_video.values():
        modes = {candidate[2] for candidate in candidate_values}
        metric_names = {
            metric for values in candidate_values.values() for metric in values
        }
        for mode in modes:
            for metric in metric_names:
                ranking = [
                    (candidate, values[metric])
                    for candidate, values in candidate_values.items()
                    if candidate[2] == mode and metric in values
                ]
                if not ranking:
                    continue
                ranking.sort(key=lambda item: item[1], reverse=True)
                top_count = max(1, math.ceil(len(ranking) * 0.2))
                top_candidates = {
                    candidate for candidate, _value in ranking[:top_count]
                }
                for candidate, _value in ranking:
                    key = (*candidate, metric)
                    totals[key] += 1
                    if candidate in top_candidates:
                        hits[key] += 1
    return {key: hits[key] / count for key, count in totals.items() if count}


def _build_report(
    per_video: Mapping[str, Mapping[tuple[int, int, str], Mapping[str, float]]],
    sample_counts: Mapping[str, Mapping[tuple[int, int, str], int]],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    values: dict[tuple[int, int, str, str], list[float]] = defaultdict(list)
    total_samples: dict[tuple[int, int, str], int] = defaultdict(int)
    for video_name, candidate_values in per_video.items():
        for candidate, candidate_metrics in candidate_values.items():
            for metric, value in candidate_metrics.items():
                values[(*candidate, metric)].append(value)
        for candidate, count in sample_counts.get(video_name, {}).items():
            total_samples[candidate] += count

    top20 = _top20_fractions(per_video)
    candidates = sorted({key[:3] for key in values})
    head_rows: list[dict[str, Any]] = []
    for layer, head, mode in candidates:
        summaries: dict[str, Any] = {}
        for key, metric_values in values.items():
            if key[:3] != (layer, head, mode):
                continue
            metric = key[3]
            summary = _distribution(metric_values)
            summary["top20_video_fraction"] = top20.get(key, 0.0)
            summaries[metric] = summary
        head_rows.append(
            {
                "layer": layer,
                "head": head,
                "mode": mode,
                "samples": total_samples[(layer, head, mode)],
                "metrics": summaries,
            }
        )

    rankings: dict[str, dict[str, list[dict[str, float | int]]]] = {}
    for mode in ("qk", "kk"):
        rankings[mode] = {}
        for label, metric in (
            ("instance_semantic", "instance_lift"),
            ("category_semantic", "category_lift"),
            ("position_control", "position_lift"),
        ):
            rows = []
            for row in head_rows:
                if row["mode"] != mode or metric not in row["metrics"]:
                    continue
                summary = row["metrics"][metric]
                rows.append(
                    {
                        "layer": row["layer"],
                        "head": row["head"],
                        "mean": summary["mean"],
                        "median": summary["median"],
                        "top20_video_fraction": summary["top20_video_fraction"],
                    }
                )
            rows.sort(
                key=lambda row: (row["mean"], row["top20_video_fraction"]),
                reverse=True,
            )
            rankings[mode][label] = rows

    return {
        "schema": "heft.ovis_semantic_heads",
        "schema_version": 1,
        "config": dict(config),
        "evaluated_videos": list(per_video),
        "heads": head_rows,
        "rankings": rankings,
    }


def build_parser() -> argparse.ArgumentParser:
    workspace = _workspace_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=workspace / "annotations_valid_withgt.json.tos-download",
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=workspace / "features" / "ovis_wan" / "layers_all_heads_all",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=workspace / "eval" / "ovis_wan_all_layers_semantics.json",
    )
    parser.add_argument("--videos", default=None, help="comma-separated names or ids")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--layers", default=None, help="default: all captured layers")
    parser.add_argument("--heads", default=None, help="default: all captured heads")
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="device used when --devices is not supplied",
    )
    parser.add_argument(
        "--devices",
        default=os.getenv("HEFT_EVAL_DEVICES", "cuda:2,cuda:3"),
        help="comma-separated devices for concurrent evaluation, e.g. cuda:2,cuda:3",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("HEFT_EVAL_WORKERS", "16")),
        help="total workers, distributed evenly across --devices (default: 16)",
    )
    parser.add_argument("--frame-gap", type=int, default=4)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--queries-per-instance", type=int, default=4)
    parser.add_argument("--query-min-coverage", type=float, default=0.8)
    parser.add_argument("--region-min-coverage", type=float, default=0.5)
    parser.add_argument("--position-radius", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query-batch-size", type=int, default=256)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cv2.setNumThreads(1)
    if args.frame_gap <= 0 or args.frame_stride <= 0:
        raise SystemExit("--frame-gap and --frame-stride must be positive")
    if args.queries_per_instance <= 0 or args.query_batch_size <= 0:
        raise SystemExit("query counts must be positive")
    if not 0 < args.query_min_coverage <= 1:
        raise SystemExit("--query-min-coverage must be in (0, 1]")
    if not 0 < args.region_min_coverage <= 1:
        raise SystemExit("--region-min-coverage must be in (0, 1]")
    if args.position_radius < 0 or args.temperature <= 0 or args.epsilon <= 0:
        raise SystemExit(
            "radius must be non-negative; temperature/epsilon must be positive"
        )

    requested_layers = _parse_optional_indices(args.layers)
    requested_heads = _parse_optional_indices(args.heads)
    dataset = OvisDataset(args.annotations)
    videos = dataset.select(_comma_values(args.videos))
    if args.max_videos is not None:
        if args.max_videos <= 0:
            raise SystemExit("--max-videos must be positive")
        videos = videos[: args.max_videos]

    settings = EvaluationSettings(
        frame_gap=args.frame_gap,
        frame_stride=args.frame_stride,
        queries_per_instance=args.queries_per_instance,
        query_min_coverage=args.query_min_coverage,
        region_min_coverage=args.region_min_coverage,
        position_radius=args.position_radius,
        temperature=args.temperature,
        epsilon=args.epsilon,
        seed=args.seed,
        query_batch_size=args.query_batch_size,
    )
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    device_names = _comma_values(args.devices) or (args.device,)
    devices = tuple(torch.device(name) for name in device_names)
    per_video: dict[str, dict[tuple[int, int, str], dict[str, float]]] = {}
    sample_counts: dict[str, dict[tuple[int, int, str], int]] = {}
    jobs: list[tuple[OvisVideo, FeatureVideo, tuple[int, ...]]] = []

    for index, video in enumerate(videos, start=1):
        feature_dir = args.feature_root / video.name
        manifest = feature_dir / "manifest.json"
        if not manifest.is_file():
            print(f"[{index}/{len(videos)}] skip {video.name}: no manifest")
            continue
        feature_video = FeatureVideo.open(feature_dir)
        if feature_video.num_frames != video.length:
            raise ValueError(
                f"{video.name}: {feature_video.num_frames} feature frames != "
                f"{video.length} OVIS frames"
            )
        layers = feature_video.layers if requested_layers is None else requested_layers
        missing_layers = [
            layer for layer in layers if layer not in feature_video.layers
        ]
        if missing_layers:
            raise ValueError(f"{video.name}: layers {missing_layers} were not captured")
        if requested_heads is not None and feature_video.heads is not None:
            missing_heads = [
                head for head in requested_heads if head not in feature_video.heads
            ]
            if missing_heads:
                raise ValueError(
                    f"{video.name}: heads {missing_heads} were not captured"
                )
        jobs.append((video, feature_video, layers))

    if not jobs:
        raise SystemExit(f"no feature manifests found under {args.feature_root}")
    worker_count = min(args.workers, len(jobs))
    active_device_count = min(len(devices), worker_count)
    active_devices = devices[:active_device_count]
    worker_limits = {
        device: worker_count // active_device_count
        + (index < worker_count % active_device_count)
        for index, device in enumerate(active_devices)
    }
    print(
        f"evaluate {len(jobs)} videos with {worker_count} workers: "
        + ", ".join(f"{device}={worker_limits[device]}" for device in active_devices)
        + f"; query_batch_size={settings.query_batch_size}"
    )
    with ExitStack() as stack:
        executors = {
            device: stack.enter_context(
                ThreadPoolExecutor(max_workers=worker_limits[device])
            )
            for device in active_devices
        }
        future_to_video = {}
        ordered_jobs = sorted(
            jobs,
            key=lambda job: (len(job[1].chunks) * len(job[2]), job[0].length),
            reverse=True,
        )
        for index, (video, feature_video, layers) in enumerate(ordered_jobs):
            device = active_devices[index % active_device_count]
            future = executors[device].submit(
                _evaluate_video,
                video,
                feature_video,
                layers=layers,
                heads=requested_heads,
                device=device,
                settings=settings,
            )
            future_to_video[future] = (video.name, str(device))

        for completed, future in enumerate(as_completed(future_to_video), start=1):
            video_name, device_name = future_to_video[future]
            metrics, counts = future.result()
            if metrics:
                per_video[video_name] = metrics
                sample_counts[video_name] = counts
            print(
                f"[{completed}/{len(jobs)}] done {video_name} on {device_name} "
                f"samples={sum(counts.values())}",
                flush=True,
            )

    if not per_video:
        raise SystemExit(f"no evaluable features found under {args.feature_root}")

    config = {
        "annotations": str(args.annotations),
        "feature_root": str(args.feature_root),
        "devices": [str(device) for device in active_devices],
        "workers": worker_count,
        "workers_per_device": {
            str(device): worker_limits[device] for device in active_devices
        },
        "frame_gap": settings.frame_gap,
        "frame_stride": settings.frame_stride,
        "queries_per_instance": settings.queries_per_instance,
        "query_min_coverage": settings.query_min_coverage,
        "region_min_coverage": settings.region_min_coverage,
        "position_radius": settings.position_radius,
        "temperature": settings.temperature,
        "epsilon": settings.epsilon,
        "seed": settings.seed,
        "query_batch_size": settings.query_batch_size,
        "aggregation": "query mean -> instance mean -> video mean -> dataset mean",
        "region_priority": "P -> I -> G -> O (OVIS has no point correspondence C)",
    }
    report = _build_report(per_video, sample_counts, config=config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for mode in ("qk", "kk"):
        for ranking_name in ("instance_semantic", "category_semantic"):
            print(f"\n{mode.upper()} {ranking_name} top heads")
            for row in report["rankings"][mode][ranking_name][:10]:
                print(
                    f"L{row['layer']:02d}H{row['head']:02d} "
                    f"mean={row['mean']:.6f} median={row['median']:.6f} "
                    f"top20={row['top20_video_fraction']:.3f}"
                )
    print(f"\n{args.output}")


if __name__ == "__main__":
    main()
