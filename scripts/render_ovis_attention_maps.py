"""Render reusable publication-style OVIS Q-K/K-K spatial heatmaps as PNG files.

The renderer accepts any compatible ``FeatureVideo`` directory, infers captured
layers/heads when possible, can automatically select representative samples, and
also accepts a JSON case file for fully reproducible hand-picked examples.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import zipfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
from evaluate_ovis_semantics import (
    EvaluationSettings,
    SemanticSample,
    _build_samples,
    _decode_grid_mask,
    _feature_path,
    _score_regions,
    _segmentation,
    _similarity_maps,
)
from ovis_data import OvisDataset, OvisVideo
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils
from safetensors import safe_open

from heft import FeatureVideo, FrameRange

Color = tuple[int, int, int]

BACKGROUND: Color = (246, 248, 252)
PANEL: Color = (255, 255, 255)
TEXT: Color = (27, 37, 54)
MUTED: Color = (96, 109, 130)
BORDER: Color = (222, 228, 237)
RED: Color = (224, 64, 78)
CYAN: Color = (20, 202, 220)
YELLOW: Color = (255, 193, 7)
MAGENTA: Color = (218, 92, 214)
BLUE: Color = (51, 112, 221)

FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


@dataclass(frozen=True, slots=True)
class Candidate:
    """A candidate query/target pair generated with the evaluation protocol."""

    chunk: FrameRange
    sample: SemanticSample
    structural_score: float


@dataclass(frozen=True, slots=True)
class ManualCase:
    """Optional explicit sample specification loaded from JSON."""

    video: str
    source_frame: int
    target_frame: int
    track_id: int
    source_token: int | None = None
    query_xy: tuple[float, float] | None = None
    label: str | None = None


@dataclass(slots=True)
class RenderCase:
    """All spatial data needed to render one qualitative example."""

    video: OvisVideo
    feature_video: FeatureVideo
    chunk: FrameRange
    sample: SemanticSample
    category_name: str
    source_frame: int
    target_frame: int
    source: Image.Image
    target: Image.Image
    source_mask: np.ndarray
    instance_mask: np.ndarray
    category_mask: np.ndarray
    query_xy: tuple[float, float]
    maps: dict[str, dict[int, np.ndarray]]
    metrics: dict[str, dict[int, dict[str, float]]]
    selection_score: float | None
    label: str | None = None


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), size)


def _parse_indices(value: str | None) -> tuple[int, ...] | None:
    if value is None or value.strip().lower() == "all":
        return None
    indices: set[int] = set()
    for raw_item in value.split(","):
        item = raw_item.strip()
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
    if not indices:
        raise SystemExit("head selection must not be empty")
    return tuple(sorted(indices))


def _comma_values(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    return values or None


def _parse_modes(value: str) -> tuple[str, ...]:
    modes = tuple(dict.fromkeys(item.strip().lower() for item in value.split(",")))
    invalid = [mode for mode in modes if mode not in {"qk", "kk"}]
    if not modes or invalid:
        raise SystemExit(f"--modes must contain qk and/or kk; invalid={invalid}")
    return modes


def _parse_percentiles(value: str) -> tuple[float, float]:
    parts = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(parts) != 2 or not 0 <= parts[0] < parts[1] <= 100:
        raise SystemExit("--percentiles must be LOW,HIGH with 0 <= LOW < HIGH <= 100")
    return parts


def _available_heads(feature_video: FeatureVideo, layer: int) -> tuple[int, ...]:
    if feature_video.heads is not None:
        return feature_video.heads
    chunk = feature_video.chunks[0]
    path = _feature_path(feature_video, chunk, layer, "query")
    with safe_open(path, framework="pt", device="cpu") as feature_file:
        heads = []
        keys = feature_file.keys()
        for key in keys:
            if not key.startswith("head_") or not key[5:].isdigit():
                raise ValueError(f"invalid feature tensor key in {path}: {key}")
            heads.append(int(key[5:]))
    return tuple(sorted(heads))


def _load_manual_cases(path: Path) -> tuple[ManualCase, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_cases = payload.get("cases") if isinstance(payload, Mapping) else payload
    if not isinstance(raw_cases, list):
        raise TypeError("case JSON must be a list or an object containing a 'cases' list")
    cases: list[ManualCase] = []
    for raw in raw_cases:
        if not isinstance(raw, Mapping):
            raise TypeError(f"invalid case entry: {raw!r}")
        raw_xy = raw.get("query_xy")
        query_xy = None
        if raw_xy is not None:
            if not isinstance(raw_xy, Sequence) or len(raw_xy) != 2:
                raise ValueError("query_xy must be [x, y]")
            query_xy = (float(raw_xy[0]), float(raw_xy[1]))
        cases.append(
            ManualCase(
                video=str(raw["video"]),
                source_frame=int(raw["source_frame"]),
                target_frame=int(raw["target_frame"]),
                track_id=int(raw["track_id"]),
                source_token=(
                    None if raw.get("source_token") is None else int(raw["source_token"])
                ),
                query_xy=query_xy,
                label=None if raw.get("label") is None else str(raw["label"]),
            )
        )
    return tuple(cases)


def _decode_mask(segmentation: Mapping[str, Any] | None) -> np.ndarray | None:
    if segmentation is None:
        return None
    decoded = mask_utils.decode(cast(Any, dict(segmentation)))
    if decoded.ndim == 3:
        decoded = decoded.any(axis=2)
    return decoded.astype(bool)


def _read_frame(
    archive: zipfile.ZipFile,
    members: frozenset[str],
    file_name: str,
) -> Image.Image:
    for candidate in (file_name, f"valid/{file_name}"):
        if candidate in members:
            with Image.open(io.BytesIO(archive.read(candidate))) as image:
                return image.convert("RGB")
    raise FileNotFoundError(file_name)


def _align_mask(mask: np.ndarray, image: Image.Image) -> np.ndarray:
    if mask.shape == (image.height, image.width):
        return mask
    resized = cv2.resize(
        mask.astype(np.uint8),
        (image.width, image.height),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)


def _position_disk(
    source_token: int,
    *,
    feature_size: tuple[int, int],
    radius: float,
) -> np.ndarray:
    height, width = feature_size
    y, x = divmod(source_token, width)
    grid_y, grid_x = np.ogrid[:height, :width]
    return ((grid_y - y) ** 2 + (grid_x - x) ** 2 <= radius**2).reshape(-1)


def _annotation_by_id(video: OvisVideo, track_id: int) -> Mapping[str, Any]:
    for annotation in video.annotations:
        if int(annotation["id"]) == track_id:
            return annotation
    raise KeyError(f"{video.name}: unknown track id {track_id}")


def _deepest_token(mask: np.ndarray) -> int:
    if not mask.any():
        raise ValueError("cannot choose a query token from an empty source mask")
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    return int(distance.argmax())


def _make_manual_candidate(
    spec: ManualCase,
    video: OvisVideo,
    feature_video: FeatureVideo,
    settings: EvaluationSettings,
) -> Candidate:
    chunk = next(
        (
            item
            for item in feature_video.chunks
            if item.start <= spec.source_frame < item.stop
            and item.start <= spec.target_frame < item.stop
        ),
        None,
    )
    if chunk is None:
        raise ValueError(
            f"{video.name}: source/target frames must be inside the same feature chunk"
        )
    annotation = _annotation_by_id(video, spec.track_id)
    source_grid = _decode_grid_mask(
        _segmentation(annotation, spec.source_frame),
        feature_size=feature_video.feature_size,
        min_coverage=settings.query_min_coverage,
    )
    target_grid = _decode_grid_mask(
        _segmentation(annotation, spec.target_frame),
        feature_size=feature_video.feature_size,
        min_coverage=settings.region_min_coverage,
    )
    if source_grid is None or target_grid is None or not source_grid.any():
        raise ValueError(f"{video.name}: queried track is not visible in both frames")

    height, width = feature_video.feature_size
    if spec.source_token is not None:
        source_token = spec.source_token
    elif spec.query_xy is not None:
        x = min(width - 1, max(0, int(spec.query_xy[0] * width / video.width)))
        y = min(height - 1, max(0, int(spec.query_xy[1] * height / video.height)))
        source_token = y * width + x
    else:
        source_token = _deepest_token(source_grid)
    if not 0 <= source_token < height * width:
        raise ValueError(f"{video.name}: source token {source_token} is outside the grid")
    if not source_grid.reshape(-1)[source_token]:
        raise ValueError(f"{video.name}: source token {source_token} is outside the track mask")

    category_id = int(annotation["category_id"])
    category_grid = np.zeros(feature_video.feature_size, dtype=bool)
    for other in video.annotations:
        if (
            int(other["id"]) == spec.track_id
            or int(other["category_id"]) != category_id
        ):
            continue
        other_grid = _decode_grid_mask(
            _segmentation(other, spec.target_frame),
            feature_size=feature_video.feature_size,
            min_coverage=settings.region_min_coverage,
        )
        if other_grid is not None:
            category_grid |= other_grid

    position = _position_disk(
        source_token,
        feature_size=feature_video.feature_size,
        radius=settings.position_radius,
    )
    instance = target_grid.reshape(-1) & ~position
    category = category_grid.reshape(-1) & ~position & ~instance
    other = ~(position | instance | category)
    if not instance.any() or not other.any():
        raise ValueError(f"{video.name}: manual case has empty I or O region")
    return Candidate(
        chunk=chunk,
        sample=SemanticSample(
            track_id=spec.track_id,
            source_frame=spec.source_frame - chunk.start,
            target_frame=spec.target_frame - chunk.start,
            source_token=source_token,
            position=position,
            instance=instance,
            category=category,
            other=other,
        ),
        structural_score=0.0,
    )


def _candidate_structure_score(
    sample: SemanticSample,
    *,
    feature_size: tuple[int, int],
    objective: str,
) -> float:
    spatial_tokens = feature_size[0] * feature_size[1]
    instance_fraction = float(sample.instance.sum()) / spatial_tokens
    category_fraction = float(sample.category.sum()) / spatial_tokens
    if instance_fraction <= 0:
        return -math.inf
    if objective in {"balanced", "category"} and category_fraction <= 0:
        return -math.inf

    target_points = np.flatnonzero(sample.instance)
    source_y, source_x = divmod(sample.source_token, feature_size[1])
    target_y, target_x = np.unravel_index(target_points, feature_size)
    motion = math.hypot(
        source_y - float(target_y.mean()),
        source_x - float(target_x.mean()),
    ) / math.hypot(*feature_size)

    instance_quality = math.exp(-abs(math.log(max(instance_fraction, 1e-6) / 0.10)))
    category_quality = (
        math.exp(-abs(math.log(max(category_fraction, 1e-6) / 0.16)))
        if category_fraction > 0
        else 0.0
    )
    if objective == "instance":
        return 1.4 * instance_quality + 0.6 * motion
    if objective == "category":
        return 1.5 * category_quality + 0.4 * instance_quality + 0.4 * motion
    return instance_quality + category_quality + 0.5 * motion


def _load_head_tensors(
    feature_video: FeatureVideo,
    chunk: FrameRange,
    *,
    layer: int,
    heads: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    query_path = _feature_path(feature_video, chunk, layer, "query")
    key_path = _feature_path(feature_video, chunk, layer, "key")
    with (
        safe_open(query_path, framework="pt", device="cpu") as query_file,
        safe_open(key_path, framework="pt", device="cpu") as key_file,
    ):
        query_keys = frozenset(query_file.keys())
        key_keys = frozenset(key_file.keys())
        missing = [
            head
            for head in heads
            if f"head_{head:03d}" not in query_keys
            or f"head_{head:03d}" not in key_keys
        ]
        if missing:
            raise ValueError(f"heads {missing} are missing from {query_path}")
        query = torch.cat(
            [query_file.get_tensor(f"head_{head:03d}") for head in heads], dim=0
        )
        key = torch.cat(
            [key_file.get_tensor(f"head_{head:03d}") for head in heads], dim=0
        )
    return query, key


def _candidate_metric_score(
    metrics: Mapping[int, Mapping[str, float]],
    *,
    objective: str,
    instance_head: int,
    semantic_head: int,
    control_head: int | None,
) -> float:
    def value(head: int, name: str, default: float = 0.0) -> float:
        raw = float(metrics.get(head, {}).get(name, default))
        return raw if math.isfinite(raw) else default

    instance_score = (
        value(instance_head, "instance_lift")
        + 1.5 * value(instance_head, "instance_coverage")
        - 0.12 * max(0.0, value(instance_head, "position_lift"))
    )
    category_score = (
        value(semantic_head, "category_lift")
        + 1.8 * value(semantic_head, "category_coverage")
        - 0.10 * max(0.0, value(semantic_head, "position_lift"))
    )
    contrast = 0.0
    if control_head is not None:
        contrast = max(
            -2.0,
            min(
                4.0,
                value(semantic_head, "category_lift")
                - value(control_head, "category_lift"),
            ),
        )
    if objective == "instance":
        return instance_score + 0.15 * contrast
    if objective == "category":
        return category_score + 0.25 * contrast
    return instance_score + category_score + 0.20 * contrast


def _auto_select_candidate(
    video: OvisVideo,
    feature_video: FeatureVideo,
    *,
    layer: int,
    selection_heads: tuple[int, ...],
    instance_head: int,
    semantic_head: int,
    control_head: int | None,
    objective: str,
    selection_mode: str,
    candidate_limit: int,
    device: torch.device,
    settings: EvaluationSettings,
) -> tuple[Candidate, float]:
    candidates: list[Candidate] = []
    for chunk in feature_video.chunks:
        for sample in _build_samples(
            video,
            chunk,
            feature_size=feature_video.feature_size,
            settings=settings,
        ):
            score = _candidate_structure_score(
                sample,
                feature_size=feature_video.feature_size,
                objective=objective,
            )
            if math.isfinite(score):
                candidates.append(
                    Candidate(chunk=chunk, sample=sample, structural_score=score)
                )
    if not candidates:
        raise ValueError(f"{video.name}: no valid {objective} visualization samples")
    candidates.sort(key=lambda item: item.structural_score, reverse=True)
    candidates = candidates[:candidate_limit]

    best: Candidate | None = None
    best_score = -math.inf
    grouped: dict[int, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.chunk.index].append(candidate)
    for chunk_candidates in grouped.values():
        chunk = chunk_candidates[0].chunk
        query, key = _load_head_tensors(
            feature_video,
            chunk,
            layer=layer,
            heads=selection_heads,
        )
        offset = 0
        for batch, qk_maps, kk_maps in _similarity_maps(
            query,
            key,
            [candidate.sample for candidate in chunk_candidates],
            feature_size=feature_video.feature_size,
            device=device,
            batch_size=min(32, len(chunk_candidates)),
        ):
            selected_maps = qk_maps if selection_mode == "qk" else kk_maps
            for batch_index, sample in enumerate(batch):
                candidate = chunk_candidates[offset + batch_index]
                per_head = {
                    head: _score_regions(
                        selected_maps[batch_index, head_index],
                        sample,
                        temperature=settings.temperature,
                        epsilon=settings.epsilon,
                    )
                    for head_index, head in enumerate(selection_heads)
                }
                metric_score = _candidate_metric_score(
                    per_head,
                    objective=objective,
                    instance_head=instance_head,
                    semantic_head=semantic_head,
                    control_head=control_head,
                )
                total = metric_score + 0.15 * candidate.structural_score
                if total > best_score:
                    best, best_score = candidate, total
            offset += len(batch)
        del query, key
    if best is None:
        raise RuntimeError(f"{video.name}: failed to score visualization candidates")
    return best, best_score


def _compute_case_maps(
    feature_video: FeatureVideo,
    candidate: Candidate,
    *,
    layer: int,
    heads: tuple[int, ...],
    modes: tuple[str, ...],
    device: torch.device,
    settings: EvaluationSettings,
) -> tuple[
    dict[str, dict[int, np.ndarray]],
    dict[str, dict[int, dict[str, float]]],
]:
    query, key = _load_head_tensors(
        feature_video,
        candidate.chunk,
        layer=layer,
        heads=heads,
    )
    _batch, qk_maps, kk_maps = next(
        iter(
            _similarity_maps(
                query,
                key,
                [candidate.sample],
                feature_size=feature_video.feature_size,
                device=device,
                batch_size=1,
            )
        )
    )
    raw_by_mode = {"qk": qk_maps[0], "kk": kk_maps[0]}
    maps: dict[str, dict[int, np.ndarray]] = {}
    metrics: dict[str, dict[int, dict[str, float]]] = {}
    for mode in modes:
        maps[mode] = {}
        metrics[mode] = {}
        for head_index, head in enumerate(heads):
            values = raw_by_mode[mode][head_index]
            maps[mode][head] = values.reshape(feature_video.feature_size)
            metrics[mode][head] = _score_regions(
                values,
                candidate.sample,
                temperature=settings.temperature,
                epsilon=settings.epsilon,
            )
    return maps, metrics


def _load_render_case(
    video: OvisVideo,
    feature_video: FeatureVideo,
    candidate: Candidate,
    *,
    dataset: OvisDataset,
    archive: zipfile.ZipFile,
    members: frozenset[str],
    layer: int,
    heads: tuple[int, ...],
    modes: tuple[str, ...],
    device: torch.device,
    settings: EvaluationSettings,
    selection_score: float | None,
    label: str | None,
) -> RenderCase:
    source_frame = candidate.chunk.start + candidate.sample.source_frame
    target_frame = candidate.chunk.start + candidate.sample.target_frame
    source = _read_frame(archive, members, video.file_names[source_frame])
    target = _read_frame(archive, members, video.file_names[target_frame])
    annotation = _annotation_by_id(video, candidate.sample.track_id)
    category_id = int(annotation["category_id"])
    category_name = dataset.categories.get(category_id, f"category {category_id}")

    source_mask = _decode_mask(_segmentation(annotation, source_frame))
    instance_mask = _decode_mask(_segmentation(annotation, target_frame))
    if source_mask is None or instance_mask is None:
        raise ValueError(f"{video.name}: selected instance mask unexpectedly disappeared")
    category_mask = np.zeros_like(instance_mask)
    for other in video.annotations:
        if (
            int(other["id"]) == candidate.sample.track_id
            or int(other["category_id"]) != category_id
        ):
            continue
        other_mask = _decode_mask(_segmentation(other, target_frame))
        if other_mask is not None:
            category_mask |= other_mask
    source_mask = _align_mask(source_mask, source)
    instance_mask = _align_mask(instance_mask, target)
    category_mask = _align_mask(category_mask, target)

    feature_height, feature_width = feature_video.feature_size
    token_y, token_x = divmod(candidate.sample.source_token, feature_width)
    query_xy = (
        (token_x + 0.5) * source.width / feature_width,
        (token_y + 0.5) * source.height / feature_height,
    )
    maps, metrics = _compute_case_maps(
        feature_video,
        candidate,
        layer=layer,
        heads=heads,
        modes=modes,
        device=device,
        settings=settings,
    )
    return RenderCase(
        video=video,
        feature_video=feature_video,
        chunk=candidate.chunk,
        sample=candidate.sample,
        category_name=category_name,
        source_frame=source_frame,
        target_frame=target_frame,
        source=source,
        target=target,
        source_mask=source_mask,
        instance_mask=instance_mask,
        category_mask=category_mask,
        query_xy=query_xy,
        maps=maps,
        metrics=metrics,
        selection_score=selection_score,
        label=label,
    )


def _draw_boundaries(
    rgb: np.ndarray,
    masks_and_colors: Sequence[tuple[np.ndarray, Color]],
) -> np.ndarray:
    output = rgb.copy()
    thickness = max(2, rgb.shape[1] // 450)
    for mask, color in masks_and_colors:
        if not mask.any():
            continue
        contours, _ = cv2.findContours(
            mask.astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(output, contours, -1, color, thickness=thickness)
    return output


def _source_overlay(case: RenderCase) -> Image.Image:
    rgb = np.asarray(case.source).copy()
    overlay = rgb.astype(np.float32)
    overlay[case.source_mask] = (
        0.83 * overlay[case.source_mask]
        + 0.17 * np.asarray(CYAN, dtype=np.float32)
    )
    output = _draw_boundaries(
        np.clip(overlay, 0, 255).astype(np.uint8),
        ((case.source_mask, CYAN),),
    )
    image = Image.fromarray(output)
    draw = ImageDraw.Draw(image)
    x, y = case.query_xy
    radius = max(10, image.width // 75)
    outer_width = max(6, image.width // 180)
    inner_width = max(3, image.width // 320)
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        outline=(255, 255, 255),
        width=outer_width,
    )
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        outline=RED,
        width=inner_width,
    )
    draw.line((x - 1.45 * radius, y, x + 1.45 * radius, y), fill=RED, width=3)
    draw.line((x, y - 1.45 * radius, x, y + 1.45 * radius), fill=RED, width=3)
    return image


def _target_overlay(case: RenderCase) -> Image.Image:
    rgb = np.asarray(case.target).copy()
    overlay = rgb.astype(np.float32)
    if case.category_mask.any():
        overlay[case.category_mask] = (
            0.63 * overlay[case.category_mask]
            + 0.37 * np.asarray(YELLOW, dtype=np.float32)
        )
    overlay[case.instance_mask] = (
        0.63 * overlay[case.instance_mask]
        + 0.37 * np.asarray(CYAN, dtype=np.float32)
    )
    output = _draw_boundaries(
        np.clip(overlay, 0, 255).astype(np.uint8),
        ((case.category_mask, YELLOW), (case.instance_mask, CYAN)),
    )
    return Image.fromarray(output)


def _heatmap_overlay(
    case: RenderCase,
    heatmap: np.ndarray,
    *,
    percentiles: tuple[float, float],
) -> Image.Image:
    values = heatmap.astype(np.float32)
    low, high = np.percentile(values, percentiles)
    normalized = np.clip(
        (values - low) / max(float(high - low), 1e-8),
        0.0,
        1.0,
    )
    normalized = normalized**1.12
    resized = cv2.resize(
        normalized,
        (case.target.width, case.target.height),
        interpolation=cv2.INTER_CUBIC,
    )
    resized = np.clip(resized, 0.0, 1.0)
    colored = cv2.applyColorMap(
        np.round(resized * 255).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).astype(np.float32)
    base = np.asarray(case.target).astype(np.float32)
    alpha = (0.03 + 0.70 * resized**1.35)[..., None]
    output = np.clip(base * (1.0 - alpha) + colored * alpha, 0, 255).astype(np.uint8)
    output = _draw_boundaries(
        output,
        ((case.category_mask, YELLOW), (case.instance_mask, CYAN)),
    )
    return Image.fromarray(output)


def _raw_source_point(case: RenderCase) -> Image.Image:
    """Return the untouched source frame with only the requested query point."""
    image = case.source.copy()
    draw = ImageDraw.Draw(image)
    x, y = case.query_xy
    radius = max(4, min(image.size) // 100)
    outline = max(1, radius // 3)
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=RED,
        outline=(255, 255, 255),
        width=outline,
    )
    return image


def _raw_heatmap(
    case: RenderCase,
    heatmap: np.ndarray,
    *,
    percentiles: tuple[float, float],
) -> tuple[Image.Image, np.ndarray]:
    """Return a plain full-resolution color heatmap and its normalized values."""
    values = heatmap.astype(np.float32)
    low, high = np.percentile(values, percentiles)
    normalized = np.clip(
        (values - low) / max(float(high - low), 1e-8),
        0.0,
        1.0,
    )
    normalized = cv2.resize(
        normalized,
        (case.target.width, case.target.height),
        interpolation=cv2.INTER_CUBIC,
    )
    normalized = np.clip(normalized, 0.0, 1.0)
    colored = cv2.applyColorMap(
        np.round(normalized * 255).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return Image.fromarray(colored), normalized


def _raw_heatmap_overlay(
    case: RenderCase,
    heatmap: Image.Image,
    normalized: np.ndarray,
    *,
    alpha: float,
) -> Image.Image:
    """Blend a heatmap onto the original target without labels or boundaries."""
    base = np.asarray(case.target).astype(np.float32)
    colored = np.asarray(heatmap).astype(np.float32)
    blend = (alpha * normalized)[..., None]
    output = np.clip(base * (1.0 - blend) + colored * blend, 0, 255).astype(
        np.uint8
    )
    return Image.fromarray(output)


def _save_raw_case(
    case: RenderCase,
    *,
    output_dir: Path,
    layer: int,
    heads: tuple[int, ...],
    modes: tuple[str, ...],
    percentiles: tuple[float, float],
    overlay_alpha: float,
) -> None:
    """Save native-resolution frames and one independent PNG per map."""
    case_dir = output_dir / case.video.name
    source_path = case_dir / f"source_f{case.source_frame:04d}_point.png"
    target_path = case_dir / f"target_f{case.target_frame:04d}.png"
    _save_png(_raw_source_point(case), source_path)
    _save_png(case.target, target_path)
    print(source_path)
    print(target_path)

    for mode in modes:
        for head in heads:
            heatmap, normalized = _raw_heatmap(
                case,
                case.maps[mode][head],
                percentiles=percentiles,
            )
            stem = f"layer{layer:02d}_h{head:02d}_{mode}"
            heatmap_path = case_dir / f"{stem}_heatmap.png"
            overlay_path = case_dir / f"{stem}_overlay.png"
            _save_png(heatmap, heatmap_path)
            _save_png(
                _raw_heatmap_overlay(
                    case,
                    heatmap,
                    normalized,
                    alpha=overlay_alpha,
                ),
                overlay_path,
            )
            print(heatmap_path)
            print(overlay_path)


def _fit_image(image: Image.Image, size: tuple[int, int]) -> tuple[Image.Image, tuple[int, int]]:
    target_width, target_height = size
    scale = min(target_width / image.width, target_height / image.height)
    resized = image.resize(
        (round(image.width * scale), round(image.height * scale)),
        Image.Resampling.LANCZOS,
    )
    return resized, (
        (target_width - resized.width) // 2,
        (target_height - resized.height) // 2,
    )


def _paste_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    *,
    box: tuple[int, int, int, int],
    title: str,
    subtitle: str,
) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=18, fill=PANEL, outline=BORDER, width=2)
    draw.text((left + 16, top + 12), title, fill=TEXT, font=_font(20, bold=True))
    draw.text((left + 16, top + 42), subtitle, fill=MUTED, font=_font(14))
    image_top = top + 72
    image_size = (right - left - 20, bottom - image_top - 12)
    fitted, offset = _fit_image(image, image_size)
    canvas.paste(fitted, (left + 10 + offset[0], image_top + offset[1]))


def _metric_value(metrics: Mapping[str, float], name: str) -> str:
    value = metrics.get(name)
    return "n/a" if value is None or not math.isfinite(value) else f"{value:.2f}"


def _metric_caption(metrics: Mapping[str, float]) -> str:
    return (
        f"I-lift {_metric_value(metrics, 'instance_lift')} · "
        f"G-lift {_metric_value(metrics, 'category_lift')} · "
        f"G-cov {_metric_value(metrics, 'category_coverage')}"
    )


def _role_labels(
    heads: tuple[int, ...],
    *,
    instance_head: int,
    semantic_head: int,
    control_head: int | None,
) -> dict[int, str]:
    labels = {head: "comparison" for head in heads}
    labels[instance_head] = "instance / matching"
    labels[semantic_head] = "category semantic"
    if control_head is not None:
        labels[control_head] = "weak-response control"
    return labels


def _render_strip(
    case: RenderCase,
    *,
    layer: int,
    heads: tuple[int, ...],
    mode: str,
    roles: Mapping[int, str],
    percentiles: tuple[float, float],
) -> Image.Image:
    panel_width = 500
    panel_height = 470
    gap = 22
    margin = 48
    columns = 2 + len(heads)
    width = 2 * margin + columns * panel_width + (columns - 1) * gap
    height = 690
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    mode_label = "Q–K attention" if mode == "qk" else "K–K similarity"
    title = case.label or f"{case.category_name} · video {case.video.name}"
    draw.text((margin, 24), title, fill=TEXT, font=_font(31, bold=True))
    draw.text(
        (margin, 67),
        (
            f"Layer {layer} · {mode_label} · track {case.sample.track_id} · "
            f"frames {case.source_frame} → {case.target_frame}"
        ),
        fill=MUTED,
        font=_font(18),
    )

    panels: list[tuple[Image.Image, str, str]] = [
        (
            _source_overlay(case),
            "Source query",
            f"t={case.source_frame} · red marker inside I",
        ),
        (
            _target_overlay(case),
            "Target regions",
            f"t={case.target_frame} · I cyan · G yellow",
        ),
    ]
    for head in heads:
        panels.append(
            (
                _heatmap_overlay(case, case.maps[mode][head], percentiles=percentiles),
                f"H{head} · {roles[head]}",
                _metric_caption(case.metrics[mode][head]),
            )
        )
    panel_top = 112
    for column, (panel_image, panel_title, panel_subtitle) in enumerate(panels):
        left = margin + column * (panel_width + gap)
        _paste_panel(
            canvas,
            draw,
            panel_image,
            box=(left, panel_top, left + panel_width, panel_top + panel_height),
            title=panel_title,
            subtitle=panel_subtitle,
        )

    footer_y = 608
    draw.rounded_rectangle(
        (margin, footer_y, width - margin, height - 24),
        radius=15,
        fill=(236, 242, 250),
    )
    draw.text(
        (margin + 20, footer_y + 14),
        (
            f"Heatmap: independent p{percentiles[0]:g}–p{percentiles[1]:g} normalization "
            "for spatial readability; displayed lift/coverage values use raw similarities."
        ),
        fill=MUTED,
        font=_font(15),
    )
    legend_x = width - margin - 540
    draw.line((legend_x, footer_y + 43, legend_x + 34, footer_y + 43), fill=CYAN, width=6)
    draw.text((legend_x + 43, footer_y + 32), "same instance (I)", fill=TEXT, font=_font(14))
    draw.line(
        (legend_x + 205, footer_y + 43, legend_x + 239, footer_y + 43),
        fill=YELLOW,
        width=6,
    )
    draw.text((legend_x + 248, footer_y + 32), "same category (G)", fill=TEXT, font=_font(14))
    return canvas


def _render_overview(
    strips: Sequence[Image.Image],
    cases: Sequence[RenderCase],
    *,
    layer: int,
    mode: str,
) -> Image.Image:
    if not strips:
        raise ValueError("cannot render an empty overview")
    width = max(image.width for image in strips)
    header_height = 104
    gap = 14
    height = header_height + sum(image.height for image in strips) + gap * (len(strips) - 1)
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    mode_label = "Q–K attention" if mode == "qk" else "K–K similarity"
    draw.text(
        (48, 22),
        f"OVIS qualitative head comparison · Layer {layer} · {mode_label}",
        fill=TEXT,
        font=_font(32, bold=True),
    )
    draw.text(
        (48, 64),
        f"{len(cases)} automatically or manually selected query-target examples",
        fill=MUTED,
        font=_font(18),
    )
    top = header_height
    for strip in strips:
        canvas.paste(strip, (0, top))
        top += strip.height + gap
    return canvas


def _save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)


def _case_record(case: RenderCase, *, layer: int, heads: tuple[int, ...]) -> dict[str, Any]:
    return {
        "video": case.video.name,
        "category": case.category_name,
        "label": case.label,
        "layer": layer,
        "heads": list(heads),
        "chunk": case.chunk.index,
        "source_frame": case.source_frame,
        "target_frame": case.target_frame,
        "track_id": case.sample.track_id,
        "source_token": case.sample.source_token,
        "query_xy_pixels": [round(value, 3) for value in case.query_xy],
        "selection_score": case.selection_score,
        "region_patches": {
            "position": int(case.sample.position.sum()),
            "instance": int(case.sample.instance.sum()),
            "category": int(case.sample.category.sum()),
            "other": int(case.sample.other.sum()),
        },
        "metrics": {
            mode: {
                str(head): {
                    name: float(value)
                    for name, value in head_metrics.items()
                    if math.isfinite(float(value))
                }
                for head, head_metrics in mode_metrics.items()
            }
            for mode, mode_metrics in case.metrics.items()
        },
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
        "--frames-zip",
        type=Path,
        default=workspace / "valid.zip.tos-download",
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        required=True,
        help="directory containing one FeatureVideo subdirectory per OVIS video",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_repo_root() / "reports" / "ovis_attention_maps",
    )
    parser.add_argument("--videos", default=None, help="comma-separated video names or ids")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--layer", type=int, default=None, help="default: infer single layer")
    parser.add_argument("--heads", default="all", help="comma/range selection or all")
    parser.add_argument("--modes", default="qk,kk", help="qk, kk, or qk,kk")
    parser.add_argument("--instance-head", type=int, default=None)
    parser.add_argument("--semantic-head", type=int, default=None)
    parser.add_argument("--control-head", type=int, default=None)
    parser.add_argument(
        "--selection-objective",
        choices=("balanced", "instance", "category"),
        default="balanced",
    )
    parser.add_argument("--selection-mode", choices=("qk", "kk"), default="qk")
    parser.add_argument("--candidate-limit", type=int, default=48)
    parser.add_argument(
        "--case-file",
        type=Path,
        default=None,
        help="optional JSON list with video/source_frame/target_frame/track_id",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="device used only for similarity calculation",
    )
    parser.add_argument("--frame-gap", type=int, default=4)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--queries-per-instance", type=int, default=8)
    parser.add_argument("--query-min-coverage", type=float, default=0.8)
    parser.add_argument("--region-min-coverage", type=float, default=0.5)
    parser.add_argument("--position-radius", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--percentiles",
        default="5,99.5",
        help="per-map display normalization percentiles",
    )
    parser.add_argument(
        "--style",
        choices=("raw", "panel", "both"),
        default="raw",
        help="raw saves native-resolution files; panel saves annotated comparison sheets",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.70,
        help="maximum heatmap opacity for raw overlays",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cv2.setNumThreads(1)
    modes = _parse_modes(args.modes)
    requested_heads = _parse_indices(args.heads)
    percentiles = _parse_percentiles(args.percentiles)
    if args.max_videos is not None and args.max_videos <= 0:
        raise SystemExit("--max-videos must be positive")
    if args.candidate_limit <= 0 or args.queries_per_instance <= 0:
        raise SystemExit("candidate/query counts must be positive")
    if args.frame_gap <= 0 or args.frame_stride <= 0:
        raise SystemExit("frame gap/stride must be positive")
    if not 0 < args.query_min_coverage <= 1 or not 0 < args.region_min_coverage <= 1:
        raise SystemExit("mask coverage thresholds must be in (0, 1]")
    if not 0 <= args.overlay_alpha <= 1:
        raise SystemExit("--overlay-alpha must be in [0, 1]")

    dataset = OvisDataset(args.annotations)
    feature_names = tuple(
        sorted(
            path.name
            for path in args.feature_root.iterdir()
            if path.is_dir() and (path / "manifest.json").is_file()
        )
    )
    requested_videos = _comma_values(args.videos) or feature_names
    videos = dataset.select(requested_videos)
    if args.max_videos is not None:
        videos = videos[: args.max_videos]
    if not videos:
        raise SystemExit("no videos selected")

    feature_videos = {
        video.name: FeatureVideo.open(args.feature_root / video.name) for video in videos
    }
    first_feature = feature_videos[videos[0].name]
    if args.layer is None:
        if len(first_feature.layers) != 1:
            raise SystemExit(
                f"feature root captures layers {first_feature.layers}; pass --layer explicitly"
            )
        layer = first_feature.layers[0]
    else:
        layer = args.layer
    available_heads = _available_heads(first_feature, layer)
    heads = available_heads if requested_heads is None else requested_heads
    missing_heads = [head for head in heads if head not in available_heads]
    if missing_heads:
        raise SystemExit(f"requested heads were not captured: {missing_heads}")

    instance_head = args.instance_head if args.instance_head is not None else heads[0]
    semantic_head = (
        args.semantic_head
        if args.semantic_head is not None
        else heads[min(1, len(heads) - 1)]
    )
    control_head = (
        args.control_head
        if args.control_head is not None
        else (heads[-1] if len(heads) >= 3 else None)
    )
    role_heads = tuple(
        dict.fromkeys(
            head
            for head in (instance_head, semantic_head, control_head)
            if head is not None
        )
    )
    invalid_roles = [head for head in role_heads if head not in heads]
    if invalid_roles:
        raise SystemExit(f"role heads must be included in --heads: {invalid_roles}")

    for video in videos:
        feature_video = feature_videos[video.name]
        if layer not in feature_video.layers:
            raise ValueError(f"{video.name}: layer {layer} was not captured")
        video_heads = _available_heads(feature_video, layer)
        missing = [head for head in heads if head not in video_heads]
        if missing:
            raise ValueError(f"{video.name}: heads {missing} were not captured")
        if feature_video.num_frames != video.length:
            raise ValueError(
                f"{video.name}: {feature_video.num_frames} feature frames != "
                f"{video.length} OVIS frames"
            )

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
        query_batch_size=32,
    )
    device = torch.device(args.device)
    manual_cases = _load_manual_cases(args.case_file) if args.case_file else ()
    manual_by_video = {case.video: case for case in manual_cases}
    unknown_manual = sorted(set(manual_by_video) - {video.name for video in videos})
    if unknown_manual:
        raise ValueError(f"case-file videos are not selected/present: {unknown_manual}")

    print(
        f"render {len(videos)} videos · layer {layer} · heads {heads} · "
        f"modes {modes} · device {device}"
    )
    render_cases: list[RenderCase] = []
    with zipfile.ZipFile(args.frames_zip) as archive:
        members = frozenset(archive.namelist())
        for index, video in enumerate(videos, start=1):
            feature_video = feature_videos[video.name]
            manual = manual_by_video.get(video.name)
            if manual is not None:
                candidate = _make_manual_candidate(manual, video, feature_video, settings)
                selection_score = None
                label = manual.label
                selection_note = "manual"
            else:
                candidate, selection_score = _auto_select_candidate(
                    video,
                    feature_video,
                    layer=layer,
                    selection_heads=role_heads,
                    instance_head=instance_head,
                    semantic_head=semantic_head,
                    control_head=control_head,
                    objective=args.selection_objective,
                    selection_mode=args.selection_mode,
                    candidate_limit=args.candidate_limit,
                    device=device,
                    settings=settings,
                )
                label = None
                selection_note = f"auto score={selection_score:.3f}"
            case = _load_render_case(
                video,
                feature_video,
                candidate,
                dataset=dataset,
                archive=archive,
                members=members,
                layer=layer,
                heads=heads,
                modes=modes,
                device=device,
                settings=settings,
                selection_score=selection_score,
                label=label,
            )
            render_cases.append(case)
            print(
                f"[{index}/{len(videos)}] {video.name}: {case.category_name} "
                f"track={case.sample.track_id} frames={case.source_frame}->{case.target_frame} "
                f"token={case.sample.source_token} ({selection_note})",
                flush=True,
            )

    roles = _role_labels(
        heads,
        instance_head=instance_head,
        semantic_head=semantic_head,
        control_head=control_head,
    )
    if args.style in {"raw", "both"}:
        for case in render_cases:
            _save_raw_case(
                case,
                output_dir=args.output_dir,
                layer=layer,
                heads=heads,
                modes=modes,
                percentiles=percentiles,
                overlay_alpha=args.overlay_alpha,
            )

    if args.style in {"panel", "both"}:
        strips_by_mode: dict[str, list[Image.Image]] = {mode: [] for mode in modes}
        individual_dir = args.output_dir / "individual"
        for case in render_cases:
            for mode in modes:
                strip = _render_strip(
                    case,
                    layer=layer,
                    heads=heads,
                    mode=mode,
                    roles=roles,
                    percentiles=percentiles,
                )
                strips_by_mode[mode].append(strip)
                output = (
                    individual_dir / f"{case.video.name}_layer{layer:02d}_{mode}.png"
                )
                _save_png(strip, output)
                print(output)

        for mode, strips in strips_by_mode.items():
            overview = _render_overview(strips, render_cases, layer=layer, mode=mode)
            output = args.output_dir / f"overview_layer{layer:02d}_{mode}.png"
            _save_png(overview, output)
            print(output)

    metadata = {
        "feature_root": str(args.feature_root),
        "annotations": str(args.annotations),
        "frames_zip": str(args.frames_zip),
        "layer": layer,
        "heads": list(heads),
        "modes": list(modes),
        "roles": {str(head): label for head, label in roles.items()},
        "selection": {
            "objective": args.selection_objective,
            "mode": args.selection_mode,
            "candidate_limit": args.candidate_limit,
            "frame_gap": args.frame_gap,
            "frame_stride": args.frame_stride,
            "queries_per_instance": args.queries_per_instance,
            "query_min_coverage": args.query_min_coverage,
            "region_min_coverage": args.region_min_coverage,
            "position_radius": args.position_radius,
            "seed": args.seed,
        },
        "output_style": args.style,
        "overlay_alpha": args.overlay_alpha,
        "display_percentiles": list(percentiles),
        "cases": [
            _case_record(case, layer=layer, heads=heads) for case in render_cases
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / "selection_and_metrics.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(metadata_path)


if __name__ == "__main__":
    main()
