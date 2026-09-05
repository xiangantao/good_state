"""Render publication-style OVIS semantic-head figures without matplotlib."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import zipfile
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
    _feature_path,
    _score_regions,
)
from ovis_data import OvisDataset, OvisVideo
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils
from safetensors import safe_open

from heft import FeatureVideo

Color = tuple[int, int, int]
Report = dict[str, Any]

BACKGROUND: Color = (246, 248, 252)
PANEL: Color = (255, 255, 255)
TEXT: Color = (27, 37, 54)
MUTED: Color = (101, 113, 133)
GRID: Color = (222, 228, 237)
BLUE: Color = (51, 112, 221)
ORANGE: Color = (239, 142, 42)
PURPLE: Color = (129, 92, 246)
GREEN: Color = (20, 156, 119)
RED: Color = (224, 79, 95)
CYAN: Color = (23, 190, 207)
YELLOW: Color = (255, 193, 7)
GRAY: Color = (154, 163, 178)
DARK_GRAY: Color = (73, 84, 103)

FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


@dataclass(frozen=True, slots=True)
class QualitativeSpec:
    video: str
    chunk: int
    source_frame: int
    target_frame: int
    source_token: int
    track_id: int
    label: str


@dataclass(frozen=True, slots=True)
class QualitativeExample:
    spec: QualitativeSpec
    source: Image.Image
    target: Image.Image
    instance_mask: np.ndarray
    category_mask: np.ndarray
    query_xy: tuple[float, float]
    qk_maps: dict[int, np.ndarray]
    qk_metrics: dict[int, dict[str, float]]


QUALITATIVE_SPECS = (
    QualitativeSpec(
        video="d084134f",
        chunk=2,
        source_frame=54,
        target_frame=58,
        source_token=914,
        track_id=75,
        label="Horse",
    ),
    QualitativeSpec(
        video="0d0030a7",
        chunk=4,
        source_frame=104,
        target_frame=108,
        source_token=895,
        track_id=96,
        label="Zebra",
    ),
)


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), size)


def _text_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return round(box[2] - box[0]), round(box[3] - box[1])


def _blend(first: Color, second: Color, amount: float) -> Color:
    amount = min(max(amount, 0.0), 1.0)
    return (
        round(first[0] + (second[0] - first[0]) * amount),
        round(first[1] + (second[1] - first[1]) * amount),
        round(first[2] + (second[2] - first[2]) * amount),
    )


def _panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    radius: int = 28,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=PANEL, outline=(232, 236, 243), width=2)


def _save(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, optimize=True)


def _head_rows(report: Report, mode: str) -> dict[int, dict[str, Any]]:
    return {
        int(row["head"]): row
        for row in report["heads"]
        if str(row["mode"]) == mode
    }


def _metric_summary(
    report: Report,
    mode: str,
    head: int,
    metric: str,
) -> dict[str, float | int]:
    return cast(dict[str, float | int], _head_rows(report, mode)[head]["metrics"][metric])


def _metric(
    report: Report,
    mode: str,
    head: int,
    metric: str,
    field: str = "mean",
) -> float:
    return float(_metric_summary(report, mode, head, metric)[field])


def _nice_ticks(maximum: float, count: int = 5) -> list[float]:
    if maximum <= 0:
        return [0.0]
    raw = maximum / count
    magnitude = 10 ** math.floor(math.log10(raw))
    normalized = raw / magnitude
    if normalized <= 1:
        step = magnitude
    elif normalized <= 2:
        step = 2 * magnitude
    elif normalized <= 2.5:
        step = 2.5 * magnitude
    elif normalized <= 5:
        step = 5 * magnitude
    else:
        step = 10 * magnitude
    stop = math.ceil(maximum / step) * step
    return [index * step for index in range(round(stop / step) + 1)]


def _draw_header(
    draw: ImageDraw.ImageDraw,
    title: str,
    subtitle: str,
    *,
    width: int,
) -> None:
    draw.text((80, 52), title, fill=TEXT, font=_font(50, bold=True))
    draw.text((82, 118), subtitle, fill=MUTED, font=_font(24))
    draw.line((80, 164, width - 80, 164), fill=GRID, width=2)


def _draw_bar_panel(
    draw: ImageDraw.ImageDraw,
    report: Report,
    *,
    box: tuple[int, int, int, int],
    mode: str,
    metric: str,
    title: str,
    color: Color,
) -> None:
    left, top, right, bottom = box
    _panel(draw, box)
    draw.text((left + 32, top + 24), title, fill=TEXT, font=_font(28, bold=True))
    draw.text(
        (left + 32, top + 62),
        f"{mode.upper()}  ·  mean lift over O",
        fill=MUTED,
        font=_font(18),
    )

    rows = []
    for head in range(12):
        summary = _metric_summary(report, mode, head, metric)
        rows.append((head, summary))
    rows.sort(key=lambda item: float(item[1]["mean"]), reverse=True)

    chart_left = left + 105
    chart_right = right - 80
    chart_top = top + 112
    chart_bottom = bottom - 58
    maximum = max(float(summary["q75"]) for _, summary in rows) * 1.08
    ticks = _nice_ticks(maximum)
    axis_max = ticks[-1]

    for tick in ticks:
        x = chart_left + (chart_right - chart_left) * tick / axis_max
        draw.line((x, chart_top, x, chart_bottom), fill=GRID, width=2)
        label = f"{tick:.1f}" if axis_max >= 1 else f"{tick:.2f}"
        tw, _ = _text_size(draw, label, _font(15))
        draw.text((x - tw / 2, chart_bottom + 12), label, fill=MUTED, font=_font(15))

    row_height = (chart_bottom - chart_top) / len(rows)
    bar_height = max(16, int(row_height * 0.54))
    for rank, (head, summary) in enumerate(rows):
        mean = float(summary["mean"])
        q25 = float(summary["q25"])
        q75 = float(summary["q75"])
        median = float(summary["median"])
        y = chart_top + (rank + 0.5) * row_height
        head_font = _font(17, bold=(rank == 0 or head == 7))
        label_color = GREEN if head == 7 else TEXT
        draw.text((left + 38, y - 11), f"H{head}", fill=label_color, font=head_font)

        x_mean = chart_left + (chart_right - chart_left) * mean / axis_max
        x_q25 = chart_left + (chart_right - chart_left) * q25 / axis_max
        x_q75 = chart_left + (chart_right - chart_left) * q75 / axis_max
        x_median = chart_left + (chart_right - chart_left) * median / axis_max
        bar_color = GREEN if head == 7 else color
        draw.rounded_rectangle(
            (chart_left, y - bar_height / 2, x_mean, y + bar_height / 2),
            radius=bar_height // 2,
            fill=_blend(bar_color, PANEL, 0.08 if rank < 3 else 0.28),
        )
        draw.line((x_q25, y, x_q75, y), fill=DARK_GRAY, width=3)
        draw.line((x_q25, y - 7, x_q25, y + 7), fill=DARK_GRAY, width=2)
        draw.line((x_q75, y - 7, x_q75, y + 7), fill=DARK_GRAY, width=2)
        draw.ellipse(
            (x_median - 4, y - 4, x_median + 4, y + 4),
            fill=PANEL,
            outline=DARK_GRAY,
            width=2,
        )
        value_label = f"{mean:.3f}"
        draw.text(
            (min(x_mean + 8, chart_right - 52), y - 10),
            value_label,
            fill=TEXT,
            font=_font(15, bold=rank < 3),
        )


def render_rankings(report: Report, output: Path) -> None:
    width, height = 2500, 1740
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    _draw_header(
        draw,
        "Layer 15 head ranking on OVIS",
        "Bars: video-level mean  ·  dot: median  ·  whisker: interquartile range",
        width=width,
    )
    panels = (
        ((80, 205, 1225, 925), "qk", "instance_lift", "Instance semantic (I)", BLUE),
        ((1275, 205, 2420, 925), "qk", "category_lift", "Category semantic (G)", ORANGE),
        ((80, 965, 1225, 1685), "kk", "instance_lift", "Instance representation (I)", PURPLE),
        ((1275, 965, 2420, 1685), "kk", "category_lift", "Category representation (G)", GREEN),
    )
    for box, mode, metric, title, color in panels:
        _draw_bar_panel(
            draw,
            report,
            box=box,
            mode=mode,
            metric=metric,
            title=title,
            color=color,
        )
    _save(image, output)


def _scatter_ticks(minimum: float, maximum: float, count: int = 5) -> list[float]:
    return [minimum + (maximum - minimum) * index / count for index in range(count + 1)]


def _draw_scatter_panel(
    draw: ImageDraw.ImageDraw,
    report: Report,
    *,
    box: tuple[int, int, int, int],
    mode: str,
) -> None:
    left, top, right, bottom = box
    _panel(draw, box)
    draw.text(
        (left + 34, top + 26),
        f"{mode.upper()} functional map",
        fill=TEXT,
        font=_font(31, bold=True),
    )
    draw.text(
        (left + 34, top + 70),
        "Right = stronger queried-instance response; up = stronger same-category transfer",
        fill=MUTED,
        font=_font(17),
    )

    heads = list(range(12))
    xs = np.array([_metric(report, mode, head, "instance_lift") for head in heads])
    ys = np.array([_metric(report, mode, head, "category_lift") for head in heads])
    x_span = float(np.ptp(xs)) or 1.0
    y_span = float(np.ptp(ys)) or 1.0
    x_min, x_max = float(xs.min() - 0.12 * x_span), float(xs.max() + 0.12 * x_span)
    y_min, y_max = float(ys.min() - 0.15 * y_span), float(ys.max() + 0.15 * y_span)

    plot_left, plot_right = left + 105, right - 55
    plot_top, plot_bottom = top + 135, bottom - 105

    def px(value: float) -> float:
        return plot_left + (value - x_min) / (x_max - x_min) * (plot_right - plot_left)

    def py(value: float) -> float:
        return plot_bottom - (value - y_min) / (y_max - y_min) * (plot_bottom - plot_top)

    for tick in _scatter_ticks(x_min, x_max):
        x = px(tick)
        draw.line((x, plot_top, x, plot_bottom), fill=GRID, width=2)
        label = f"{tick:.2f}"
        tw, _ = _text_size(draw, label, _font(16))
        draw.text((x - tw / 2, plot_bottom + 16), label, fill=MUTED, font=_font(16))
    for tick in _scatter_ticks(y_min, y_max):
        y = py(tick)
        draw.line((plot_left, y, plot_right, y), fill=GRID, width=2)
        label = f"{tick:.2f}"
        tw, th = _text_size(draw, label, _font(16))
        draw.text((plot_left - tw - 13, y - th / 2), label, fill=MUTED, font=_font(16))

    slope, intercept = np.polyfit(xs, ys, 1)
    line_start = float(slope * x_min + intercept)
    line_stop = float(slope * x_max + intercept)
    draw.line(
        (px(x_min), py(line_start), px(x_max), py(line_stop)),
        fill=(167, 174, 186),
        width=3,
    )

    special_colors = {7: GREEN, 1: RED, 3: PURPLE, 4: ORANGE, 0: BLUE, 5: CYAN}
    offsets = {
        0: (15, -28),
        1: (15, -8),
        2: (14, 9),
        3: (14, -26),
        4: (14, 12),
        5: (14, -27),
        6: (14, 8),
        7: (16, -32),
        8: (13, -25),
        9: (14, 9),
        10: (14, -27),
        11: (14, 8),
    }
    for head, x_value, y_value in zip(heads, xs, ys, strict=True):
        robustness = _metric(
            report,
            mode,
            head,
            "category_lift",
            "top20_video_fraction",
        )
        radius = 11 + round(11 * robustness)
        center_x, center_y = px(float(x_value)), py(float(y_value))
        point_color = special_colors.get(head, GRAY)
        draw.ellipse(
            (
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius,
            ),
            fill=_blend(point_color, PANEL, 0.05),
            outline=TEXT if head == 7 else PANEL,
            width=4 if head == 7 else 2,
        )
        dx, dy = offsets[head]
        draw.text(
            (center_x + dx, center_y + dy),
            f"H{head}",
            fill=point_color if head in special_colors else DARK_GRAY,
            font=_font(18, bold=head in special_colors),
        )

    x_label = "Instance lift (I vs O)"
    tw, _ = _text_size(draw, x_label, _font(19, bold=True))
    draw.text(
        ((plot_left + plot_right - tw) / 2, bottom - 47),
        x_label,
        fill=TEXT,
        font=_font(19, bold=True),
    )
    draw.text(
        (left + 20, plot_top - 33),
        "Category lift",
        fill=TEXT,
        font=_font(18, bold=True),
    )
    draw.text(
        (plot_left + 12, plot_top + 12),
        "above trend: category-selective",
        fill=PURPLE,
        font=_font(16, bold=True),
    )
    draw.text(
        (plot_right - 285, plot_bottom - 35),
        "below trend: instance/position-biased",
        fill=RED,
        font=_font(15, bold=True),
    )


def render_function_map(report: Report, output: Path) -> None:
    width, height = 2500, 1260
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    _draw_header(
        draw,
        "Semantic strength and specialization",
        "Bubble size encodes category-lift top-20% frequency across videos; line is the across-head trend",
        width=width,
    )
    _draw_scatter_panel(draw, report, box=(80, 205, 1225, 1190), mode="qk")
    _draw_scatter_panel(draw, report, box=(1275, 205, 2420, 1190), mode="kk")
    _save(image, output)


def _heat_color(value: float, minimum: float, maximum: float, color: Color) -> Color:
    fraction = 0.5 if maximum == minimum else (value - minimum) / (maximum - minimum)
    return _blend((244, 247, 251), color, 0.14 + 0.78 * fraction)


def render_heatmap(report: Report, output: Path) -> None:
    width, height = 2500, 1450
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    _draw_header(
        draw,
        "Complete functional profile of all Layer 15 heads",
        "P: same-position control  ·  I: queried instance  ·  G: same category, different instance",
        width=width,
    )
    outer = (80, 205, 2420, 1370)
    _panel(draw, outer)

    columns = [
        ("qk", "position_lift", "P lift", PURPLE, ".2f"),
        ("qk", "instance_lift", "I lift", BLUE, ".2f"),
        ("qk", "category_lift", "G lift", ORANGE, ".2f"),
        ("qk", "category_lift", "G top20", GREEN, ".0%"),
        ("kk", "position_lift", "P lift", PURPLE, ".3f"),
        ("kk", "instance_lift", "I lift", BLUE, ".3f"),
        ("kk", "category_lift", "G lift", ORANGE, ".3f"),
        ("kk", "category_lift", "G top20", GREEN, ".0%"),
    ]
    readings = {
        0: "K-space object",
        1: "position + instance",
        2: "Q-K semantic",
        3: "category-selective",
        4: "position",
        5: "K-space object",
        6: "secondary semantic",
        7: "category-semantic",
        8: "weak / control",
        9: "mixed semantic",
        10: "secondary",
        11: "secondary",
    }

    label_left = 120
    grid_left = 330
    grid_top = 390
    cell_width = 205
    cell_height = 68
    reading_left = grid_left + len(columns) * cell_width + 35

    draw.text((label_left, 305), "Head", fill=MUTED, font=_font(19, bold=True))
    draw.text(
        (grid_left + cell_width * 2 - 48, 245),
        "Q-K attention",
        fill=TEXT,
        font=_font(26, bold=True),
    )
    draw.text(
        (grid_left + cell_width * 6 - 48, 245),
        "K-K representation",
        fill=TEXT,
        font=_font(26, bold=True),
    )
    draw.line(
        (grid_left + 4 * cell_width + 3, 250, grid_left + 4 * cell_width + 3, 1320),
        fill=GRID,
        width=4,
    )
    for index, (_, _, label, _, _) in enumerate(columns):
        tw, _ = _text_size(draw, label, _font(18, bold=True))
        draw.text(
            (grid_left + index * cell_width + (cell_width - tw) / 2, 318),
            label,
            fill=MUTED,
            font=_font(18, bold=True),
        )
    draw.text((reading_left, 318), "Interpretation", fill=MUTED, font=_font(18, bold=True))

    values_by_column: list[list[float]] = []
    for mode, metric, label, _color, _format in columns:
        field = "top20_video_fraction" if label == "G top20" else "mean"
        values_by_column.append(
            [_metric(report, mode, head, metric, field) for head in range(12)]
        )

    for head in range(12):
        y = grid_top + head * cell_height
        if head % 2 == 0:
            draw.rounded_rectangle(
                (105, y - 6, 2388, y + cell_height - 7),
                radius=12,
                fill=(249, 250, 253),
            )
        head_color = GREEN if head == 7 else TEXT
        draw.text((label_left, y + 14), f"H{head}", fill=head_color, font=_font(23, bold=True))
        for column_index, (_, _, _, color, number_format) in enumerate(columns):
            values = values_by_column[column_index]
            value = values[head]
            x0 = grid_left + column_index * cell_width + 7
            x1 = x0 + cell_width - 14
            y0 = y + 5
            y1 = y + cell_height - 10
            fill = _heat_color(value, min(values), max(values), color)
            draw.rounded_rectangle((x0, y0, x1, y1), radius=12, fill=fill)
            label = format(value, number_format)
            tw, th = _text_size(draw, label, _font(19, bold=True))
            draw.text(
                ((x0 + x1 - tw) / 2, (y0 + y1 - th) / 2 - 2),
                label,
                fill=TEXT,
                font=_font(19, bold=True),
            )
            if value == max(values):
                draw.rounded_rectangle(
                    (x0, y0, x1, y1),
                    radius=12,
                    outline=(255, 255, 255),
                    width=4,
                )
        reading_color = GREEN if head == 7 else (RED if head in (1, 4) else DARK_GRAY)
        draw.text(
            (reading_left, y + 16),
            readings[head],
            fill=reading_color,
            font=_font(19, bold=head in (1, 3, 4, 7)),
        )

    draw.text(
        (120, 1315),
        "Cell color is normalized within each column; annotated values are the original dataset means.",
        fill=MUTED,
        font=_font(17),
    )
    _save(image, output)


def _decode_mask(segmentation: Any) -> np.ndarray:
    decoded = mask_utils.decode(cast(Any, dict(segmentation)))
    if decoded.ndim == 3:
        decoded = decoded.any(axis=2)
    return decoded.astype(bool)


def _read_frame(
    archive: zipfile.ZipFile,
    members: frozenset[str],
    file_name: str,
) -> Image.Image:
    candidates = (file_name, f"valid/{file_name}")
    for candidate in candidates:
        if candidate in members:
            return Image.open(io.BytesIO(archive.read(candidate))).convert("RGB")
    raise FileNotFoundError(file_name)


def _find_sample(
    video: OvisVideo,
    feature_video: FeatureVideo,
    spec: QualitativeSpec,
    settings: EvaluationSettings,
) -> tuple[Any, SemanticSample]:
    chunk = next(item for item in feature_video.chunks if item.index == spec.chunk)
    samples = _build_samples(
        video,
        chunk,
        feature_size=feature_video.feature_size,
        settings=settings,
    )
    for sample in samples:
        if (
            chunk.start + sample.source_frame == spec.source_frame
            and chunk.start + sample.target_frame == spec.target_frame
            and sample.source_token == spec.source_token
            and sample.track_id == spec.track_id
        ):
            return chunk, sample
    raise ValueError(f"qualitative sample not found: {spec}")


def _load_qualitative_example(
    dataset: OvisDataset,
    archive: zipfile.ZipFile,
    members: frozenset[str],
    feature_root: Path,
    spec: QualitativeSpec,
    *,
    heads: tuple[int, ...],
) -> QualitativeExample:
    video = dataset.select((spec.video,))[0]
    feature_video = FeatureVideo.open(feature_root / spec.video)
    settings = EvaluationSettings(
        frame_gap=4,
        frame_stride=4,
        queries_per_instance=4,
        query_min_coverage=0.8,
        region_min_coverage=0.5,
        position_radius=1.0,
        temperature=1.0,
        epsilon=1e-12,
        seed=42,
        query_batch_size=32,
    )
    chunk, sample = _find_sample(video, feature_video, spec, settings)
    source = _read_frame(archive, members, video.file_names[spec.source_frame])
    target = _read_frame(archive, members, video.file_names[spec.target_frame])

    annotations = {int(annotation["id"]): annotation for annotation in video.annotations}
    queried = annotations[spec.track_id]
    category_id = int(queried["category_id"])
    instance_mask = _decode_mask(queried["segmentations"][spec.target_frame])
    category_mask = np.zeros_like(instance_mask)
    for track_id, annotation in annotations.items():
        if track_id == spec.track_id or int(annotation["category_id"]) != category_id:
            continue
        segmentation = annotation["segmentations"][spec.target_frame]
        if segmentation is not None:
            category_mask |= _decode_mask(segmentation)

    query_path = _feature_path(feature_video, chunk, 15, "query")
    key_path = _feature_path(feature_video, chunk, 15, "key")
    qk_maps: dict[int, np.ndarray] = {}
    qk_metrics: dict[int, dict[str, float]] = {}
    feature_height, feature_width = feature_video.feature_size
    spatial_tokens = feature_height * feature_width
    with (
        safe_open(query_path, framework="pt", device="cpu") as query_file,
        safe_open(key_path, framework="pt", device="cpu") as key_file,
    ):
        for head in heads:
            query = query_file.get_tensor(f"head_{head:03d}")[0]
            key = key_file.get_tensor(f"head_{head:03d}")[0]
            channels = query.shape[-1]
            source_index = sample.source_frame * spatial_tokens + sample.source_token
            target_start = sample.target_frame * spatial_tokens
            target_key = key[target_start : target_start + spatial_tokens].float()
            source_query = query[source_index].float()
            qk = torch.mv(target_key, source_query) * channels**-0.5
            values = qk.numpy()
            qk_maps[head] = values.reshape(feature_video.feature_size)
            qk_metrics[head] = _score_regions(
                values,
                sample,
                temperature=1.0,
                epsilon=1e-12,
            )

    token_y, token_x = divmod(spec.source_token, feature_width)
    query_xy = (
        (token_x + 0.5) * source.width / feature_width,
        (token_y + 0.5) * source.height / feature_height,
    )
    return QualitativeExample(
        spec=spec,
        source=source,
        target=target,
        instance_mask=instance_mask,
        category_mask=category_mask,
        query_xy=query_xy,
        qk_maps=qk_maps,
        qk_metrics=qk_metrics,
    )


def _fit_image(image: Image.Image, size: tuple[int, int]) -> tuple[Image.Image, tuple[int, int]]:
    target_width, target_height = size
    scale = min(target_width / image.width, target_height / image.height)
    resized = image.resize(
        (round(image.width * scale), round(image.height * scale)),
        Image.Resampling.LANCZOS,
    )
    return resized, ((target_width - resized.width) // 2, (target_height - resized.height) // 2)


def _mask_overlay(
    image: Image.Image,
    instance_mask: np.ndarray,
    category_mask: np.ndarray,
) -> Image.Image:
    rgb = np.asarray(image).copy()
    overlay = rgb.astype(np.float32)
    for mask, color in ((category_mask, YELLOW), (instance_mask, CYAN)):
        overlay[mask] = 0.56 * overlay[mask] + 0.44 * np.asarray(color, dtype=np.float32)
    result = np.clip(overlay, 0, 255).astype(np.uint8)
    result = _draw_mask_boundaries(result, instance_mask, category_mask)
    return Image.fromarray(result)


def _draw_mask_boundaries(
    rgb: np.ndarray,
    instance_mask: np.ndarray,
    category_mask: np.ndarray,
) -> np.ndarray:
    output = rgb.copy()
    for mask, color in ((category_mask, YELLOW), (instance_mask, CYAN)):
        mask_u8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, color, thickness=max(2, rgb.shape[1] // 450))
    return output


def _heatmap_overlay(
    image: Image.Image,
    heatmap: np.ndarray,
    instance_mask: np.ndarray,
    category_mask: np.ndarray,
) -> Image.Image:
    values = heatmap.astype(np.float32)
    low, high = np.percentile(values, (5.0, 99.5))
    normalized = np.clip((values - low) / max(float(high - low), 1e-8), 0.0, 1.0)
    normalized = normalized**1.15
    resized = cv2.resize(
        normalized,
        (image.width, image.height),
        interpolation=cv2.INTER_CUBIC,
    )
    resized = np.clip(resized, 0.0, 1.0)
    colored = cv2.applyColorMap((resized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).astype(np.float32)
    base = np.asarray(image).astype(np.float32)
    alpha = (0.08 + 0.64 * resized**1.4)[..., None]
    output = np.clip(base * (1.0 - alpha) + colored * alpha, 0, 255).astype(np.uint8)
    output = _draw_mask_boundaries(output, instance_mask, category_mask)
    return Image.fromarray(output)


def _draw_query_marker(image: Image.Image, point: tuple[float, float]) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    x, y = point
    radius = max(10, image.width // 70)
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        outline=(255, 255, 255),
        width=max(6, image.width // 180),
    )
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        outline=RED,
        width=max(3, image.width // 300),
    )
    draw.line((x - radius * 1.4, y, x + radius * 1.4, y), fill=RED, width=3)
    draw.line((x, y - radius * 1.4, x, y + radius * 1.4), fill=RED, width=3)
    return result


def _paste_panel_image(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    *,
    box: tuple[int, int, int, int],
    title: str,
    subtitle: str,
) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=22, fill=PANEL, outline=(226, 232, 240), width=2)
    draw.text((left + 18, top + 15), title, fill=TEXT, font=_font(22, bold=True))
    draw.text((left + 18, top + 49), subtitle, fill=MUTED, font=_font(16))
    image_top = top + 82
    fitted, offset = _fit_image(image, (right - left - 24, bottom - image_top - 16))
    canvas.paste(fitted, (left + 12 + offset[0], image_top + offset[1]))


def render_qualitative(
    examples: tuple[QualitativeExample, ...],
    output: Path,
) -> None:
    width, height = 2700, 1600
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    _draw_header(
        draw,
        "Qualitative Q-K attention: instance focus vs category transfer",
        "Cyan boundary = queried instance  ·  yellow boundary = same-category other instances  ·  red cross = source query patch",
        width=width,
    )
    panel_width = 500
    gap = 25
    left_margin = 55
    row_height = 625
    row_tops = (225, 890)
    heads = (1, 7, 8)

    for row_index, example in enumerate(examples):
        row_top = row_tops[row_index]
        source_marked = _draw_query_marker(example.source, example.query_xy)
        target_gt = _mask_overlay(
            example.target,
            example.instance_mask,
            example.category_mask,
        )
        panels: list[tuple[Image.Image, str, str]] = [
            (
                source_marked,
                "(a) Source query",
                f"{example.spec.label} · t={example.spec.source_frame}",
            ),
            (
                target_gt,
                "(b) Target regions",
                f"t+4={example.spec.target_frame} · I=cyan, G=yellow",
            ),
        ]
        for panel_index, head in enumerate(heads, start=2):
            metrics = example.qk_metrics[head]
            attention = _heatmap_overlay(
                example.target,
                example.qk_maps[head],
                example.instance_mask,
                example.category_mask,
            )
            panels.append(
                (
                    attention,
                    f"({chr(ord('a') + panel_index)}) Head {head} Q-K",
                    (
                        f"I-lift {metrics.get('instance_lift', float('nan')):.2f}  ·  "
                        f"G-lift {metrics.get('category_lift', float('nan')):.2f}"
                    ),
                )
            )
        for column, (panel_image, title, subtitle) in enumerate(panels):
            left = left_margin + column * (panel_width + gap)
            _paste_panel_image(
                image,
                draw,
                panel_image,
                box=(left, row_top, left + panel_width, row_top + row_height),
                title=title,
                subtitle=subtitle,
            )

    draw.rounded_rectangle((55, 1530, 2645, 1578), radius=18, fill=(237, 244, 250))
    draw.text(
        (78, 1541),
        "Heatmaps use per-head 5–99.5 percentile normalization for spatial readability; quantitative lift values remain unnormalized.",
        fill=DARK_GRAY,
        font=_font(17),
    )
    _save(image, output)


def _write_csv(report: Report, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "head",
        "qk_position_lift",
        "qk_instance_lift",
        "qk_category_lift",
        "qk_category_top20",
        "kk_position_lift",
        "kk_instance_lift",
        "kk_category_lift",
        "kk_category_top20",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for head in range(12):
            writer.writerow(
                {
                    "head": head,
                    "qk_position_lift": _metric(report, "qk", head, "position_lift"),
                    "qk_instance_lift": _metric(report, "qk", head, "instance_lift"),
                    "qk_category_lift": _metric(report, "qk", head, "category_lift"),
                    "qk_category_top20": _metric(
                        report,
                        "qk",
                        head,
                        "category_lift",
                        "top20_video_fraction",
                    ),
                    "kk_position_lift": _metric(report, "kk", head, "position_lift"),
                    "kk_instance_lift": _metric(report, "kk", head, "instance_lift"),
                    "kk_category_lift": _metric(report, "kk", head, "category_lift"),
                    "kk_category_top20": _metric(
                        report,
                        "kk",
                        head,
                        "category_lift",
                        "top20_video_fraction",
                    ),
                }
            )


def _ranking_table(report: Report, mode: str, metric: str, limit: int = 5) -> str:
    rows = sorted(
        range(12),
        key=lambda head: _metric(report, mode, head, metric),
        reverse=True,
    )[:limit]
    lines = ["| Rank | Head | Mean | Median | Top-20% videos |", "|---:|---:|---:|---:|---:|"]
    for rank, head in enumerate(rows, start=1):
        summary = _metric_summary(report, mode, head, metric)
        lines.append(
            f"| {rank} | H{head} | {float(summary['mean']):.4f} | "
            f"{float(summary['median']):.4f} | "
            f"{100 * float(summary['top20_video_fraction']):.1f}% |"
        )
    return "\n".join(lines)


def _write_markdown(report: Report, output_dir: Path) -> None:
    samples = int(_head_rows(report, "qk")[0]["samples"])
    text = f"""# OVIS Layer 15 Head 功能分析

**生成日期：2026-09-04**  
**数据：OVIS validation，140 个视频 / 8,784 帧 / 815 个实例轨迹 / 25 个类别**  
**有效查询：每个 head、每种模式 {samples:,} 个 query-target 样本**

## 一句话结论

> **H7 是目前最可信的类别语义头**：Q-K 与 K-K 的同类别跨实例指标均排名第 1，而且分别在 83.8% 和 75.7% 的视频中进入 top-20%。H1/H4 更像位置与实例响应头；H3 相对偏类别选择性；H0/H5 的语义主要体现在 K 空间。

## 核心图

### 1. 全部 head 排名与跨视频分布

![Head rankings](figures/01_head_rankings.png)

这张图不只展示均值，还展示了跨视频的中位数和四分位区间。H7 在两套类别语义指标上都排名第一，说明它不是少数视频拉高的偶然结果。

### 2. 实例语义与类别语义的功能分布

![Functional map](figures/02_function_map.png)

- **右上角**：实例和类别信号都强；H7 最突出。
- **趋势线上方**：相对于实例响应，更偏类别泛化；H3 是值得继续验证的“类别选择型”候选。
- **趋势线下方**：更偏实例或位置；H1/H4 需要结合 P 指标解释，不能直接叫纯语义头。
- Q-K 与 K-K 数值尺度不同，只应在各自模式内比较排名和相对位置。

### 3. P / I / G 完整功能热力图

![Head profile heatmap](figures/03_head_metric_heatmap.png)

这张图把位置控制 P、查询实例 I、同类别其他实例 G 放在一起看，能避免把位置响应误判为语义：

- **H7：类别语义主候选。** G 强且稳定，P 不占优势。
- **H1：位置 + 实例。** Q-K 的 P 和 I 都是第 1，但 G 只有第 8。
- **H4：位置候选。** Q-K P 第 2、K-K P 第 1，而 G 明显偏弱。
- **H3：类别选择性候选。** 绝对 G 强度不是第一，但相对于自身 I/P 更偏 G。
- **H0/H5：K-space object heads。** K-K 的 I/G 很强，Q-K 不同程度偏弱。
- **H2：Q-K semantic / matching 混合候选。** Q-K G 第 2，但 K-K G 第 9；结合此前 TAP-Vid matching 结果，不应把它简单定义为“纯语义头”。
- **H8：弱响应对照。** 四组语义排名基本垫底。

### 4. 真实视频上的注意力可视化

![Qualitative attention](figures/04_qualitative_attention.png)

图中青色边界是查询实例 I，黄色边界是同类别不同实例 G。H7 在 horse 和 zebra 两个场景中都会把高响应扩展到其他同类实例；H1 更集中于查询实例或原坐标附近；H8 作为弱响应对照。

> 注意：热图为了看清空间结构，逐 head 使用 5–99.5 百分位归一化；图中标注的 I/G lift 是未归一化的真实指标。

## 排名表

### Q-K instance semantic

{_ranking_table(report, 'qk', 'instance_lift')}

### Q-K category semantic

{_ranking_table(report, 'qk', 'category_lift')}

### K-K instance semantic

{_ranking_table(report, 'kk', 'instance_lift')}

### K-K category semantic

{_ranking_table(report, 'kk', 'category_lift')}

## 可汇报结论

1. **Layer 15 已出现明显的 head 功能分化，而不是 12 个 head 做同一件事。**
2. **H7 是跨 Q-K attention 与 K-K representation 都成立的类别语义头。**
3. **H1/H4 的高实例响应受到位置效应显著影响，必须通过 P 指标排除混淆。**
4. **H2 的 Q-K 语义指标很强，但 K-K 证据弱；结合 matching 结果，更可能是多功能或路由型 head。**
5. **语义头不应只按单个平均分判断，应同时看绝对 G lift、跨视频 top-20% 稳定性、I/G/P 相对结构，以及 Q-K/K-K 一致性。**

## 实验边界

- OVIS 没有点级真实对应 C，因此本报告不能单独判定 matching head；matching 仍需 TAP-Vid。
- G 表示同类别、不同实例；140 个视频中有 136 个对该指标提供有效样本。
- 当前结论仅针对 Wan2.1、Layer 15、step 49；跨层结论需要后续提取其他层验证。
- 所有汇总遵循：query mean → instance mean → video mean → dataset mean，避免长视频或大实例支配结果。

## 文件

- `figures/01_head_rankings.png`：四组排名及 IQR
- `figures/02_function_map.png`：语义功能分布图
- `figures/03_head_metric_heatmap.png`：P/I/G 全量热力图
- `figures/04_qualitative_attention.png`：真实视频注意力案例
- `tables/head_metrics.csv`：12 个 head 的可复用数值表
- `report.html`：适合浏览器直接展示的版本
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def _write_html(report: Report, output_dir: Path) -> None:
    samples = int(_head_rows(report, "qk")[0]["samples"])
    table_rows = []
    for head in range(12):
        table_rows.append(
            "<tr>"
            f"<td><b>H{head}</b></td>"
            f"<td>{_metric(report, 'qk', head, 'position_lift'):.3f}</td>"
            f"<td>{_metric(report, 'qk', head, 'instance_lift'):.3f}</td>"
            f"<td>{_metric(report, 'qk', head, 'category_lift'):.3f}</td>"
            f"<td>{_metric(report, 'kk', head, 'position_lift'):.3f}</td>"
            f"<td>{_metric(report, 'kk', head, 'instance_lift'):.3f}</td>"
            f"<td>{_metric(report, 'kk', head, 'category_lift'):.3f}</td>"
            "</tr>"
        )
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OVIS Layer 15 Head 功能分析</title>
<style>
:root{{--ink:#1b2536;--muted:#657185;--card:#fff;--bg:#f4f7fb;--green:#149c77;--blue:#3370dd;}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.7 Arial,"Noto Sans CJK SC",sans-serif}}
main{{max-width:1260px;margin:auto;padding:48px 28px 80px}} h1{{font-size:38px;margin:0 0 8px}} h2{{margin-top:48px;font-size:27px}}
.meta{{color:var(--muted)}} .hero,.card{{background:var(--card);border:1px solid #e3e8f0;border-radius:18px;padding:26px 30px;box-shadow:0 8px 30px #23334d0d}}
.hero{{border-left:7px solid var(--green);font-size:20px;margin:28px 0}} img{{display:block;width:100%;border-radius:16px;border:1px solid #e3e8f0;margin:18px 0 36px}}
.grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:18px}} .tag{{display:inline-block;padding:4px 10px;border-radius:999px;background:#e8f7f2;color:#08795b;font-weight:700}}
table{{width:100%;border-collapse:collapse;background:white;border-radius:14px;overflow:hidden}} th,td{{padding:11px 14px;text-align:right;border-bottom:1px solid #edf0f4}} th:first-child,td:first-child{{text-align:left}} th{{background:#eef3fa}}
li{{margin:8px 0}} code{{background:#edf1f6;padding:2px 6px;border-radius:5px}} @media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>OVIS Layer 15 Head 功能分析</h1>
<div class="meta">2026-09-04 · 140 videos · 8,784 frames · 815 tracks · 25 categories · {samples:,} query-target samples/head/mode</div>
<div class="hero"><span class="tag">核心结论</span><br><b>H7 是目前最可信的类别语义头：</b>Q-K 与 K-K 的 G 指标均排名第 1，且跨视频稳定。H1/H4 更偏位置与实例；H3 更偏类别选择；H0/H5 的对象语义主要存在于 K 空间。</div>
<h2>1. Head 排名与跨视频分布</h2><img src="figures/01_head_rankings.png">
<h2>2. 实例—类别功能分布</h2><img src="figures/02_function_map.png">
<div class="grid"><div class="card"><b>H7</b><br>绝对 G 强度和跨视频稳定性都最高，是主语义头。</div><div class="card"><b>H3</b><br>相对自身实例响应更偏类别，适合作为类别选择型候选。</div><div class="card"><b>H1 / H4</b><br>P 指标非常强，不能把实例高分直接解释成纯语义。</div><div class="card"><b>H2</b><br>Q-K G 第 2、K-K G 第 9，可能是 matching/semantic 混合路由。</div></div>
<h2>3. P / I / G 完整剖面</h2><img src="figures/03_head_metric_heatmap.png">
<h2>4. 真实 OVIS 案例</h2><img src="figures/04_qualitative_attention.png">
<p class="meta">青色：查询实例 I；黄色：同类别其他实例 G；红色十字：source query patch。热图仅为显示进行逐 head 百分位归一化，I/G lift 数值未归一化。</p>
<h2>完整数值</h2><table><thead><tr><th>Head</th><th>QK-P</th><th>QK-I</th><th>QK-G</th><th>KK-P</th><th>KK-I</th><th>KK-G</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
<h2>汇报时建议这样说</h2><ol><li>Layer 15 已经出现稳定的 head 功能分化。</li><li>H7 同时通过 Q-K 与 K-K 的类别跨实例检验，是最稳健的类别语义头。</li><li>H1/H4 说明仅看实例响应会被位置效应混淆，P/I/G 必须联合判断。</li><li>H2 与此前 matching 证据结合后更像多功能或 attention-routing head，不宜简单贴单一标签。</li><li>OVIS 不提供点级对应 C，因此 matching 结论仍由 TAP-Vid 支撑。</li></ol>
</main></body></html>"""
    (output_dir / "report.html").write_text(html, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    workspace = _workspace_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        type=Path,
        default=workspace / "eval" / "ovis_wan_layer15_semantics.json",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=workspace / "annotations_valid_withgt.json.tos-download",
    )
    parser.add_argument(
        "--frames",
        type=Path,
        default=workspace / "valid.zip.tos-download",
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=workspace / "features" / "ovis_wan" / "layers_015_heads_all",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_repo_root() / "reports" / "ovis_layer15_semantics",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report: Report = json.loads(args.result.read_text(encoding="utf-8"))
    output_dir: Path = args.output
    figures = output_dir / "figures"
    tables = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("render quantitative figures")
    render_rankings(report, figures / "01_head_rankings.png")
    render_function_map(report, figures / "02_function_map.png")
    render_heatmap(report, figures / "03_head_metric_heatmap.png")

    print("load qualitative examples")
    dataset = OvisDataset(args.annotations)
    with zipfile.ZipFile(args.frames) as archive:
        members = frozenset(archive.namelist())
        examples = tuple(
            _load_qualitative_example(
                dataset,
                archive,
                members,
                args.feature_root,
                spec,
                heads=(1, 7, 8),
            )
            for spec in QUALITATIVE_SPECS
        )
    render_qualitative(examples, figures / "04_qualitative_attention.png")

    _write_csv(report, tables / "head_metrics.csv")
    _write_markdown(report, output_dir)
    _write_html(report, output_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
