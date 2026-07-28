"""Evaluation result value objects."""

from __future__ import annotations

from dataclasses import dataclass

from .config import EvaluationConfig


@dataclass(frozen=True, slots=True)
class MetricValues:
    """TAP-Vid-style metrics for one video or an aggregate."""

    occlusion_accuracy: float
    pts_within: dict[float, float]
    jaccard: dict[float, float]
    average_pts_within_thresh: float
    average_jaccard: float


@dataclass(frozen=True, slots=True)
class VideoMetrics:
    """Metrics associated with one dataset video."""

    video_id: str
    metrics: MetricValues


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Aggregate metrics and their per-video components."""

    aggregate: MetricValues
    videos: tuple[VideoMetrics, ...]
    config: EvaluationConfig
