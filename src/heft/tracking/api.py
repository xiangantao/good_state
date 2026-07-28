"""Convenience Python APIs for single- and multi-GPU feature tracking."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .config import TrackingConfig, TrackingResult, TrackingTask
from .pool import TrackingPool


def track_feature(
    task: TrackingTask,
    *,
    config: TrackingConfig | None = None,
    gpu_ids: Sequence[int],
) -> TrackingResult:
    """Track one video on the first dynamically available selected GPU."""

    with TrackingPool(config=config, gpu_ids=gpu_ids) as pool:
        return pool.map([task])[0]


def track_features(
    tasks: Iterable[TrackingTask],
    *,
    config: TrackingConfig | None = None,
    gpu_ids: Sequence[int],
) -> tuple[TrackingResult, ...]:
    """Track multiple videos with one concurrent task per selected GPU."""

    with TrackingPool(config=config, gpu_ids=gpu_ids) as pool:
        return pool.map(tasks)
