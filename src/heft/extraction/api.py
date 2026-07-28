"""Convenience Python APIs for feature extraction."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from .config import ExtractionConfig, ExtractionResult, ExtractionTask, ModelConfig
from .pool import FeatureExtractionPool


def extract_feature(
    task: ExtractionTask,
    *,
    model: ModelConfig,
    config: ExtractionConfig,
    gpu_ids: Sequence[int],
    cache_dir: str | Path | None = None,
) -> ExtractionResult:
    """Extract one video's chunks across the selected GPUs."""

    with FeatureExtractionPool(
        model=model,
        config=config,
        gpu_ids=gpu_ids,
        cache_dir=cache_dir,
    ) as pool:
        return pool.map([task])[0]


def extract_features(
    tasks: Iterable[ExtractionTask],
    *,
    model: ModelConfig,
    config: ExtractionConfig,
    gpu_ids: Sequence[int],
    cache_dir: str | Path | None = None,
) -> tuple[ExtractionResult, ...]:
    """Extract multiple videos using one persistent worker per selected GPU."""

    with FeatureExtractionPool(
        model=model,
        config=config,
        gpu_ids=gpu_ids,
        cache_dir=cache_dir,
    ) as pool:
        return pool.map(tasks)
