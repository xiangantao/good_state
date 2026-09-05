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
    workers_per_gpu: int = 1,
    cache_dir: str | Path | None = None,
    prompt_embeddings_path: str | Path | None = None,
) -> ExtractionResult:
    """Extract one video's chunks across the selected GPUs."""

    with FeatureExtractionPool(
        model=model,
        config=config,
        gpu_ids=gpu_ids,
        workers_per_gpu=workers_per_gpu,
        cache_dir=cache_dir,
        prompt_embeddings_path=prompt_embeddings_path,
    ) as pool:
        return pool.map([task])[0]


def extract_features(
    tasks: Iterable[ExtractionTask],
    *,
    model: ModelConfig,
    config: ExtractionConfig,
    gpu_ids: Sequence[int],
    workers_per_gpu: int = 1,
    cache_dir: str | Path | None = None,
    prompt_embeddings_path: str | Path | None = None,
) -> tuple[ExtractionResult, ...]:
    """Extract multiple videos using persistent workers on selected GPUs."""

    with FeatureExtractionPool(
        model=model,
        config=config,
        gpu_ids=gpu_ids,
        workers_per_gpu=workers_per_gpu,
        cache_dir=cache_dir,
        prompt_embeddings_path=prompt_embeddings_path,
    ) as pool:
        return pool.map(tasks)
