"""Persistent Diffusers executor owned by one GPU worker process."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from heft.attn_hook import (
    AttentionFeatureCapture,
    CaptureContext,
    CapturedFeature,
    FeatureKind,
    FeatureRecorder,
    FeatureSinkRouter,
    SafetensorsFeatureStorage,
)

from .config import ChunkExecutionMetadata, ChunkJob, ExtractionConfig, ModelConfig


@dataclass(frozen=True, slots=True)
class DiffusersChunkExecutorFactory:
    """Picklable factory used to initialize one persistent GPU executor."""

    model: ModelConfig
    config: ExtractionConfig
    cache_dir: Path | None = None

    def __call__(self, gpu_id: int) -> DiffusersChunkExecutor:
        return DiffusersChunkExecutor(
            gpu_id=gpu_id,
            model=self.model,
            config=self.config,
            cache_dir=self.cache_dir,
        )


class DiffusersChunkExecutor:
    """Load a pipeline once and extract independently scheduled chunks."""

    def __init__(
        self,
        *,
        gpu_id: int,
        model: ModelConfig,
        config: ExtractionConfig,
        cache_dir: str | os.PathLike[str] | None,
    ) -> None:
        self._model = model
        self._config = config
        self._device = torch.device("cuda", gpu_id)
        torch.cuda.set_device(self._device)
        cv2.setNumThreads(1)

        module = importlib.import_module(model.pipeline_module)
        pipeline_class = getattr(module, model.pipeline_class)
        load_kwargs: dict[str, Any] = {"torch_dtype": model.dtype}
        if cache_dir is not None:
            load_kwargs["cache_dir"] = str(cache_dir)
        self._pipeline = pipeline_class.from_pretrained(model.model_id, **load_kwargs)
        self._pipeline.to(self._device)

        transformer = self._pipeline.transformer
        self._blocks = getattr(transformer, model.blocks_attribute)
        first_layer = min(config.capture.layers)
        attention = self._blocks[first_layer].attn1
        self._heads = (
            tuple(range(attention.heads))
            if config.capture.heads is None
            else tuple(sorted(config.capture.heads))
        )

        self._sink = FeatureSinkRouter()
        capture = AttentionFeatureCapture(
            spec=config.capture,
            context=CaptureContext(start_step=model.start_step),
            sink=self._sink,
        )
        self._session = capture.attach(self._blocks, processor=model.processor)

    def __call__(self, job: ChunkJob) -> ChunkExecutionMetadata:
        storage = SafetensorsFeatureStorage(
            job.output_dir,
            expected_heads=self._heads,
            overwrite=self._config.overwrite,
        )
        with (
            storage,
            FeatureRecorder(
                sink=storage,
                max_pending=self._config.max_pending_features,
            ) as recorder,
        ):
            sink = recorder
            if job.padding_frames:
                feature_h = (
                    self._model.resolution[0] // self._model.feature_spatial_stride[0]
                )
                feature_w = (
                    self._model.resolution[1] // self._model.feature_spatial_stride[1]
                )
                actual_tokens = job.num_frames * feature_h * feature_w
                sink = lambda feature: recorder(_trim_feature(feature, actual_tokens))
            self._sink.bind(sink)
            self._session.begin_chunk(
                chunk=job.chunk,
                start_step=self._model.start_step,
            )
            try:
                generator = torch.manual_seed(job.seed)
                height, width = self._model.resolution
                input_video = _prepare_input_video(
                    job.input_video,
                    self._model.resolution,
                )
                self._pipeline(
                    prompt=job.prompt,
                    height=height,
                    width=width,
                    num_frames=job.pipeline_num_frames,
                    num_inference_steps=self._model.num_inference_steps,
                    guidance_scale=self._model.guidance_scale,
                    generator=generator,
                    input_video=input_video,
                    start_step=self._model.start_step,
                    output_type="latent",
                )
            finally:
                self._session.end_chunk()
                self._sink.unbind()
        captured_kind = next(iter(self._config.capture.features))
        shape = recorder.feature_shape(captured_kind)
        if len(shape) != 3:
            raise ValueError(
                f"expected captured per-head features to be rank 3, got {shape}"
            )
        return ChunkExecutionMetadata(heads=self._heads, feature_tokens=shape[1])


def _prepare_input_video(
    video: torch.Tensor | None,
    size: tuple[int, int],
) -> torch.Tensor | None:
    if video is None:
        return video
    height, width = size
    frames = video.permute(0, 2, 3, 1).numpy()
    if tuple(video.shape[-2:]) != size:
        frames = np.stack(
            [
                cv2.resize(
                    frame,
                    (width, height),
                    interpolation=cv2.INTER_LANCZOS4,
                )
                for frame in frames
            ]
        )
    prepared = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
    if prepared.dtype is torch.uint8:
        prepared = prepared.to(dtype=torch.float32).div_(255.0)
    return prepared


def _trim_feature(feature: CapturedFeature, tokens: int) -> CapturedFeature:
    tensor = (
        feature.tensor[:, :tokens, :tokens]
        if feature.kind is FeatureKind.ATTENTION_MAP
        else feature.tensor[:, :tokens]
    )
    return replace(feature, tensor=tensor)
