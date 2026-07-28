"""Canonical data exchanged between dataset adapters and the tracking system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor


class VideoSource(Protocol):
    """Lazily provide RGB video frames on the CPU."""

    @property
    def num_frames(self) -> int: ...

    @property
    def frame_size(self) -> tuple[int, int]: ...

    def read(self, start: int = 0, stop: int | None = None) -> Tensor:
        """Return contiguous RGB uint8 frames shaped ``[T, C, H, W]``."""

        ...


@dataclass(frozen=True, slots=True)
class TrackAnnotations:
    """Resolution-independent point trajectories for one video."""

    tracks: Tensor
    visibility: Tensor

    def __post_init__(self) -> None:
        tracks = self.tracks.detach().to(device="cpu", dtype=torch.float32).contiguous()
        visibility = (
            self.visibility.detach().to(device="cpu", dtype=torch.bool).contiguous()
        )
        if tracks.ndim != 3 or tracks.shape[-1] != 2:
            raise ValueError("tracks must have shape [track, frame, xy]")
        if visibility.ndim != 2:
            raise ValueError("visibility must have shape [track, frame]")
        if visibility.shape != tracks.shape[:2]:
            raise ValueError("visibility must match the track and frame dimensions")
        object.__setattr__(self, "tracks", tracks)
        object.__setattr__(self, "visibility", visibility)


@dataclass(frozen=True, slots=True)
class AnnotatedVideo:
    """One dataset video and all of its trajectory annotations."""

    video_id: str
    video: VideoSource
    annotations: TrackAnnotations

    def __post_init__(self) -> None:
        if not self.video_id:
            raise ValueError("video_id must not be empty")
        if self.video.num_frames != self.annotations.tracks.shape[1]:
            raise ValueError(
                "video and annotations must have the same number of frames"
            )


class TrackingDataset(Protocol):
    """Dataset port providing videos and ground truth to HeFT."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> AnnotatedVideo: ...
