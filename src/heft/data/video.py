"""Lazy video sources used by dataset adapters."""

from __future__ import annotations

import io
from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor


class ArrayVideoSource:
    """Expose an in-memory ``[T, H, W, C]`` array without copying it eagerly."""

    def __init__(self, frames: np.ndarray | Tensor) -> None:
        tensor = torch.from_numpy(frames) if isinstance(frames, np.ndarray) else frames
        if tensor.ndim != 4 or tensor.shape[-1] != 3:
            raise ValueError("array video must have shape [frame, height, width, 3]")
        if tensor.dtype is not torch.uint8:
            raise ValueError("array video must use uint8 RGB frames")
        if tensor.shape[0] == 0:
            raise ValueError("video must contain at least one frame")
        self._frames = tensor.permute(0, 3, 1, 2)

    @property
    def num_frames(self) -> int:
        return self._frames.shape[0]

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._frames.shape[2], self._frames.shape[3]

    def read(self, start: int = 0, stop: int | None = None) -> Tensor:
        resolved_stop = self.num_frames if stop is None else stop
        _validate_frame_range(start, resolved_stop, self.num_frames)
        return self._frames[start:resolved_stop].contiguous()


class JpegVideoSource:
    """Decode only the requested encoded frames."""

    def __init__(self, frames: Sequence[bytes]) -> None:
        if len(frames) == 0:
            raise ValueError("video must contain at least one frame")
        self._frames = frames
        self._first_frame = _decode_image(frames[0])
        self._frame_size = self._first_frame.shape[1], self._first_frame.shape[2]

    @property
    def num_frames(self) -> int:
        return len(self._frames)

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._frame_size

    def read(self, start: int = 0, stop: int | None = None) -> Tensor:
        resolved_stop = self.num_frames if stop is None else stop
        _validate_frame_range(start, resolved_stop, self.num_frames)
        frames = [
            self._first_frame if index == 0 else _decode_image(self._frames[index])
            for index in range(start, resolved_stop)
        ]
        if not frames:
            height, width = self.frame_size
            return torch.empty((0, 3, height, width), dtype=torch.uint8)
        return torch.stack(frames)


def _decode_image(encoded: bytes) -> Tensor:
    with Image.open(io.BytesIO(encoded)) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _validate_frame_range(start: int, stop: int, num_frames: int) -> None:
    if start < 0 or stop < start or stop > num_frames:
        raise IndexError(
            f"invalid frame range [{start}, {stop}) for {num_frames} frames"
        )
