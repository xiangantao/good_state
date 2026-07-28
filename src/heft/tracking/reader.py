"""Read selected feature groups directly from safetensors storage."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from safetensors import safe_open
from torch import Tensor

from heft.attn_hook import FeatureKind

from .config import FeatureSelection
from .features import FeatureVideo, FrameRange


class FeatureReader:
    """Read only the layer, heads, and channels needed by one tracker."""

    def __init__(
        self,
        video: FeatureVideo,
        *,
        device: torch.device,
        pin_memory: bool = False,
    ) -> None:
        self.video = video
        self.device = device
        self.pin_memory = pin_memory

    def load_chunk(
        self,
        chunk: FrameRange,
        *,
        selection: FeatureSelection,
        kinds: Iterable[FeatureKind],
        frequency_range: tuple[float, float],
    ) -> dict[FeatureKind, Tensor]:
        """Load feature volumes as float32 ``[time, channel, height, width]``."""

        if selection.layer not in self.video.layers:
            raise ValueError(f"layer {selection.layer} was not captured")
        requested_kinds = tuple(dict.fromkeys(FeatureKind(kind) for kind in kinds))
        missing_kinds = [
            kind.value
            for kind in requested_kinds
            if kind.value not in self.video.feature_kinds
        ]
        if missing_kinds:
            raise ValueError(f"features were not captured: {missing_kinds}")

        channel_indices = self.video.rope.channel_indices(frequency_range)
        return {
            kind: self._load_group(
                chunk,
                selection=selection,
                kind=kind,
                channel_indices=channel_indices,
            )
            for kind in requested_kinds
        }

    def prefetch_chunks(
        self,
        chunks: Iterable[FrameRange],
        *,
        selection: FeatureSelection,
        kinds: Iterable[FeatureKind],
        frequency_range: tuple[float, float],
    ) -> Iterator[tuple[FrameRange, dict[FeatureKind, Tensor]]]:
        """Overlap reading and transfer of the next chunk with current tracking."""

        ordered_chunks = tuple(chunks)
        if not ordered_chunks:
            return
        requested_kinds = tuple(kinds)
        host_reader = FeatureReader(
            self.video,
            device=torch.device("cpu"),
            pin_memory=self.device.type == "cuda",
        )
        copy_stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda"
            else None
        )

        def load(chunk: FrameRange) -> dict[FeatureKind, Tensor]:
            volumes = host_reader.load_chunk(
                chunk,
                selection=selection,
                kinds=requested_kinds,
                frequency_range=frequency_range,
            )
            if copy_stream is None:
                return volumes
            with torch.cuda.device(self.device), torch.cuda.stream(copy_stream):
                transferred = {
                    kind: volume.to(self.device, non_blocking=True)
                    for kind, volume in volumes.items()
                }
            copy_stream.synchronize()
            return transferred

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(load, ordered_chunks[0])
            for index, chunk in enumerate(ordered_chunks):
                volumes = future.result()
                if index + 1 < len(ordered_chunks):
                    future = executor.submit(load, ordered_chunks[index + 1])
                yield chunk, volumes

    def _load_group(
        self,
        chunk: FrameRange,
        *,
        selection: FeatureSelection,
        kind: FeatureKind,
        channel_indices: tuple[int, ...],
    ) -> Tensor:
        path = self._path(chunk.index, selection.layer, kind)
        selected_heads = self._selected_heads(path, selection.head)
        feature_h, feature_w = self.video.feature_size
        expected_tokens = chunk.feature_frames * feature_h * feature_w
        selected_channels = len(channel_indices)
        output = torch.empty(
            (
                chunk.feature_frames,
                len(selected_heads) * selected_channels,
                feature_h,
                feature_w,
            ),
            device=self.device,
            dtype=torch.float32,
            pin_memory=self.pin_memory,
        )

        with safe_open(path, framework="pt", device="cpu") as file:
            for head_index, head in enumerate(selected_heads):
                tensor = file.get_tensor(f"head_{head:03d}")
                if tensor.ndim != 3 or tensor.shape[0] != 1:
                    raise ValueError(
                        f"invalid feature tensor shape in {path}: {tensor.shape}"
                    )
                if tensor.shape[1] != expected_tokens:
                    raise ValueError(
                        f"invalid token count in {path}: expected {expected_tokens}, "
                        f"got {tensor.shape[1]}"
                    )
                if tensor.shape[2] != self.video.rope.channels:
                    raise ValueError(
                        f"feature channels in {path} do not match the model RoPE layout"
                    )
                selected = (
                    tensor[0]
                    if selected_channels == self.video.rope.channels
                    else tensor[0, :, channel_indices]
                )
                volume = selected.reshape(
                    chunk.feature_frames, feature_h, feature_w, -1
                ).permute(0, 3, 1, 2)
                channel_start = head_index * selected_channels
                output[:, channel_start : channel_start + selected_channels].copy_(
                    volume
                )

        return output

    def _selected_heads(self, path: Path, head: int | None) -> tuple[int, ...]:
        if head is not None:
            if self.video.heads is not None and head not in self.video.heads:
                raise ValueError(f"head {head} was not captured")
            return (head,)
        if self.video.heads is not None:
            return self.video.heads
        with safe_open(path, framework="pt", device="cpu") as file:
            return tuple(sorted(_head_from_key(key) for key in file))

    def _path(self, chunk: int, layer: int, kind: FeatureKind) -> Path:
        return (
            self.video.root
            / f"chunk_{chunk:03d}"
            / f"step_{self.video.step:03d}"
            / f"layer_{layer:03d}"
            / f"{kind.value}.safetensors"
        )


def _head_from_key(key: str) -> int:
    prefix = "head_"
    if not key.startswith(prefix) or not key[len(prefix) :].isdigit():
        raise ValueError(f"invalid feature tensor key: {key!r}")
    return int(key[len(prefix) :])
