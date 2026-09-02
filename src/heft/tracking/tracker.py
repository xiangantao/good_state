"""Forward/backward-consistency point tracking over extracted attention features."""

from __future__ import annotations

import os

import torch
from torch import Tensor

from heft.attn_hook import FeatureKind

from .config import (
    FeatureSelection,
    TrackingConfig,
    TrackingResult,
)
from .features import FeatureVideo, FrameRange
from .ops import DescriptorState, TrackingOps
from .reader import FeatureReader


class FeatureTracker:
    """Run the original tracking algorithm on one explicitly selected device."""

    def __init__(
        self,
        *,
        config: TrackingConfig | None = None,
        device: str | torch.device = "cuda:0",
    ) -> None:
        self.config = TrackingConfig() if config is None else config
        self.device = torch.device(device)

    @torch.inference_mode()
    def track(
        self,
        features: FeatureVideo | str | os.PathLike[str],
        query_points: Tensor,
        selection: FeatureSelection,
        *,
        video_id: str | None = None,
    ) -> TrackingResult:
        """Track frame-zero query points and return CPU tensors without side effects."""

        video = (
            features
            if isinstance(features, FeatureVideo)
            else FeatureVideo.open(features)
        )
        points = query_points.to(device=self.device, dtype=torch.float32).contiguous()
        self._validate_inputs(video, points, selection)
        tracks = torch.zeros(
            (points.shape[0], video.num_frames, 2),
            device=self.device,
            dtype=torch.float32,
        )
        visibility = torch.zeros(
            (points.shape[0], video.num_frames),
            device=self.device,
            dtype=torch.bool,
        )
        if points.shape[0] == 0:
            return self._result(video, points, tracks, visibility, selection, video_id)

        tracks[:, 0] = points[:, :2]
        visibility[:, 0] = True
        reader = FeatureReader(video, device=self.device)
        ops = TrackingOps(
            frame_size=video.frame_size,
            config=self.config,
            device=self.device,
        )
        required_kinds = self._required_kinds()
        updated_descriptors: DescriptorState | None = None

        for chunk, volumes in reader.prefetch_chunks(
            video.chunks,
            selection=selection,
            kinds=required_kinds,
            frequency_range=self.config.frequency_range,
        ):
            local_start, local_frames = _local_window(chunk)
            volumes = {
                kind: self._window_volume(volume, chunk, local_frames, ops)
                for kind, volume in volumes.items()
            }
            local_tracks = tracks[:, local_start : chunk.stop]
            local_visibility = visibility[:, local_start : chunk.stop]

            if updated_descriptors is None:
                query_frame = ops.frame(volumes[self.config.query_feature], 0)
                updated_descriptors = ops.make_descriptors(query_frame, points[:, :2])

            for frame_index in range(1, local_frames):
                target_frame = ops.frame(
                    volumes[self.config.target_feature], frame_index
                )
                predicted = ops.predict(
                    updated_descriptors,
                    target_frame,
                    previous_positions=local_tracks[:, frame_index - 1],
                    previous_visibility=local_visibility[:, frame_index - 1],
                    apply_search_mask=selection.head is not None,
                    softmax_before_resize=True,
                )
                query_frame = (
                    target_frame
                    if self.config.query_feature is self.config.target_feature
                    else ops.frame(volumes[self.config.query_feature], frame_index)
                )
                local_visibility[:, frame_index] = True
                current_visibility = self._visibility(
                    predicted,
                    frame_index=frame_index,
                    query_frame=query_frame,
                    volumes=volumes,
                    tracks=local_tracks,
                    visibility=local_visibility,
                    ops=ops,
                )
                local_tracks[:, frame_index] = predicted
                local_visibility[:, frame_index] = current_visibility

                update_kind = self._active_update_feature
                if update_kind is not None:
                    if update_kind is self.config.target_feature:
                        update_frame = target_frame
                    elif update_kind is self.config.query_feature:
                        update_frame = query_frame
                    else:
                        update_frame = ops.frame(volumes[update_kind], frame_index)
                    ops.update_descriptor(
                        updated_descriptors,
                        update_frame,
                        predicted,
                        current_visibility,
                    )

        return self._result(video, points, tracks, visibility, selection, video_id)

    def _window_volume(
        self,
        volume: Tensor,
        chunk: FrameRange,
        local_frames: int,
        ops: TrackingOps,
    ) -> Tensor:
        """Lay a chunk's feature volume onto its local window without shifting it.

        Chunks after the first extend one frame backwards so the previous chunk's
        result can seed the search, but their volume only covers
        ``[chunk.start, chunk.stop)``. Resampling it across the longer window
        would slide every frame's features towards later source frames, so the
        volume is resampled onto its own frames and the seed slot is filled with
        a copy of the first one.
        """

        if not self.config.align_chunk_features:
            return ops.interpolate_time(volume, local_frames)
        volume = ops.interpolate_time(volume, chunk.stop - chunk.start)
        if volume.shape[0] == local_frames:
            return volume
        return torch.cat((volume[:1], volume))

    def _visibility(
        self,
        target_points: Tensor,
        *,
        frame_index: int,
        query_frame: Tensor,
        volumes: dict[FeatureKind, Tensor],
        tracks: Tensor,
        visibility: Tensor,
        ops: TrackingOps,
    ) -> Tensor:
        backward_descriptors = ops.make_descriptors(query_frame, target_points)
        active_indices = torch.arange(
            target_points.shape[0], device=self.device, dtype=torch.long
        )
        backward_positions = target_points.clone()
        squared_error = torch.zeros(
            target_points.shape[0], device=self.device, dtype=torch.float32
        )
        visible_count = (
            visibility[:, :frame_index].sum(dim=1, dtype=torch.float32).clamp_min_(1.0)
        )
        failure_limit = visible_count * self.config.visibility_threshold**2
        update_kind = self._active_update_feature

        for previous_index in range(frame_index - 1, -1, -1):
            if active_indices.numel() == 0:
                break
            target_frame = ops.frame(
                volumes[self.config.target_feature], previous_index
            )
            backward_positions = ops.predict(
                backward_descriptors,
                target_frame,
                previous_positions=backward_positions,
                previous_visibility=visibility[active_indices, previous_index + 1],
                apply_search_mask=True,
                softmax_before_resize=False,
            )
            valid = visibility[active_indices, previous_index]
            difference = backward_positions - tracks[active_indices, previous_index]
            squared_error.add_(difference.square().sum(dim=1) * valid)

            keep = squared_error <= failure_limit[active_indices]
            kept = keep.nonzero(as_tuple=False).flatten()
            active_indices = active_indices.index_select(0, kept)
            backward_positions = backward_positions.index_select(0, kept)
            backward_descriptors = backward_descriptors.index_select(kept)
            squared_error = squared_error.index_select(0, kept)
            valid = valid.index_select(0, kept)

            if update_kind is not None and active_indices.numel() > 0:
                update_frame = (
                    target_frame
                    if update_kind is self.config.target_feature
                    else ops.frame(volumes[update_kind], previous_index)
                )
                ops.update_descriptor(
                    backward_descriptors,
                    update_frame,
                    backward_positions,
                    valid,
                )

        result = torch.zeros(
            target_points.shape[0], device=self.device, dtype=torch.bool
        )
        result[active_indices] = True
        return result

    @property
    def _active_update_feature(self) -> FeatureKind | None:
        if self.config.feature_ema_alpha == 0.0:
            return None
        return self.config.update_feature

    def _required_kinds(self) -> tuple[FeatureKind, ...]:
        kinds = [self.config.query_feature, self.config.target_feature]
        update_kind = self._active_update_feature
        if update_kind is not None:
            kinds.append(update_kind)
        return tuple(dict.fromkeys(kinds))

    def _validate_inputs(
        self,
        video: FeatureVideo,
        points: Tensor,
        selection: FeatureSelection,
    ) -> None:
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("query_points must have shape [point, 3]")
        if selection.layer not in video.layers:
            raise ValueError(f"layer {selection.layer} was not captured")
        if selection.head is not None and (
            video.heads is not None and selection.head not in video.heads
        ):
            raise ValueError(f"head {selection.head} was not captured")
        if points.numel() == 0:
            return
        if not torch.equal(points[:, 2], torch.zeros_like(points[:, 2])):
            raise ValueError(
                "the current tracking algorithm requires frame-zero queries"
            )
        if (
            points[:, 0].amin() < 0
            or points[:, 0].amax() > video.frame_size[1] - 1
            or points[:, 1].amin() < 0
            or points[:, 1].amax() > video.frame_size[0] - 1
        ):
            raise ValueError("query point coordinates are outside the video frame")

    @staticmethod
    def _result(
        video: FeatureVideo,
        points: Tensor,
        tracks: Tensor,
        visibility: Tensor,
        selection: FeatureSelection,
        video_id: str | None,
    ) -> TrackingResult:
        return TrackingResult(
            video_id=video.name if video_id is None else video_id,
            frame_size=video.frame_size,
            query_points=points.cpu(),
            tracks=tracks.cpu(),
            visibility=visibility.cpu(),
            selection=selection,
        )


def _local_window(chunk: FrameRange) -> tuple[int, int]:
    start = chunk.start if chunk.index == 0 else chunk.start - 1
    return start, chunk.stop - start
