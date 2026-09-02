"""Tests for feature reading, tracking, and dynamic GPU routing."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from heft.attn_hook import FeatureKind
from heft.tracking import (
    FeatureSelection,
    FeatureTracker,
    FeatureVideo,
    FrameRange,
    RopePairing,
    RopeSpec,
    TrackingConfig,
    TrackingPool,
    TrackingResult,
    TrackingTask,
)
from heft.tracking.ops import DescriptorState, TrackingOps
from heft.tracking.reader import FeatureReader
from heft.tracking.tracker import _local_window


@dataclass(frozen=True, slots=True)
class FakeTrackingExecutor:
    gpu_id: int
    delays: tuple[float, ...]

    def __call__(self, task: TrackingTask) -> TrackingResult:
        assert task.video_id is not None
        time.sleep(self.delays[int(task.video_id)])
        return TrackingResult(
            video_id=task.video_id,
            frame_size=(1, 1),
            query_points=task.query_points,
            tracks=torch.zeros(1, 1, 2),
            visibility=torch.ones(1, 1, dtype=torch.bool),
            selection=task.selection,
            gpu_id=self.gpu_id,
        )


@dataclass(frozen=True, slots=True)
class FakeTrackingExecutorFactory:
    delays: tuple[float, ...]

    def __call__(self, gpu_id: int) -> FakeTrackingExecutor:
        return FakeTrackingExecutor(gpu_id=gpu_id, delays=self.delays)


def _write_group(
    video: FeatureVideo,
    kind: FeatureKind,
    tensor: torch.Tensor,
    *,
    head: int = 0,
    chunk: int = 0,
) -> None:
    path = (
        video.root
        / f"chunk_{chunk:03d}"
        / f"step_{video.step:03d}"
        / f"layer_{video.layers[0]:03d}"
        / f"{kind.value}.safetensors"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({f"head_{head:03d}": tensor.contiguous()}, path)


def _legacy_predict(
    ops: TrackingOps,
    descriptors: torch.Tensor,
    target: torch.Tensor,
    positions: torch.Tensor,
    visibility: torch.Tensor,
    *,
    apply_search_mask: bool,
    softmax_before_resize: bool,
) -> torch.Tensor:
    source_h, source_w = target.shape[-2:]
    correlation = F.normalize(descriptors, dim=1) @ F.normalize(
        target.flatten(1), dim=0
    )
    if softmax_before_resize:
        probability = correlation.softmax(dim=1).reshape(-1, source_h, source_w)
        if (source_h, source_w) != (ops.frame_h, ops.frame_w):
            probability = F.interpolate(
                probability[:, None],
                size=(ops.frame_h, ops.frame_w),
                mode="bilinear",
                align_corners=False,
            )[:, 0]
    else:
        correlation = correlation.reshape(-1, source_h, source_w)
        if (source_h, source_w) != (ops.frame_h, ops.frame_w):
            correlation = F.interpolate(
                correlation[:, None],
                size=(ops.frame_h, ops.frame_w),
                mode="bilinear",
                align_corners=False,
            )[:, 0]
        probability = (
            correlation.flatten(1).softmax(dim=1).reshape(-1, ops.frame_h, ops.frame_w)
        )

    if apply_search_mask:
        x_distance = ops.x_coordinates[None, None, :] - positions[:, 0, None, None]
        y_distance = ops.y_coordinates[None, :, None] - positions[:, 1, None, None]
        search_mask = (
            x_distance.square() + y_distance.square() <= ops.config.search_radius**2
        )
        search_mask |= ~visibility[:, None, None]
        probability = probability.masked_fill(~search_mask, 0.0)

    flat_argmax = probability.flatten(1).argmax(dim=1)
    centers = torch.stack(
        (
            flat_argmax.remainder(ops.frame_w),
            torch.div(flat_argmax, ops.frame_w, rounding_mode="floor"),
        ),
        dim=1,
    ).float()
    x_distance = ops.x_coordinates[None, None, :] - centers[:, 0, None, None]
    y_distance = ops.y_coordinates[None, :, None] - centers[:, 1, None, None]
    local = probability.masked_fill(
        x_distance.square() + y_distance.square() > ops.config.argmax_radius**2,
        0.0,
    )
    denominator = local.sum(dim=(1, 2)).clamp_min(torch.finfo(torch.float32).tiny)
    x = local.sum(dim=1) @ ops.x_coordinates / denominator
    y = local.sum(dim=2) @ ops.y_coordinates / denominator
    return torch.stack((x, y), dim=1)


def test_cosmos_reader_keeps_split_half_rope_pairs(tmp_path: Path) -> None:
    video = FeatureVideo(
        root=tmp_path / "cosmos",
        name="cosmos",
        model="cosmos2",
        num_frames=1,
        frame_size=(1, 1),
        feature_size=(1, 1),
        chunks=(FrameRange(0, 0, 1, feature_frames=1),),
        step=4,
        layers=(0,),
        heads=(0,),
        feature_kinds=("query",),
        rope=RopeSpec(RopePairing.SPLIT_HALF, 2, 2, 2),
    )
    tensor = torch.arange(12, dtype=torch.float32).reshape(1, 1, 12)
    _write_group(video, FeatureKind.QUERY, tensor)

    reader = FeatureReader(video, device=torch.device("cpu"))
    loaded = reader.load_chunk(
        video.chunks[0],
        selection=FeatureSelection(layer=0, head=0),
        kinds=(FeatureKind.QUERY,),
        frequency_range=(0.0, 0.5),
    )[FeatureKind.QUERY]
    prefetched = dict(
        reader.prefetch_chunks(
            video.chunks,
            selection=FeatureSelection(layer=0, head=0),
            kinds=(FeatureKind.QUERY,),
            frequency_range=(0.0, 0.5),
        )
    )[video.chunks[0]][FeatureKind.QUERY]

    torch.testing.assert_close(
        loaded[:, :, 0, 0],
        torch.tensor([[0.0, 2.0, 4.0, 6.0, 8.0, 10.0]]),
    )
    assert torch.equal(prefetched, loaded)


def test_region_matching_matches_legacy_full_probability_map() -> None:
    torch.manual_seed(7)
    config = TrackingConfig(
        argmax_radius=2.2,
        search_radius=3.5,
        point_batch_size=2,
    )
    ops = TrackingOps(frame_size=(9, 11), config=config, device=torch.device("cpu"))
    target = torch.randn(6, 9, 11)
    descriptors = torch.randn(4, 6)
    positions = torch.tensor([[4.2, 5.1], [1.5, 2.5], [8.1, 7.4], [5.0, 1.0]])
    visibility = torch.tensor([True, False, True, True])

    expected = _legacy_predict(
        ops,
        descriptors,
        target,
        positions,
        visibility,
        apply_search_mask=True,
        softmax_before_resize=True,
    )
    actual = ops.predict(
        DescriptorState.from_values(descriptors),
        target,
        previous_positions=positions,
        previous_visibility=visibility,
        apply_search_mask=True,
        softmax_before_resize=True,
    )

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_region_kernel_matches_legacy_full_probability_map() -> None:
    torch.manual_seed(17)
    device = torch.device("cuda:0")
    config = TrackingConfig(argmax_radius=2.2, search_radius=3.5)
    ops = TrackingOps(frame_size=(9, 11), config=config, device=device)
    target = torch.randn(6, 9, 11, device=device)
    descriptors = torch.randn(4, 6, device=device)
    positions = torch.tensor(
        [[4.2, 5.1], [1.5, 2.5], [8.1, 7.4], [5.0, 1.0]], device=device
    )
    visibility = torch.ones(4, dtype=torch.bool, device=device)

    expected = _legacy_predict(
        ops,
        descriptors,
        target,
        positions,
        visibility,
        apply_search_mask=True,
        softmax_before_resize=True,
    )
    actual = ops.predict(
        DescriptorState.from_values(descriptors),
        target,
        previous_positions=positions,
        previous_visibility=visibility,
        apply_search_mask=True,
        softmax_before_resize=True,
    )

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_resized_matching_preserves_both_legacy_orders() -> None:
    torch.manual_seed(11)
    config = TrackingConfig(
        upsample_features=False,
        argmax_radius=2.0,
        search_radius=4.0,
    )
    ops = TrackingOps(frame_size=(9, 11), config=config, device=torch.device("cpu"))
    target = torch.randn(6, 4, 5)
    descriptors = torch.randn(3, 6)
    state = DescriptorState.from_values(descriptors)
    positions = torch.tensor([[4.0, 4.0], [1.0, 2.0], [9.0, 7.0]])
    visibility = torch.tensor([True, False, True])

    for softmax_before_resize in (True, False):
        expected = _legacy_predict(
            ops,
            descriptors,
            target,
            positions,
            visibility,
            apply_search_mask=True,
            softmax_before_resize=softmax_before_resize,
        )
        actual = ops.predict(
            state,
            target,
            previous_positions=positions,
            previous_visibility=visibility,
            apply_search_mask=True,
            softmax_before_resize=softmax_before_resize,
        )
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_descriptor_state_renormalizes_only_updated_rows() -> None:
    values = torch.tensor([[3.0, 4.0], [5.0, 12.0], [8.0, 15.0]])
    state = DescriptorState.from_values(values.clone())
    unchanged = state.normalized[1].clone()
    indices = torch.tensor([0, 2])
    replacements = torch.tensor([[6.0, 8.0], [7.0, 24.0]])

    state.update(indices, replacements, 0.25)

    expected_values = values.clone()
    expected_values[indices] = torch.lerp(values[indices], replacements, 0.25)
    torch.testing.assert_close(state.values, expected_values)
    torch.testing.assert_close(state.normalized, F.normalize(expected_values, dim=1))
    assert torch.equal(state.normalized[1], unchanged)


def test_tracker_follows_moving_feature_and_returns_visibility(tmp_path: Path) -> None:
    frames = 3
    height = width = 3
    channels = 6
    video = FeatureVideo(
        root=tmp_path / "moving",
        name="moving",
        model="fake",
        num_frames=frames,
        frame_size=(height, width),
        feature_size=(height, width),
        chunks=(FrameRange(0, 0, frames, feature_frames=frames),),
        step=2,
        layers=(0,),
        heads=(0,),
        feature_kinds=("query", "key"),
        rope=RopeSpec(RopePairing.ADJACENT, 1, 1, 1),
    )
    feature = torch.zeros(frames, height, width, channels)
    for frame in range(frames):
        feature[frame, 1, frame, 0] = 1.0
    stored = feature.reshape(1, frames * height * width, channels)
    _write_group(video, FeatureKind.QUERY, stored)
    _write_group(video, FeatureKind.KEY, stored)

    config = TrackingConfig(
        argmax_radius=0.1,
        search_radius=2.0,
        visibility_threshold=0.1,
        point_batch_size=1,
    )
    result = FeatureTracker(config=config, device="cpu").track(
        video,
        torch.tensor([[0.0, 1.0, 0.0]]),
        FeatureSelection(layer=0, head=0),
    )

    assert config.upsample_features is True
    torch.testing.assert_close(
        result.tracks,
        torch.tensor([[[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]]),
    )
    assert result.visibility.tolist() == [[True, True, True]]
    assert result.tracks.device.type == "cpu"


def test_tracker_handles_a_short_final_chunk(tmp_path: Path) -> None:
    height = width = 3
    channels = 6
    video = FeatureVideo(
        root=tmp_path / "short-tail",
        name="short-tail",
        model="fake",
        num_frames=5,
        frame_size=(height, width),
        feature_size=(height, width),
        chunks=(
            FrameRange(0, 0, 3, feature_frames=3),
            FrameRange(1, 3, 5, feature_frames=2),
        ),
        step=2,
        layers=(0,),
        heads=(0,),
        feature_kinds=("query", "key"),
        rope=RopeSpec(RopePairing.ADJACENT, 1, 1, 1),
    )
    for chunk, frames in enumerate((3, 2)):
        feature = torch.zeros(frames, height, width, channels)
        feature[:, 1, 1, 0] = 1.0
        stored = feature.reshape(1, frames * height * width, channels)
        _write_group(video, FeatureKind.QUERY, stored, chunk=chunk)
        _write_group(video, FeatureKind.KEY, stored, chunk=chunk)

    result = FeatureTracker(
        config=TrackingConfig(
            argmax_radius=0.1,
            search_radius=2.0,
            visibility_threshold=0.1,
        ),
        device="cpu",
    ).track(
        video,
        torch.tensor([[1.0, 1.0, 0.0]]),
        FeatureSelection(layer=0, head=0),
    )

    torch.testing.assert_close(
        result.tracks,
        torch.tensor([[[1.0, 1.0]] * 5]),
    )
    assert result.visibility.all()


def test_tracking_pool_routes_new_video_to_the_next_idle_gpu(tmp_path: Path) -> None:
    selection = FeatureSelection(layer=0, head=0)
    tasks = [
        TrackingTask(
            features=tmp_path / "unused",
            query_points=torch.tensor([[0.0, 0.0, 0.0]]),
            selection=selection,
            video_id=str(index),
        )
        for index in range(5)
    ]

    with TrackingPool(
        gpu_ids=(2, 6),
        executor_factory=FakeTrackingExecutorFactory(
            delays=(0.30, 0.02, 0.02, 0.02, 0.02)
        ),
    ) as pool:
        results = pool.map(tasks)

    assert [result.video_id for result in results] == ["0", "1", "2", "3", "4"]
    tasks_per_gpu = {
        gpu_id: sum(result.gpu_id == gpu_id for result in results) for gpu_id in (2, 6)
    }
    assert sorted(tasks_per_gpu.values()) == [1, 4]


def _window_volume(chunk: FrameRange, *, align: bool) -> torch.Tensor:
    """Run the tracker's own windowing on a volume whose frame t holds the value t."""

    tracker = FeatureTracker(
        config=TrackingConfig(align_chunk_features=align), device="cpu"
    )
    ops = TrackingOps(
        frame_size=(3, 3),
        config=tracker.config,
        device=torch.device("cpu"),
    )
    volume = (
        torch.arange(chunk.start, chunk.stop, dtype=torch.float32)
        .reshape(-1, 1, 1, 1)
        .expand(-1, 2, 3, 3)
        .contiguous()
    )
    _, local_frames = _local_window(chunk)
    return tracker._window_volume(volume, chunk, local_frames, ops)[:, 0, 0, 0]


def test_chunk_window_keeps_features_on_their_own_source_frames() -> None:
    """Local window slot ``i`` must carry the features of source frame ``start + i``."""

    chunk = FrameRange(index=1, start=5, stop=10, feature_frames=5)
    local_start, local_frames = _local_window(chunk)
    assert (local_start, local_frames) == (4, 6)

    windowed = _window_volume(chunk, align=True)

    # Slot 0 seeds the search from the previous chunk and has no features here,
    # so it holds a copy of the first one; every other slot names its own frame.
    assert windowed[0].item() == float(chunk.start)
    torch.testing.assert_close(
        windowed[1:],
        torch.arange(local_start + 1, chunk.stop, dtype=torch.float32),
    )


def test_first_chunk_window_is_untouched() -> None:
    chunk = FrameRange(index=0, start=0, stop=5, feature_frames=5)

    for align in (True, False):
        torch.testing.assert_close(
            _window_volume(chunk, align=align),
            torch.arange(0, 5, dtype=torch.float32),
        )


def test_disabling_alignment_restores_the_original_drift() -> None:
    """The legacy path stretched the volume, running features up to a frame ahead."""

    chunk = FrameRange(index=1, start=5, stop=10, feature_frames=5)
    local_start, _ = _local_window(chunk)

    windowed = _window_volume(chunk, align=False)

    drift = windowed - torch.arange(local_start, chunk.stop, dtype=torch.float32)
    # Slot 0 stands in for the previous chunk under either path. The legacy drift
    # over the remaining slots peaks right after the seam and decays to zero by
    # the end of the chunk; aligning removes it entirely.
    torch.testing.assert_close(
        drift, torch.tensor([1.0, 0.75, 0.5833333, 0.4166667, 0.25, 0.0])
    )
