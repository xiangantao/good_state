"""Tests for the unified annotated-video dataset interface."""

from __future__ import annotations

import io
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from heft.data import (
    ArrayVideoSource,
    JpegVideoSource,
    PointOdysseyDataset,
    TapVidDavisDataset,
    TapVidKineticsDataset,
    TapVidRGBStackDataset,
    TrackingDataset,
)


def _record(*, encoded: bool = False) -> dict[str, object]:
    video = np.arange(3 * 4 * 5 * 3, dtype=np.uint8).reshape(3, 4, 5, 3)
    if encoded:
        encoded_frames: list[bytes] = []
        for frame in video:
            buffer = io.BytesIO()
            Image.fromarray(frame).save(buffer, format="PNG")
            encoded_frames.append(buffer.getvalue())
        stored_video: object = encoded_frames
    else:
        stored_video = video
    return {
        "video": stored_video,
        "points": np.array(
            [
                [[0.0, 0.0], [0.2, 0.3], [0.4, 0.5]],
                [[0.5, 0.5], [0.6, 0.7], [1.0, 1.0]],
            ],
            dtype=np.float32,
        ),
        "occluded": np.array(
            [
                [False, False, True],
                [True, False, False],
            ]
        ),
    }


def _dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(value, file)


def test_array_video_source_reads_only_the_requested_frames() -> None:
    frames = np.arange(4 * 3 * 5 * 3, dtype=np.uint8).reshape(4, 3, 5, 3)
    source = ArrayVideoSource(frames)

    actual = source.read(1, 3)

    assert source.num_frames == 4
    assert source.frame_size == (3, 5)
    assert actual.shape == (2, 3, 3, 5)
    assert actual.dtype is torch.uint8
    assert actual.is_contiguous()
    torch.testing.assert_close(actual[0], torch.from_numpy(frames[1]).permute(2, 0, 1))


def test_jpeg_video_source_decodes_only_the_requested_frames() -> None:
    record = _record(encoded=True)
    encoded = record["video"]
    assert isinstance(encoded, list)
    source = JpegVideoSource(encoded)

    actual = source.read(1, 2)

    assert source.num_frames == 3
    assert source.frame_size == (4, 5)
    assert actual.shape == (1, 3, 4, 5)
    assert actual.dtype is torch.uint8


def test_all_pickle_datasets_return_the_same_annotated_video_contract(
    tmp_path: Path,
) -> None:
    record = _record()
    _dump(tmp_path / "davis" / "tapvid_davis.pkl", {"scene": record})
    _dump(tmp_path / "rgb" / "tapvid_rgb_stacking.pkl", [record])
    _dump(tmp_path / "kinetics" / "000.pkl", [_record(encoded=True)])
    _dump(tmp_path / "point" / "point_odyssey.pkl", {"clip": record})

    datasets: list[TrackingDataset] = [
        TapVidDavisDataset(tmp_path / "davis"),
        TapVidRGBStackDataset(tmp_path / "rgb"),
        TapVidKineticsDataset(tmp_path / "kinetics"),
        PointOdysseyDataset(tmp_path / "point"),
    ]

    assert [dataset[0].video_id for dataset in datasets] == [
        "scene",
        "0000",
        "0000",
        "clip",
    ]
    for dataset in datasets:
        sample = dataset[0]
        assert sample.annotations.tracks.shape == (2, 3, 2)
        assert sample.annotations.tracks.dtype is torch.float32
        assert sample.annotations.visibility.dtype is torch.bool
        assert sample.video.num_frames == 3
        assert sample.video.frame_size == (4, 5)
        assert sample.video.read(0, 1).shape == (1, 3, 4, 5)
        assert sample.annotations.visibility.tolist() == [
            [True, True, False],
            [False, True, True],
        ]


def test_dataset_does_not_apply_query_policy_or_mutate_normalized_tracks(
    tmp_path: Path,
) -> None:
    record = _record()
    _dump(tmp_path / "tapvid_davis.pkl", {"scene": record})
    dataset = TapVidDavisDataset(tmp_path)

    first = dataset[0]
    second = dataset[0]

    assert first.annotations.tracks.shape[0] == 2
    torch.testing.assert_close(first.annotations.tracks, second.annotations.tracks)
    assert first.annotations.tracks.max().item() == 1.0
