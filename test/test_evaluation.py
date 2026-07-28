"""Tests for dataset-backed tracking evaluation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from heft.data import AnnotatedVideo, ArrayVideoSource, TrackAnnotations
from heft.evaluation import EvaluationConfig, evaluate_dataset
from heft.tracking import FeatureSelection, TrackingResult


class MemoryDataset:
    def __init__(self, samples: tuple[AnnotatedVideo, ...]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> AnnotatedVideo:
        return self.samples[index]


def _sample(
    video_id: str,
    tracks: torch.Tensor,
    visibility: torch.Tensor,
) -> AnnotatedVideo:
    frames = np.zeros((tracks.shape[1], 4, 5, 3), dtype=np.uint8)
    return AnnotatedVideo(
        video_id=video_id,
        video=ArrayVideoSource(frames),
        annotations=TrackAnnotations(tracks=tracks, visibility=visibility),
    )


def _result(
    sample: AnnotatedVideo,
    *,
    visibility: torch.Tensor | None = None,
    frame_size: tuple[int, int] = (11, 21),
) -> TrackingResult:
    selected = sample.annotations.visibility[:, 0]
    normalized_tracks = sample.annotations.tracks[selected]
    height, width = frame_size
    scale = torch.tensor((width - 1, height - 1), dtype=torch.float32)
    tracks = normalized_tracks * scale
    query_points = torch.cat(
        (
            tracks[:, 0],
            torch.zeros((tracks.shape[0], 1), dtype=torch.float32),
        ),
        dim=1,
    )
    return TrackingResult(
        video_id=sample.video_id,
        frame_size=frame_size,
        query_points=query_points,
        tracks=tracks,
        visibility=(
            sample.annotations.visibility[selected]
            if visibility is None
            else visibility
        ),
        selection=FeatureSelection(layer=0),
    )


def test_evaluation_loads_gt_from_dataset_and_selects_first_frame_tracks() -> None:
    sample = _sample(
        "scene",
        tracks=torch.tensor(
            [
                [[0.25, 0.50], [0.50, 0.75]],
                [[0.10, 0.20], [0.20, 0.30]],
            ]
        ),
        visibility=torch.tensor(
            [
                [True, True],
                [False, True],
            ]
        ),
    )

    report = evaluate_dataset((_result(sample),), MemoryDataset((sample,)))

    assert report.videos[0].video_id == "scene"
    assert report.aggregate.occlusion_accuracy == 1.0
    assert report.aggregate.average_pts_within_thresh == 1.0
    assert report.aggregate.average_jaccard == 1.0


def test_evaluation_matches_video_ids_and_supports_macro_or_micro_average() -> None:
    first = _sample(
        "first",
        tracks=torch.tensor([[[0.0, 0.0], [0.0, 0.0]]]),
        visibility=torch.tensor([[True, False]]),
    )
    second = _sample(
        "second",
        tracks=torch.tensor(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, 1.0], [1.0, 1.0]],
            ]
        ),
        visibility=torch.ones((2, 2), dtype=torch.bool),
    )
    first_result = _result(first, visibility=torch.tensor([[True, True]]))
    second_result = _result(second)
    dataset = MemoryDataset((second, first))

    macro = evaluate_dataset((first_result, second_result), dataset)
    micro = evaluate_dataset(
        (first_result, second_result),
        dataset,
        config=EvaluationConfig(aggregation="micro"),
    )

    assert [item.video_id for item in macro.videos] == ["first", "second"]
    assert macro.aggregate.occlusion_accuracy == 0.75
    assert micro.aggregate.occlusion_accuracy == pytest.approx(5 / 6)
    assert macro.videos[0].metrics.jaccard[1.0] == 0.5


def test_evaluation_rejects_unaligned_queries_and_frame_counts() -> None:
    sample = _sample(
        "scene",
        tracks=torch.tensor([[[0.25, 0.50], [0.50, 0.75]]]),
        visibility=torch.ones((1, 2), dtype=torch.bool),
    )
    result = _result(sample)
    shifted_queries = result.query_points.clone()
    shifted_queries[0, 0] += 1

    with pytest.raises(ValueError, match="query points"):
        evaluate_dataset(
            (replace(result, query_points=shifted_queries),),
            MemoryDataset((sample,)),
        )

    with pytest.raises(ValueError, match="identical shapes"):
        evaluate_dataset(
            (replace(result, tracks=result.tracks[:, :1]),),
            MemoryDataset((sample,)),
        )


def test_evaluation_crops_gt_to_discarded_feature_tail() -> None:
    sample = _sample(
        "scene",
        tracks=torch.tensor([[[0.25, 0.50], [0.50, 0.75], [0.75, 1.00]]]),
        visibility=torch.ones((1, 3), dtype=torch.bool),
    )
    result = _result(sample)
    result = replace(
        result,
        tracks=result.tracks[:, :2],
        visibility=result.visibility[:, :2],
    )

    report = evaluate_dataset((result,), MemoryDataset((sample,)))

    assert report.aggregate.occlusion_accuracy == 1.0
    assert report.aggregate.average_pts_within_thresh == 1.0
