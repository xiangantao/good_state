"""Adapters for the annotated video datasets used by HeFT."""

from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from .types import AnnotatedVideo, TrackAnnotations
from .video import ArrayVideoSource, JpegVideoSource


class _RecordDataset:
    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        video_ids: Sequence[str],
        *,
        encoded_video: bool,
    ) -> None:
        if len(records) != len(video_ids):
            raise ValueError("records and video_ids must have the same length")
        self._records = records
        self._video_ids = tuple(video_ids)
        self._encoded_video = encoded_video

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> AnnotatedVideo:
        record = self._records[index]
        points = torch.as_tensor(np.asarray(record["points"]), dtype=torch.float32)
        occluded = torch.as_tensor(np.asarray(record["occluded"]), dtype=torch.bool)
        annotations = TrackAnnotations(tracks=points, visibility=~occluded)
        raw_video = record["video"]
        video = (
            JpegVideoSource(cast(Sequence[bytes], raw_video))
            if self._encoded_video
            else ArrayVideoSource(cast(np.ndarray, raw_video))
        )
        return AnnotatedVideo(
            video_id=self._video_ids[index],
            video=video,
            annotations=annotations,
        )


class TapVidDavisDataset(_RecordDataset):
    """TAP-Vid DAVIS adapter."""

    def __init__(self, root: str | Path) -> None:
        data = _load_pickle(Path(root) / "tapvid_davis.pkl")
        if not isinstance(data, Mapping):
            raise TypeError("tapvid_davis.pkl must contain a mapping")
        raw_keys = sorted(data, key=str)
        records = [cast(Mapping[str, Any], data[key]) for key in raw_keys]
        super().__init__(records, [str(key) for key in raw_keys], encoded_video=False)


class TapVidRGBStackDataset(_RecordDataset):
    """TAP-Vid RGB Stacking adapter."""

    def __init__(self, root: str | Path) -> None:
        records = _as_record_sequence(
            _load_pickle(Path(root) / "tapvid_rgb_stacking.pkl"),
            source="tapvid_rgb_stacking.pkl",
        )
        video_ids = [f"{index:04d}" for index in range(len(records))]
        super().__init__(records, video_ids, encoded_video=False)


class TapVidKineticsDataset(_RecordDataset):
    """TAP-Vid Kinetics adapter without experiment-specific filtering."""

    def __init__(self, root: str | Path) -> None:
        records: list[Mapping[str, Any]] = []
        for path in sorted(Path(root).rglob("*.pkl")):
            records.extend(_as_record_sequence(_load_pickle(path), source=str(path)))
        video_ids = [f"{index:04d}" for index in range(len(records))]
        super().__init__(records, video_ids, encoded_video=True)


class PointOdysseyDataset(_RecordDataset):
    """Adapter for the prepared Point Odyssey pickle."""

    def __init__(self, root: str | Path) -> None:
        data = _load_pickle(Path(root) / "point_odyssey.pkl")
        if not isinstance(data, Mapping):
            raise TypeError("point_odyssey.pkl must contain a mapping")
        raw_keys = sorted(data, key=str)
        records = [cast(Mapping[str, Any], data[key]) for key in raw_keys]
        super().__init__(records, [str(key) for key in raw_keys], encoded_video=False)


def _load_pickle(path: Path) -> object:
    with path.open("rb") as file:
        return pickle.load(file)


def _as_record_sequence(value: object, *, source: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence):
        raise TypeError(f"{source} must contain a sequence")
    return [cast(Mapping[str, Any], record) for record in value]
