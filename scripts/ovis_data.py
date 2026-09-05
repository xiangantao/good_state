"""Small OVIS helpers shared by the extraction and semantic-evaluation scripts."""

from __future__ import annotations

import io
import json
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch import Tensor


@dataclass(frozen=True, slots=True)
class OvisVideo:
    """One video entry and its track annotations from the OVIS JSON file."""

    id: int
    name: str
    width: int
    height: int
    length: int
    file_names: tuple[str, ...]
    annotations: tuple[Mapping[str, Any], ...]


class OvisDataset:
    """In-memory index over an OVIS annotation JSON file."""

    def __init__(self, annotation_path: str | Path) -> None:
        self.annotation_path = Path(annotation_path)
        with self.annotation_path.open(encoding="utf-8") as file:
            payload: dict[str, Any] = json.load(file)

        self.categories = {
            int(category["id"]): str(category["name"])
            for category in payload["categories"]
        }
        annotations_by_video: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for annotation in payload["annotations"]:
            annotations_by_video[int(annotation["video_id"])].append(annotation)

        videos: list[OvisVideo] = []
        for raw_video in payload["videos"]:
            file_names = tuple(str(name) for name in raw_video["file_names"])
            if not file_names:
                raise ValueError(f"OVIS video {raw_video['id']} has no frames")
            name = Path(file_names[0]).parts[0]
            if any(Path(frame).parts[0] != name for frame in file_names):
                raise ValueError(f"OVIS video {raw_video['id']} spans multiple folders")
            length = int(raw_video["length"])
            if len(file_names) != length:
                raise ValueError(
                    f"OVIS video {raw_video['id']} declares {length} frames but lists "
                    f"{len(file_names)}"
                )
            videos.append(
                OvisVideo(
                    id=int(raw_video["id"]),
                    name=name,
                    width=int(raw_video["width"]),
                    height=int(raw_video["height"]),
                    length=length,
                    file_names=file_names,
                    annotations=tuple(
                        sorted(
                            annotations_by_video[int(raw_video["id"])],
                            key=lambda annotation: int(annotation["id"]),
                        )
                    ),
                )
            )
        self.videos = tuple(sorted(videos, key=lambda video: video.id))
        self._by_name = {video.name: video for video in self.videos}
        self._by_id = {video.id: video for video in self.videos}

    def select(self, names_or_ids: Iterable[str] | None = None) -> tuple[OvisVideo, ...]:
        """Return all videos or a caller-specified sequence of names/numeric ids."""

        if names_or_ids is None:
            return self.videos
        selected: list[OvisVideo] = []
        for value in names_or_ids:
            value = value.strip()
            if not value:
                continue
            video = self._by_name.get(value)
            if video is None and value.isdigit():
                video = self._by_id.get(int(value))
            if video is None:
                raise KeyError(f"unknown OVIS video: {value}")
            selected.append(video)
        return tuple(selected)


class OvisFrameArchive:
    """Read OVIS JPEG frames directly from the downloaded validation ZIP."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._archive: zipfile.ZipFile | None = None
        self._members: frozenset[str] = frozenset()

    def __enter__(self) -> OvisFrameArchive:  # noqa: PYI034
        self._archive = zipfile.ZipFile(self.path)
        self._members = frozenset(self._archive.namelist())
        return self

    def __exit__(self, *_: object) -> None:
        if self._archive is not None:
            self._archive.close()
        self._archive = None
        self._members = frozenset()

    def read_video(
        self,
        video: OvisVideo,
        *,
        size: tuple[int, int] | None = None,
    ) -> Tensor:
        """Decode one video as contiguous RGB uint8 ``[T, C, H, W]``.

        When ``size`` is supplied, each image is resized with the same Lanczos
        interpolation used by HeFT's extraction worker. Resizing before stacking
        prevents high-resolution OVIS videos from occupying unnecessary host RAM.
        """

        archive = self._require_open()
        output_height, output_width = size or (video.height, video.width)
        frames: list[Tensor] = []
        for file_name in video.file_names:
            member = self._resolve_member(file_name)
            encoded = archive.read(member)
            with Image.open(io.BytesIO(encoded)) as image:
                frame = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
            if frame.shape[:2] != (output_height, output_width):
                frame = cv2.resize(
                    frame,
                    (output_width, output_height),
                    interpolation=cv2.INTER_LANCZOS4,
                )
            frames.append(torch.from_numpy(frame).permute(2, 0, 1))
        return torch.stack(frames).contiguous()

    def _resolve_member(self, file_name: str) -> str:
        candidates = (file_name, f"valid/{file_name}")
        for candidate in candidates:
            if candidate in self._members:
                return candidate
        raise FileNotFoundError(f"{file_name} is missing from {self.path}")

    def _require_open(self) -> zipfile.ZipFile:
        if self._archive is None:
            raise RuntimeError("OVIS frame archive is not open")
        return self._archive
