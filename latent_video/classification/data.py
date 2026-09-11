"""SSv2 manifests and strict, reproducible adapters around V-JEPA2 video loading."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .config import RunConfig

IMAGENET_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def sample_seed(*parts) -> int:
    value = json.dumps(parts, separators=(",", ":"), ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little") % 2**63


@contextmanager
def seeded_sample(seed: int):
    py_state, np_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.random.set_rng_state(torch.Generator().manual_seed(seed).get_state())
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


@dataclass(frozen=True)
class Sample:
    id: str
    path: str
    label: int


def read_manifest(path: Path, config: RunConfig) -> list[Sample]:
    records = []
    if path.suffix == ".json":
        if config.data.labels is None or config.data.videos is None:
            raise ValueError("Official SSv2 JSON needs data.labels and data.videos")
        with config.path(config.data.labels).open() as stream:
            mapping = json.load(stream)
        if not isinstance(mapping, dict) or any(
            type(v) is not int and not (isinstance(v, str) and v.isdecimal())
            for v in mapping.values()
        ):
            raise ValueError("Invalid SSv2 label mapping")
        mapping = {key: int(value) for key, value in mapping.items()}
        if len(mapping) != config.data.num_classes or set(mapping.values()) != set(
            range(config.data.num_classes)
        ):
            raise ValueError("SSv2 label IDs must uniquely cover 0 through 173")
        with path.open() as stream:
            rows = json.load(stream)
        if not isinstance(rows, list):
            raise ValueError("SSv2 annotations must be a JSON list")
        for row in rows:
            identifier = str(row["id"])
            if not identifier.isdecimal():
                raise ValueError("SSv2 video IDs must be decimal identifiers")
            template = row["template"].replace("[", "").replace("]", "")
            video = config.path(config.data.videos) / (
                identifier + config.data.extension
            )
            records.append(Sample(identifier, str(video), mapping[template]))
    elif path.suffix == ".csv":
        with path.open(newline="") as stream:
            for row in csv.reader(stream, delimiter=" ", skipinitialspace=True):
                if not row:
                    continue
                if len(row) != 2:
                    raise ValueError("Expected a space-delimited video-path/label CSV")
                video = Path(row[0]).expanduser()
                if not video.is_absolute():
                    video = path.parent / video
                records.append(Sample(video.stem, str(video.resolve()), int(row[1])))
    else:
        raise ValueError("Use official SSv2 JSON or a V-JEPA2 path/label .csv")
    if not records:
        raise ValueError(f"Empty data split: {path}")
    if len({r.id for r in records}) != len(records) or len(
        {r.path for r in records}
    ) != len(records):
        raise ValueError(f"Duplicate video IDs or paths in {path}")
    for record in records:
        if not 0 <= record.label < config.data.num_classes:
            raise ValueError(f"Out-of-range label for video {record.id}")
    return records


def manifest_identity(records: list[Sample]) -> str:
    raw = json.dumps([asdict(r) for r in records], sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def write_manifest(path: Path, records: list[Sample]):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter=" ", lineterminator="\n")
        writer.writerows((r.path, r.label) for r in records)


class WanRGBTransform:
    """Use official augmentations, then undo ImageNet normalization for the Wan VAE."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, frames):
        mean, std = (
            torch.tensor(v).reshape(3, 1, 1, 1) for v in IMAGENET_NORMALIZATION
        )
        return [
            ((view * std + mean) * 255)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(1, 0, 2, 3)
            .contiguous()
            for view in self.transform(frames)
        ]


class OnlineVideoDataset(Dataset):
    def __init__(self, dataset, records: list[Sample], *, seed: int, training: bool):
        self.dataset, self.records = dataset, records
        self.seed, self.training, self.epoch = seed, training, 0

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __getitem__(self, index):
        record = self.records[index]
        epoch = self.epoch if self.training else 0
        phase = "train" if self.training else "val"
        seed = sample_seed(self.seed, phase, epoch, record.id)
        try:
            with seeded_sample(seed):
                # Bypass the upstream random replacement on failure.
                output = self.dataset.get_item_video(index)
        except Exception as error:
            raise RuntimeError(
                f"Failed to decode SSv2 video {record.id}: {record.path}"
            ) from error
        if output is None:
            raise RuntimeError(
                f"Failed to decode SSv2 video {record.id}: {record.path}"
            )
        clips, label, indices = output
        if int(label) != record.label:
            raise ValueError(f"Label mismatch for {record.id}")
        if not clips or not clips[0] or len(indices) != len(clips):
            raise ValueError(f"Invalid decoded clip layout for {record.id}")
        views = len(clips[0])
        if any(len(segment) != views for segment in clips):
            raise ValueError("Every temporal segment must have the same spatial views")
        return {
            "clips": clips,
            "label": record.label,
            "id": record.id,
            "frame_indices": [torch.as_tensor(i, dtype=torch.long) for i in indices],
            "feature_seeds": torch.tensor(
                [
                    [
                        sample_seed(seed, "latent", segment, view)
                        for view in range(views)
                    ]
                    for segment in range(len(clips))
                ],
                dtype=torch.long,
            ),
        }


def build_dataset(
    runtime, manifest: Path, records: list[Sample], config: RunConfig, *, training: bool
):
    transform = runtime.make_transforms(
        training=training,
        num_views_per_clip=1 if training else config.data.num_views_per_segment,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 4 / 3),
        random_resize_scale=(0.08, 1.0),
        reprob=0.25,
        auto_augment=True,
        motion_shift=False,
        crop_size=config.data.crop_size,
        normalize=IMAGENET_NORMALIZATION,
    )
    dataset = runtime.video_dataset(
        data_paths=[str(manifest)],
        frames_per_clip=config.clip.frames,
        frame_step=config.data.frame_step,
        num_clips=config.data.num_segments,
        random_clip_sampling=True,
        allow_clip_overlap=True,
        transform=WanRGBTransform(transform),
    )
    if list(dataset.samples) != [r.path for r in records] or list(dataset.labels) != [
        r.label for r in records
    ]:
        raise ValueError("V-JEPA2 read a different video/label manifest")
    return OnlineVideoDataset(dataset, records, seed=config.seed, training=training)


class EvaluationSampler(Sampler):
    """Partition validation without DistributedSampler's repeated padding examples."""

    def __init__(self, size: int, rank: int, world_size: int):
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid evaluation rank/world size")
        self.indices = range(rank, size, world_size)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
