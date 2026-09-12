"""Configuration for one online, frozen-Wan classification experiment."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

from ..config import ClipConfig

HEFT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DataConfig:
    train: str
    val: str
    videos: str | None = None
    labels: str | None = None
    extension: str = ".webm"
    num_classes: int = 174
    num_segments: int = 2
    num_views_per_segment: int = 3
    frame_step: int = 4
    crop_size: int = 256

    def __post_init__(self):
        if self.num_classes != 174:
            raise ValueError("This experiment requires the 174 SSv2 classes")
        for name in (
            "num_segments",
            "num_views_per_segment",
            "frame_step",
            "crop_size",
        ):
            positive_int(name, getattr(self, name))
        if self.extension not in (".webm", ".mp4", ".avi"):
            raise ValueError("extension must be .webm, .mp4, or .avi")


@dataclass(frozen=True)
class OptimizationConfig:
    batch_size: int = 4
    num_epochs: int = 20
    lr: float = 0.0003
    weight_decay: float = 0.1
    warmup: float = 0.0
    final_lr: float = 0.0
    amp_dtype: str = "float16"

    def __post_init__(self):
        positive_int("batch_size", self.batch_size)
        positive_int("num_epochs", self.num_epochs)
        for name in ("lr", "weight_decay", "warmup", "final_lr"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.lr == 0 or self.final_lr > self.lr or self.warmup >= self.num_epochs:
            raise ValueError("Require lr > 0, final_lr <= lr, and warmup < num_epochs")
        if self.amp_dtype not in ("float16", "bfloat16", "float32"):
            raise ValueError("amp_dtype must be float16, bfloat16, or float32")

    def upstream_kwargs(self) -> list[dict]:
        return [
            {
                "ref_wd": self.weight_decay,
                "final_wd": self.weight_decay,
                "start_lr": self.lr,
                "ref_lr": self.lr,
                "final_lr": self.final_lr,
                "warmup": self.warmup,
            }
        ]


def positive_int(name: str, value: int):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool = False
    project: str = "heft-ssv2-latents"
    entity: str | None = None
    name: str | None = None
    group: str | None = None
    log_every: int = 10
    mode: str = "offline"

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise TypeError("wandb.enabled must be a boolean")
        if self.mode not in ("online", "offline"):
            raise ValueError("wandb.mode must be online or offline")
        for name in ("project", "entity", "name", "group"):
            value = getattr(self, name)
            if value is None and name != "project":
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"wandb.{name} must be a non-empty string")
        positive_int("wandb.log_every", self.log_every)


@dataclass(frozen=True)
class RunConfig:
    data: DataConfig
    model_path: str = "../models/Wan2.1-T2V-1.3B-Diffusers"
    vjepa_root: str = "../vjepa2"
    channel_mask: str = (
        "reports/scannet_channels/73f4e810824b58cb/global_256/masks.json"
    )
    held_out: str | None = None
    output_dir: str = "../runs/ssv2_wan_fused_online"
    seed: int = 42
    num_workers: int = 4
    extract_batch_size: int = 1
    num_heads: int = 16
    num_probe_blocks: int = 4
    clip: ClipConfig = field(default_factory=ClipConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self):
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0,2**63)")
        if type(self.num_workers) is not int or self.num_workers < 0:
            raise ValueError("num_workers must be a non-negative integer")
        positive_int("num_heads", self.num_heads)
        positive_int("extract_batch_size", self.extract_batch_size)
        positive_int("num_probe_blocks", self.num_probe_blocks)
        if 896 % self.num_heads:
            raise ValueError("num_heads must divide the 896 fused channels")
        if self.clip.frames != 16:
            raise ValueError("The classification protocol uses 16 frames per segment")

    def path(self, value: str) -> Path:
        path = Path(value).expanduser()
        return (path if path.is_absolute() else HEFT_ROOT / path).resolve()

    def metadata(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path) -> RunConfig:
        with Path(path).open() as stream:
            value = yaml.safe_load(stream)
        if not isinstance(value, dict):
            raise TypeError("The run configuration must be a YAML mapping")
        value = dict(value)
        for key, factory in (
            ("data", DataConfig),
            ("clip", ClipConfig),
            ("optimization", OptimizationConfig),
            ("wandb", WandbConfig),
        ):
            if key in value:
                entry = value[key]
                if not isinstance(entry, dict):
                    raise TypeError(f"{key} must be a mapping")
                unknown = set(entry) - {f.name for f in fields(factory)}
                if unknown:
                    raise ValueError(f"Unknown {key} settings: {sorted(unknown)}")
                value[key] = factory(**entry)
        unknown = set(value) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown run settings: {sorted(unknown)}")
        return cls(**value)
