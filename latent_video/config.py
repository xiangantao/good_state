"""Fixed feature/noise definitions and explicit selection of existing channels."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClipConfig:
    frames: int = 16
    resolution: tuple[int, int] = (480, 832)
    grid: tuple[int, int] = (14, 14)
    stop_after_capture: bool = True

    def __post_init__(self):
        if type(self.frames) is not int or self.frames < 1:
            raise ValueError("frames must be a positive integer")
        if type(self.stop_after_capture) is not bool:
            raise TypeError("stop_after_capture must be a boolean")
        for name, sizes in (("resolution", self.resolution), ("grid", self.grid)):
            if len(sizes) != 2 or any(type(v) is not int or v < 1 for v in sizes):
                raise ValueError(f"{name} must contain two positive integers")
            object.__setattr__(self, name, tuple(sizes))
        if any(v % 16 for v in self.resolution):
            raise ValueError("Wan input height and width must be divisible by 16")
        if any(g > r // 16 for g, r in zip(self.grid, self.resolution, strict=True)):
            raise ValueError("The pooled grid must not exceed the input patch grid")


@dataclass(frozen=True)
class NoiseSpec:
    name: str
    mode: str
    requested_timestep: int
    shift: float
    actual_timestep: int
    sigma: float


NOISE_SPECS = (
    NoiseSpec("t57", "heft", 49, 3.0, 57, 0.05763683095574379),
    NoiseSpec("t299", "legacy", 300, 5.0, 299, 0.299923837184906),
)

# Keys are (zero-based layer, feature kind, zero-based head).
QK_TARGETS = {
    "t57": {
        (15, "key", 2): "l15h2_k_t57",
        (13, "key", 10): "l13h10_k_t57",
        (15, "query", 7): "l15h7_q_t57",
        (15, "key", 7): "l15h7_k_t57",
    },
    "t299": {(15, "key", 2): "l15h2_k_t299"},
}

HIDDEN_NAME = "b15_hidden256_t299"
CANDIDATES = {
    "l15h2_k_t57": ("l15h2_k_t57",),
    "l13h10_k_t57": ("l13h10_k_t57",),
    "l15h7_qk_t57": ("l15h7_q_t57", "l15h7_k_t57"),
    HIDDEN_NAME: (HIDDEN_NAME,),
    "l15h2_k_t299": ("l15h2_k_t299",),
}


@dataclass(frozen=True)
class ChannelSelection:
    indices: tuple[int, ...]
    source: str
    source_sha256: str
    key: str
    held_out: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "indices", tuple(self.indices))
        if len(self.indices) != 256:
            raise ValueError("The hidden feature requires exactly 256 channels")
        if any(type(i) is not int or not 0 <= i < 1536 for i in self.indices):
            raise ValueError("Channel indices must be integers in [0,1536)")
        if len(set(self.indices)) != 256:
            raise ValueError("Channel indices must be unique")

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        held_out: str | None = None,
        key: str = "ablation_iterative_256",
    ) -> ChannelSelection:
        """Read a fixed masks.json, or explicitly select a legacy validation fold."""
        path = Path(path).resolve(strict=True)
        raw = path.read_bytes()
        payload = json.loads(raw)
        if isinstance(payload, list):
            if held_out is None:
                raise ValueError("A fold report requires an explicit held_out scene")
            matches = [
                row
                for row in payload
                if isinstance(row, dict) and row.get("held_out") == held_out
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one channel table for held_out={held_out!r}"
                )
            tables = matches[0].get("masks")
        elif isinstance(payload, dict):
            if held_out is not None:
                raise ValueError(
                    "held_out selects a fold report; omit it for a fixed masks.json"
                )
            tables = payload
        else:
            raise TypeError("Expected a fold report or a channel-mask mapping")
        if not isinstance(tables, dict) or key not in tables:
            raise ValueError(f"Channel table {key!r} is missing")
        if not isinstance(tables[key], list):
            raise TypeError("A channel table must be a JSON list")
        return cls(
            tuple(tables[key]),
            str(path),
            hashlib.sha256(raw).hexdigest(),
            key,
            held_out,
        )

    def metadata(self) -> dict:
        result = asdict(self)
        result["indices"] = list(self.indices)
        return result
