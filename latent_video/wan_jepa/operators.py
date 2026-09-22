"""Frozen portable operators; verified artifacts, no fitting at runtime."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_artifacts(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"Artifact SHA256 mismatch: {name}")
    with np.load(directory / "wan_group_zca.npz", allow_pickle=False) as f:
        wan = {k: f[k].copy() for k in ("mean", "std", "matrix")}
    with np.load(directory / "jepa_supervised256.npz", allow_pickle=False) as f:
        if (
            str(f["source"]) != "jepa_b17+jepa_finalnorm"
            or str(f["plan_sha256"]) != manifest["calvin_plan_sha256"]
        ):
            raise ValueError("Incorrect supervised projection source/split")
        jepa = {k: f[k].copy() for k in ("mean", "std", "projection")}
    for arrays, shapes in (
        (wan, {"mean": (896,), "std": (896,), "matrix": (896, 896)}),
        (jepa, {"mean": (2048,), "std": (2048,), "projection": (2048, 256)}),
    ):
        for key, shape in shapes.items():
            if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
                raise ValueError(f"Invalid {key}: {arrays[key].shape}")
        if not (arrays["std"] > 0).all():
            raise ValueError("Nonpositive std")
    return wan, jepa, manifest


class FixedOperators(torch.nn.Module):
    def __init__(self, directory):
        super().__init__()
        wan, jepa, self.manifest = load_artifacts(directory)
        for prefix, arrays in (("w", wan), ("j", jepa)):
            for key, array in arrays.items():
                self.register_buffer(f"{prefix}_{key}", torch.from_numpy(array).float())

    def wan(self, raw):
        if raw.ndim != 5 or raw.shape[1:] != (896, 8, 16, 16):
            raise ValueError("Require Wan896x8x16x16")
        with torch.autocast(raw.device.type, enabled=False):
            x = (
                raw.half().float() - self.w_mean[None, :, None, None, None]
            ) / self.w_std[None, :, None, None, None]
            return (
                (x.flatten(2).transpose(1, 2) @ self.w_matrix)
                .transpose(1, 2)
                .reshape_as(x)
            )

    def jepa(self, raw):
        if raw.ndim != 3 or raw.shape[1:] != (2048, 2048):
            raise ValueError("Require2048 JEPA tokens of2048 channels")
        with torch.autocast(raw.device.type, enabled=False):
            return ((raw.half().float() - self.j_mean) / self.j_std) @ self.j_projection
