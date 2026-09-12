"""Atomic probe-only checkpoints with optimizer, scheduler, and rank RNG state."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .engine import ProbeOptimization, unwrap

SCHEMA = "heft.latent_video.online_probe.v1"


def publish_checkpoint_alias(source: Path, destination: Path):
    """Atomically point latest/best at a completed immutable epoch checkpoint."""
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    try:
        os.link(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def rng_state(device) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def save_checkpoint(
    path: Path,
    classifier,
    optimization: ProbeOptimization,
    *,
    epoch: int,
    best_top1: float,
    protocol: dict,
    device,
):
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    local_rng = rng_state(device)
    states: list[dict | None] = [None] * world_size
    if distributed:
        dist.all_gather_object(states, local_rng)
    else:
        states[0] = local_rng
    if rank != 0:
        return
    payload = {
        "schema": SCHEMA,
        "classifiers": [unwrap(classifier).state_dict()],
        "opt": [optimization.optimizer.state_dict()],
        "scaler": None
        if optimization.scaler is None
        else [optimization.scaler.state_dict()],
        "scheduler_step": optimization.scheduler._step,
        "wd_scheduler_step": optimization.wd_scheduler._step,
        "epoch": epoch,
        "best_top1": best_top1,
        "world_size": world_size,
        "rng": states,
        "protocol": protocol,
    }
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: Path,
    classifier,
    *,
    protocol: dict,
    device,
    optimization: ProbeOptimization | None = None,
) -> tuple[int, float]:
    # These are local checkpoints written by this runner, including Python/NumPy RNG state.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA or len(payload.get("classifiers", [])) != 1:
        raise ValueError("Expected this runner's single-classifier checkpoint")
    recorded = payload["protocol"]
    keys = (
        ("representation", "validation_manifest")
        if optimization is None
        else tuple(protocol)
    )
    if any(recorded.get(key) != protocol[key] for key in keys):
        raise ValueError(
            "Checkpoint protocol differs from the current data, features, or configuration"
        )
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    if optimization is not None and payload["world_size"] != world_size:
        raise ValueError(
            "Resume with the same world size to preserve batches and random state"
        )
    unwrap(classifier).load_state_dict(payload["classifiers"][0], strict=True)
    if optimization is None:
        return payload["epoch"], payload["best_top1"]
    optimization.optimizer.load_state_dict(payload["opt"][0])
    optimization.scheduler._step = payload["scheduler_step"]
    optimization.wd_scheduler._step = payload["wd_scheduler_step"]
    if optimization.scaler is not None:
        if payload["scaler"] is None:
            raise ValueError("Checkpoint is missing the FP16 gradient scaler")
        optimization.scaler.load_state_dict(payload["scaler"][0])
    state = payload["rng"][rank]
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)
    return payload["epoch"], payload["best_top1"]
