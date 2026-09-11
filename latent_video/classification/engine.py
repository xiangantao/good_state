# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted from V-JEPA2 evals/video_classification_frozen/eval.py (MIT).
"""Single-probe training and multi-view evaluation with online frozen features."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

logger = logging.getLogger(__name__)


@dataclass
class ProbeOptimization:
    optimizer: Any
    scaler: Any
    scheduler: Any
    wd_scheduler: Any

    @classmethod
    def create(cls, runtime, classifier, config, iterations_per_epoch):
        opt, scaler, scheduler, wd_scheduler = runtime.init_opt(
            classifiers=[classifier],
            iterations_per_epoch=iterations_per_epoch,
            opt_kwargs=config.upstream_kwargs(),
            num_epochs=config.num_epochs,
            use_bfloat16=config.amp_dtype == "float16",
        )
        if any(len(items) != 1 for items in (opt, scaler, scheduler, wd_scheduler)):
            raise ValueError("Exactly one classifier and optimizer are required")
        return cls(opt[0], scaler[0], scheduler[0], wd_scheduler[0])


def unwrap(classifier):
    return (
        classifier.module
        if isinstance(classifier, DistributedDataParallel)
        else classifier
    )


def probabilities(logits: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([F.softmax(output.float(), dim=1) for output in logits]).mean(0)


def run_epoch(
    classifier,
    encoder,
    loader,
    *,
    training: bool,
    amp_dtype: str = "float16",
    optimization: ProbeOptimization | None = None,
    predictions: Path | None = None,
    on_step: Callable[[dict], None] | None = None,
    log_every: int = 10,
    epoch: int = 0,
) -> dict:
    if training and optimization is None:
        raise ValueError("Training requires an optimizer")
    device = encoder.device
    classifier.train(training)
    # Validation ranks may have unequal batch counts; bypass DDP forward collectives.
    model = classifier if training else unwrap(classifier)
    stats = torch.zeros(4, device=device, dtype=torch.float64)
    previous_stats = torch.zeros_like(stats)
    previous_iteration = 0
    window_start = time.perf_counter()
    if on_step is not None and training and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    amp_enabled = device.type == "cuda" and amp_dtype != "float32"
    dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
    temporary = (
        None
        if predictions is None
        else predictions.with_name(predictions.name + f".{os.getpid()}.tmp")
    )
    stream = None
    if temporary is not None:
        stream = temporary.open("w")
    try:
        for iteration, batch in enumerate(loader):
            labels = batch["label"].to(device)
            views = len(batch["clips"][0])
            if training and views != 1:
                raise ValueError(
                    "Official training uses one spatial view per temporal segment"
                )
            if optimization is not None and training:
                optimization.scheduler.step()
                optimization.wd_scheduler.step()
                optimization.optimizer.zero_grad(set_to_none=True)
            logits = []
            with nullcontext() if training else torch.no_grad():
                for view in range(views):
                    with torch.no_grad(), torch.autocast(device.type, enabled=False):
                        tokens = encoder.encode_view(batch, view)
                    # Classifier autocast must never change the Wan extraction arithmetic.
                    with torch.autocast(device.type, dtype=dtype, enabled=amp_enabled):
                        logits.append(model(tokens if amp_enabled else tokens.float()))
                losses = [F.cross_entropy(output.float(), labels) for output in logits]
                loss = torch.stack(losses).sum()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite classification loss")
            if training and optimization is not None:
                scaler = optimization.scaler
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimization.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimization.optimizer.step()
            with torch.no_grad():
                probs = probabilities(logits)
                if not torch.isfinite(probs).all():
                    raise ValueError("Non-finite predicted probabilities")
                top = probs.topk(min(5, probs.shape[1]), dim=1).indices
                count = len(labels)
                stats[0] += count
                stats[1] += loss.detach().double() * count / views
                stats[2] += top[:, 0].eq(labels).sum()
                stats[3] += top.eq(labels[:, None]).any(dim=1).sum()
                if stream is not None:
                    for identifier, label, indices, scores in zip(
                        batch["id"],
                        labels.tolist(),
                        top.tolist(),
                        probs.gather(1, top).tolist(),
                        strict=True,
                    ):
                        stream.write(
                            json.dumps(
                                {
                                    "id": identifier,
                                    "label": label,
                                    "top5": indices,
                                    "probabilities": scores,
                                }
                            )
                            + "\n"
                        )
            # All training ranks enter these collectives; only rank zero writes logs.
            # Validation has uneven shards and reports only after the epoch reduction.
            if (
                training
                and on_step is not None
                and ((iteration + 1) % log_every == 0 or iteration + 1 == len(loader))
            ):
                window_stats = stats - previous_stats
                previous_stats.copy_(stats)
                telemetry = torch.tensor(
                    [
                        time.perf_counter() - window_start,
                        torch.cuda.max_memory_allocated(device) / 1024**3
                        if device.type == "cuda"
                        else 0.0,
                    ],
                    device=device,
                    dtype=torch.float64,
                )
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(window_stats, op=dist.ReduceOp.SUM)
                    dist.all_reduce(telemetry, op=dist.ReduceOp.MAX)
                samples, loss_sum, correct1, correct5 = window_stats.tolist()
                elapsed, peak_gib = telemetry.tolist()
                assert optimization is not None
                group = optimization.optimizer.param_groups[0]
                values = {
                    "step": epoch * len(loader) + iteration + 1,
                    "epoch": epoch + (iteration + 1) / len(loader),
                    "batch/samples": int(samples),
                    "batch/loss": loss_sum / samples,
                    "batch/top1": 100 * correct1 / samples,
                    "batch/top5": 100 * correct5 / samples,
                    "optim/lr": group["lr"],
                    "optim/weight_decay": group["weight_decay"],
                    "timing/iteration_seconds": elapsed
                    / (iteration + 1 - previous_iteration),
                    "timing/videos_per_second": samples / max(elapsed, 1e-9),
                    "memory/peak_allocated_gib": peak_gib,
                }
                if optimization.scaler is not None:
                    values["optim/grad_scale"] = optimization.scaler.get_scale()
                on_step(values)
                previous_iteration = iteration + 1
                window_start = time.perf_counter()
            if iteration % 10 == 0:
                logger.info(
                    "batch=%d samples=%d loss=%.5f top1=%.3f",
                    iteration,
                    int(stats[0]),
                    float(stats[1] / stats[0]),
                    float(100 * stats[2] / stats[0]),
                )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        count, loss_sum, correct1, correct5 = stats.cpu().tolist()
        if count == 0:
            raise ValueError("The data loader produced no samples")
        result = {
            "samples": int(count),
            "loss": loss_sum / count,
            "top1": 100 * correct1 / count,
            "top5": 100 * correct5 / count,
        }
        if stream is not None:
            stream.close()
            stream = None
            assert temporary is not None and predictions is not None
            os.replace(temporary, predictions)
        return result
    finally:
        if stream is not None:
            stream.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)
