"""Train or evaluate one attentive classifier on freshly extracted Wan features."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import random
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from ..config import ChannelSelection
from ..extractor import WanLatentExtractor
from .checkpoint import load_checkpoint, publish_checkpoint_alias, save_checkpoint
from .config import RunConfig
from .data import (
    EvaluationSampler,
    build_dataset,
    manifest_identity,
    read_manifest,
    write_manifest,
)
from .encoder import OnlineWanEncoder
from .engine import ProbeOptimization, run_epoch
from .execution import ExtractionPool
from .monitoring import WandbMonitor
from .upstream import load_upstream, missing_dependencies

logger = logging.getLogger(__name__)


def preflight(config: RunConfig, *, evaluation: bool = False) -> list[str]:
    issues = []
    required = {
        "model_path": config.model_path,
        "vjepa_root": config.vjepa_root,
        "channel_mask": config.channel_mask,
        "data.val": config.data.val,
    }
    if not evaluation:
        required["data.train"] = config.data.train
    for name, value in (
        ("data.labels", config.data.labels),
        ("data.videos", config.data.videos),
    ):
        if value is not None:
            required[name] = value
    for name, value in required.items():
        if not config.path(value).exists():
            issues.append(f"{name} does not exist: {config.path(value)}")
    if config.path(config.channel_mask).is_file():
        try:
            ChannelSelection.load(
                config.path(config.channel_mask), held_out=config.held_out
            )
        except (ValueError, TypeError, KeyError) as error:
            issues.append(f"channel_mask: {error}")
    missing = missing_dependencies()
    if config.wandb.enabled and importlib.util.find_spec("wandb") is None:
        missing.append("wandb")
    if missing:
        issues.append(
            f"Missing dependencies: {', '.join(missing)} (no automatic installation)"
        )
    return issues


def distributed_device():
    if not torch.cuda.is_available():
        raise RuntimeError("The online Wan checkpoint runner requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    return torch.device("cuda", local_rank), rank, world_size


def make_loader(
    dataset, config: RunConfig, *, rank: int, world_size: int, training: bool
):
    sampler = (
        DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=config.seed
        )
        if training
        else EvaluationSampler(len(dataset), rank, world_size)
    )
    return DataLoader(
        dataset,
        batch_size=config.optimization.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False,
        generator=torch.Generator().manual_seed(config.seed + rank),
        multiprocessing_context="spawn" if config.num_workers else None,
    ), sampler


def protocol_record(config, encoder, runtime, train_records, val_records, world_size):
    sources = sorted(
        p for p in Path(__file__).parent.glob("*.py") if not p.name.startswith("test_")
    )
    return {
        "representation": {
            "clip": asdict(config.clip),
            "encoder": encoder.metadata(),
            "probe": {
                "embed_dim": encoder.embed_dim,
                "heads": config.num_heads,
                "depth": config.num_probe_blocks,
                "classes": config.data.num_classes,
            },
            "preprocessing": {
                "crop_size": config.data.crop_size,
                "frame_step": config.data.frame_step,
                "segments": config.data.num_segments,
                "validation_views": config.data.num_views_per_segment,
                "rgb_adapter": "official augmentation; inverse ImageNet normalization; round/clamp uint8; Wan resize",
                "sampling": "per-video seeded original sampler; validation epoch fixed to zero",
                "seed": config.seed,
            },
            "vjepa2": runtime.provenance,
            "runner_sources": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
            },
        },
        "training": {
            "optimization": asdict(config.optimization),
            "validate_every": config.validate_every,
            "world_size": world_size,
            "manifest": manifest_identity(train_records)
            if train_records is not None
            else None,
        },
        "validation_manifest": manifest_identity(val_records),
    }


def run(
    config: RunConfig, *, resume: Path | None = None, evaluate: Path | None = None,
    extractor_class=None, encoder_factory=None,
):
    extractor_class = WanLatentExtractor if extractor_class is None else extractor_class
    encoder_factory = OnlineWanEncoder if encoder_factory is None else encoder_factory
    issues = preflight(config, evaluation=evaluate is not None)
    if issues:
        raise ValueError("\n".join(issues))
    train_records = (
        None
        if evaluate is not None
        else read_manifest(config.path(config.data.train), config)
    )
    val_records = read_manifest(config.path(config.data.val), config)
    if train_records is not None and {r.id for r in train_records} & {
        r.id for r in val_records
    }:
        raise ValueError("Training and validation video IDs overlap")
    output = config.path(config.output_dir)
    if evaluate is not None:
        output = output / "evaluations" / evaluate.stem
    if (output / "latest.pt").exists() and resume is None and evaluate is None:
        raise ValueError(
            "Output already has a checkpoint; use --resume or a new output_dir"
        )
    device, rank, world_size = distributed_device()
    monitor = None
    pool = None
    succeeded = False
    try:
        if rank == 0:
            for record in (train_records or []) + val_records:
                if not Path(record.path).is_file():
                    raise FileNotFoundError(
                        f"Video is not available yet: {record.path}"
                    )
            output.mkdir(parents=True, exist_ok=True)
            if train_records is not None:
                write_manifest(output / "train_paths.csv", train_records)
            write_manifest(output / "val_paths.csv", val_records)
        if dist.is_initialized():
            dist.barrier()
        monitor = WandbMonitor(
            config.wandb,
            output,
            rank=rank,
            metadata=config.metadata(),
            job_type="evaluate" if evaluate is not None else "train",
        )
        runtime = load_upstream(config.path(config.vjepa_root))
        val_loader = None
        if evaluate is not None or config.validate_every:
            val_dataset = build_dataset(
                runtime, output / "val_paths.csv", val_records, config, training=False
            )
            val_loader, _ = make_loader(
                val_dataset, config, rank=rank, world_size=world_size, training=False
            )
        train_loader = train_sampler = train_dataset = None
        if train_records is not None:
            train_dataset = build_dataset(
                runtime,
                output / "train_paths.csv",
                train_records,
                config,
                training=True,
            )
            train_loader, train_sampler = make_loader(
                train_dataset, config, rank=rank, world_size=world_size, training=True
            )
        extractor = extractor_class.from_local(
            config.path(config.model_path),
            channel_mask=config.path(config.channel_mask),
            held_out=config.held_out,
            device=str(device),
            config=config.clip,
        )
        # Clone before installing compiled forwards; never share hooks or VAE caches.
        extractors = [extractor] + [
            extractor.replica() for _ in range(config.extract_lanes - 1)
        ]
        pool = ExtractionPool(extractors, compile_blocks=config.compile_blocks)
        encoder = encoder_factory(
            extractor, extract_batch_size=config.extract_batch_size, pool=pool
        )
        random.seed(config.seed)
        np.random.seed(config.seed % 2**32)
        torch.manual_seed(config.seed)
        classifier = runtime.classifier(
            embed_dim=encoder.embed_dim,
            num_heads=config.num_heads,
            depth=config.num_probe_blocks,
            num_classes=config.data.num_classes,
            use_activation_checkpointing=True,
        ).to(device)
        if dist.is_initialized():
            classifier = DistributedDataParallel(
                classifier, device_ids=[device.index], static_graph=True
            )
        protocol = protocol_record(
            config, encoder, runtime, train_records, val_records, world_size
        )
        monitor.update_config({"protocol": protocol})
        if evaluate is not None:
            assert val_loader is not None
            epoch, _ = load_checkpoint(
                evaluate, classifier, protocol=protocol, device=device
            )
            metrics = run_epoch(
                classifier,
                encoder,
                val_loader,
                training=False,
                amp_dtype=config.optimization.amp_dtype,
                predictions=output / f"predictions_r{rank}.jsonl",
            )
            if metrics["samples"] != len(val_records):
                raise RuntimeError(
                    "Validation did not cover exactly the requested split"
                )
            monitor.update_config({"evaluation_checkpoint": str(evaluate.resolve())})
            monitor.log(
                {"epoch": epoch, **{f"evaluation/{k}": v for k, v in metrics.items()}}
            )
            if rank == 0:
                (output / "evaluation.json").write_text(
                    json.dumps(
                        {
                            "checkpoint": str(evaluate.resolve()),
                            "epoch": epoch,
                            "metrics": metrics,
                            "protocol": protocol,
                        },
                        indent=2,
                    )
                )
            succeeded = True
            return metrics
        assert (
            train_loader is not None
            and train_sampler is not None
            and train_dataset is not None
        )
        optimization = ProbeOptimization.create(
            runtime, classifier, config.optimization, len(train_loader)
        )
        start_epoch, best = 0, float("-inf")
        if resume is not None:
            start_epoch, best = load_checkpoint(
                resume,
                classifier,
                optimization=optimization,
                protocol=protocol,
                device=device,
            )
        monitor.update_config(
            {
                "start_epoch": start_epoch,
                "resume_checkpoint": str(resume.resolve())
                if resume is not None
                else None,
            }
        )
        if rank == 0:
            (output / "config.json").write_text(json.dumps(config.metadata(), indent=2))
            (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
        for epoch in range(start_epoch, config.optimization.num_epochs):
            assert isinstance(train_sampler, DistributedSampler)
            train_sampler.set_epoch(epoch)
            train_dataset.set_epoch(epoch)
            train_metrics = run_epoch(
                classifier,
                encoder,
                train_loader,
                training=True,
                amp_dtype=config.optimization.amp_dtype,
                optimization=optimization,
                on_step=monitor.log if config.wandb.enabled else None,
                log_every=config.wandb.log_every,
                epoch=epoch,
            )
            val_metrics = None
            improved = False
            if config.validate_every and (epoch + 1) % config.validate_every == 0:
                assert val_loader is not None
                val_metrics = run_epoch(
                    classifier,
                    encoder,
                    val_loader,
                    training=False,
                    amp_dtype=config.optimization.amp_dtype,
                )
                if val_metrics["samples"] != len(val_records):
                    raise RuntimeError(
                        "Validation did not cover exactly the requested split"
                    )
                improved = val_metrics["top1"] > best
                best = max(best, val_metrics["top1"])
            monitor.log(
                {
                    "step": (epoch + 1) * len(train_loader),
                    "epoch": epoch + 1,
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    **(
                        {f"val/{k}": v for k, v in val_metrics.items()}
                        if val_metrics
                        else {}
                    ),
                    **({"val/best_top1": best} if val_metrics else {}),
                }
            )
            save_args = {
                "epoch": epoch + 1,
                "best_top1": best,
                "protocol": protocol,
                "device": device,
            }
            epoch_path = output / f"epoch_{epoch + 1:04d}.pt"
            if epoch_path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite retained checkpoint: {epoch_path}"
                )
            save_checkpoint(epoch_path, classifier, optimization, **save_args)
            if rank == 0:
                publish_checkpoint_alias(epoch_path, output / "latest.pt")
                if improved:
                    publish_checkpoint_alias(epoch_path, output / "best.pt")
            if rank == 0:
                with (output / "metrics.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "epoch": epoch + 1,
                                "train": train_metrics,
                                "val": val_metrics,
                                "best_top1": best if np.isfinite(best) else None,
                            }
                        )
                        + "\n"
                    )
                logger.info(
                    "epoch=%d train=%s val=%s", epoch + 1, train_metrics, val_metrics
                )
        succeeded = True
    finally:
        try:
            if pool is not None:
                pool.close()
        finally:
            try:
                if monitor is not None:
                    monitor.finish(exit_code=0 if succeeded else 1)
            finally:
                if dist.is_initialized():
                    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", type=Path)
    mode.add_argument("--evaluate", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check configuration and local prerequisites without loading models",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    config = RunConfig.load(args.config)
    if args.model_path is not None:
        config = replace(config, model_path=str(args.model_path.resolve()))
    if args.output_dir is not None:
        config = replace(config, output_dir=str(args.output_dir.resolve()))
    if args.check:
        issues = preflight(config, evaluation=args.evaluate is not None)
        print(
            json.dumps(
                {"ready": not issues, "issues": issues, "config": config.metadata()},
                indent=2,
            )
        )
        return 1 if issues else 0
    run(config, resume=args.resume, evaluate=args.evaluate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
