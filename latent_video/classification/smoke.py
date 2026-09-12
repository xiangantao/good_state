"""Bounded local-data extraction/DDP check; never starts a full training run."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import replace
from itertools import islice
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import default_collate

from ..extractor import WanLatentExtractor
from .config import RunConfig
from .data import Sample, build_dataset, write_manifest
from .encoder import OnlineWanEncoder
from .engine import ProbeOptimization, run_epoch, unwrap
from .execution import ExtractionPool
from .train import distributed_device
from .upstream import load_upstream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--extract-batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    if min(args.batch_size, args.extract_batch_size, args.steps) < 1:
        parser.error("Batch sizes and steps must be positive")
    config = RunConfig.load(args.config)
    config = replace(
        config,
        optimization=replace(config.optimization, batch_size=args.batch_size),
        extract_batch_size=args.extract_batch_size,
    )
    device, rank, world = distributed_device()
    pool = None
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    def event(name, **values):
        print(
            json.dumps(dict(event=name, rank=rank, time=time.time(), **values)),
            flush=True,
        )

    event("start", pid=os.getpid(), world_size=world, device=str(device))
    try:
        runtime = load_upstream(config.path(config.vjepa_root))
        manifest = config.path(config.data.train)
        with manifest.open() as stream:
            rows = list(
                islice(
                    csv.reader(stream, delimiter=" ", skipinitialspace=True),
                    rank * args.batch_size,
                    (rank + 1) * args.batch_size,
                )
            )
        records = [
            Sample(Path(path).stem, str((manifest.parent / path).resolve()), int(label))
            for path, label in rows
        ]
        subset = output / f"train_r{rank}.csv"
        write_manifest(subset, records)
        dataset = build_dataset(runtime, subset, records, config, training=True)
        batch = default_collate([dataset[i] for i in range(len(dataset))])
        event("decoded", videos=len(records))
        source = WanLatentExtractor.from_local(
            args.model_path,
            channel_mask=config.path(config.channel_mask),
            held_out=config.held_out,
            device=str(device),
            config=config.clip,
        )
        extractors = [source] + [
            source.replica() for _ in range(config.extract_lanes - 1)
        ]
        pool = ExtractionPool(extractors, compile_blocks=config.compile_blocks)
        encoder = OnlineWanEncoder(
            source, extract_batch_size=config.extract_batch_size, pool=pool
        )
        original = pool.extract_groups
        pool.extract_groups = lambda requests: original(requests, concurrent=False)
        reference = encoder.encode_view(batch, 0).cpu()
        pool.extract_groups = original
        actual = encoder.encode_view(batch, 0).cpu()
        exact = torch.equal(reference, actual)
        delta = actual.float() - reference.float()
        comparison = {
            "exact": exact,
            "max_abs": delta.abs().max().item(),
            "relative_rms": (delta.square().sum() / reference.float().square().sum())
            .sqrt()
            .item(),
        }
        event("concurrency_comparison", **comparison)
        if not exact:
            raise AssertionError(
                f"Parallel extraction differs from the same-lane serial reference: {comparison}"
            )
        del reference, actual, delta
        torch.manual_seed(config.seed)
        classifier = runtime.classifier(
            embed_dim=896,
            num_heads=config.num_heads,
            depth=config.num_probe_blocks,
            num_classes=174,
            use_activation_checkpointing=True,
        ).to(device)
        if dist.is_initialized():
            classifier = DistributedDataParallel(
                classifier, device_ids=[device.index], static_graph=True
            )
        optimization = ProbeOptimization.create(
            runtime, classifier, config.optimization, args.steps
        )
        before = next(unwrap(classifier).parameters()).detach().clone()
        steps = []

        def log_step(values):
            steps.append(values)
            event("training_step", **values)

        event("training_start")
        metrics = run_epoch(
            classifier,
            encoder,
            [batch] * args.steps,
            training=True,
            optimization=optimization,
            amp_dtype=config.optimization.amp_dtype,
            on_step=log_step,
            log_every=1,
        )
        after = next(unwrap(classifier).parameters()).detach()
        assert not torch.equal(before, after), "Classifier did not update"
        assert all(
            p.grad is None and not p.requires_grad
            for e in extractors
            for module in (e.pipeline.vae, e.pipeline.transformer)
            for p in module.parameters()
        )
        rank_agreement = True
        if dist.is_initialized():
            copies = [torch.empty_like(after) for _ in range(world)]
            dist.all_gather(copies, after.contiguous())
            rank_agreement = all(torch.equal(after, value) for value in copies)
            assert rank_agreement, "DDP classifier parameters diverged"
        report = {
            "complete": True,
            "world_size": world,
            "rank": rank,
            "videos_per_rank": args.batch_size,
            "execution": pool.metadata(),
            "extract_batch_size": config.extract_batch_size,
            "resolution": list(config.clip.resolution),
            "comparison": comparison,
            "rank_agreement": rank_agreement,
            "metrics": metrics,
            "steps": steps,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            "measurement": "Predecoded real clips repeated for bounded training steps; excludes loader throughput",
            "torch_version": torch.__version__,
            "extractor": source.provenance,
        }
        (output / f"result_r{rank}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        event("complete", metrics=metrics, rank_agreement=rank_agreement)
    finally:
        if pool is not None:
            pool.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
