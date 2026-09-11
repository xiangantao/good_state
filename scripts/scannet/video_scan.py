"""Scan joint-video Wan block hidden and Q/K with HEFT's inference settings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from heft.extraction.config import WAN_2_1

from .dataset import SensScene, discover_scenes
from .extract import offline_environment
from .metrics import evaluate_pair
from .report import pca_preview, write_json
from .scan import atomic_npz, digest, model_identity, parser as block_parser, prepare_scene, source_identity
from .video_extract import WanVideoExtractor, chunk_records, noise_protocol, video_jobs
from .video_report import generate_video


def read_json(path):
    return json.loads(Path(path).read_text())


def runtime_sources():
    repo = Path(__file__).resolve().parents[2]
    paths = [
        "src/heft/extraction/config.py", "src/heft/extraction/pool.py",
        "src/heft/extraction/worker.py", "src/heft/attn_hook/capture.py",
        "diffusers/src/diffusers/pipelines/wan/pipeline_wan.py",
        "diffusers/src/diffusers/models/autoencoders/autoencoder_kl_wan.py",
        "diffusers/src/diffusers/models/transformers/transformer_wan.py",
        "diffusers/src/diffusers/schedulers/scheduling_unipc_multistep.py",
        "diffusers/src/diffusers/video_processor.py", "diffusers/src/diffusers/image_processor.py",
    ]
    return {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in paths}


def parser():
    workspace = Path(__file__).resolve().parents[3]
    p = block_parser()
    p.description = __doc__
    p.set_defaults(
        blocks=[14, 15], timesteps=[WAN_2_1.start_step], shift=3.0,
        output_root=workspace / "eval/scannet_video",
        report_root=workspace / "heft/reports/scannet_video",
    )
    p.add_argument("--chunk-size", type=int, default=25)
    p.add_argument("--heads", nargs="+", type=int, default=list(range(12)))
    p.add_argument("--modes", nargs="+", choices=["kk", "qk"], default=["kk", "qk"])
    p.add_argument("--noise-mode", choices=["heft", "legacy"], default="heft",
                   help="HEFT checkpoint noise, or one exact noise point from the legacy schedule")
    return p


def validate_args(args):
    if args.chunk_size < 1 or (args.chunk_size - 1) % 4:
        raise ValueError("The local video VAE requires chunk_size=4n+1 (default 25)")
    if args.frames < 2 or tuple(args.size) != WAN_2_1.resolution:
        raise ValueError("Require >=2 sampled frames and HEFT's 480x832 resolution")
    if args.dtype != "bf16" or args.batch_size != 1 or args.save_prepool:
        raise ValueError("Video mode uses HEFT bf16, one video per forward, pooled caches")
    if args.noise_mode == "heft":
        if args.timesteps != [WAN_2_1.start_step] or args.shift != 3.0:
            raise ValueError("HEFT mode uses capture step 49 and shift 3; select --noise-mode legacy for VEGA noise")
    elif len(args.timesteps) != 1 or not 0 <= args.timesteps[0] <= 1000 or not np.isfinite(args.shift) or args.shift <= 0:
        raise ValueError("Legacy noise requires one requested timestep in [0,1000] and positive finite shift")
    if not args.blocks or min(args.blocks) < 0 or max(args.blocks) >= 30:
        raise ValueError("Block indices must be in [0,29]")
    if not args.heads or min(args.heads) < 0 or max(args.heads) >= 12:
        raise ValueError("Head indices must be in [0,11]")
    if any(n <= 0 for n in args.grid) or not 0 < args.min_coverage <= 1:
        raise ValueError("Invalid grid or depth coverage")
    if min(args.voxel_size, args.hit_distance, args.negative_min_distance) <= 0:
        raise ValueError("Distance thresholds must be positive")
    args.blocks, args.heads, args.modes = sorted(set(args.blocks)), sorted(set(args.heads)), sorted(set(args.modes))


def score_features(features, pairs, info, args, noise, block, query=None, head=None, mode=None):
    rows = []
    for i, pair in enumerate(pairs):
        options = {}
        if mode == "qk":
            options = {"similarity_kind": "scaled_dot", "query_features_i": query[i], "query_features_j": query[i + 1]}
        values = evaluate_pair(
            features[i], features[i + 1], pair, args.seed + i,
            args.negative_min_distance, args.hit_distance, **options,
        )
        record = {
            "scene": info["scene"], "timestep": noise["requested_timestep"], "block": block,
            "frame_i": info["sampled_frame_ids"][i], "frame_j": info["sampled_frame_ids"][i + 1],
            "actual_timestep": noise["actual_timestep"], "sigma": noise["sigma"], **values,
        }
        if head is not None:
            record.update(head=head, mode=mode)
        rows.append(record)
    return rows


def run(args):
    validate_args(args)
    torch.set_num_threads(2)
    workspace = Path(__file__).resolve().parents[3]
    offline_environment(workspace)
    args.model = args.model.resolve()
    scenes = [SensScene(path) for path in discover_scenes(args.data_root, args.scenes)]
    noise = noise_protocol(args.model, args.noise_mode, args.timesteps[0], args.shift)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in {"cache_root", "output_root", "report_root", "no_previews", "geometry_only"}}
    config.update(
        schema="heft.scannet_video", schema_version=1, model_identity=model_identity(args.model),
        scene_sources=[scene.signature() for scene in scenes], requested_noise=[noise],
        timestep_kind=noise["timestep_kind"], comparison="adjacent sampled frames; chunk boundaries marked",
        aggregation="pair macro -> scene macro", code=source_identity([
            "dataset.py", "geometry.py", "metrics.py", "scan.py", "report.py",
            "video_extract.py", "video_noise.py", "extract.py", "video_scan.py", "video_report.py",
        ]), runtime_code=runtime_sources(),
        video_protocol={
            "pipeline": "WanPipeline", "num_inference_steps": noise["num_inference_steps"],
            "capture_step": noise["capture_step"], "guidance_scale": WAN_2_1.guidance_scale,
            "active_denoising_steps": 1, "updates_before_capture": 0,
            "vae": "joint video encoding; temporal downsampling disabled, causal convolution retained",
            "latent_distribution": "sample", "prompt": "", "negative_prompt": "",
            "tail_policy": "repeat last RGB frame before VAE; trim descriptors after capture",
            "sampling": "uniform valid observations, preserving sampled-frame order",
            "capture": "conditional full block output and per-head Q/K after normalization and RoPE",
        },
    )
    out, report_dir = args.output_root / digest(config), args.report_root / digest(config)
    write_json(out / "config.json", config)
    write_json(out / "status.json", {"status": "running"})
    print(f"Video results: {out}\nVideo report: {report_dir}", flush=True)
    started, sampled, extractor = time.monotonic(), [], None
    try:
        for scene in scenes:
            images, pairs, info = prepare_scene(scene, args, args.cache_root)
            jobs = video_jobs(images, scene.name, args.seed, args.chunk_size)
            info["video_chunks"] = chunk_records(jobs, info["sampled_frame_ids"])
            identity = {
                "geometry": info["geometry_cache"], "model": config["model_identity"],
                "code": config["code"], "runtime_code": config["runtime_code"],
                "protocol": config["video_protocol"], "noise": noise,
                "chunks": info["video_chunks"], "dtype": args.dtype, "device": args.device,
                "blocks": args.blocks, "heads": args.heads, "grid": args.grid,
            }
            key = digest(identity)
            info.update(video_feature_key=key, video_feature_identity=identity)
            sampled.append(info)
            write_json(out / "sampled_frames.json", sampled)
            print(f"{scene.name}: {len(images)} frames, {len(jobs)} video chunks, "
                  f"{sum(bool(p['shared']) for p in pairs)}/{len(pairs)} valid pairs", flush=True)
            if args.geometry_only:
                continue
            cache = args.cache_root / "video_features" / key
            complete = cache / "metadata.json"
            if complete.exists():
                meta = read_json(complete)
                if meta["identity"] != identity or meta["feature_key"] != key:
                    raise ValueError("Video feature cache identity mismatch")
            if not complete.exists() or not all((cache / f"block_{b}.npz").exists() for b in args.blocks):
                audits = []
                for job in jobs:
                    chunk_dir = cache / "chunks" / str(job.chunk)
                    marker = chunk_dir / "audit.json"
                    if not marker.exists() or not all((chunk_dir / f"block_{b}.npz").exists() for b in args.blocks):
                        if extractor is None:
                            extractor = WanVideoExtractor(args.model, args.device, noise)
                        captured, audit = extractor.extract(job, args.blocks, args.heads, tuple(args.grid))
                        for forward in audit["transformer_forwards"]:
                            if (forward["timestep"] != noise["actual_timestep"]
                                    or forward["sigma"] != noise["sigma"]
                                    or forward["updates_before_forward"] != 0):
                                raise ValueError("Observed pipeline noise differs from the single-step probe")
                        for block, tensors in captured.items():
                            atomic_npz(chunk_dir / f"block_{block}.npz", **{k: t.numpy() for k, t in tensors.items()})
                        write_json(marker, {"feature_key": key, "chunk": info["video_chunks"][job.chunk], **audit})
                        del captured
                    audit = read_json(marker)
                    if audit["feature_key"] != key or audit["chunk"] != info["video_chunks"][job.chunk]:
                        raise ValueError("Chunk provenance changed")
                    audits.append(audit)
                    print(f"{scene.name}: chunk {job.chunk} captured as {audit['transformer_forwards'][0]['shape']}", flush=True)
                for block in args.blocks:
                    arrays = {kind: [] for kind in ("features", "query", "key")}
                    for job in jobs:
                        with np.load(cache / "chunks" / str(job.chunk) / f"block_{block}.npz", allow_pickle=False) as archive:
                            for kind in arrays:
                                arrays[kind].append(archive[kind].copy())
                    merged = {kind: np.concatenate(values) for kind, values in arrays.items()}
                    atomic_npz(cache / f"block_{block}.npz", **merged)
                    del arrays, merged
                write_json(complete, {"identity": identity, "feature_key": key, "noise": noise,
                                      "heads": args.heads, "audits": audits,
                                      "versions": extractor.versions if extractor else {}})
            for block in args.blocks:
                with np.load(cache / f"block_{block}.npz", allow_pickle=False) as archive:
                    hidden, query, keys = [archive[k].copy() for k in ("features", "query", "key")]
                if hidden.shape != (len(images), 1536, *args.grid) or query.shape != (len(images), len(args.heads), 128, *args.grid) or keys.shape != query.shape:
                    raise ValueError("Cached video descriptor shape mismatch")
                if any(not np.isfinite(x).all() for x in (hidden, query, keys)):
                    raise ValueError("Non-finite cached video descriptors")
                rows = score_features(hidden.transpose(0, 2, 3, 1), pairs, info, args, noise, block)
                write_json(out / "pairs" / f"{scene.name}_k{noise['requested_timestep']}_b{block}.json", rows)
                for index, head in enumerate(args.heads):
                    q, k = query[:, index].transpose(0, 2, 3, 1), keys[:, index].transpose(0, 2, 3, 1)
                    for mode in args.modes:
                        rows = score_features(k, pairs, info, args, noise, block, query=q, head=head, mode=mode)
                        write_json(out / "head_pairs" / f"{scene.name}_b{block}_h{head}_{mode}.json", rows)
                if not args.no_previews:
                    report_dir.mkdir(parents=True, exist_ok=True)
                    pca_preview(hidden, images, report_dir / f"{scene.name}_b{block}_pca.png")
                del hidden, query, keys
            generate_video(out, report_dir)
        if not args.geometry_only:
            generate_video(out, report_dir)
        write_json(out / "status.json", {
            "status": "geometry_complete" if args.geometry_only else "complete",
            "elapsed_seconds": time.monotonic() - started,
            "scenes_prepared": len(sampled), "scenes_requested": len(scenes),
            "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated(args.device) / 2**20 if extractor else 0,
        })
    except Exception as error:
        write_json(out / "status.json", {"status": "failed", "error": str(error),
                                         "elapsed_seconds": time.monotonic() - started})
        raise
    print(f"Video scan complete: {out}", flush=True)
    return out


if __name__ == "__main__":
    run(parser().parse_args())
