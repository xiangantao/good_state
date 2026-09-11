"""Offline ScanNet QK/KK head scan; run from heft with python -m scripts.scannet.head_scan."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from .dataset import SensScene, discover_scenes
from .extract import offline_environment, selected_noise
from .head_extract import WanHeadExtractor
from .head_report import generate_heads
from .metrics import evaluate_pair
from .report import matching_preview, pca_preview, write_json
from .scan import (
    atomic_npz,
    digest,
    model_identity,
    prepare_scene,
    source_identity,
)
from .scan import (
    parser as block_parser,
)


def parser():
    workspace = Path(__file__).resolve().parents[3]
    p = block_parser()
    p.description = "离线 ScanNet / Wan 两层全部 attention head 的 QK/KK 空间扫描"
    p.set_defaults(
        blocks=[14, 15],
        output_root=workspace / "eval/scannet_heads",
        report_root=workspace / "heft/reports/scannet_heads",
    )
    p.add_argument("--heads", type=int, nargs="+", default=list(range(12)))
    p.add_argument("--modes", nargs="+", choices=["qk", "kk"], default=["qk", "kk"])
    return p


def run(args):
    workspace = Path(__file__).resolve().parents[3]
    offline_environment(workspace)
    torch.set_num_threads(4)
    if args.geometry_only or args.save_prepool:
        raise ValueError(
            "Use block scan for geometry-only; head scan stores pooled Q/K only"
        )
    if args.frames < 2 or any(n <= 0 or n % 16 for n in args.size):
        raise ValueError(
            "Require frames>=2 and image dimensions positive and divisible by16"
        )
    if any(n <= 0 for n in args.grid) or not 0 < args.min_coverage <= 1:
        raise ValueError("Invalid grid or depth coverage")
    if (
        args.batch_size < 1
        or min(args.voxel_size, args.hit_distance, args.negative_min_distance) <= 0
    ):
        raise ValueError("Batch size and distances must be positive")
    for values, limit in [(args.blocks, 30), (args.heads, 12)]:
        if not values or any(i < 0 or i >= limit for i in values):
            raise ValueError("Invalid block/head indices")
    args.blocks, args.heads, args.modes = (
        sorted(set(x)) for x in (args.blocks, args.heads, args.modes)
    )
    for timestep in args.timesteps:
        selected_noise(timestep, args.shift)
    scenes = [SensScene(d) for d in discover_scenes(args.data_root, args.scenes)]
    model_id = model_identity(args.model.resolve())
    local_transformer = (
        workspace
        / "heft/diffusers/src/diffusers/models/transformers/transformer_wan.py"
    )
    processor_hash = hashlib.sha256(local_transformer.read_bytes()).hexdigest()
    config = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k not in {"output_root", "report_root", "cache_root", "no_previews"}
    }
    config.update(
        schema_version=1,
        model_identity=model_id,
        scene_sources=[s.signature() for s in scenes],
        code=source_identity(
            [
                "dataset.py",
                "geometry.py",
                "extract.py",
                "metrics.py",
                "scan.py",
                "report.py",
                "head_extract.py",
                "head_report.py",
                "head_scan.py",
            ]
        ),
        local_wan_processor_sha256=processor_hash,
        capture="attn1 Q/K after normalization and RoPE; pool each head independently",
        qk="scaled_dot; separate Q_i K_j and Q_j K_i",
        kk="cosine K_i K_j",
        comparison="adjacent sampled frames",
        aggregation="pair macro -> scene macro",
        requested_noise=[selected_noise(k, args.shift) for k in args.timesteps],
    )
    run_id = digest(config)
    out, report_dir = args.output_root / run_id, args.report_root / run_id
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", config)
    write_json(out / "status.json", {"status": "running"})
    print(f"Head实验目录：{out}\nHead报告目录：{report_dir}", flush=True)
    sampled, errors, extractor = [], [], None
    started = time.monotonic()
    for scene in scenes:
        try:
            images, pairs, info = prepare_scene(scene, args, args.cache_root)
            sampled.append(info)
            write_json(out / "sampled_frames.json", sampled)
            print(
                f"{scene.name}：{len(images)}帧，{sum(bool(p['shared']) for p in pairs)}/{len(pairs)}对有共享体素。",
                flush=True,
            )
            for timestep in args.timesteps:

                def pair_path(block, head, mode, scene_name=scene.name, step=timestep):
                    return (
                        out
                        / "pairs"
                        / f"{scene_name}_k{step}_b{block}_h{head}_{mode}.json"
                    )

                # Keep the layer scan's latent identity exactly, preserving its cache.
                identity = {
                    "geometry": info["geometry_cache"],
                    "model": model_id,
                    "code": source_identity(["extract.py"]),
                    "dtype": args.dtype,
                    "device": args.device,
                }
                feature_identity = {
                    **identity,
                    "seed": args.seed,
                    "k": timestep,
                    "shift": args.shift,
                    "batch": args.batch_size,
                    "head_code": source_identity(["head_extract.py"]),
                    "processor": processor_hash,
                    "capture": config["capture"],
                }
                cache = args.cache_root / "head_features" / digest(feature_identity)
                cache.mkdir(parents=True, exist_ok=True)
                needed = [
                    b
                    for b in args.blocks
                    if not all(
                        pair_path(b, h, m).exists()
                        for h in args.heads
                        for m in args.modes
                    )
                ]
                if not needed and args.no_previews:
                    print(f"{scene.name} k={timestep}：评分齐全，跳过。", flush=True)
                    continue
                # Existing feature caches also let a resume restore missing previews.
                missing = [
                    b for b in args.blocks if not (cache / f"block_{b}.npz").exists()
                ]
                noise_info = selected_noise(timestep, args.shift)
                if missing:
                    if extractor is None:
                        extractor = WanHeadExtractor(
                            args.model, args.device, args.dtype
                        )
                    latent_path = (
                        args.cache_root / "latents" / (digest(identity) + ".npz")
                    )
                    if latent_path.exists():
                        with np.load(latent_path, allow_pickle=False) as f:
                            latents = torch.from_numpy(f["latents"].copy())
                        print(f"{scene.name}：复用已有VAE缓存。", flush=True)
                    else:
                        latents = extractor.latents(images)
                        atomic_npz(latent_path, latents=latents.numpy())
                    captured, noise_info = extractor.heads(
                        latents,
                        missing,
                        timestep,
                        args.shift,
                        args.seed,
                        scene.name,
                        info["sampled_frame_ids"],
                        args.batch_size,
                        tuple(args.grid),
                    )
                    for block, kinds in captured.items():
                        atomic_npz(
                            cache / f"block_{block}.npz",
                            **{k: v.numpy() for k, v in kinds.items()},
                        )
                    write_json(
                        cache / "metadata.json",
                        {
                            "identity": feature_identity,
                            "noise": noise_info,
                            "versions": extractor.versions,
                        },
                    )
                    del latents, captured
                for block in args.blocks:
                    with np.load(cache / f"block_{block}.npz", allow_pickle=False) as f:
                        queries, keys = f["query"].copy(), f["key"].copy()
                    for head in args.heads:
                        # T,D,H,W -> T,H,W,D, one actual attention head.
                        q = queries[:, head].transpose(0, 2, 3, 1)
                        k = keys[:, head].transpose(0, 2, 3, 1)
                        pca_path = (
                            report_dir
                            / f"{scene.name}_k{timestep}_b{block}_h{head}_key_pca.png"
                        )
                        if not args.no_previews and not pca_path.exists():
                            pca_preview(keys[:, head], images, pca_path)
                        for mode in args.modes:
                            path = pair_path(block, head, mode)
                            similarity_kind = "scaled_dot" if mode == "qk" else "cosine"
                            if path.exists():
                                import json

                                rows = json.loads(path.read_text())
                            else:
                                rows = []
                                for i, geometry in enumerate(pairs):
                                    options = {"similarity_kind": similarity_kind}
                                    if mode == "qk":
                                        options.update(
                                            query_features_i=q[i],
                                            query_features_j=q[i + 1],
                                        )
                                    values = evaluate_pair(
                                        k[i],
                                        k[i + 1],
                                        geometry,
                                        args.seed + i,
                                        args.negative_min_distance,
                                        args.hit_distance,
                                        **options,
                                    )
                                    rows.append(
                                        {
                                            "scene": scene.name,
                                            "timestep": timestep,
                                            "block": block,
                                            "head": head,
                                            "mode": mode,
                                            "frame_i": info["sampled_frame_ids"][i],
                                            "frame_j": info["sampled_frame_ids"][i + 1],
                                            "actual_timestep": noise_info[
                                                "actual_timestep"
                                            ],
                                            "sigma": noise_info["sigma"],
                                            **values,
                                        }
                                    )
                                write_json(path, rows)
                            if not args.no_previews:
                                eligible = [
                                    (i, r["retrieval_world_error_mean"])
                                    for i, r in enumerate(rows)
                                    if r["retrieval_world_error_mean"] is not None
                                ]
                                if eligible:
                                    for label, idx in [
                                        ("best", min(eligible, key=lambda x: x[1])[0]),
                                        ("worst", max(eligible, key=lambda x: x[1])[0]),
                                    ]:
                                        preview = (
                                            report_dir / f"{path.stem}_{label}.png"
                                        )
                                        if not preview.exists():
                                            matching_preview(
                                                k[idx],
                                                k[idx + 1],
                                                images[idx : idx + 2],
                                                pairs[idx],
                                                preview,
                                                args.hit_distance,
                                                query_features=q[idx]
                                                if mode == "qk"
                                                else None,
                                                similarity_kind=similarity_kind,
                                            )
                        print(
                            f"{scene.name} block={block} head={head}：评分及效果图已保存。",
                            flush=True,
                        )
                    del queries, keys
                generate_heads(out, report_dir)
        except Exception as error:  # noqa: BLE001 — record experiment failure and stop
            import traceback

            errors.append(
                {"scene": scene.name, "type": type(error).__name__, "error": str(error)}
            )
            write_json(out / "errors.json", errors)
            traceback.print_exc()
            break
    generate_heads(out, report_dir)
    write_json(
        out / "status.json",
        {
            "status": "failed" if errors else "complete",
            "errors": errors,
            "elapsed_seconds": time.monotonic() - started,
            "scenes_prepared": len(sampled),
            "scenes_requested": len(scenes),
        },
    )
    if errors:
        raise SystemExit(1)
    print("Head扫描完成，结果与效果图已保存。", flush=True)
    return out


if __name__ == "__main__":
    run(parser().parse_args())
