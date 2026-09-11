"""Run from heft/: python -m scripts.scannet.scan --help."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .dataset import SensScene, discover_scenes
from .geometry import pair_geometry, register_geometry, resize_rgb
from .metrics import evaluate_pair
from .report import generate, matching_preview, pca_preview, write_json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def source_identity(names):
    base = Path(__file__).parent
    return {
        name: hashlib.sha256((base / name).read_bytes()).hexdigest() for name in names
    }


def model_identity(model):
    files = []
    for p in sorted(model.rglob("*")):
        if p.is_file() and p.suffix in {".json", ".safetensors", ".model"}:
            s = p.stat()
            files.append([str(p.relative_to(model)), s.st_size, s.st_mtime_ns])
    if not files:
        raise ValueError(f"No local checkpoint files: {model}")
    return {"directory": str(model), "files": files}


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as f:
        np.savez_compressed(f, **arrays)
    temporary.replace(path)


def prepare_scene(scene, args, cache_root):
    selected = scene.sample(args.frames)
    info = scene.describe(selected)
    key = digest(
        {
            "source": scene.signature(),
            "ids": info["sampled_frame_ids"],
            "size": args.size,
            "grid": args.grid,
            "coverage": args.min_coverage,
            "code": source_identity(["dataset.py", "geometry.py"]),
        }
    )
    directory = cache_root / "geometry" / key
    directory.mkdir(parents=True, exist_ok=True)
    geometry_path = directory / "geometry.npz"
    image_paths = [directory / f"rgb_{f.index:06d}.png" for f in selected]
    if not geometry_path.exists() or not all(p.exists() for p in image_paths):
        coords, valid, coverage = [], [], []
        for frame, path in zip(selected, image_paths):
            rgb, depth = scene.decode(frame)
            xyz, mask, ratio = register_geometry(
                scene,
                frame,
                depth,
                tuple(args.size),
                tuple(args.grid),
                args.min_coverage,
            )
            coords.append(xyz)
            valid.append(mask)
            coverage.append(ratio)
            resize_rgb(rgb, tuple(args.size)).save(path)
        atomic_npz(
            geometry_path,
            xyz=np.stack(coords),
            valid=np.stack(valid),
            coverage=np.stack(coverage),
        )
    with np.load(geometry_path, allow_pickle=False) as data:
        xyz, valid, coverage = [data[k].copy() for k in ["xyz", "valid", "coverage"]]
    images = [Image.open(p).convert("RGB") for p in image_paths]
    pairs = [
        pair_geometry(xyz[i], xyz[i + 1], valid[i], valid[i + 1], args.voxel_size)
        for i in range(len(selected) - 1)
    ]
    info.update(
        geometry_cache=key,
        valid_tokens_per_frame=valid.reshape(len(valid), -1).sum(1).tolist(),
        coverage_mean_per_frame=coverage.reshape(len(coverage), -1).mean(1).tolist(),
        shared_voxels_per_adjacent_pair=[len(p["shared"]) for p in pairs],
    )
    return images, pairs, info


def parser():
    workspace = Path(__file__).resolve().parents[3]
    p = argparse.ArgumentParser(
        description="离线 ScanNet / Wan 相邻采样帧、多层空间特征扫描"
    )
    p.add_argument(
        "--data-root", type=Path, default=workspace / "datasets/ScanNet/v2/scans_test"
    )
    p.add_argument(
        "--model", type=Path, default=workspace / "models/Wan2.1-T2V-1.3B-Diffusers"
    )
    p.add_argument("--scenes", nargs="+")
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--blocks", type=int, nargs="+", default=[10, 12, 15, 20, 25, 28])
    p.add_argument("--timesteps", type=int, nargs="+", default=[300])
    p.add_argument("--size", type=int, nargs=2, default=[480, 832], metavar=("H", "W"))
    p.add_argument("--grid", type=int, nargs=2, default=[14, 14], metavar=("H", "W"))
    p.add_argument("--voxel-size", type=float, default=0.1)
    p.add_argument("--min-coverage", type=float, default=0.1)
    p.add_argument("--negative-min-distance", type=float, default=0.2)
    p.add_argument("--hit-distance", type=float, default=0.1)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--cache-root", type=Path, default=workspace / "cache/scannet")
    p.add_argument("--output-root", type=Path, default=workspace / "eval/scannet")
    p.add_argument(
        "--report-root", type=Path, default=workspace / "heft/reports/scannet"
    )
    p.add_argument(
        "--geometry-only", action="store_true", help="只检查数据与几何，不加载模型"
    )
    p.add_argument("--save-prepool", action="store_true")
    p.add_argument("--no-previews", action="store_true")
    return p


def run(args):
    import torch

    torch.set_num_threads(4)
    if args.frames < 2 or any(n <= 0 or n % 16 for n in args.size):
        raise ValueError(
            "frames>=2 and positive image dimensions divisible by 16 required"
        )
    if any(n <= 0 for n in args.grid) or not 0 < args.min_coverage <= 1:
        raise ValueError("Invalid feature grid or depth coverage")
    if (
        args.batch_size < 1
        or min(args.voxel_size, args.hit_distance, args.negative_min_distance) <= 0
    ):
        raise ValueError("Batch size and distance thresholds must be positive")
    if not args.blocks or min(args.blocks) < 0 or max(args.blocks) >= 30:
        raise ValueError("Wan-1.3B block indices must be in [0,29]")
    from .extract import offline_environment, selected_noise

    offline_environment(Path(__file__).resolve().parents[3])
    for k in args.timesteps:
        selected_noise(k, args.shift)
    directories = discover_scenes(args.data_root, args.scenes)
    scenes = [SensScene(d) for d in directories]
    args.model = args.model.resolve()
    model_id = model_identity(args.model)
    config = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k
        not in {
            "geometry_only",
            "no_previews",
            "output_root",
            "report_root",
            "cache_root",
        }
    }
    config.update(
        schema_version=1,
        scene_sources=[s.signature() for s in scenes],
        code=source_identity(
            [
                "dataset.py",
                "geometry.py",
                "extract.py",
                "metrics.py",
                "scan.py",
                "report.py",
            ]
        ),
        model_identity=model_id,
        comparison="adjacent sampled frames",
        aggregation="pair macro -> scene macro (not paper all-view-pair score)",
        requested_noise=[selected_noise(k, args.shift) for k in args.timesteps],
    )
    run_id = digest(config)
    out = args.output_root / run_id
    report_dir = args.report_root / run_id
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", config)
    print(f"实验目录：{out}", flush=True)
    print(f"报告目录：{report_dir}", flush=True)
    extractor = None
    sampled, errors = [], []
    start = time.monotonic()
    for scene in scenes:
        try:
            print(f"正在准备 {scene.name} 的采样帧与几何对应。", flush=True)
            images, pairs, info = prepare_scene(scene, args, args.cache_root)
            sampled.append(info)
            write_json(out / "sampled_frames.json", sampled)
            print(
                f"{scene.name}：原始{len(scene.frames)}帧；采样{len(images)}帧；"
                f"共享体素数={[len(p['shared']) for p in pairs]}。",
                flush=True,
            )
            if args.geometry_only:
                continue
            for timestep in args.timesteps:
                if all(
                    (out / "pairs" / f"{scene.name}_k{timestep}_b{block}.json").exists()
                    for block in args.blocks
                ):
                    print(
                        f"{scene.name} k={timestep}：已有完整结果，跳过模型计算。",
                        flush=True,
                    )
                    continue
                identity = {
                    "geometry": info["geometry_cache"],
                    "model": model_id,
                    "code": source_identity(["extract.py"]),
                    "dtype": args.dtype,
                    "device": args.device,
                }
                latent_key = digest(identity)
                feature_key = digest(
                    {
                        **identity,
                        "seed": args.seed,
                        "k": timestep,
                        "shift": args.shift,
                        "batch": args.batch_size,
                    }
                )
                cache = args.cache_root / "features" / feature_key
                cache.mkdir(parents=True, exist_ok=True)
                missing = [
                    b
                    for b in args.blocks
                    if not (cache / f"block_{b}.npz").exists()
                    or (
                        args.save_prepool
                        and not (cache / f"block_{b}_prepool.npz").exists()
                    )
                ]
                noise_info = selected_noise(timestep, args.shift)
                if missing:
                    import torch

                    from .extract import WanExtractor

                    if extractor is None:
                        extractor = WanExtractor(args.model, args.device, args.dtype)
                    latent_path = args.cache_root / "latents" / (latent_key + ".npz")
                    if latent_path.exists():
                        with np.load(latent_path, allow_pickle=False) as f:
                            latents = torch.from_numpy(f["latents"].copy())
                    else:
                        print(
                            f"{scene.name}：正在编码采样帧的 VAE latent。", flush=True
                        )
                        latents = extractor.latents(images)
                        atomic_npz(latent_path, latents=latents.numpy())
                    features, prepool, noise_info = extractor.blocks(
                        latents,
                        missing,
                        timestep,
                        args.shift,
                        args.seed,
                        scene.name,
                        info["sampled_frame_ids"],
                        args.batch_size,
                        tuple(args.grid),
                        args.save_prepool,
                    )
                    for block, feature in features.items():
                        atomic_npz(
                            cache / f"block_{block}.npz", features=feature.numpy()
                        )
                    for block, feature in prepool.items():
                        atomic_npz(
                            cache / f"block_{block}_prepool.npz",
                            features=feature.float().numpy().astype(np.float16),
                        )
                    write_json(
                        cache / "metadata.json",
                        {
                            "identity": identity,
                            "noise": noise_info,
                            "feature_key": feature_key,
                            "versions": extractor.versions,
                        },
                    )
                    del latents, features, prepool
                for block in args.blocks:
                    pair_path = (
                        out / "pairs" / f"{scene.name}_k{timestep}_b{block}.json"
                    )
                    if pair_path.exists():
                        continue
                    with np.load(cache / f"block_{block}.npz", allow_pickle=False) as f:
                        features = f["features"].copy()
                    tokens = features.transpose(0, 2, 3, 1)
                    rows = []
                    for i, pair in enumerate(pairs):
                        values = evaluate_pair(
                            tokens[i],
                            tokens[i + 1],
                            pair,
                            args.seed + i,
                            args.negative_min_distance,
                            args.hit_distance,
                        )
                        rows.append(
                            {
                                "scene": scene.name,
                                "timestep": timestep,
                                "block": block,
                                "frame_i": info["sampled_frame_ids"][i],
                                "frame_j": info["sampled_frame_ids"][i + 1],
                                "actual_timestep": noise_info["actual_timestep"],
                                "sigma": noise_info["sigma"],
                                **values,
                            }
                        )
                    write_json(pair_path, rows)
                    if not args.no_previews:
                        prefix = report_dir / f"{scene.name}_k{timestep}_b{block}"
                        pca_preview(features, images, str(prefix) + "_pca.png")
                        # Show both a best and worst geometric-error pair where available.
                        eligible = [
                            (i, row["retrieval_world_error_mean"])
                            for i, row in enumerate(rows)
                            if row["retrieval_world_error_mean"] is not None
                        ]
                        if eligible:
                            for label, idx in [
                                ("best", min(eligible, key=lambda x: x[1])[0]),
                                ("worst", max(eligible, key=lambda x: x[1])[0]),
                            ]:
                                matching_preview(
                                    tokens[idx],
                                    tokens[idx + 1],
                                    images[idx : idx + 2],
                                    pairs[idx],
                                    str(prefix) + f"_{label}.png",
                                    hit_distance=args.hit_distance,
                                )
                    print(
                        f"{scene.name} k={timestep} block={block}：评分已保存。",
                        flush=True,
                    )
                generate(out, report_dir)
        except Exception as error:  # noqa: BLE001 — persist scene failure before stopping
            import traceback

            errors.append(
                {"scene": scene.name, "type": type(error).__name__, "error": str(error)}
            )
            write_json(out / "errors.json", errors)
            traceback.print_exc()
            # Runtime/dependency failures are shared across scenes; stop instead
            # of retrying the same failed model load for every scene.
            if not args.geometry_only:
                break
    if not args.geometry_only:
        generate(out, report_dir)
    write_json(
        out / "status.json",
        {
            "status": "failed"
            if errors
            else "geometry_complete"
            if args.geometry_only
            else "complete",
            "elapsed_seconds": time.monotonic() - start,
            "errors": errors,
            "scenes_prepared": len(sampled),
            "scenes_requested": len(scenes),
        },
    )
    print(
        "几何检查完成。" if args.geometry_only else "扫描结束，结果和报告已保存。",
        flush=True,
    )
    if errors:
        raise SystemExit(1)
    return out


if __name__ == "__main__":
    run(parser().parse_args())
