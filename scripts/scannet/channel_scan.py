"""Cache-only, leave-one-scene-out scan of original block-hidden channels."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .channel_selection import (
    ABLATION_METRICS, AblationScene, fit_pca, rank_channels, scene_ablation,
    scene_moments, scene_retrieval, training_scene_names,
)
from .geometry import pair_geometry
from .metrics import evaluate_pair
from .report import METRICS, write_json
from .scan import atomic_npz, digest, source_identity


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def read_json(path):
    return json.loads(Path(path).read_text())


def file_identity(path):
    return {
        "path": str(path.resolve()), "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def load_cached_scenes(args):
    original = read_json(args.baseline_run / "config.json")
    sampled = read_json(args.baseline_run / "sampled_frames.json")
    if args.block not in original["blocks"] or args.timestep not in original["timesteps"]:
        raise ValueError("Requested block/timestep is absent from the baseline run")
    scenes, provenance = {}, []
    for info in sampled:
        name = info["scene"]
        if name in scenes:
            raise ValueError(f"Repeated scene: {name}")
        identity = {
            "geometry": info["geometry_cache"], "model": original["model_identity"],
            "code": {"extract.py": original["code"]["extract.py"]},
            "dtype": original["dtype"], "device": original["device"],
        }
        key = digest({
            **identity, "seed": original["seed"], "k": args.timestep,
            "shift": original["shift"], "batch": original["batch_size"],
        })
        feature_dir = args.cache_root / "features" / key
        metadata = read_json(feature_dir / "metadata.json")
        expected_noise = next(
            item for item in original["requested_noise"]
            if item["requested_timestep"] == args.timestep
        )
        if (metadata["identity"] != identity or metadata["feature_key"] != key
                or metadata["noise"] != expected_noise):
            raise ValueError(f"Feature cache identity/noise mismatch for {name}")
        feature_path = feature_dir / f"block_{args.block}.npz"
        geometry_path = args.cache_root / "geometry" / info["geometry_cache"] / "geometry.npz"
        archive_path = args.baseline_run / "pairs" / f"{name}_k{args.timestep}_b{args.block}.json"
        with np.load(feature_path, allow_pickle=False) as f:
            features = f["features"].copy().transpose(0, 2, 3, 1)
        with np.load(geometry_path, allow_pickle=False) as g:
            xyz, valid = g["xyz"].copy(), g["valid"].copy()
        ids = info["sampled_frame_ids"]
        if features.shape[:3] != (len(ids), *original["grid"]):
            raise ValueError(f"Unexpected feature grid for {name}: {features.shape}")
        if xyz.shape != (*features.shape[:3], 3) or valid.shape != features.shape[:3]:
            raise ValueError(f"Geometry/feature shape mismatch for {name}")
        if not np.isfinite(features).all():
            raise ValueError(f"Non-finite features for {name}")
        if valid.reshape(len(ids), -1).sum(1).tolist() != info["valid_tokens_per_frame"]:
            raise ValueError(f"Changed valid-token support for {name}")
        pairs = [
            pair_geometry(xyz[i], xyz[i + 1], valid[i], valid[i + 1], original["voxel_size"])
            for i in range(len(ids) - 1)
        ]
        if [len(p["shared"]) for p in pairs] != info["shared_voxels_per_adjacent_pair"]:
            raise ValueError(f"Changed shared-voxel support for {name}")
        scenes[name] = {
            "features": features, "pairs": pairs, "valid": valid,
            "frame_ids": ids, "archived": read_json(archive_path),
        }
        provenance.append({
            "scene": name, "feature_key": key, "metadata": metadata,
            "geometry_key": info["geometry_cache"], "frame_ids": ids,
            "shape_thwc": list(features.shape),
            "files": [file_identity(p) for p in (feature_path, geometry_path, archive_path)],
        })
        log(f"Loaded {name}: {features.shape}, {sum(bool(p['shared']) for p in pairs)} valid pairs")
    if len(scenes) < 3 or len({s["features"].shape[-1] for s in scenes.values()}) != 1:
        raise ValueError("Need >= 3 scenes with the same channel count")
    return original, scenes, provenance


def score_scene(name, scene, features, protocol, method, repeat=0):
    rows = []
    for i, pair in enumerate(scene["pairs"]):
        values = evaluate_pair(
            features[i], features[i + 1], pair, protocol["seed"] + i,
            protocol["negative_min_distance"], protocol["hit_distance"],
        )
        rows.append({
            "scene": name, "method": method, "dimensions": features.shape[-1],
            "repeat": repeat, "frame_i": scene["frame_ids"][i],
            "frame_j": scene["frame_ids"][i + 1], **values,
        })
    return rows


def check_baseline(scenes, protocol, out, tolerance=1e-6):
    failures, rows = [], []
    max_delta = dict.fromkeys(METRICS, 0.0)
    for name, scene in scenes.items():
        actual = score_scene(name, scene, scene["features"], protocol, "original")
        if len(actual) != len(scene["archived"]):
            raise ValueError(f"Archived pair count mismatch for {name}")
        for index, (new, old) in enumerate(zip(actual, scene["archived"])):
            for key, expected in old.items():
                if key in {"timestep", "block", "actual_timestep", "sigma"}:
                    continue
                value = new[key]
                if key in METRICS and value is not None and expected is not None:
                    delta = abs(value - expected)
                    max_delta[key] = max(max_delta[key], delta)
                    equal = delta <= tolerance
                else:
                    equal = value == expected
                if not equal:
                    failures.append({
                        "scene": name, "pair": index, "field": key,
                        "expected": expected, "actual": value,
                    })
        write_json(out / "pairs" / f"{name}_original_{scene['features'].shape[-1]}_0.json", actual)
        rows.extend(actual)
    write_json(out / "baseline_check.json", {
        "passed": not failures, "pairs_checked": len(rows),
        "valid_pairs": sum(r["status"] == "ok" for r in rows),
        "absolute_tolerance": tolerance, "maximum_metric_delta": max_delta,
        "failures": failures,
    })
    if failures:
        raise ValueError(f"Baseline mismatch: {len(failures)} fields; see baseline_check.json")
    log(f"Baseline verified: {len(rows)} pairs; max delta {max(max_delta.values()):.3g}")
    return rows


def cached_score(out, name, scene, features, protocol, method, repeat=0):
    path = out / "pairs" / f"{name}_{method}_{features.shape[-1]}_{repeat}.json"
    if path.exists():
        rows = read_json(path)
        if len(rows) != len(scene["pairs"]):
            raise ValueError(f"Incomplete score file: {path}")
        return rows
    rows = score_scene(name, scene, features, protocol, method, repeat)
    write_json(path, rows)
    return rows


def cached_ablation(path, scene, channels, args, protocol):
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if not np.array_equal(data["channels"], channels):
                raise ValueError(f"Ablation channels changed: {path}")
            return {k: data[k].copy() for k in ("full", "deleted", "valid_pairs")}
    result = scene_ablation(scene, channels, args.channel_batch, protocol["hit_distance"])
    atomic_npz(path, channels=channels, **result)
    return result


def parser():
    workspace = Path(__file__).resolve().parents[3]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-run", type=Path, default=workspace / "eval/scannet/07c02b354a694662")
    p.add_argument("--cache-root", type=Path, default=workspace / "cache/scannet")
    p.add_argument("--output-root", type=Path, default=workspace / "eval/scannet_channels")
    p.add_argument("--report-root", type=Path, default=workspace / "heft/reports/scannet_channels")
    p.add_argument("--block", type=int, default=15)
    p.add_argument("--timestep", type=int, default=300)
    p.add_argument("--dimensions", type=int, nargs="+", default=[1024, 768, 512, 384, 256, 128])
    p.add_argument("--random-seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    p.add_argument("--device", default="cpu")
    p.add_argument("--channel-batch", type=int, default=64)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--baseline-only", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.threads < 1 or args.channel_batch < 1:
        raise ValueError("Thread and batch counts must be positive")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(0.04, args.device)
    started = time.time()
    protocol, scenes, provenance = load_cached_scenes(args)
    names = list(scenes)
    dimensions = next(iter(scenes.values()))["features"].shape[-1]
    if args.dimensions != sorted(set(args.dimensions), reverse=True):
        raise ValueError("Dimension schedule must be unique and strictly decreasing")
    if min(args.dimensions) < 2 or max(args.dimensions) >= dimensions:
        raise ValueError("Retained dimensions must be between 2 and C - 1")
    if len(set(args.random_seeds)) != len(args.random_seeds):
        raise ValueError("Random seeds must be distinct")
    config = {
        "schema_version": 1, "baseline_run": str(args.baseline_run.resolve()),
        "block": args.block, "timestep": args.timestep,
        "dimensions": args.dimensions, "random_seeds": args.random_seeds,
        "device": args.device, "channel_batch": args.channel_batch, "threads": args.threads,
        "protocol": protocol, "provenance": provenance,
        "validation": "leave-one-scene-out; dimensions fixed before scoring",
        "selection": "deletion importance: voxel hit, then matching margin, then channel index",
        "selection_precision": "float64; final scores use unchanged float32 evaluate_pair",
        "pca": "scene-equal valid-token covariance; centered, no whitening; training scenes only",
        "code": source_identity([
            "channel_selection.py", "channel_scan.py", "channel_report.py",
            "geometry.py", "metrics.py", "report.py", "scan.py",
        ]),
        "versions": {"numpy": np.__version__, "torch": torch.__version__},
    }
    out = args.output_root / digest(config)
    report = args.report_root / out.name
    write_json(out / "config.json", config)
    write_json(out / "status.json", {"status": "baseline_check", "started": started})
    log(f"Output: {out}")
    try:
        rows = check_baseline(scenes, protocol, out)
        if args.baseline_only:
            write_json(out / "status.json", {"status": "baseline_verified", "seconds": time.time() - started})
            return
        prepared, full_stats, moments = {}, {}, {}
        all_channels = np.arange(dimensions)
        for name, scene in scenes.items():
            prepared[name] = AblationScene.from_arrays(
                name, scene["features"], scene["pairs"], scene["valid"], args.device
            )
            log(f"Full-channel deletion scan: {name}")
            full_stats[name] = cached_ablation(
                out / "ablation" / f"{name}_full.npz",
                prepared[name], all_channels, args, protocol,
            )
            moments[name] = scene_moments(prepared[name])
        permutations = {
            seed: np.random.default_rng(seed).permutation(dimensions)
            for seed in args.random_seeds
        }
        for fold, held_out in enumerate(names):
            train = training_scene_names(names, held_out)
            scene = scenes[held_out]
            fold_dir = out / "folds" / held_out
            write_json(fold_dir / "split.json", {"held_out": held_out, "training_scenes": train})
            write_json(out / "status.json", {"status": "running", "fold": fold + 1, "held_out": held_out})
            log(f"Fold {fold + 1}/{len(names)}: hold out {held_out}; fit on {', '.join(train)}")
            ranked, full, deleted, importance = rank_channels(all_channels, [full_stats[n] for n in train])
            atomic_npz(
                fold_dir / "initial_ranking.npz", channels=all_channels,
                ranked_channels=ranked, full=full, deleted=deleted, importance=importance,
                training_scenes=np.array(train), metrics=np.array(ABLATION_METRICS),
            )
            current, masks, traces = all_channels.copy(), {}, []
            for k in args.dimensions:
                if len(current) == dimensions:
                    stage_rank = ranked
                else:
                    stats = [
                        cached_ablation(
                            fold_dir / "ablation" / f"{name}_from_{len(current)}.npz",
                            prepared[name], current, args, protocol,
                        ) for name in train
                    ]
                    stage_rank, before, stage_deleted, stage_importance = rank_channels(current, stats)
                    atomic_npz(
                        fold_dir / f"ranking_from_{len(current)}.npz", channels=current,
                        ranked_channels=stage_rank, full=before, deleted=stage_deleted,
                        importance=stage_importance, training_scenes=np.array(train),
                        metrics=np.array(ABLATION_METRICS),
                    )
                previous_count = len(current)
                current = np.sort(stage_rank[:k])
                for method, channels in {"ablation_once": np.sort(ranked[:k]), "ablation_iterative": current}.items():
                    masks[f"{method}_{k}"] = channels.tolist()
                    train_values = np.mean([
                        scene_retrieval(prepared[n], channels, protocol["hit_distance"])
                        for n in train
                    ], axis=0)
                    traces.append({
                        "method": method, "dimensions": k,
                        "from_dimensions": dimensions if method == "ablation_once" else previous_count,
                        "training_scores": dict(zip(ABLATION_METRICS, train_values.tolist())),
                        "held_out": held_out, "training_scenes": train,
                    })
                    rows.extend(cached_score(out, held_out, scene, scene["features"][..., channels], protocol, method))
                write_json(fold_dir / "masks.json", masks)
                write_json(fold_dir / "training_trace.json", traces)
                log(f"{held_out}: pruning to {k} complete; training voxel hit {train_values[0]:.4f}")
            log(f"{held_out}: fitting PCA on training scenes and scoring controls")
            pca = fit_pca([moments[n] for n in train])
            atomic_npz(fold_dir / "pca.npz", **pca, training_scenes=np.array(train))
            centered = (scene["features"] - pca["mean"].astype(np.float32)).astype(np.float32)
            rows.extend(cached_score(out, held_out, scene, centered, protocol, "centered_full"))
            projected = centered @ pca["components"].astype(np.float32)
            for k in args.dimensions:
                rows.extend(cached_score(out, held_out, scene, projected[..., :k], protocol, "pca"))
                for seed, permutation in permutations.items():
                    channels = np.sort(permutation[:k])
                    masks[f"random_{k}_seed{seed}"] = channels.tolist()
                    rows.extend(cached_score(out, held_out, scene, scene["features"][..., channels], protocol, "random", seed))
            write_json(fold_dir / "masks.json", masks)
            write_json(fold_dir / "pca_variance.json", {
                str(k): float(pca["eigenvalues"][:k].sum() / pca["eigenvalues"].sum())
                for k in args.dimensions
            })
            log(f"Fold {fold + 1}/{len(names)} complete")
        from .channel_report import generate

        generate(out, report, rows)
        status = {"status": "complete", "seconds": time.time() - started, "pair_rows": len(rows)}
        if args.device.startswith("cuda"):
            status["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
        write_json(out / "status.json", status)
        log(f"Complete in {status['seconds']:.1f}s. Report: {report / 'report.md'}")
    except Exception as error:
        write_json(out / "status.json", {"status": "failed", "error": repr(error), "seconds": time.time() - started})
        raise


if __name__ == "__main__":
    main()
