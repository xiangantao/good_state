"""Channel deletion probes and PCA fitted without held-out scene access."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


ABLATION_METRICS = (
    "retrieval_voxel_hit",
    "retrieval_hit_distance",
    "retrieval_world_error_mean",
    "matching_margin",
)


@dataclass
class RetrievalGeometry:
    ids_i: torch.Tensor
    ids_j: torch.Tensor
    same: torch.Tensor
    distance: torch.Tensor
    eligible_i: torch.Tensor
    eligible_j: torch.Tensor

    @classmethod
    def from_pair(cls, pair, device):
        same = (pair["vox_i"][:, None] == pair["vox_j"][None]).all(-1)
        distance = np.linalg.norm(
            pair["xyz_i"][:, None] - pair["xyz_j"][None], axis=-1
        )
        return cls(
            *[
                torch.as_tensor(value, device=device)
                for value in (
                    pair["ids_i"], pair["ids_j"], same, distance,
                    np.flatnonzero(same.any(1)), np.flatnonzero(same.any(0)),
                )
            ]
        )


@dataclass
class AblationScene:
    name: str
    features: torch.Tensor
    pairs: list
    valid: np.ndarray

    @classmethod
    def from_arrays(cls, name, features, pairs, valid, device="cpu"):
        if not np.isfinite(features).all():
            raise ValueError("Non-finite cached features")
        return cls(
            name,
            torch.as_tensor(
                np.ascontiguousarray(features), dtype=torch.float64, device=device
            ),
            [
                RetrievalGeometry.from_pair(pair, device) if pair["shared"] else None
                for pair in pairs
            ],
            valid,
        )


def retrieval_values(similarity, geometry, hit_distance=0.1):
    """Batched bidirectional pair scores, with the evaluator's 1e-6 tie rule.

    The secondary ranking margin is best correct minus best incorrect cosine;
    queries without an incorrect candidate contribute zero to this diagnostic.
    """
    total = torch.zeros(
        (len(similarity), len(ABLATION_METRICS)),
        dtype=similarity.dtype, device=similarity.device,
    )
    count = 0
    for sim, same, dist, eligible in (
        (similarity, geometry.same, geometry.distance, geometry.eligible_i),
        (similarity.transpose(-1, -2), geometry.same.T,
         geometry.distance.T, geometry.eligible_j),
    ):
        if not len(eligible):
            continue
        sim, same, dist = sim[:, eligible], same[eligible], dist[eligible]
        tied = (sim - sim.amax(-1, keepdim=True)).abs() <= 1e-6
        weights = tied / tied.sum(-1, keepdim=True).to(sim.dtype)
        best_positive = sim.masked_fill(~same, -torch.inf).amax(-1)
        best_negative = sim.masked_fill(same, -torch.inf).amax(-1)
        margin = torch.where(
            (~same).any(-1), best_positive - best_negative, 0.0
        )
        total += torch.stack(
            (
                (weights * same).sum((-1, -2)),
                (weights * (dist <= hit_distance)).sum((-1, -2)),
                (weights * dist).sum((-1, -2)),
                margin.sum(-1),
            ), dim=-1,
        )
        count += len(eligible)
    if not count:
        raise ValueError("Ablation requires shared-voxel queries")
    return total / count


def pair_ablation(a, b, geometry, batch_size=64, hit_distance=0.1):
    """Recompute cosine after each deletion using float64 subtraction.

    a/b contain valid tokens and the currently retained channels. The model is
    never run here. Each batch bounds the channel x query x target allocation.
    """
    if a.shape[1] != b.shape[1] or a.shape[1] < 2 or batch_size < 1:
        raise ValueError("Need matching dimensions >= 2 and a positive batch size")
    a, b = a.to(torch.float64), b.to(torch.float64)
    dot = a @ b.T
    sa, sb = a.square().sum(-1), b.square().sum(-1)
    baseline = dot / (
        sa.sqrt().clamp_min(1e-12)[:, None]
        * sb.sqrt().clamp_min(1e-12)[None]
    )
    full = retrieval_values(baseline[None], geometry, hit_distance)[0]
    deleted = []
    for start in range(0, a.shape[1], batch_size):
        ac = a[:, start:start + batch_size].T
        bc = b[:, start:start + batch_size].T
        na = (sa[None] - ac.square()).clamp_min(0).sqrt().clamp_min(1e-12)
        nb = (sb[None] - bc.square()).clamp_min(0).sqrt().clamp_min(1e-12)
        sim = (dot[None] - ac[:, :, None] * bc[:, None, :]) / (
            na[:, :, None] * nb[:, None, :]
        )
        deleted.append(retrieval_values(sim, geometry, hit_distance))
    return full, torch.cat(deleted)


@torch.inference_mode()
def scene_ablation(scene, channels, batch_size=64, hit_distance=0.1):
    channels = np.asarray(channels, dtype=np.int64)
    if len(np.unique(channels)) != len(channels):
        raise ValueError("Repeated channel indices")
    index = torch.as_tensor(channels, device=scene.features.device)
    full_sum = scene.features.new_zeros(len(ABLATION_METRICS))
    deleted_sum = scene.features.new_zeros((len(channels), len(ABLATION_METRICS)))
    count = 0
    for i, pair in enumerate(scene.pairs):
        if pair is None:
            continue
        a = scene.features[i].reshape(-1, scene.features.shape[-1])[pair.ids_i]
        b = scene.features[i + 1].reshape(-1, scene.features.shape[-1])[pair.ids_j]
        full, deleted = pair_ablation(
            a[:, index], b[:, index], pair, batch_size, hit_distance
        )
        full_sum += full
        deleted_sum += deleted
        count += 1
    if not count:
        raise ValueError(f"No valid training pairs in {scene.name}")
    return {
        "full": (full_sum / count).cpu().numpy(),
        "deleted": (deleted_sum / count).cpu().numpy(),
        "valid_pairs": count,
    }


@torch.inference_mode()
def scene_retrieval(scene, channels, hit_distance=0.1):
    index = torch.as_tensor(channels, device=scene.features.device)
    values = []
    for i, pair in enumerate(scene.pairs):
        if pair is None:
            continue
        a = scene.features[i].reshape(-1, scene.features.shape[-1])[pair.ids_i][:, index]
        b = scene.features[i + 1].reshape(-1, scene.features.shape[-1])[pair.ids_j][:, index]
        a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        values.append(retrieval_values((a @ b.T)[None], pair, hit_distance)[0])
    if not values:
        raise ValueError(f"No valid training pairs in {scene.name}")
    return torch.stack(values).mean(0).cpu().numpy()


def rank_channels(channels, scene_stats):
    """Scene-macro deletion importance; hit first, then margin, then index."""
    if not scene_stats:
        raise ValueError("At least one training scene is required")
    full = np.mean([s["full"] for s in scene_stats], axis=0)
    deleted = np.mean([s["deleted"] for s in scene_stats], axis=0)
    importance = full[None] - deleted
    channels = np.asarray(channels, dtype=np.int64)
    order = np.lexsort((channels, -importance[:, 3], -importance[:, 0]))
    return channels[order], full, deleted, importance


def training_scene_names(names, held_out):
    if len(set(names)) != len(names) or held_out not in names or len(names) < 3:
        raise ValueError("Need distinct scenes including the held-out scene")
    return [name for name in names if name != held_out]


@torch.inference_mode()
def scene_moments(scene):
    """Raw valid-token moments; fitting gives each training scene equal weight."""
    x = scene.features[torch.as_tensor(scene.valid, device=scene.features.device)]
    if not len(x):
        raise ValueError("No valid PCA tokens")
    return {
        "mean": x.mean(0).cpu().numpy(),
        "second": (x.T @ x / len(x)).cpu().numpy(),
        "tokens": len(x),
    }


def fit_pca(moments):
    if not moments:
        raise ValueError("PCA needs training scenes")
    mean = np.mean([m["mean"] for m in moments], axis=0)
    covariance = np.mean([m["second"] for m in moments], axis=0) - np.outer(mean, mean)
    values, vectors = np.linalg.eigh((covariance + covariance.T) * 0.5)
    values, vectors = values[::-1].clip(0), vectors[:, ::-1].copy()
    signs = np.sign(vectors[np.abs(vectors).argmax(0), np.arange(len(mean))])
    vectors *= np.where(signs == 0, 1, signs)
    return {"mean": mean, "components": vectors, "eigenvalues": values}
