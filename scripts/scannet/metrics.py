"""Adjacent-frame metrics; geometry defines evaluation labels, never predictions."""

from __future__ import annotations

import numpy as np


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def mean_or_none(values):
    a = np.asarray(values, dtype=np.float64)
    return float(a.mean()) if a.size else None


def evaluate_pair(
    features_i,
    features_j,
    geometry,
    seed=42,
    negative_min_distance=0.2,
    hit_distance=0.1,
    *,
    query_features_i=None,
    query_features_j=None,
    similarity_kind="cosine",
):
    """Use the same geometry/negative sample for every candidate layer.

    Retrieval is bidirectional. Ground-truth eligibility requires the query voxel
    to exist in the other view. Predictions search ALL valid target tokens, with
    fractional tie credit and expected error rather than optimistic tie breaking.
    """
    if negative_min_distance <= 0 or hit_distance <= 0:
        raise ValueError("Distance thresholds must be positive")
    if features_i.shape[-1] != features_j.shape[-1]:
        raise ValueError("Feature channel counts must match")
    if query_features_i is not None and query_features_i.shape != features_i.shape:
        raise ValueError("Query/key shapes must match in view i")
    if query_features_j is not None and query_features_j.shape != features_j.shape:
        raise ValueError("Query/key shapes must match in view j")
    ni, nj = geometry["ids_i"], geometry["ids_j"]
    fi = np.asarray(features_i, dtype=np.float32).reshape(-1, features_i.shape[-1])[ni]
    fj = np.asarray(features_j, dtype=np.float32).reshape(-1, features_j.shape[-1])[nj]
    if not np.isfinite(fi).all() or not np.isfinite(fj).all():
        raise ValueError("Non-finite features")
    if (query_features_i is None) != (query_features_j is None):
        raise ValueError("Both directional query features are required")
    qi = (
        fi
        if query_features_i is None
        else np.asarray(query_features_i, dtype=np.float32).reshape(-1, fi.shape[-1])[
            ni
        ]
    )
    qj = (
        fj
        if query_features_j is None
        else np.asarray(query_features_j, dtype=np.float32).reshape(-1, fj.shape[-1])[
            nj
        ]
    )
    if not np.isfinite(qi).all() or not np.isfinite(qj).all():
        raise ValueError("Non-finite query features")
    if similarity_kind not in {"cosine", "scaled_dot"}:
        raise ValueError("Unsupported similarity kind")
    project = normalize if similarity_kind == "cosine" else np.asarray
    scale = 1.0 if similarity_kind == "cosine" else fi.shape[-1] ** -0.5
    result = {
        "valid_tokens_i": len(ni),
        "valid_tokens_j": len(nj),
        "shared_voxels": len(geometry["shared"]),
        "retrieval_queries": 0,
        "positive_similarity": None,
        "negative_similarity": None,
        "similarity_gap": None,
        "retrieval_voxel_hit": None,
        "retrieval_hit_distance": None,
        "retrieval_world_error_mean": None,
        "retrieval_world_error_median": None,
        "negative_pairs": 0,
        "zero_norm_tokens": int(
            (np.linalg.norm(fi, axis=-1) == 0).sum()
            + (np.linalg.norm(fj, axis=-1) == 0).sum()
        ),
        "zero_norm_queries": int(
            (np.linalg.norm(qi, axis=-1) == 0).sum()
            + (np.linalg.norm(qj, axis=-1) == 0).sum()
        ),
        "status": "no_shared_voxels",
    }
    if not len(ni) or not len(nj):
        result["status"] = "no_valid_tokens"
        return result
    if not geometry["shared"]:
        return result
    # Match the appendix: average raw features first, then normalize.
    pi = project(np.stack([fi[a].mean(0) for a, _ in geometry["shared"]]))
    pj = project(np.stack([fj[b].mean(0) for _, b in geometry["shared"]]))
    pqi = project(np.stack([qi[a].mean(0) for a, _ in geometry["shared"]]))
    pqj = project(np.stack([qj[b].mean(0) for _, b in geometry["shared"]]))
    result["positive_similarity"] = float(
        0.5 * ((pqi * pj).sum(-1).mean() + (pqj * pi).sum(-1).mean()) * scale
    )
    similarity = (project(qi) @ project(fj).T) * scale
    reverse_similarity = (project(qj) @ project(fi).T) * scale
    same = (geometry["vox_i"][:, None] == geometry["vox_j"][None]).all(-1)
    distance = np.linalg.norm(
        geometry["xyz_i"][:, None] - geometry["xyz_j"][None], axis=-1
    )
    # Negatives use the same per-view voxel-prototype unit as the positives.
    # Restrict to shared voxels so the compared support is explicit and fixed.
    rng = np.random.default_rng(seed)
    negatives, hits, errors, voxel_hits = [], [], [], []
    xi = np.stack([geometry["xyz_i"][a].mean(0) for a, _ in geometry["shared"]])
    xj = np.stack([geometry["xyz_j"][b].mean(0) for _, b in geometry["shared"]])
    pdist = np.linalg.norm(xi[:, None] - xj[None], axis=-1)
    psim = (pqi @ pj.T) * scale
    reverse_psim = (pqj @ pi.T) * scale
    for sim, dist in [(psim, pdist), (reverse_psim, pdist.T)]:
        for i in range(len(sim)):
            candidate = np.flatnonzero(
                (np.arange(len(sim)) != i) & (dist[i] >= negative_min_distance)
            )
            if len(candidate):
                negatives.append(sim[i, rng.choice(candidate)])
    for sim, match, dist in [
        (similarity, same, distance),
        (reverse_similarity, same.T, distance.T),
    ]:
        for i in np.flatnonzero(match.any(1)):
            tied = np.flatnonzero(np.isclose(sim[i], sim[i].max(), rtol=0, atol=1e-6))
            voxel_hits.append(match[i, tied].mean())
            hits.append((dist[i, tied] <= hit_distance).mean())
            errors.append(dist[i, tied].mean())
    result.update(
        status="ok",
        negative_similarity=mean_or_none(negatives),
        negative_pairs=len(negatives),
        retrieval_queries=len(errors),
        retrieval_voxel_hit=mean_or_none(voxel_hits),
        retrieval_hit_distance=mean_or_none(hits),
        retrieval_world_error_mean=mean_or_none(errors),
        retrieval_world_error_median=float(np.median(errors)) if errors else None,
    )
    # Diagnostic positive-minus-negative contrast, not a calibrated loss.
    if negatives:
        result["similarity_gap"] = (
            result["positive_similarity"] - result["negative_similarity"]
        )
    return result
