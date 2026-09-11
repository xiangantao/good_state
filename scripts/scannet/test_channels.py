"""Correctness checks for deletion scoring, scene weighting and PCA isolation."""

import numpy as np
import pytest
import torch

from .channel_report import summarize
from .channel_scan import check_baseline, score_scene
from .channel_selection import (
    AblationScene, RetrievalGeometry, fit_pca, pair_ablation, rank_channels,
    scene_ablation, scene_moments, scene_retrieval, training_scene_names,
)
from .geometry import pair_geometry
from .metrics import evaluate_pair
from .report import METRICS


def fixture_pair():
    x = np.array([[[0, 0, 1], [.04, 0, 1], [.3, 0, 1], [.6, 0, 1]]], np.float32)
    y = np.array([[[.3, 0, 1], [0, 0, 1], [.9, 0, 1], [.6, 0, 1]]], np.float32)
    valid = np.ones((1, 4), bool)
    return pair_geometry(x, y, valid, valid), np.stack([valid, valid])


@pytest.mark.parametrize("kind", ["random", "constant", "zero", "sparse"])
@pytest.mark.parametrize("batch_size", [1, 3, 16])
def test_accelerated_deletion_matches_real_slices(kind, batch_size):
    pair, _ = fixture_pair()
    rng = np.random.default_rng(71)
    a, b = rng.normal(size=(2, 1, 4, 7)).astype(np.float32)
    if kind == "constant":
        a[:], b[:] = 1, 1
    elif kind == "zero":
        a[:], b[:] = 0, 0
    elif kind == "sparse":
        a[..., 1:], b[..., 1:] = 0, 0
    full, deleted = pair_ablation(
        torch.from_numpy(a.reshape(4, 7)), torch.from_numpy(b.reshape(4, 7)),
        RetrievalGeometry.from_pair(pair, "cpu"), batch_size,
    )
    for index in range(8):
        channels = np.arange(7) if index == 7 else np.delete(np.arange(7), index)
        result = evaluate_pair(a[..., channels], b[..., channels], pair)
        actual = full if index == 7 else deleted[index]
        np.testing.assert_allclose(actual[:3], [result[m] for m in METRICS[3:6]], atol=1e-7)
        assert torch.isfinite(actual).all()


def test_nuisance_deletion_recovers_correspondence():
    pair, _ = fixture_pair()
    signal_a = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    signal_b = np.array([[0, 1, 0], [1, 0, 0], [-1, -1, -1], [0, 0, 1]], np.float32)
    nuisance_a = np.array([100, -100, 100, -100], np.float32)[:, None]
    nuisance_b = np.array([100, -100, -100, 100], np.float32)[:, None]
    a, b = [torch.from_numpy(np.concatenate(items, axis=1)) for items in ((signal_a, nuisance_a), (signal_b, nuisance_b))]
    full, deleted = pair_ablation(a, b, RetrievalGeometry.from_pair(pair, "cpu"))
    ranked, _, _, _ = rank_channels(np.arange(4), [{"full": full.numpy(), "deleted": deleted.numpy()}])
    assert deleted[3, 0] == 1
    assert full[0] < deleted[3, 0]
    assert ranked[-1] == 3


def test_scene_scores_and_moments():
    pair, valid = fixture_pair()
    features = np.random.default_rng(7).normal(size=(2, 1, 4, 5)).astype(np.float32)
    scene = AblationScene.from_arrays("test", features, [pair], valid)
    stats = scene_ablation(scene, np.arange(5), batch_size=2)
    np.testing.assert_allclose(stats["full"], scene_retrieval(scene, np.arange(5)), atol=1e-12)
    moments = scene_moments(scene)
    x = features[valid].astype(np.float64)
    np.testing.assert_allclose(moments["mean"], x.mean(0))
    np.testing.assert_allclose(moments["second"], x.T @ x / len(x))


def test_scene_weighting_and_pca_heldout_isolation():
    rng = np.random.default_rng(13)
    moments = {}
    for name, count in (("a", 20), ("b", 5), ("held", 200)):
        x = rng.normal(size=(count, 6)) + (20 if name == "held" else 0)
        moments[name] = {"mean": x.mean(0), "second": x.T @ x / count}
    train = training_scene_names(list(moments), "held")
    pca = fit_pca([moments[n] for n in train])
    moments["held"]["mean"][:] = 9999
    repeated = fit_pca([moments[n] for n in train])
    np.testing.assert_array_equal(pca["components"], repeated["components"])
    np.testing.assert_allclose(pca["mean"], (moments["a"]["mean"] + moments["b"]["mean"]) / 2)
    np.testing.assert_allclose(pca["components"].T @ pca["components"], np.eye(6), atol=1e-12)
    covariance = sum(moments[n]["second"] for n in train) / 2 - np.outer(pca["mean"], pca["mean"])
    np.testing.assert_allclose(pca["components"] @ np.diag(pca["eigenvalues"]) @ pca["components"].T, covariance, atol=1e-12)


def test_macro_missing_pairs_and_random_seed_variation():
    rows = []
    for repeat, offset in ((42, 0), (43, .2)):
        for scene, values in (("a", [.2 + offset]), ("b", [.6 + offset] * 3 + [None])):
            for value in values:
                rows.append({"scene": scene, "method": "random", "dimensions": 2, "repeat": repeat,
                             "status": "ok" if value is not None else "no_shared_voxels",
                             **dict.fromkeys(METRICS, value)})
    _, _, summary = summarize(rows)
    result = summary[0]
    assert result["retrieval_voxel_hit"] == pytest.approx(.5)
    assert result["retrieval_voxel_hit_seed_std"] == pytest.approx(.1)
    assert result["retrieval_voxel_hit_scene_std"] == pytest.approx(.2)


def test_baseline_gate_detects_changed_score(tmp_path):
    pair, valid = fixture_pair()
    features = np.random.default_rng(3).normal(size=(2, 1, 4, 5)).astype(np.float32)
    scene = {"features": features, "pairs": [pair], "valid": valid, "frame_ids": [0, 1]}
    protocol = {"seed": 42, "negative_min_distance": .2, "hit_distance": .1}
    scene["archived"] = score_scene("test", scene, features, protocol, "original")
    check_baseline({"test": scene}, protocol, tmp_path)
    scene["archived"][0]["retrieval_voxel_hit"] += .1
    with pytest.raises(ValueError, match="Baseline mismatch"):
        check_baseline({"test": scene}, protocol, tmp_path)
