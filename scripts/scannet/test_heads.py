"""Head ordering, asymmetric QK, callback fidelity and grouping regressions."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from .geometry import pair_geometry
from .head_extract import WanHeadExtractor, pool_heads
from .head_report import generate_heads
from .metrics import evaluate_pair
from .report import write_json


def test_pool_heads_keeps_head_and_token_order():
    x = torch.arange(2 * 3 * 8 * 5).reshape(2, 3, 8, 5).float()
    pooled = pool_heads(x, (2, 4), (1, 2))
    assert pooled.shape == (2, 3, 5, 1, 2)
    for b in range(2):
        for h in range(3):
            torch.testing.assert_close(
                pooled[b, h, :, 0, 0], x[b, h, [0, 1, 4, 5]].mean(0)
            )
            torch.testing.assert_close(
                pooled[b, h, :, 0, 1], x[b, h, [2, 3, 6, 7]].mean(0)
            )


def two_point_geometry():
    xyz = np.array([[[0.0, 0, 2], [1.0, 0, 2]]])
    return pair_geometry(xyz, xyz, np.ones((1, 2), bool), np.ones((1, 2), bool))


def test_qk_reverse_is_independent_not_forward_transpose():
    k = np.eye(2, dtype=np.float32)[None]
    result = evaluate_pair(
        k,
        k,
        two_point_geometry(),
        query_features_i=k,
        query_features_j=k[:, ::-1],
        similarity_kind="scaled_dot",
    )
    # Forward is perfect, reverse swaps both points. Transposing forward would give1.
    assert result["retrieval_voxel_hit"] == pytest.approx(0.5)
    assert result["retrieval_hit_distance"] == pytest.approx(0.5)
    assert result["retrieval_world_error_mean"] == pytest.approx(0.5)
    assert result["positive_similarity"] == pytest.approx(0.5 / np.sqrt(2))
    assert result["negative_similarity"] == pytest.approx(0.5 / np.sqrt(2))
    assert result["similarity_gap"] == pytest.approx(0)


def test_qk_scaled_dot_and_collapsed_ties():
    k = np.eye(2, dtype=np.float32)[None]
    result = evaluate_pair(
        3 * k,
        3 * k,
        two_point_geometry(),
        query_features_i=2 * k,
        query_features_j=2 * k,
        similarity_kind="scaled_dot",
    )
    assert result["positive_similarity"] == pytest.approx(6 / np.sqrt(2))
    assert result["retrieval_voxel_hit"] == 1
    collapsed = evaluate_pair(
        np.ones_like(k),
        np.ones_like(k),
        two_point_geometry(),
        query_features_i=np.ones_like(k),
        query_features_j=np.ones_like(k),
        similarity_kind="scaled_dot",
    )
    assert collapsed["retrieval_voxel_hit"] == pytest.approx(0.5)
    assert collapsed["similarity_gap"] == pytest.approx(0)


def test_capture_restored_on_success_and_failure():
    extractor = WanHeadExtractor.__new__(WanHeadExtractor)
    previous = object()
    attention = SimpleNamespace(_heft_capture=previous)
    extractor.transformer = SimpleNamespace(
        config=SimpleNamespace(patch_size=(1, 1, 1)),
        blocks=[SimpleNamespace(attn1=attention)],
    )
    tensor = torch.arange(12 * 4 * 128).reshape(1, 12, 4, 128).float()

    def fake_forward(*args):
        attention._heft_capture(query=tensor, key=tensor + 1)
        return {}, {}, {"sigma": 0.1}

    extractor.blocks = fake_forward
    latents = torch.ones(1, 16, 1, 2, 2)
    result, _ = extractor.heads(latents, [0], 300, 5, 42, "test", [0], grid_size=(2, 2))
    torch.testing.assert_close(result[0]["key"], pool_heads(tensor + 1, (2, 2), (2, 2)))
    assert attention._heft_capture is previous
    del attention._heft_capture

    def fail(*args):
        raise RuntimeError("expected forward failure")

    extractor.blocks = fail
    with pytest.raises(RuntimeError, match="expected forward failure"):
        extractor.heads(latents, [0], 300, 5, 42, "test", [0])
    assert not hasattr(attention, "_heft_capture")


def test_local_processor_capture_is_post_norm_rope_and_passive():
    from diffusers.models.attention_processor import Attention
    from diffusers.models.transformers.transformer_wan import WanAttnProcessor2_0

    torch.manual_seed(1)
    attn = Attention(
        query_dim=24,
        heads=3,
        kv_heads=3,
        dim_head=8,
        qk_norm="rms_norm_across_heads",
        processor=WanAttnProcessor2_0(),
    ).eval()
    x = torch.randn(1, 4, 24)
    angles = torch.randn(1, 1, 4, 4, dtype=torch.float64)
    rope = torch.polar(torch.ones_like(angles), angles)
    captured = {}
    with torch.inference_mode():
        baseline = attn(x, rotary_emb=rope)
        attn._heft_capture = lambda **values: captured.update(values)
        actual = attn(x, rotary_emb=rope)
        expected = attn.norm_q(attn.to_q(x)).unflatten(2, (3, 8)).transpose(1, 2)
        expected = (
            torch.view_as_real(
                torch.view_as_complex(expected.double().unflatten(3, (-1, 2))) * rope
            )
            .flatten(3, 4)
            .float()
        )
    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
    torch.testing.assert_close(captured["query"], expected, rtol=0, atol=0)


def test_head_report_keeps_modes_heads_and_scene_macro(tmp_path):
    out, report = tmp_path / "out", tmp_path / "report"
    write_json(out / "config.json", {"hit_distance": 0.1})
    rows = []
    for head in [0, 1]:
        for mode in ["qk", "kk"]:
            for scene, value, count in [("a", 1.0, 4), ("b", 0.0, 1)]:
                rows.extend(
                    [
                        {
                            "scene": scene,
                            "timestep": 300,
                            "block": 14,
                            "head": head,
                            "mode": mode,
                            "status": "ok",
                            "positive_similarity": value,
                            "retrieval_voxel_hit": value,
                            "retrieval_hit_distance": value,
                        }
                    ]
                    * count
                )
    write_json(out / "pairs" / "test.json", rows)
    summary = generate_heads(out, report)
    assert len(summary) == 4
    assert all(
        r["positive_similarity"] == 0.5 and r["positive_similarity_count"] == 2
        for r in summary
    )
    assert "体素命中率" in (report / "report.md").read_text()
