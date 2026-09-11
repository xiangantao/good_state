"""Run with: python -m pytest scripts/scannet/test_scannet.py"""

from __future__ import annotations

import io
import struct
import zlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from .dataset import SensScene
from .extract import frame_seed, selected_noise
from .geometry import pair_geometry, register_geometry, unproject, valid_pool
from .metrics import evaluate_pair
from .report import aggregate
from .scan import digest


def test_unprojection_and_valid_pool():
    d = np.array([[2, 0], [2, 0]], dtype=np.float32)
    k = np.eye(4)
    world = unproject(d, k)
    world[d == 0] = np.nan
    xyz, coverage = valid_pool(world, d > 0, (1, 1))
    np.testing.assert_allclose(xyz[0, 0], [0, 1, 2])
    assert coverage[0, 0] == 0.5


def test_registration_camera_translation_and_z_buffer():
    # Two points project onto a single output RGB pixel. Nearest surface wins.
    scene = SimpleNamespace(
        depth_intrinsic=np.eye(4),
        depth_to_color=np.eye(4),
        color_intrinsic=np.eye(4),
        color_width=2,
        color_height=1,
        axis_alignment=np.eye(4),
    )
    pose = np.eye(4)
    pose[0, 3] = 3
    frame = SimpleNamespace(pose=pose)
    xyz, mask, _ = register_geometry(
        scene, frame, np.array([[2.0, 4.0]]), (1, 1), (1, 1), 0.1
    )
    assert mask.all()
    np.testing.assert_allclose(xyz[0, 0], [3, 0, 2])


def test_correct_permuted_geometry_retrieval_and_collapse():
    x = np.array([[[0.0, 0, 2], [1, 0, 2], [2, 0, 2]]])
    y = x[:, [2, 0, 1]]
    g = pair_geometry(x, y, np.ones((1, 3), bool), np.ones((1, 3), bool))
    f = np.eye(3, dtype=np.float32)[None]
    correct = evaluate_pair(f, f[:, [2, 0, 1]], g)
    assert correct["retrieval_voxel_hit"] == 1
    assert correct["retrieval_world_error_mean"] == 0
    assert correct["positive_similarity"] == pytest.approx(1)
    assert correct["negative_similarity"] == pytest.approx(0)
    collapsed = evaluate_pair(np.ones_like(f), np.ones_like(f), g)
    assert collapsed["positive_similarity"] == pytest.approx(1)
    assert collapsed["similarity_gap"] == pytest.approx(0, abs=1e-6)
    assert collapsed["retrieval_voxel_hit"] == pytest.approx(1 / 3)
    assert collapsed["retrieval_world_error_mean"] > 0.5


def test_no_overlap_is_missing_not_zero():
    a = np.zeros((1, 1, 3))
    b = np.ones((1, 1, 3))
    g = pair_geometry(a, b, np.ones((1, 1), bool), np.ones((1, 1), bool))
    result = evaluate_pair(np.ones((1, 1, 2)), np.ones((1, 1, 2)), g)
    assert result["status"] == "no_shared_voxels"
    assert result["positive_similarity"] is None


def test_prototype_averages_raw_features_before_normalizing():
    a = np.array([[[0.0, 0, 2], [0.01, 0, 2]]])
    b = np.array([[[0.0, 0, 2]]])
    g = pair_geometry(a, b, np.ones((1, 2), bool), np.ones((1, 1), bool))
    result = evaluate_pair(np.array([[[10.0, 0], [0, 1]]]), np.array([[[1.0, 0]]]), g)
    assert result["positive_similarity"] == pytest.approx(10 / np.sqrt(101))


def test_scene_macro_and_missing_counts():
    rows = [
        {
            "scene": "a",
            "block": 1,
            "timestep": 300,
            "positive_similarity": 1.0,
            "status": "ok",
        }
    ] * 4
    rows += [
        {
            "scene": "b",
            "block": 1,
            "timestep": 300,
            "positive_similarity": 0.0,
            "status": "ok",
        }
    ]
    scenes = aggregate(rows, ["scene", "block", "timestep"])
    final = aggregate(scenes, ["block", "timestep"])[0]
    assert final["positive_similarity"] == 0.5
    assert final["positive_similarity_count"] == 2
    assert final["negative_similarity"] is None


def test_noise_matches_local_vega_schedule():
    import importlib.util
    from pathlib import Path

    file = (
        Path(__file__).resolve().parents[3]
        / "VEGA-3D/llava/model/multimodal_generative_encoder/wan/utils/fm_solvers_unipc.py"
    )
    # Import this single scheduler file, without initializing the LLaVA package.
    spec = importlib.util.spec_from_file_location("reference_flow_scheduler", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scheduler = module.FlowUniPCMultistepScheduler(
        num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
    )
    scheduler.set_timesteps(1000, device="cpu", shift=5)
    for k in [0, 100, 300, 600, 1000]:
        index = int(torch.argmin(torch.abs(scheduler.timesteps - k)))
        actual = selected_noise(k)
        assert actual["actual_timestep"] == scheduler.timesteps[index].item()
        assert actual["sigma"] == pytest.approx(
            scheduler.sigmas[index].item(), abs=1e-7
        )


def test_frame_noise_and_cache_identity():
    assert frame_seed(42, "scene1", 10) == frame_seed(42, "scene1", 10)
    assert frame_seed(42, "scene1", 10) != frame_seed(42, "scene1", 11)
    assert digest({"blocks": [10], "size": [480, 832]}) != digest(
        {"blocks": [20], "size": [480, 832]}
    )


def write_tiny_sens(tmp_path):
    scene = tmp_path / "scene0000_00"
    scene.mkdir()
    identity = np.eye(4, dtype="<f4").tobytes()
    (scene / "scene0000_00.txt").write_text(
        "colorToDepthExtrinsics = " + " ".join(map(str, np.eye(4).ravel()))
    )
    rgb = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(rgb, format="JPEG")
    color = rgb.getvalue()
    depth = zlib.compress(np.full((2, 2), 2000, dtype="<u2").tobytes())
    with (scene / "scene0000_00.sens").open("wb") as f:
        f.write(struct.pack("<IQ", 4, 4))
        f.write(b"test")
        f.write(identity * 4)
        f.write(struct.pack("<iiIIIIfQ", 2, 1, 2, 2, 2, 2, 1000, 3))
        for i in range(3):
            f.write(identity)
            f.write(struct.pack("<QQQQ", i, i, len(color), len(depth)))
            f.write(color)
            f.write(depth)
    return scene


def test_sens_index_decode_sampling_and_truncation(tmp_path):
    directory = write_tiny_sens(tmp_path)
    scene = SensScene(directory)
    assert [f.index for f in scene.sample(2)] == [0, 2]
    assert len(scene.sample(32)) == 3  # never duplicate a short sequence
    rgb, depth = scene.decode(scene.frames[1])
    assert rgb.size == (2, 2)
    np.testing.assert_array_equal(depth, np.full((2, 2), 2.0))
    with scene.path.open("r+b") as f:
        f.truncate(scene.path.stat().st_size - 10)
    with pytest.raises(ValueError, match="incomplete"):
        SensScene(directory)
