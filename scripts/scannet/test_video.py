"""Temporal context, CFG capture, padding and geometry-summary regressions."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from heft.extraction.config import WAN_2_1, ChunkJob

from .video_extract import WanVideoExtractor, chunk_records, pool_video_tokens, video_jobs
from .video_noise import video_scheduler
from .video_report import summarize_video
from .video_scan import parser, validate_args


def test_video_chunks_use_time_and_pad_only_the_tail():
    images = [Image.fromarray(np.full((2, 2, 3), i, np.uint8)) for i in range(32)]
    jobs = video_jobs(images, "scene", 42, 25)
    records = chunk_records(jobs, list(range(0, 64, 2)))
    assert [(job.num_frames, job.pipeline_num_frames) for job in jobs] == [(25, 25), (7, 25)]
    assert [job.seed for job in jobs] == [42, 42]
    assert jobs[1].input_video[:, 0, 0, 0].tolist() == list(range(25, 32)) + [31] * 18
    assert records[1]["frame_ids"] == list(range(50, 64, 2))
    assert records[1]["input_frame_ids"][-18:] == [62] * 18
    with pytest.raises(ValueError, match="4n\\+1"):
        video_jobs(images, "scene", 42, 2)


def test_pooling_keeps_time_order_and_excludes_padding():
    maps = torch.arange(5 * 2 * 4 * 3).reshape(5, 2, 4, 3).float()
    pooled = pool_video_tokens(maps.reshape(1, -1, 3), 5, (2, 4), (1, 2), 3)
    assert pooled.shape == (3, 3, 1, 2)
    for frame in range(3):
        for x in range(2):
            torch.testing.assert_close(pooled[frame, :, 0, x], maps[frame, :, 2*x:2*x+2].mean((0, 1)))
    with pytest.raises(ValueError, match="joint video"):
        pool_video_tokens(maps.reshape(5, 8, 3), 5, (2, 4), (1, 2), 3)


class TinyVideoPipeline:
    def __init__(self):
        from diffusers import WanTransformer3DModel

        torch.manual_seed(13)
        self.transformer = WanTransformer3DModel(
            patch_size=(1, 1, 1), num_attention_heads=2, attention_head_dim=12,
            in_channels=4, out_channels=4, text_dim=8, freq_dim=8, ffn_dim=48,
            num_layers=2, rope_max_seq_len=16,
        ).eval()
        self.latents = torch.randn(1, 4, 5, 2, 2)
        self.context = torch.randn(1, 3, 8)
        self.scheduler = SimpleNamespace(
            sigmas=torch.tensor([0.017]), index_for_timestep=lambda t: 0, lower_order_nums=0,
        )
        self.fail = False

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        self.outputs = []
        for branch in range(2):
            self.outputs.append(self.transformer(
                hidden_states=self.latents, timestep=torch.tensor([17]),
                encoder_hidden_states=self.context + branch,
            )[0])
            if self.fail:
                raise RuntimeError("injected failure")


def test_real_transformer_context_cfg_and_capture_cleanup():
    pipe = TinyVideoPipeline()
    model = replace(WAN_2_1, resolution=(32, 32))
    extractor = WanVideoExtractor.__new__(WanVideoExtractor)
    extractor.pipeline, extractor.model = pipe, model
    extractor.prompt_embeds, extractor.negative_prompt_embeds = pipe.context, pipe.context + 1
    job = ChunkJob(0, "test", None, 0, 0, 3, torch.zeros(5, 3, 32, 32), "", 42, 2)
    references = []
    hook = pipe.transformer.blocks[1].register_forward_hook(lambda m, i, o: references.append(o.detach().clone()))
    with torch.inference_mode():
        baseline = pipe.transformer(hidden_states=pipe.latents, timestep=torch.tensor([17]), encoder_hidden_states=pipe.context)[0]
    hook.remove()
    captured, audit = extractor.extract(job, [0, 1], [0, 1], (2, 2))
    torch.testing.assert_close(pipe.outputs[0], baseline, rtol=0, atol=0)
    torch.testing.assert_close(captured[1]["features"], pool_video_tokens(references[0], 5, (2, 2), (2, 2), 3), rtol=0, atol=0)
    assert captured[1]["key"].shape == (3, 2, 12, 2, 2)
    assert [r["shape"] for r in audit["transformer_forwards"]] == [[1, 4, 5, 2, 2]] * 2
    assert pipe.kwargs["num_frames"] == 5 and pipe.kwargs["start_step"] == 49
    assert pipe.kwargs["guidance_scale"] == 5 and pipe.kwargs["num_inference_steps"] == 50
    pipe.latents[:, :, -1] += 3
    changed, _ = extractor.extract(job, [0, 1], [0, 1], (2, 2))
    assert not torch.allclose(changed[1]["features"][0], captured[1]["features"][0])
    assert not torch.allclose(changed[1]["key"][0], captured[1]["key"][0])
    previous = object()
    pipe.transformer.blocks[0].attn1._heft_capture = previous
    pipe.fail = True
    with pytest.raises(RuntimeError, match="injected failure"):
        extractor.extract(job, [0, 1], [0, 1], (2, 2))
    assert pipe.transformer.blocks[0].attn1._heft_capture is previous
    assert not hasattr(pipe.transformer.blocks[1].attn1, "_heft_capture")
    assert not pipe.transformer._forward_pre_hooks
    assert all(not block._forward_hooks for block in pipe.transformer.blocks)


def test_video_summary_separates_chunk_boundaries_and_scene_weights():
    sampled = [{"scene": scene, "video_chunks": [
        {"index": 0, "frame_ids": [0, 1]}, {"index": 1, "frame_ids": [2, 3]},
    ]} for scene in ("a", "b")]
    rows = []
    for scene in ("a", "b"):
        for i in range(3):
            value = 1.0 if scene == "a" and i != 1 else 0.0
            rows.append({"scene": scene, "frame_i": i, "frame_j": i + 1,
                         "feature": "hidden", "timestep": 49, "block": 15,
                         "head": -1, "mode": "cosine", "status": "ok",
                         "positive_similarity": value, "retrieval_voxel_hit": value})
    expanded, _, summary = summarize_video(rows, sampled)
    assert len(expanded) == 10
    values = {row["scope"]: row["retrieval_voxel_hit"] for row in summary}
    assert values == pytest.approx({"within_chunk": 0.5, "all_pairs": 1 / 3})


def test_video_cli_rejects_mixed_legacy_settings():
    args = parser().parse_args([])
    validate_args(args)
    assert args.chunk_size == 25 and args.timesteps == [49]
    args.timesteps = [300]
    with pytest.raises(ValueError, match="capture step 49"):
        validate_args(args)
    args = parser().parse_args(["--noise-mode", "legacy", "--timesteps", "300", "--shift", "5"])
    validate_args(args)
    assert args.timesteps == [300] and args.shift == 5
    args.timesteps = [300, 400]
    with pytest.raises(ValueError, match="one requested timestep"):
        validate_args(args)


def test_exact_video_noise_matches_legacy_and_resets_each_chunk(tmp_path):
    from diffusers import UniPCMultistepScheduler
    from .extract import selected_noise

    original = UniPCMultistepScheduler(
        use_flow_sigmas=True, flow_shift=3.0, prediction_type="flow_prediction",
    )
    original.save_pretrained(tmp_path / "scheduler")
    default, default_info = video_scheduler(tmp_path)
    original.set_timesteps(50)
    assert default_info["actual_timestep"] == float(original.timesteps[49])
    assert default_info["sigma"] == float(original.sigmas[49])
    assert default_info["capture_step"] == 49
    scheduler, info = video_scheduler(tmp_path, "legacy", 300, 5.0)
    legacy = selected_noise(300, 5.0)
    assert info["actual_timestep"] == legacy["actual_timestep"] == 299
    assert info["sigma"] == legacy["sigma"] == 0.299923837184906
    assert info["capture_step"] == 0 and info["num_inference_steps"] == 1
    assert scheduler.config.flow_shift == 5.0
    for dtype in (torch.float32, torch.bfloat16):
        clean = torch.tensor([[[[[1.25, -0.5]]]]], dtype=dtype)
        noise = torch.tensor([[[[[-0.75, 0.25]]]]], dtype=dtype)
        scheduler.set_timesteps(1)
        sigma = torch.tensor(legacy["sigma"], dtype=dtype)
        actual = scheduler.add_noise(clean, noise, scheduler.timesteps)
        torch.testing.assert_close(actual, (1 - sigma) * clean + sigma * noise, rtol=0, atol=0)
        result = scheduler.step(torch.zeros_like(actual), scheduler.timesteps[0], actual)[0]
        assert torch.isfinite(result).all() and scheduler.step_index == 1
        scheduler.set_timesteps(1)
        assert scheduler.step_index is None and scheduler.lower_order_nums == 0
        torch.testing.assert_close(scheduler.add_noise(clean, noise, scheduler.timesteps), actual, rtol=0, atol=0)
    with pytest.raises(ValueError, match="one active step"):
        scheduler.set_timesteps(50)


def test_exact_noise_uses_one_capture_and_records_requested_timestep():
    from .geometry import pair_geometry
    from .video_scan import score_features

    pipe = TinyVideoPipeline()
    extractor = WanVideoExtractor.__new__(WanVideoExtractor)
    extractor.pipeline = pipe
    extractor.model = replace(WAN_2_1, resolution=(32, 32), num_inference_steps=1, start_step=0)
    extractor.prompt_embeds, extractor.negative_prompt_embeds = pipe.context, pipe.context + 1
    job = ChunkJob(0, "test", None, 0, 0, 3, torch.zeros(5, 3, 32, 32), "", 42, 2)
    _, audit = extractor.extract(job, [0, 1], [0, 1], (2, 2))
    assert pipe.kwargs["num_inference_steps"] == 1 and pipe.kwargs["start_step"] == 0
    assert len(audit["transformer_forwards"]) == 2
    args = parser().parse_args(["--noise-mode", "legacy", "--timesteps", "300", "--shift", "5"])
    validate_args(args)
    xyz = np.array([[[0., 0., 1.], [1., 0., 1.]]])
    valid = np.ones((1, 2), bool)
    pair = pair_geometry(xyz, xyz, valid, valid, 0.1)
    features = np.tile(np.eye(2, dtype=np.float32)[None, None], (2, 1, 1, 1))
    rows = score_features(features, [pair], {"scene": "a", "sampled_frame_ids": [0, 1]}, args,
                          {"requested_timestep": 300, "actual_timestep": 299, "sigma": 0.299923837184906}, 15)
    assert rows[0]["timestep"] == 300 and rows[0]["actual_timestep"] == 299


def test_single_point_adapter_preserves_checkpoint_noise_inputs(tmp_path):
    from diffusers import UniPCMultistepScheduler

    UniPCMultistepScheduler(
        use_flow_sigmas=True, flow_shift=3.0, prediction_type="flow_prediction",
    ).save_pretrained(tmp_path / "scheduler")
    checkpoint, info = video_scheduler(tmp_path)
    probe, _ = video_scheduler(tmp_path, "legacy", 300, 5.0)
    probe.probe_timestep, probe.probe_sigma = info["actual_timestep"], info["sigma"]
    probe.set_timesteps(1)
    generator = torch.Generator().manual_seed(21)
    clean = torch.randn(1, 4, 5, 2, 2, generator=generator)
    noise = torch.randn(1, 4, 5, 2, 2, generator=generator)
    for dtype in (torch.float32, torch.bfloat16):
        old = checkpoint.add_noise(clean.to(dtype), noise.to(dtype), checkpoint.timesteps[49:50])
        new = probe.add_noise(clean.to(dtype), noise.to(dtype), probe.timesteps)
        torch.testing.assert_close(new, old, rtol=0, atol=0)
    assert float(probe.timesteps[0]) == float(checkpoint.timesteps[49])


def test_channel_loader_accepts_video_cache_and_rejects_changed_provenance(tmp_path):
    from .channel_scan import check_baseline, load_cached_scenes
    from .geometry import pair_geometry
    from .metrics import evaluate_pair
    from .report import write_json
    from .scan import atomic_npz, digest

    baseline, cache = tmp_path / "run", tmp_path / "cache"
    noise = {"requested_timestep": 49, "actual_timestep": 17.0, "sigma": 0.017}
    config = {
        "schema": "heft.scannet_video", "blocks": [15], "heads": [0], "timesteps": [49],
        "model_identity": {}, "code": {}, "runtime_code": {}, "video_protocol": {},
        "requested_noise": [noise], "dtype": "bf16", "device": "cuda:4", "grid": [1, 2],
        "voxel_size": 0.1, "seed": 42, "negative_min_distance": 0.2, "hit_distance": 0.1,
    }
    features = np.tile(np.eye(2, dtype=np.float32).reshape(1, 2, 1, 2), (2, 1, 1, 1))
    xyz = np.array([[[[0., 0., 1.], [1., 0., 1.]]]] * 2)
    valid = np.ones((2, 1, 2), bool)
    geometry = pair_geometry(xyz[0], xyz[1], valid[0], valid[1])
    sampled = []
    for scene in ("a", "b", "c"):
        chunks = [{"index": 0, "frame_ids": [0, 1], "seed": 42}]
        identity = {"geometry": scene, "model": {}, "code": {}, "runtime_code": {},
                    "protocol": {}, "noise": noise, "chunks": chunks,
                    "dtype": "bf16", "device": "cuda:4", "blocks": [15], "heads": [0], "grid": [1, 2]}
        key = digest(identity)
        sampled.append({"scene": scene, "geometry_cache": scene, "sampled_frame_ids": [0, 1],
                        "valid_tokens_per_frame": [2, 2], "shared_voxels_per_adjacent_pair": [2],
                        "video_chunks": chunks, "video_feature_key": key, "video_feature_identity": identity})
        atomic_npz(cache / "geometry" / scene / "geometry.npz", xyz=xyz, valid=valid)
        atomic_npz(cache / "video_features" / key / "block_15.npz", features=features)
        write_json(cache / "video_features" / key / "metadata.json", {"identity": identity, "feature_key": key, "noise": noise})
        tokens = features.transpose(0, 2, 3, 1)
        write_json(baseline / "pairs" / f"{scene}_k49_b15.json", [{
            "scene": scene, "timestep": 49, "block": 15, "frame_i": 0, "frame_j": 1,
            "actual_timestep": 17.0, "sigma": 0.017,
            **evaluate_pair(tokens[0], tokens[1], geometry, 42, 0.2, 0.1),
        }])
    write_json(baseline / "config.json", config)
    write_json(baseline / "status.json", {"status": "complete"})
    write_json(baseline / "sampled_frames.json", sampled)
    args = SimpleNamespace(baseline_run=baseline, cache_root=cache, block=15, timestep=49)
    protocol, scenes, provenance = load_cached_scenes(args)
    assert len(provenance) == 3
    assert len(check_baseline(scenes, protocol, tmp_path / "checked")) == 3
    write_json(baseline / "status.json", {"status": "running"})
    with pytest.raises(ValueError, match="must be complete"):
        load_cached_scenes(args)
    write_json(baseline / "status.json", {"status": "complete"})
    sampled[0]["video_feature_identity"]["device"] = "cuda:5"
    write_json(baseline / "sampled_frames.json", sampled)
    with pytest.raises(ValueError, match="Video provenance mismatch"):
        load_cached_scenes(args)
