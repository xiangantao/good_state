"""CPU regressions with real, randomly initialized small Wan models."""

import json
from dataclasses import replace

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from diffusers.pipelines.wan.pipeline_wan import WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

from heft.extraction.config import WAN_2_1, ChunkJob
from heft.extraction.worker import _prepare_input_video
from scripts.scannet.video_extract import WanVideoExtractor
from scripts.scannet.video_noise import video_scheduler

from .config import (
    CANDIDATES,
    HIDDEN_NAME,
    NOISE_SPECS,
    QK_TARGETS,
    ChannelSelection,
    ClipConfig,
)
from .extractor import WanLatentExtractor, load_noise_branches


@pytest.fixture(scope="module")
def components(tmp_path_factory):
    root = tmp_path_factory.mktemp("wan_components")
    scheduler = UniPCMultistepScheduler(
        use_flow_sigmas=True,
        flow_shift=3.0,
        prediction_type="flow_prediction",
    )
    scheduler.save_pretrained(root / "scheduler")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        vae = AutoencoderKLWan(
            base_dim=4,
            z_dim=4,
            # The vendored constructor incorrectly annotates this list as Tuple[int].
            dim_mult=[1, 1, 1, 1],  # pyright: ignore[reportArgumentType]
            num_res_blocks=1,
            latents_mean=[0.1, -0.2, 0.3, -0.4],
            latents_std=[0.8, 1.1, 1.4, 1.7],
        ).eval()
        transformer = WanTransformer3DModel(
            num_attention_heads=12,
            attention_head_dim=24,
            in_channels=4,
            out_channels=4,
            text_dim=8,
            freq_dim=8,
            ffn_dim=32,
            num_layers=17,
            rope_max_seq_len=32,
        ).eval()
        prompt = torch.randn(1, 3, 8)
        frames = torch.randint(0, 256, (25, 3, 16, 16), dtype=torch.uint8)
    # Precomputed prompts need no text modules; Wan supports UniPC despite its annotation.
    pipeline = WanPipeline(
        tokenizer=None,  # pyright: ignore[reportArgumentType]
        text_encoder=None,  # pyright: ignore[reportArgumentType]
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,  # pyright: ignore[reportArgumentType]
    )
    pipeline.set_progress_bar_config(disable=True)
    mask_path = root / "masks.json"
    # Deliberately preserve a non-sorted order to check channel identity, not just shape.
    mask_path.write_text(
        json.dumps({"ablation_iterative_256": list(range(255, -1, -1))})
    )
    yield root, pipeline, prompt, frames, ChannelSelection.load(mask_path)
    torch.set_num_threads(previous_threads)


def make_extractor(components, *, frames=16, stop=True):
    root, pipeline, prompt, _, channels = components
    return WanLatentExtractor(
        pipeline,
        prompt,
        channels,
        load_noise_branches(root),
        config=ClipConfig(
            frames=frames, resolution=(16, 16), grid=(1, 1), stop_after_capture=stop
        ),
        provenance={"model": "small random CPU Wan models"},
    )


def test_channel_selection_requires_a_named_fold_and_keeps_order(tmp_path):
    path = tmp_path / "fold_masks.json"
    rows = [
        {"held_out": name, "masks": {"ablation_iterative_256": indices}}
        for name, indices in (("a", list(range(256))), ("b", list(range(255, -1, -1))))
    ]
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="explicit held_out"):
        ChannelSelection.load(path)
    with pytest.raises(ValueError, match="Expected one"):
        ChannelSelection.load(path, held_out="missing")
    selected = ChannelSelection.load(path, held_out="b")
    assert selected.indices == tuple(range(255, -1, -1))
    assert selected.held_out == "b" and len(selected.source_sha256) == 64
    assert selected.metadata()["indices"] == list(selected.indices)
    rows[1]["masks"]["ablation_iterative_256"].reverse()
    path.write_text(json.dumps(rows))
    assert (
        ChannelSelection.load(path, held_out="b").source_sha256
        != selected.source_sha256
    )


@pytest.mark.parametrize(
    "indices",
    [list(range(255)), [0] * 256, list(range(255)) + [1536], list(range(255)) + [True]],
)
def test_channel_selection_rejects_invalid_tables(tmp_path, indices):
    path = tmp_path / "masks.json"
    path.write_text(json.dumps({"ablation_iterative_256": indices}))
    with pytest.raises(ValueError):
        ChannelSelection.load(path)


def test_local_loader_rejects_remote_model_ids_before_loading(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("Attempted to load a model for a non-local path")

    monkeypatch.setattr(WanPipeline, "from_pretrained", forbidden)
    with pytest.raises(ValueError, match="existing local"):
        WanLatentExtractor.from_local(
            tmp_path / "missing", channel_mask=tmp_path / "absent.json"
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_exact_noise_values_and_arithmetic(components, dtype):
    branches = load_noise_branches(components[0])
    clean = torch.tensor([1.25, -0.5], dtype=dtype).reshape(1, 1, 1, 1, 2)
    noise = torch.tensor([-0.75, 0.25], dtype=dtype).reshape_as(clean)
    for spec in NOISE_SPECS:
        scheduler, info = branches[spec.name]
        assert info["actual_timestep"] == spec.actual_timestep
        assert info["sigma"] == spec.sigma and info["shift"] == spec.shift
        timestep = scheduler.timesteps[info["capture_step"] : info["capture_step"] + 1]
        actual = scheduler.add_noise(clean, noise, timestep)
        sigma = torch.tensor(spec.sigma, dtype=dtype)
        torch.testing.assert_close(
            actual, (1 - sigma) * clean + sigma * noise, rtol=0, atol=0
        )


def test_real_vae_padding_is_causal_and_crop_precedes_random_sampling(components):
    _, pipe, _, frames, _ = components
    extractor = make_extractor(components)
    with torch.inference_mode():
        cropped, padding = extractor._prepare_posterior(frames[:16])
        assert padding == 1 and cropped.mean.shape == (1, 4, 16, 2, 2)
        prepared = _prepare_input_video(frames[:17], (16, 16))
        video = pipe.video_processor.preprocess_video(prepared, height=16, width=16)
        first = pipe.vae.encode(video).latent_dist.parameters.clone()
        video[:, :, 16] += 10
        second = pipe.vae.encode(video).latent_dist.parameters
        torch.testing.assert_close(first[:, :, :16], second[:, :, :16], rtol=0, atol=0)
        torch.testing.assert_close(cropped.parameters, first[:, :, :16], rtol=0, atol=0)
        expected = DiagonalGaussianDistribution(first[:, :, :16].contiguous())
        rng1, rng2 = (torch.Generator().manual_seed(31) for _ in range(2))
        torch.testing.assert_close(
            cropped.sample(rng1), expected.sample(rng2), rtol=0, atol=0
        )
        assert torch.equal(rng1.get_state(), rng2.get_state())
    assert all(value is None for value in pipe.vae._enc_feat_map)


def test_six_tensors_five_candidates_and_reproducibility(components):
    _, pipe, _, frames, channels = components
    extractor = make_extractor(components)
    counts = {"vae": 0, "transformer": 0, "after_b15": 0}

    def count_vae(module, inputs):
        counts["vae"] += 1

    def count_transformer(module, inputs):
        counts["transformer"] += 1

    def count_tail(module, inputs):
        counts["after_b15"] += 1

    handles = [
        pipe.vae.quant_conv.register_forward_pre_hook(count_vae),
        pipe.transformer.register_forward_pre_hook(count_transformer),
        pipe.transformer.blocks[16].register_forward_pre_hook(count_tail),
    ]
    state = torch.random.get_rng_state().clone()
    try:
        result = extractor.extract(
            frames[:16], seed=99, frame_ids=list(range(0, 32, 2)), clip_id="clip"
        )
    finally:
        for handle in handles:
            handle.remove()
    assert counts == {"vae": 1, "transformer": 2, "after_b15": 0}
    assert torch.equal(state, torch.random.get_rng_state())
    assert set(result.tensors) == {
        name for parts in CANDIDATES.values() for name in parts
    }
    assert len(result.tensors) == 6 and len(result.metadata["candidates"]) == 5
    assert result.candidate("l15h7_qk_t57").shape == (16, 48, 1, 1)
    assert result.tensors[HIDDEN_NAME].shape == (16, 256, 1, 1)
    for tensor in result.tensors.values():
        assert tensor.device.type == "cpu" and tensor.dtype == torch.bfloat16
        assert not tensor.requires_grad
    assert result.metadata["frame_ids"] == list(range(0, 32, 2))
    assert result.metadata["channel_selection"]["indices"] == list(channels.indices)
    assert result.metadata["vae_input_frames"] == 17
    for branch in result.metadata["noise_branches"].values():
        assert branch["forward"]["shape"] == [1, 4, 16, 2, 2]
        assert branch["forward"]["stopped_after_block"] == 15
        assert (
            branch["conditional_forwards"] == 1
            and branch["scheduler_updates_executed"] == 0
        )
    json.dumps(result.metadata, allow_nan=False)
    repeated = extractor.extract(frames[:16], seed=99)
    for name in result.tensors:
        torch.testing.assert_close(
            result.tensors[name], repeated.tensors[name], rtol=0, atol=0
        )
    assert not torch.equal(
        result.tensors["l15h2_k_t57"], result.tensors["l15h2_k_t299"]
    )


def test_early_stop_matches_full_forward_and_skips_later_blocks(components):
    _, pipe, _, frames, _ = components
    early = make_extractor(components, stop=True).extract(frames[:16], seed=11)
    calls = []
    handle = pipe.transformer.blocks[16].register_forward_pre_hook(
        lambda m, i: calls.append(16)
    )
    try:
        full = make_extractor(components, stop=False).extract(frames[:16], seed=11)
    finally:
        handle.remove()
    assert calls == [16, 16]
    for name in full.tensors:
        torch.testing.assert_close(
            early.tensors[name], full.tensors[name], rtol=0, atol=0
        )
    assert (
        full.metadata["noise_branches"]["t299"]["forward"]["stopped_after_block"]
        is None
    )


def test_25_frame_features_match_existing_complete_video_pipeline(components):
    root, pipe, prompt, frames, channels = components
    ours = make_extractor(components, frames=25).extract(frames, seed=42)
    for spec in NOISE_SPECS:
        scheduler, info = video_scheduler(
            root, spec.mode, spec.requested_timestep, spec.shift
        )
        pipe.register_modules(scheduler=scheduler)
        reference = WanVideoExtractor.__new__(WanVideoExtractor)
        reference.pipeline = pipe
        reference.model = replace(
            WAN_2_1,
            resolution=(16, 16),
            num_inference_steps=info["num_inference_steps"],
            start_step=info["capture_step"],
        )
        reference.prompt_embeds, reference.negative_prompt_embeds = prompt, prompt + 1
        job = ChunkJob(0, "reference", root, 0, 0, 25, frames, "", 42)
        heads = [2, 7, 10]
        with torch.random.fork_rng(devices=[]):
            captured, audit = reference.extract(job, [13, 15], heads, (1, 1))
        assert len(audit["transformer_forwards"]) == 2
        for (layer, kind, head), name in QK_TARGETS[spec.name].items():
            expected = captured[layer][kind][:, heads.index(head)].to(torch.bfloat16)
            torch.testing.assert_close(ours.tensors[name], expected, rtol=0, atol=0)
        if spec.name == "t299":
            expected = captured[15]["features"][:, channels.indices].to(torch.bfloat16)
            torch.testing.assert_close(
                ours.tensors[HIDDEN_NAME], expected, rtol=0, atol=0
            )


def test_extractor_preserves_temporal_context(components):
    frames = components[3][:16].clone()
    extractor = make_extractor(components)
    first = extractor.extract(frames, seed=7)
    frames[-1] = 255 - frames[-1]
    second = extractor.extract(frames, seed=7)
    assert not torch.equal(
        first.tensors[HIDDEN_NAME][0], second.tensors[HIDDEN_NAME][0]
    )
    assert not torch.equal(
        first.tensors["l15h2_k_t57"][0], second.tensors["l15h2_k_t57"][0]
    )


def test_failure_restores_hooks_and_allows_a_later_extraction(components):
    _, pipe, _, frames, _ = components
    extractor = make_extractor(components)
    previous = object()
    pipe.transformer.blocks[13].attn1._heft_capture = previous

    def fail(module, inputs):
        raise RuntimeError("injected failure")

    handle = pipe.transformer.blocks[15].register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="injected failure"):
            extractor.extract(frames[:16])
        assert pipe.transformer.blocks[13].attn1._heft_capture is previous
        assert not hasattr(pipe.transformer.blocks[15].attn1, "_heft_capture")
        assert not pipe.transformer._forward_pre_hooks
        assert not pipe.transformer.blocks[15]._forward_hooks
    finally:
        handle.remove()
        del pipe.transformer.blocks[13].attn1._heft_capture
    assert len(extractor.extract(frames[:16]).tensors) == 6


def test_invalid_clip_fails_before_vae_encoding(components, monkeypatch):
    extractor = make_extractor(components)

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid clip reached VAE encoding")

    monkeypatch.setattr(extractor.pipeline.vae, "encode", forbidden)
    with pytest.raises(ValueError, match="CPU uint8 RGB"):
        extractor.extract(torch.zeros(16, 3, 16, 16))
    with pytest.raises(ValueError, match="frame_ids"):
        extractor.extract(components[3][:16], frame_ids=[0])
