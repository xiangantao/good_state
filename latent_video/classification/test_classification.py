"""CPU checks for online data, fusion, training, metrics, and resumable probing."""

import copy
import importlib
import json
import random
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from latent_video import ClipConfig, ClipLatents, test_extractor
from latent_video.config import CANDIDATES, HIDDEN_NAME
from latent_video.extractor import WanLatentExtractor, load_noise_branches

from .checkpoint import load_checkpoint, save_checkpoint
from .config import DataConfig, OptimizationConfig, RunConfig
from .data import (
    IMAGENET_NORMALIZATION,
    EvaluationSampler,
    OnlineVideoDataset,
    Sample,
    WanRGBTransform,
    manifest_identity,
    read_manifest,
    write_manifest,
)
from .encoder import FUSION_ORDER, OnlineWanEncoder
from .engine import ProbeOptimization, probabilities, run_epoch
from .train import preflight

VJEPA_ROOT = Path(__file__).resolve().parents[3] / "vjepa2"
small_wan_components = test_extractor.components


@pytest.fixture(autouse=True)
def cpu_random_state():
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    py_state, np_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        torch.random.set_rng_state(torch.Generator().manual_seed(123).get_state())
        yield
    random.setstate(py_state)
    np.random.set_state(np_state)
    torch.set_num_threads(threads)


def test_configuration_has_one_optimizer_and_rejects_wrong_protocol(tmp_path):
    options = OptimizationConfig()
    assert options.upstream_kwargs() == [
        {
            "ref_wd": 0.1,
            "final_wd": 0.1,
            "start_lr": 0.0003,
            "ref_lr": 0.0003,
            "final_lr": 0.0,
            "warmup": 0.0,
        }
    ]
    cfg = RunConfig(DataConfig("train.csv", "val.csv"))
    assert (
        cfg.clip.stop_after_capture
        and cfg.num_heads == 16
        and cfg.num_probe_blocks == 4
    )
    with pytest.raises(ValueError, match="16 frames"):
        replace(cfg, clip=ClipConfig(frames=17))
    with pytest.raises(ValueError, match="divide"):
        replace(cfg, num_heads=12)
    with pytest.raises(ValueError, match="finite"):
        OptimizationConfig(lr=float("nan"))
    path = tmp_path / "invalid.yaml"
    path.write_text("data: {train: train.csv, val: val.csv}\ncache: true\n")
    with pytest.raises(ValueError, match="Unknown"):
        RunConfig.load(path)


def test_official_json_and_path_csv_preserve_ids_labels_and_spaces(tmp_path):
    videos = tmp_path / "videos with spaces"
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({f"Doing something {i}": str(i) for i in range(174)}))
    manifest = tmp_path / "train.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "id": "123",
                    "template": "Doing [something] 17",
                    "label": "free form text",
                },
                {"id": "456", "template": "Doing [something] 2"},
            ]
        )
    )
    config = RunConfig(
        DataConfig(str(manifest), "val.csv", videos=str(videos), labels=str(labels))
    )
    records = read_manifest(manifest, config)
    assert [(r.id, r.label) for r in records] == [("123", 17), ("456", 2)]
    csv_path = tmp_path / "paths.csv"
    write_manifest(csv_path, records)
    assert read_manifest(csv_path, config) == records
    assert manifest_identity(records) != manifest_identity(records[::-1])
    write_manifest(csv_path, records + records[:1])
    with pytest.raises(ValueError, match="Duplicate"):
        read_manifest(csv_path, config)


class RandomVideoSource:
    def get_item_video(self, index):
        clip = torch.randint(0, 256, (2, 3, 2, 2), dtype=torch.uint8)
        frame_index = np.random.randint(0, 100)
        return [[clip]], index, [np.array([frame_index, frame_index + 1])]


def test_training_resamples_each_epoch_validation_is_fixed_and_rng_is_restored():
    records = [Sample("a", "a.webm", 0)]
    dataset = OnlineVideoDataset(RandomVideoSource(), records, seed=42, training=True)
    before = torch.random.get_rng_state().clone()
    np_before, py_before = np.random.get_state(), random.getstate()
    first, repeated = dataset[0], dataset[0]
    assert torch.equal(first["clips"][0][0], repeated["clips"][0][0])
    assert torch.equal(before, torch.random.get_rng_state())
    np.testing.assert_equal(np_before, np.random.get_state())
    assert py_before == random.getstate()
    dataset.set_epoch(1)
    later = dataset[0]
    assert not torch.equal(first["clips"][0][0], later["clips"][0][0])
    assert not torch.equal(first["feature_seeds"], later["feature_seeds"])
    validation = OnlineVideoDataset(
        RandomVideoSource(), records, seed=42, training=False
    )
    first = validation[0]
    validation.set_epoch(9)
    assert torch.equal(first["clips"][0][0], validation[0]["clips"][0][0])
    assert torch.equal(first["feature_seeds"], validation[0]["feature_seeds"])


@pytest.mark.parametrize("raises", [False, True])
def test_failed_decode_is_reported_without_replacing_a_video(raises):
    def decode(index):
        if raises:
            raise OSError("decoder failure")

    source = SimpleNamespace(get_item_video=decode)
    dataset = OnlineVideoDataset(
        source, [Sample("42", "missing.webm", 0)], seed=0, training=True
    )
    with pytest.raises(RuntimeError, match="42"):
        dataset[0]


def test_rgb_adapter_undoes_normalization_before_wan():
    raw = torch.randint(0, 256, (3, 2, 4, 5), dtype=torch.uint8)
    mean, std = (torch.tensor(v).reshape(3, 1, 1, 1) for v in IMAGENET_NORMALIZATION)
    normalized = (raw.float() / 255 - mean) / std
    adapter = WanRGBTransform(lambda _: [normalized])
    result = adapter(None)[0]
    assert torch.equal(result, raw.permute(1, 0, 2, 3))
    assert result.dtype == torch.uint8 and result.is_contiguous()


def test_actual_vjepa_augmentations_with_wan_rgb_adapter():
    if str(VJEPA_ROOT) not in sys.path:
        sys.path.insert(0, str(VJEPA_ROOT))
    transforms = importlib.import_module("evals.video_classification_frozen.utils")
    frames = np.random.default_rng(42).integers(0, 256, (16, 24, 32, 3), dtype=np.uint8)
    for training in (True, False):
        transform = transforms.make_transforms(
            training=training,
            crop_size=16,
            num_views_per_clip=3,
            auto_augment=True,
            reprob=0.25,
            random_horizontal_flip=False,
        )
        result = WanRGBTransform(transform)(frames)
        assert len(result) == (1 if training else 3)
        assert all(
            view.shape == (16, 3, 16, 16) and view.dtype == torch.uint8
            for view in result
        )


class SyntheticExtractor:
    head_channels = 128
    device = torch.device("cpu")
    config = ClipConfig(frames=2, resolution=(16, 32), grid=(1, 2))

    def __init__(self):
        self.calls = []

    @torch.inference_mode()
    def extract(self, frames, *, seed, frame_ids, clip_id):
        self.calls.append((seed, frame_ids, clip_id))
        tensors = {}
        value = float(frames[0, 0, 0, 0])
        for i, name in enumerate(
            name for parts in CANDIDATES.values() for name in parts
        ):
            width = 256 if name == HIDDEN_NAME else 128
            tensors[name] = torch.full(
                (2, width, 1, 2), value + i, dtype=torch.bfloat16
            )
        return ClipLatents(tensors, {})


def online_batch():
    return {
        "id": ["a", "b"],
        "label": torch.tensor([0, 1]),
        "clips": [
            [
                torch.stack(
                    [
                        torch.full((2, 3, 2, 2), s * 10 + b, dtype=torch.uint8)
                        for b in range(2)
                    ]
                )
            ]
            for s in range(2)
        ],
        "frame_indices": [torch.tensor([[0, 1], [2, 3]]) for _ in range(2)],
        "feature_seeds": torch.arange(4).reshape(2, 2, 1),
    }


def test_online_fusion_keeps_channel_and_segment_order_and_reextracts():
    extractor = SyntheticExtractor()
    encoder = OnlineWanEncoder(extractor)
    batch = online_batch()
    first = encoder.encode_view(batch, 0)
    second = encoder.encode_view(batch, 0)
    assert encoder.embed_dim == 896 and first.shape == (2, 8, 896)
    assert len(extractor.calls) == 8
    assert torch.equal(first, second) and not first.is_inference()
    assert torch.equal(
        first[0, 0, torch.tensor([0, 128, 256, 384, 512, 768])].float(),
        torch.arange(6).float(),
    )
    assert torch.equal(first[0, 4], first[0, 0] + 10)
    assert FUSION_ORDER == tuple(CANDIDATES)
    head = nn.Linear(896, 3)
    head(first.float().mean(dim=1)).sum().backward()
    assert head.weight.grad is not None


class TensorEncoder:
    device = torch.device("cpu")

    def __init__(self):
        self.frozen = nn.Parameter(torch.ones(1))
        self.calls = 0

    def encode_view(self, batch, view):
        self.calls += 1
        return batch["clips"][0][view] * self.frozen


class MeanHead(nn.Module):
    def __init__(self, width=4, classes=6):
        super().__init__()
        self.linear = nn.Linear(width, classes)

    def forward(self, tokens):
        return self.linear(tokens.mean(1))


class CounterSchedule:
    def __init__(self):
        self._step = 0

    def step(self):
        self._step += 1


def make_optimization(model):
    return ProbeOptimization(
        torch.optim.AdamW(model.parameters(), lr=0.01),
        None,
        CounterSchedule(),
        CounterSchedule(),
    )


def tensor_batch(values, labels):
    return {
        "clips": [[values]],
        "label": torch.tensor(labels),
        "id": [str(i) for i in range(len(labels))],
    }


def test_training_updates_only_head_and_reextracts_on_every_epoch():
    model, encoder = MeanHead(), TensorEncoder()
    optimization = make_optimization(model)
    batches = [tensor_batch(torch.randn(2, 3, 4), [0, 1])]
    original = model.linear.weight.detach().clone()
    for _ in range(2):
        result = run_epoch(
            model,
            encoder,
            batches,
            training=True,
            optimization=optimization,
            amp_dtype="float32",
        )
        assert result["samples"] == 2
    assert encoder.calls == 2 and encoder.frozen.grad is None
    assert not torch.equal(original, model.linear.weight)
    assert optimization.scheduler._step == optimization.wd_scheduler._step == 2


def test_step_monitoring_preserves_updates_and_reports_final_partial_window():
    control = MeanHead()
    monitored = copy.deepcopy(control)
    batches = [
        tensor_batch(torch.randn(count, 3, 4), [0] * count) for count in (2, 2, 1)
    ]
    expected = run_epoch(
        control,
        TensorEncoder(),
        batches,
        training=True,
        optimization=make_optimization(control),
        amp_dtype="float32",
    )
    events = []
    actual = run_epoch(
        monitored,
        TensorEncoder(),
        batches,
        training=True,
        optimization=make_optimization(monitored),
        amp_dtype="float32",
        on_step=events.append,
        log_every=2,
        epoch=3,
    )
    assert actual == expected
    for key, value in control.state_dict().items():
        torch.testing.assert_close(value, monitored.state_dict()[key], rtol=0, atol=0)
    assert [event["step"] for event in events] == [11, 12]
    assert [event["batch/samples"] for event in events] == [4, 1]
    assert events[-1]["epoch"] == 4
    assert all(event["timing/videos_per_second"] > 0 for event in events)
    assert all(event["optim/lr"] == 0.01 for event in events)


def test_step_monitoring_aggregates_sample_counts_and_maximum_memory(monkeypatch):
    from . import engine

    control = MeanHead()
    monitored = copy.deepcopy(control)
    batches = [tensor_batch(torch.randn(2, 3, 4), [0, 1])]
    local = run_epoch(
        control,
        TensorEncoder(),
        batches,
        training=True,
        optimization=make_optimization(control),
        amp_dtype="float32",
    )
    reductions = []

    def all_reduce(value, op):
        reductions.append(op)
        if op == "sum":
            value += torch.tensor([3, 6, 2, 3], dtype=value.dtype)
        else:
            value.copy_(torch.maximum(value, torch.tensor([1, 3], dtype=value.dtype)))

    monkeypatch.setattr(
        engine,
        "dist",
        SimpleNamespace(
            is_available=lambda: True,
            is_initialized=lambda: True,
            all_reduce=all_reduce,
            ReduceOp=SimpleNamespace(SUM="sum", MAX="max"),
        ),
    )
    events = []
    result = run_epoch(
        monitored,
        TensorEncoder(),
        batches,
        training=True,
        optimization=make_optimization(monitored),
        amp_dtype="float32",
        on_step=events.append,
    )
    assert reductions == ["sum", "max", "sum"]
    assert events[0]["batch/samples"] == result["samples"] == 5
    assert events[0]["batch/loss"] == pytest.approx((local["loss"] * 2 + 6) / 5)
    assert events[0]["batch/top1"] == pytest.approx((local["top1"] * 2 + 200) / 5)
    assert events[0]["memory/peak_allocated_gib"] == 3


def test_actual_small_wan_features_support_head_backward(small_wan_components):
    root, pipeline, prompt, frames, channels = small_wan_components
    extractor = WanLatentExtractor(
        pipeline,
        prompt,
        channels,
        load_noise_branches(root),
        config=ClipConfig(resolution=(16, 16), grid=(1, 1)),
    )
    encoder = OnlineWanEncoder(extractor)
    model = MeanHead(width=encoder.embed_dim)
    batch = {
        "clips": [[frames[:16].unsqueeze(0)], [frames[9:25].unsqueeze(0)]],
        "id": ["small-wan"],
        "label": torch.tensor([0]),
        "frame_indices": [
            torch.arange(16).unsqueeze(0),
            torch.arange(9, 25).unsqueeze(0),
        ],
        "feature_seeds": torch.tensor([[[7], [8]]]),
    }
    original = model.linear.weight.detach().clone()
    metrics = run_epoch(
        model,
        encoder,
        [batch],
        training=True,
        optimization=make_optimization(model),
        amp_dtype="float32",
    )
    assert metrics["samples"] == 1
    assert not torch.equal(original, model.linear.weight)
    assert all(
        p.grad is None and not p.requires_grad
        for p in pipeline.transformer.parameters()
    )
    assert all(
        p.grad is None and not p.requires_grad for p in pipeline.vae.parameters()
    )


def test_validation_counts_last_batch_by_sample_and_saves_all_predictions(tmp_path):
    encoder = TensorEncoder()
    logits = torch.eye(6)[[0, 1, 0]] * 10
    batches = [tensor_batch(logits[:2], [0, 1]), tensor_batch(logits[2:], [2])]
    batches[1]["id"] = ["2"]
    path = tmp_path / "predictions.jsonl"
    stats = run_epoch(
        nn.Identity(),
        encoder,
        batches,
        training=False,
        amp_dtype="float32",
        predictions=path,
    )
    assert stats["samples"] == 3 and stats["top1"] == pytest.approx(200 / 3)
    assert {json.loads(line)["id"] for line in path.read_text().splitlines()} == {
        "0",
        "1",
        "2",
    }
    assert not list(tmp_path.glob("*.tmp"))


def test_multiview_prediction_averages_probabilities_not_logits():
    logits = [torch.tensor([[9.0, 0.0]]), torch.tensor([[0.0, 1.0]])]
    expected = (logits[0].softmax(1) + logits[1].softmax(1)) / 2
    torch.testing.assert_close(probabilities(logits), expected)
    assert not torch.allclose(expected, torch.stack(logits).mean(0).softmax(1))


@pytest.mark.parametrize("size,world", [(1, 8), (17, 8), (24, 8)])
def test_validation_shards_cover_each_video_exactly_once(size, world):
    shards = [EvaluationSampler(size, rank, world) for rank in range(world)]
    samples = [index for shard in shards for index in shard]
    assert sorted(samples) == list(range(size))
    assert sum(map(len, shards)) == size


def test_resume_matches_uninterrupted_optimizer_and_rejects_changed_protocol(tmp_path):
    head, encoder = MeanHead(), TensorEncoder()
    opt = make_optimization(head)
    batch = tensor_batch(torch.randn(2, 3, 4), [0, 1])
    run_epoch(
        head, encoder, [batch], training=True, optimization=opt, amp_dtype="float32"
    )
    protocol = {
        "representation": {"channels": 896},
        "training": {"lr": 0.01},
        "validation_manifest": "abc",
    }
    path = tmp_path / "latest.pt"
    save_checkpoint(
        path,
        head,
        opt,
        epoch=1,
        best_top1=50.0,
        protocol=protocol,
        device=encoder.device,
    )
    draws = torch.rand(3)
    run_epoch(
        head, encoder, [batch], training=True, optimization=opt, amp_dtype="float32"
    )
    restored = MeanHead()
    restored_opt = make_optimization(restored)
    assert load_checkpoint(
        path,
        restored,
        protocol=protocol,
        device=encoder.device,
        optimization=restored_opt,
    ) == (1, 50.0)
    assert torch.equal(draws, torch.rand(3))
    run_epoch(
        restored,
        encoder,
        [batch],
        training=True,
        optimization=restored_opt,
        amp_dtype="float32",
    )
    for key, tensor in head.state_dict().items():
        torch.testing.assert_close(tensor, restored.state_dict()[key], rtol=0, atol=0)
    assert restored_opt.scheduler._step == restored_opt.wd_scheduler._step == 2
    payload = torch.load(path, weights_only=False)
    assert len(payload["classifiers"]) == len(payload["opt"]) == 1
    assert "encoder" not in payload
    with pytest.raises(ValueError, match="protocol differs"):
        load_checkpoint(
            path,
            restored,
            protocol={**protocol, "representation": {"channels": 128}},
            device=encoder.device,
        )


def test_preflight_reports_missing_data_and_mask_without_model_loading(tmp_path):
    config = RunConfig(
        DataConfig(str(tmp_path / "train.json"), str(tmp_path / "val.json")),
        model_path=str(tmp_path / "model"),
        channel_mask=str(tmp_path / "masks.json"),
    )
    issues = preflight(config)
    assert any("data.train" in issue for issue in issues)
    assert any("model_path" in issue for issue in issues)
    assert any("channel_mask" in issue for issue in issues)


def test_original_attentive_head_accepts_fused_features_and_backward():
    pytest.importorskip(
        "timm",
        reason="Original V-JEPA2 head dependency is not installed; no downloads allowed",
    )
    if str(VJEPA_ROOT) not in sys.path:
        sys.path.insert(0, str(VJEPA_ROOT))
    module = importlib.import_module("src.models.attentive_pooler")
    head = module.AttentiveClassifier(
        embed_dim=896,
        num_heads=16,
        depth=4,
        num_classes=174,
        use_activation_checkpointing=True,
    )
    encoder = OnlineWanEncoder(SyntheticExtractor())
    logits = head(encoder.encode_view(online_batch(), 0).float())
    assert logits.shape == (2, 174)
    logits.square().mean().backward()
    assert all(p.grad is not None for p in head.parameters())
