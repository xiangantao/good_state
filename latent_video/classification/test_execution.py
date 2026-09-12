"""Ordering, ownership, lifetime, and retained-checkpoint regressions."""

import threading
from dataclasses import replace

import pytest
import torch

from ..config import ClipConfig
from ..extractor import WanLatentExtractor, load_noise_branches
from . import test_classification
from .checkpoint import publish_checkpoint_alias
from .config import DataConfig, RunConfig
from .encoder import OnlineWanEncoder
from .execution import COMPILED_BLOCKS, ExtractionPool, compile_uncaptured_blocks
from .test_classification import SyntheticExtractor, online_batch

small_wan_components = test_classification.small_wan_components


class ThreadCheckedExtractor(SyntheticExtractor):
    def __init__(self):
        super().__init__()
        self.owner = None

    def extract_batch(self, *args, **kwargs):
        current = threading.get_ident()
        if self.owner is None:
            self.owner = current
        assert current == self.owner
        return super().extract_batch(*args, **kwargs)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
def test_lane_fusion_matches_serial_including_partial_groups_and_backward(batch_size):
    extractors = [ThreadCheckedExtractor(), ThreadCheckedExtractor()]
    pool = ExtractionPool(extractors)
    try:
        serial = OnlineWanEncoder(SyntheticExtractor(), extract_batch_size=batch_size)
        parallel = OnlineWanEncoder(
            extractors[0], extract_batch_size=batch_size, pool=pool
        )
        for _ in range(2):
            expected = serial.encode_view(online_batch(), 0)
            actual = parallel.encode_view(online_batch(), 0)
            assert torch.equal(expected, actual)
            assert not actual.is_inference()
            head = torch.nn.Linear(896, 2)
            head(actual.float()).sum().backward()
            assert head.weight.grad is not None
        assert extractors[0].owner != threading.get_ident()
        if extractors[1].owner is not None:
            assert extractors[0].owner != extractors[1].owner
    finally:
        pool.close()
    pool.close()
    with pytest.raises(RuntimeError, match="closed"):
        pool.extract_groups([])


def test_pool_rejects_reusing_one_stateful_extractor():
    source = SyntheticExtractor()
    with pytest.raises(ValueError, match="distinct"):
        ExtractionPool([source, source])


def test_compile_targets_leave_capture_blocks_native(monkeypatch):
    from types import SimpleNamespace

    forwards = [object() for _ in range(30)]
    blocks = [SimpleNamespace(forward=forward) for forward in forwards]
    compiled = []

    def compile_fn(forward, *, fullgraph, dynamic):
        assert fullgraph and not dynamic
        compiled.append(forward)
        return (forward, "compiled")

    monkeypatch.setattr(torch, "compile", compile_fn)
    compile_uncaptured_blocks(
        SimpleNamespace(
            pipeline=SimpleNamespace(transformer=SimpleNamespace(blocks=blocks))
        )
    )
    assert compiled == [forwards[i] for i in COMPILED_BLOCKS]
    assert len(compiled) == 14
    for index in (13, 15, *range(16, 30)):
        assert blocks[index].forward is forwards[index]


def test_replica_owns_weights_schedulers_and_matches_native(small_wan_components):
    root, pipeline, prompt, frames, channels = small_wan_components
    source = WanLatentExtractor(
        pipeline,
        prompt,
        channels,
        load_noise_branches(root),
        config=ClipConfig(resolution=(16, 16), grid=(1, 1)),
    )
    replica = source.replica()
    for name in ("vae", "transformer"):
        a, b = getattr(source.pipeline, name), getattr(replica.pipeline, name)
        assert a is not b
        for x, y in zip(a.parameters(), b.parameters(), strict=True):
            assert torch.equal(x, y) and x.data_ptr() != y.data_ptr()
    assert all(
        source.branches[name][0] is not replica.branches[name][0]
        for name in source.branches
    )
    expected, actual = source.extract(frames[:16]), replica.extract(frames[:16])
    for key in expected.tensors:
        torch.testing.assert_close(
            expected.tensors[key], actual.tensors[key], rtol=0, atol=0
        )


def test_latest_alias_does_not_overwrite_epoch_under_async_evaluation(tmp_path):
    first, second, latest = (
        tmp_path / name for name in ("epoch_0001.pt", "epoch_0002.pt", "latest.pt")
    )
    torch.save({"epoch": 1}, first)
    publish_checkpoint_alias(first, latest)
    with latest.open("rb") as evaluating:
        torch.save({"epoch": 2}, second)
        publish_checkpoint_alias(second, latest)
        assert torch.load(evaluating, weights_only=True)["epoch"] == 1
    assert torch.load(first, weights_only=True)["epoch"] == 1
    assert torch.load(latest, weights_only=True)["epoch"] == 2


def test_execution_config_and_supplied_eight_gpu_recipe():
    config = RunConfig(DataConfig("train.csv", "val.csv"))
    for values in (
        {"extract_lanes": 0},
        {"extract_lanes": 3},
        {"compile_blocks": 1},
        {"validate_every": -1},
    ):
        with pytest.raises((ValueError, TypeError)):
            replace(config, **values)
    from pathlib import Path

    recipe = RunConfig.load(Path(__file__).parent / "configs/ssv2.yaml")
    assert recipe.extract_lanes == 2 and recipe.extract_batch_size == 32
    assert recipe.optimization.batch_size == 32 and recipe.validate_every == 0
    assert recipe.clip.resolution == (256, 256) and recipe.compile_blocks
    assert recipe.wandb.mode == "offline"


def test_train_only_retains_epochs_resumes_and_evaluates_in_separate_directory(
    tmp_path, monkeypatch
):
    import json
    from types import SimpleNamespace

    from . import train
    from .test_classification import MeanHead, TensorEncoder, make_optimization

    training, validation = tmp_path / "train.csv", tmp_path / "val.csv"
    for manifest, prefix, count in ((training, "train", 5), (validation, "val", 3)):
        rows = []
        for index in range(count):
            video = tmp_path / f"{prefix}{index}.webm"
            video.touch()
            rows.append(f"{video} {index % 2}\n")
        manifest.write_text("".join(rows))
    config = RunConfig(
        DataConfig(str(training), str(validation)),
        num_workers=0,
        validate_every=0,
        output_dir=str(tmp_path / "training"),
        optimization=replace(
            RunConfig(DataConfig("a", "b")).optimization,
            batch_size=2,
            num_epochs=2,
            amp_dtype="float32",
        ),
    )
    builds = []

    class Dataset:
        def __init__(self, records):
            self.records = records

        def __len__(self):
            return len(self.records)

        def set_epoch(self, epoch):
            pass

        def __getitem__(self, index):
            value = torch.randn(3, 4, generator=torch.Generator().manual_seed(index))
            return {
                "clips": [[value]],
                "label": self.records[index].label,
                "id": self.records[index].id,
            }

    def build(runtime, manifest, records, config, *, training):
        builds.append(training)
        return Dataset(records)

    class Encoder(TensorEncoder):
        embed_dim = 4

        def metadata(self):
            return {"test": True}

    monkeypatch.setattr(train, "preflight", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        train, "distributed_device", lambda: (torch.device("cpu"), 0, 1)
    )
    monkeypatch.setattr(train, "build_dataset", build)
    monkeypatch.setattr(
        train,
        "load_upstream",
        lambda root: SimpleNamespace(
            provenance={}, classifier=lambda **kwargs: MeanHead()
        ),
    )
    monkeypatch.setattr(
        train,
        "WanLatentExtractor",
        SimpleNamespace(from_local=lambda *args, **kwargs: object()),
    )
    monkeypatch.setattr(
        train,
        "ExtractionPool",
        lambda *args, **kwargs: SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(train, "OnlineWanEncoder", lambda *args, **kwargs: Encoder())
    monkeypatch.setattr(
        train.ProbeOptimization,
        "create",
        lambda runtime, head, config, steps: make_optimization(head),
    )
    train.run(config)
    output = config.path(config.output_dir)
    assert builds == [True]
    assert (output / "epoch_0001.pt").is_file() and (output / "epoch_0002.pt").is_file()
    assert not (output / "best.pt").exists()
    metrics = (output / "metrics.jsonl").read_text()
    assert all(json.loads(line)["val"] is None for line in metrics.splitlines())
    resumed = replace(config, output_dir=str(tmp_path / "resumed"))
    train.run(resumed, resume=output / "epoch_0001.pt")
    expected = torch.load(output / "epoch_0002.pt", weights_only=False)["classifiers"][
        0
    ]
    actual = torch.load(tmp_path / "resumed/epoch_0002.pt", weights_only=False)[
        "classifiers"
    ][0]
    for key in expected:
        assert torch.equal(expected[key], actual[key])
    result = train.run(config, evaluate=output / "epoch_0001.pt")
    assert result["samples"] == 3 and builds == [True, True, False]
    assert (output / "evaluations/epoch_0001/evaluation.json").is_file()
    assert (output / "metrics.jsonl").read_text() == metrics
