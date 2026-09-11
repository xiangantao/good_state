"""Monitoring mode selection, rank ownership, and local metric persistence."""

import json
import os
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from .config import DataConfig, RunConfig, WandbConfig
from .monitoring import WandbMonitor
from .train import preflight


class FakeRun:
    def __init__(self):
        self.id = "offline-test"
        self.config = SimpleNamespace(update=lambda *args, **kwargs: None)
        self.events = []
        self.metrics = []
        self.exit_codes = []

    def define_metric(self, name, **kwargs):
        self.metrics.append((name, kwargs))

    def log(self, values):
        self.events.append(dict(values))

    def finish(self, *, exit_code):
        self.exit_codes.append(exit_code)


@pytest.mark.parametrize("entity", [None, "test-owner"])
@pytest.mark.parametrize("mode", ["online", "offline"])
def test_monitor_uses_configured_mode_and_flushes_local_log(
    tmp_path, monkeypatch, entity, mode
):
    run = FakeRun()
    calls = []

    def initialize(**kwargs):
        assert os.environ["WANDB_MODE"] == mode
        assert os.environ["WANDB_ERROR_REPORTING"] == "false"
        calls.append(kwargs)
        return run

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=initialize, Settings=lambda **kwargs: kwargs),
    )
    monkeypatch.setenv("WANDB_MODE", "offline" if mode == "online" else "online")
    monkeypatch.setenv("WANDB_ERROR_REPORTING", "true")
    monitor = WandbMonitor(
        WandbConfig(enabled=True, entity=entity, mode=mode),
        tmp_path,
        rank=0,
        metadata={},
        job_type="train",
    )
    assert calls[0]["mode"] == mode
    assert calls[0]["entity"] == entity
    assert calls[0]["save_code"] is False
    assert calls[0]["settings"] == {"disable_git": True, "disable_code": True}
    assert "resume" not in calls[0]
    values = {"step": 100, "batch/loss": 0.25}
    monitor.log(values)
    assert json.loads((tmp_path / "monitoring.jsonl").read_text()) == {
        "run_id": "offline-test",
        **values,
    }
    assert run.events == [values]
    monitor.finish(exit_code=1)
    monitor.finish(exit_code=0)
    assert run.exit_codes == [1]


@pytest.mark.parametrize("enabled,rank", [(False, 0), (True, 1)])
def test_disabled_or_nonzero_rank_does_not_import_wandb(
    tmp_path, monkeypatch, enabled, rank
):
    monkeypatch.setitem(sys.modules, "wandb", None)
    monitor = WandbMonitor(
        WandbConfig(enabled=enabled), tmp_path, rank=rank, metadata={}, job_type="train"
    )
    monitor.log({"batch/loss": 0.25})
    monitor.finish(exit_code=0)
    assert not list(tmp_path.iterdir())


def test_monitor_loads_online_mode_and_rejects_invalid_settings(tmp_path):
    with pytest.raises(ValueError, match="log_every"):
        WandbConfig(log_every=0)
    with pytest.raises(ValueError, match="wandb.entity"):
        WandbConfig(entity=" ")
    assert WandbConfig().mode == "offline"
    with pytest.raises(ValueError, match="wandb.mode"):
        WandbConfig(mode="invalid")
    config = tmp_path / "config.yaml"
    config.write_text(
        "data: {train: train.csv, val: val.csv}\nwandb: {enabled: true, mode: online}\n"
    )
    assert RunConfig.load(config).wandb.mode == "online"


def test_preflight_requires_wandb_only_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    config = RunConfig(
        DataConfig(str(tmp_path / "train.csv"), str(tmp_path / "val.csv")),
        channel_mask=str(tmp_path / "mask.json"),
    )
    assert not any(
        "wandb" in issue
        for issue in preflight(config)
        if issue.startswith("Missing dependencies:")
    )
    enabled = replace(config, wandb=WandbConfig(enabled=True))
    assert any(
        "wandb" in issue
        for issue in preflight(enabled)
        if issue.startswith("Missing dependencies:")
    )
