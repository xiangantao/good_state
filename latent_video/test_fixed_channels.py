"""The deployed B15 descriptor has one identity across downstream samples."""

import json

import pytest

from .classification.config import HEFT_ROOT, DataConfig, RunConfig
from .config import ChannelSelection


@pytest.mark.parametrize("use_yaml", [False, True])
def test_default_classification_uses_joint_mask_without_a_fold(use_yaml):
    config = (
        RunConfig.load(HEFT_ROOT / "latent_video/classification/configs/ssv2.yaml")
        if use_yaml
        else RunConfig(data=DataConfig(train="train.csv", val="val.csv"))
    )
    assert config.held_out is None
    path = config.path(config.channel_mask)
    selected = ChannelSelection.load(path)
    assert selected.held_out is None
    assert len(selected.indices) == len(set(selected.indices)) == 256
    assert selected.indices == tuple(sorted(selected.indices))
    fit = json.loads(path.with_name("config.json").read_text())
    audit = json.loads(path.with_name("audit.json").read_text())
    assert fit["calibration_scenes"] == [
        "scene0707_00",
        "scene0708_00",
        "scene0709_00",
        "scene0710_00",
        "scene0711_00",
    ]
    assert fit["noise"]["actual_timestep"] == 299
    assert fit["noise"]["sigma"] == 0.299923837184906
    assert fit["noise"]["shift"] == 5
    assert audit["mask"]["sha256"] == selected.source_sha256
    assert audit["requires_held_out"] is False
    with pytest.raises(ValueError, match="omit it for a fixed"):
        ChannelSelection.load(path, held_out="scene0707_00")
