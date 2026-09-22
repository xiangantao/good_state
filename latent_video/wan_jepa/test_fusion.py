import shutil
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ..extractor import ClipLatents
from .config import RunConfig
from .encoder import CombinedEncoder, packed_tokens, select_wan_positions
from .operators import FixedOperators, load_artifacts

HERE = Path(__file__).parent


def test_operators_preserve_exported_formula():
    op = FixedOperators(HERE / "artifacts")
    w, j, _ = load_artifacts(HERE / "artifacts")
    gen = torch.Generator().manual_seed(87)
    raw = torch.randn(1, 896, 8, 16, 16, generator=gen)
    x = (
        raw.half().float() - torch.from_numpy(w["mean"])[None, :, None, None, None]
    ) / torch.from_numpy(w["std"])[None, :, None, None, None]
    reference = (
        (x.flatten(2).transpose(1, 2) @ torch.from_numpy(w["matrix"]))
        .transpose(1, 2)
        .reshape_as(x)
    )
    assert torch.equal(op.wan(raw), reference)
    rawj = torch.randn(1, 2048, 2048, generator=gen)
    expected = (
        (rawj.half().float() - torch.from_numpy(j["mean"]).float())
        / torch.from_numpy(j["std"]).float()
    ) @ torch.from_numpy(j["projection"]).float()
    assert torch.equal(op.jepa(rawj), expected)
    assert not list(op.parameters())


def test_stride_selection_and_channel_order():
    w = torch.arange(16.0).reshape(1, 1, 16, 1, 1).expand(1, 896, 16, 16, 16)
    selected = select_wan_positions(w)
    j = torch.arange(8.0).reshape(1, 1, 8, 1, 1).expand(1, 256, 8, 16, 16)
    out = packed_tokens(selected, j).reshape(1, 8, 16, 16, 1152)
    assert out[0, :, 0, 0, 0].tolist() == [1, 3, 5, 7, 9, 11, 13, 15]
    assert torch.equal(out[..., :896], selected.permute(0, 2, 3, 4, 1))
    assert torch.equal(out[..., 896:], j.permute(0, 2, 3, 4, 1))
    with pytest.raises(ValueError):
        select_wan_positions(w[:, :, :8])


def test_artifact_tampering_rejected(tmp_path):
    shutil.copytree(HERE / "artifacts", tmp_path / "artifacts")
    path = tmp_path / "artifacts/jepa_supervised256.npz"
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        load_artifacts(tmp_path / "artifacts")


def test_config_and_augmented_path():
    config = RunConfig.load(HERE / "ssv2.yaml")
    assert (
        config.extract_batch_size,
        config.extract_lanes,
        config.optimization.batch_size,
    ) == (16, 2, 32)
    assert config.validate_every == 0 and config.wandb.mode == "offline"
    with pytest.raises(ValueError):
        replace(config, clip=replace(config.clip, grid=(14, 14)))


def test_batch_segment_order_and_autograd_compatible_tokens():
    class Extractor:
        device = torch.device("cpu")
        head_channels = 128

        @torch.inference_mode()
        def extract_batch(self, frames, **kwargs):
            return [
                ClipLatents(
                    {
                        "wan_stride8": torch.full((896, 8, 16, 16), float(f)),
                        "jepa_supervised256": torch.full(
                            (256, 8, 16, 16), float(f) + 100
                        ),
                    },
                    {},
                )
                for f in frames
            ]

    encoder = CombinedEncoder(Extractor(), extract_batch_size=3)
    batch = {
        "id": ["a", "b"],
        "clips": [[torch.tensor([10, 20])], [torch.tensor([11, 21])]],
        "feature_seeds": torch.ones(2, 2, 1, dtype=torch.long),
        "frame_indices": [torch.arange(16).repeat(2, 1)] * 2,
    }
    tokens = encoder.encode_view(batch, 0)
    assert tokens.shape == (2, 4096, 1152) and not tokens.is_inference()
    assert tokens[:, ::2048, 0].tolist() == [[10, 11], [20, 21]]
    assert tokens[:, ::2048, 896].tolist() == [[110, 111], [120, 121]]
    weight = torch.ones(1152, 1, requires_grad=True)
    (tokens @ weight).mean().backward()
    assert torch.isfinite(weight.grad).all()
