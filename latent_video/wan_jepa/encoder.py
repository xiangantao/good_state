"""Standalone frozen extractor and attentive-head adapter for the selected latent."""

import copy
from pathlib import Path
from types import SimpleNamespace

import torch

from ..classification.baselines.vjepa2.encoder import FrozenEncoder
from ..classification.data import IMAGENET_NORMALIZATION
from ..classification.encoder import FUSION_ORDER, OnlineWanEncoder
from ..classification.upstream import load_upstream
from ..extractor import ClipLatents, WanLatentExtractor
from .operators import FixedOperators, digest

SELECTED_POSITIONS = tuple(range(1, 16, 2))


def select_wan_positions(raw):
    if raw.ndim != 5 or raw.shape[1:] != (896, 16, 16, 16):
        raise ValueError("Require Wan896x16x16x16")
    return raw[:, :, 1::2].contiguous()


def packed_tokens(wan, jepa):
    if (
        wan.shape[1:] != (896, 8, 16, 16)
        or jepa.shape[1:] != (256, 8, 16, 16)
        or len(wan) != len(jepa)
    ):
        raise ValueError("Require Wan896 and JEPA256 on native8x16x16 grid")
    return torch.cat((wan, jepa), 1).flatten(2).transpose(1, 2)


class CombinedExtractor:
    def __init__(self, wan, jepa_wrapper, jepa_metadata, artifact_dir):
        self.wan = wan
        self.jepa_wrapper = jepa_wrapper.eval().requires_grad_(False)
        self.jepa_metadata = jepa_metadata
        self.artifact_dir = Path(artifact_dir)
        self.operators = FixedOperators(artifact_dir).to(wan.device).eval()
        manifest = self.operators.manifest
        if (
            self.jepa_metadata["weights_sha256"]
            != manifest["required_jepa_weights_sha256"]
        ):
            raise ValueError("JEPA weights differ from the frozen projection fit")
        if (
            wan.channels.metadata()["source_sha256"]
            != manifest["required_channel_mask_sha256"]
        ):
            raise ValueError(
                "Wan hidden channels differ from the frozen conditioning fit"
            )

    def __getattr__(self, name):
        return getattr(self.wan, name)

    @classmethod
    def from_local(cls, *args, jepa_model_path, vjepa_root, artifact_dir, **kwargs):
        wan = WanLatentExtractor.from_local(*args, **kwargs)
        load_upstream(vjepa_root)
        config = SimpleNamespace(
            vjepa_root=str(vjepa_root),
            model_path=str(jepa_model_path),
            extract_batch_size=32,
            extract_lanes=1,
            path=lambda p: Path(p).resolve(),
        )
        owner = FrozenEncoder.from_local(config, wan.device)
        try:
            return cls(wan, owner.wrapper, owner.metadata(), artifact_dir)
        finally:
            owner.close()

    def replica(self):
        return type(self)(
            self.wan.replica(),
            copy.deepcopy(self.jepa_wrapper),
            copy.deepcopy(self.jepa_metadata),
            self.artifact_dir,
        )

    @torch.inference_mode()
    def jepa_features(self, frames):
        captured = {}
        handle = self.jepa_wrapper.model.blocks[17].register_forward_hook(
            lambda module, args, value: captured.__setitem__(
                "middle", value.detach().clone()
            )
        )
        try:
            rgb = (
                torch.stack(list(frames)).permute(0, 2, 1, 3, 4).to(self.device).float()
                / 255
            )
            mean, std = [
                torch.tensor(x, device=self.device).reshape(1, 3, 1, 1, 1)
                for x in IMAGENET_NORMALIZATION
            ]
            with torch.autocast(self.device.type, dtype=torch.bfloat16):
                final = self.jepa_wrapper([[(rgb - mean) / std]])[0]
            if captured["middle"].shape != final.shape or final.shape != (
                len(frames),
                2048,
                1024,
            ):
                raise ValueError("Unexpected JEPA native shape")
            z = self.operators.jepa(torch.cat((captured["middle"], final), -1))
            return z.transpose(1, 2).reshape(len(frames), 256, 8, 16, 16)
        finally:
            handle.remove()

    @torch.inference_mode()
    def extract_batch(self, frames, **kwargs):
        rows = self.wan.extract_batch(frames, **kwargs)
        selected = []
        for row in rows:
            full = torch.cat([row.candidate(name) for name in FUSION_ORDER], 1)
            selected.append(select_wan_positions(full.permute(1, 0, 2, 3)[None])[0])
        w = self.operators.wan(torch.stack(selected))
        j = self.jepa_features(frames)
        if not torch.isfinite(w).all() or not torch.isfinite(j).all():
            raise ValueError("Nonfinite fused features")
        return [
            ClipLatents(
                {
                    "wan_stride8": w[i].contiguous(),
                    "jepa_supervised256": j[i].contiguous(),
                },
                row.metadata,
            )
            for i, row in enumerate(rows)
        ]


class CombinedEncoder(OnlineWanEncoder):
    def __init__(self, extractor, **kwargs):
        super().__init__(extractor, **kwargs)
        self.embed_dim = 1152

    @torch.no_grad()
    def encode_view(self, batch, view):
        count = len(batch["id"])
        segments = batch["clips"]
        entries = [
            (sample, segment)
            for sample in range(count)
            for segment in range(len(segments))
        ]
        requests = []
        for start in range(0, len(entries), self.extract_batch_size):
            group = entries[start : start + self.extract_batch_size]
            requests.append(
                {
                    "frames": [segments[s][view][i] for i, s in group],
                    "seeds": [
                        int(batch["feature_seeds"][i, s, view]) for i, s in group
                    ],
                    "frame_ids": [
                        batch["frame_indices"][s][i].tolist() for i, s in group
                    ],
                    "clip_ids": [batch["id"][i] for i, s in group],
                    "output_device": self.device,
                }
            )
        with torch.autocast(self.device.type, enabled=False):
            groups = (
                self.pool.extract_groups(requests)
                if self.pool
                else [self.extractor.extract_batch(**r) for r in requests]
            )
        tokens = []
        for request, rows in zip(requests, groups, strict=True):
            if len(rows) != len(request["frames"]):
                raise ValueError("Lost clips")
            for row in rows:
                tokens.append(
                    packed_tokens(
                        row.tensors["wan_stride8"][None],
                        row.tensors["jepa_supervised256"][None],
                    )[0]
                )
        result = torch.stack(tokens).reshape(count, -1, 1152).detach()
        if result.shape != (count, len(segments) * 2048, 1152) or result.is_inference():
            raise ValueError("Invalid classifier tokens")
        return result

    def metadata(self):
        result = super().metadata()
        result.update(
            variant="wan_original5_stride8_jepa_supervised256.v1",
            wan_shape=[896, 8, 16, 16],
            jepa_shape=[256, 8, 16, 16],
            fused_shape=[1152, 8, 16, 16],
            wan_selected_positions_zero_based=list(SELECTED_POSITIONS),
            time="Wan legacy uncompressed VAE, RGB16->pad17->posterior crop16; full16 Transformer forward then select1,3,...,15; JEPA native8 unchanged",
            packing="Channel concatenation, T/H/W flatten;2048 tokens/clip;4096 tokens/2-segment video",
            conditioning="Fixed Wan channel standardization+groupZCA, same original five groups; JEPA CALVIN mean/std then shared2048x256 projection; no track mean-only",
            limits="Wan ZCA calibrated on compressed5, transferred to16. Pair endpoints vs JEPA tubelet centers are approximate correspondence.",
            operators=self.extractor.operators.manifest,
            jepa=self.extractor.jepa_metadata,
            source_hashes={
                name: digest(Path(__file__).with_name(name))
                for name in ("encoder.py", "operators.py", "config.py", "train.py")
            },
        )
        return result
