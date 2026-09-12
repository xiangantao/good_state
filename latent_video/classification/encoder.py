"""Per-batch online fusion, with no feature-cache persistence."""

from __future__ import annotations

import copy

import torch

from ..config import CANDIDATES
from .config import positive_int

FUSION_ORDER = tuple(CANDIDATES)


class OnlineWanEncoder:
    def __init__(self, extractor, *, extract_batch_size: int = 1):
        positive_int("extract_batch_size", extract_batch_size)
        self.extractor = extractor
        self.extract_batch_size = extract_batch_size
        self.device = extractor.device
        self.embed_dim = 5 * extractor.head_channels + 256

    @torch.no_grad()
    def encode_view(self, batch: dict, view: int) -> torch.Tensor:
        segments = batch["clips"]
        count = len(batch["id"])
        entries = [
            (sample, segment)
            for sample in range(count)
            for segment in range(len(segments))
        ]
        tokens = []
        for start in range(0, len(entries), self.extract_batch_size):
            group = entries[start : start + self.extract_batch_size]
            with torch.autocast(device_type=self.device.type, enabled=False):
                results = self.extractor.extract_batch(
                    [segments[segment][view][sample] for sample, segment in group],
                    seeds=[
                        int(batch["feature_seeds"][sample, segment, view])
                        for sample, segment in group
                    ],
                    frame_ids=[
                        batch["frame_indices"][segment][sample].tolist()
                        for sample, segment in group
                    ],
                    clip_ids=[batch["id"][sample] for sample, _ in group],
                    output_device=self.device,
                )
            if len(results) != len(group):
                raise ValueError("Extractor returned the wrong number of clips")
            for result in results:
                fused = torch.cat(
                    [result.candidate(name) for name in FUSION_ORDER], dim=1
                )
                expected = (
                    self.extractor.config.frames,
                    self.embed_dim,
                    *self.extractor.config.grid,
                )
                if tuple(fused.shape) != expected:
                    raise ValueError(f"Invalid fused feature shape: {fused.shape}")
                if fused.device != self.device:
                    raise ValueError(
                        "Online features must remain on the encoder device"
                    )
                tokens.append(fused.permute(0, 2, 3, 1).reshape(-1, self.embed_dim))
        # Materialize outside inference_mode: the classifier must save inputs for backward.
        return torch.stack(tokens).reshape(count, -1, self.embed_dim).detach()

    def metadata(self) -> dict:
        return {
            "fusion_order": list(FUSION_ORDER),
            "embed_dim": self.embed_dim,
            "extract_batch_size": self.extract_batch_size,
            "feature_device": "encoder",
            "latent_cache": False,
            "frozen_encoder": True,
            "extra_position_encoding": False,
            "channel_selection": self.extractor.channels.metadata(),
            "noise_branches": {
                name: copy.deepcopy(info)
                for name, (_, info) in self.extractor.branches.items()
            },
            "extractor": self.extractor.provenance,
        }
