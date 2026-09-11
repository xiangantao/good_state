"""Per-batch online fusion, with no feature-cache persistence."""

from __future__ import annotations

import copy

import torch

from ..config import CANDIDATES

FUSION_ORDER = tuple(CANDIDATES)


class OnlineWanEncoder:
    def __init__(self, extractor):
        self.extractor = extractor
        self.device = extractor.device
        self.embed_dim = 5 * extractor.head_channels + 256

    @torch.no_grad()
    def encode_view(self, batch: dict, view: int) -> torch.Tensor:
        segments = batch["clips"]
        count = len(batch["id"])
        samples = []
        for sample in range(count):
            tokens = []
            for segment, spatial_views in enumerate(segments):
                frames = spatial_views[view][sample]
                with torch.autocast(device_type=self.device.type, enabled=False):
                    result = self.extractor.extract(
                        frames,
                        seed=int(batch["feature_seeds"][sample, segment, view]),
                        frame_ids=batch["frame_indices"][segment][sample].tolist(),
                        clip_id=batch["id"][sample],
                    )
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
                tokens.append(fused.permute(0, 2, 3, 1).reshape(-1, self.embed_dim))
            samples.append(torch.cat(tokens, dim=0))
        # Materialize outside inference_mode: the classifier must save inputs for backward.
        return torch.stack(samples).to(self.device).detach().clone()

    def metadata(self) -> dict:
        return {
            "fusion_order": list(FUSION_ORDER),
            "embed_dim": self.embed_dim,
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
