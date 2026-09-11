"""HEFT self-attention Q/K capture after normalization and RoPE."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .extract import WanExtractor


def pool_heads(tensor, spatial_size, grid_size):
    """B,H,N,D -> B,H,D,Gh,Gw, preserving head and spatial ordering."""
    if tensor.ndim != 4 or tensor.shape[2] != spatial_size[0] * spatial_size[1]:
        raise ValueError("Unexpected per-head token grid")
    batch, heads, _, channels = tensor.shape
    image = tensor.permute(0, 1, 3, 2).reshape(batch * heads, channels, *spatial_size)
    return F.adaptive_avg_pool2d(image.float(), grid_size).reshape(
        batch, heads, channels, *grid_size
    )


class WanHeadExtractor(WanExtractor):
    @torch.inference_mode()
    def heads(
        self,
        latents,
        blocks,
        timestep,
        shift,
        seed,
        scene,
        frame_ids,
        batch_size=1,
        grid_size=(14, 14),
    ):
        blocks = sorted(set(blocks))
        patch = self.transformer.config.patch_size
        spatial = (latents.shape[-2] // patch[1], latents.shape[-1] // patch[2])
        captured = {b: {kind: [] for kind in ("query", "key")} for b in blocks}
        bindings = []

        def callback(block):
            def capture(*, query, key, **unused):
                for kind, tensor in (("query", query), ("key", key)):
                    captured[block][kind].append(
                        pool_heads(tensor, spatial, grid_size).cpu()
                    )

            return capture

        try:
            for block in blocks:
                attention = self.transformer.blocks[block].attn1
                had_attribute = hasattr(attention, "_heft_capture")
                previous = getattr(attention, "_heft_capture", None)
                bindings.append((attention, had_attribute, previous))
                # Direct transformer forwards contain no alternating CFG passes.
                attention._heft_capture = callback(block)
            # Reuse exactly the layer scan's latent/noise/forward path.
            block_features, prepool, noise = self.blocks(
                latents,
                blocks,
                timestep,
                shift,
                seed,
                scene,
                frame_ids,
                batch_size,
                grid_size,
                False,
            )
            del block_features, prepool
        finally:
            for attention, had_attribute, previous in bindings:
                if had_attribute:
                    attention._heft_capture = previous
                else:
                    del attention._heft_capture
        output = {}
        for block, kinds in captured.items():
            output[block] = {}
            for kind, values in kinds.items():
                if not values:
                    raise RuntimeError(
                        "Local Wan processor did not invoke HEFT capture"
                    )
                tensor = torch.cat(values)
                if tensor.shape[0] != len(latents) or tensor.shape[1:3] != (12, 128):
                    raise ValueError(f"Unexpected Q/K shape: {tensor.shape}")
                if not torch.isfinite(tensor).all():
                    raise ValueError("Non-finite Q/K features")
                output[block][kind] = tensor
        return output, noise
