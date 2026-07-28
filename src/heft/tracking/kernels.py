"""Fused CUDA kernels for spatially bounded feature matching."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from torch import Tensor

_BLOCK_CHANNELS = 32
_BLOCK_PIXELS = 64


@triton.jit
def _search_correlation_kernel(
    query,
    target,
    positions,
    output,
    channels,
    height: tl.constexpr,
    width: tl.constexpr,
    extent: tl.constexpr,
    side: tl.constexpr,
    pixels: tl.constexpr,
    radius_squared: tl.constexpr,
    block_channels: tl.constexpr,
    block_pixels: tl.constexpr,
):
    point = tl.program_id(0)
    offsets = tl.program_id(1) * block_pixels + tl.arange(0, block_pixels)
    offset_x = offsets % side - extent
    offset_y = offsets // side - extent
    position_x = tl.load(positions + point * 2)
    position_y = tl.load(positions + point * 2 + 1)
    pixel_x = tl.floor(position_x).to(tl.int32) + offset_x
    pixel_y = tl.floor(position_y).to(tl.int32) + offset_y
    distance_x = pixel_x - position_x
    distance_y = pixel_y - position_y
    distance_squared = distance_x * distance_x + distance_y * distance_y
    valid = (
        (offsets < pixels)
        & (pixel_x >= 0)
        & (pixel_x < width)
        & (pixel_y >= 0)
        & (pixel_y < height)
        & (distance_squared <= radius_squared)
    )
    accumulator = tl.zeros((block_pixels,), dtype=tl.float32)
    spatial_offset = pixel_y * width + pixel_x

    for channel_start in tl.range(0, channels, block_channels):
        channel = channel_start + tl.arange(0, block_channels)
        channel_valid = channel < channels
        query_value = tl.load(
            query + point * channels + channel,
            mask=channel_valid,
            other=0.0,
        )
        target_value = tl.load(
            target + channel[None, :] * height * width + spatial_offset[:, None],
            mask=valid[:, None] & channel_valid[None, :],
            other=0.0,
        )
        accumulator += tl.sum(target_value * query_value[None, :], axis=1)

    result = tl.where(valid, accumulator, -float("inf"))
    tl.store(output + point * pixels + offsets, result, mask=offsets < pixels)


def search_region_correlation(
    query: Tensor,
    target: Tensor,
    positions: Tensor,
    radius: float,
) -> Tensor:
    """Return cosine logits only for each point's square search neighborhood."""

    points, channels = query.shape
    height, width = target.shape[-2:]
    extent = math.ceil(radius)
    side = extent * 2 + 1
    pixels = side * side
    output = torch.empty((points, pixels), device=query.device, dtype=torch.float32)
    grid = (points, triton.cdiv(pixels, _BLOCK_PIXELS))
    _search_correlation_kernel[grid](
        query,
        target,
        positions,
        output,
        channels,
        height,
        width,
        extent,
        side,
        pixels,
        radius * radius,
        _BLOCK_CHANNELS,
        _BLOCK_PIXELS,
        num_warps=4,  # pyright: ignore[reportCallIssue]
    )
    return output
