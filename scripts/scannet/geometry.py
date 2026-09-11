"""Calibration-aware RGB/depth registration and shared adjacent-pair geometry."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def crop_parameters(original_size, output_size):
    """Return integer resize/crop shared by RGB and projected depth points."""
    ow, oh = original_size
    h, w = output_size
    scale = max(w / ow, h / oh)
    nw, nh = max(w, round(ow * scale)), max(h, round(oh * scale))
    return nw, nh, (nw - w) // 2, (nh - h) // 2


def resize_rgb(rgb, output_size):
    nw, nh, left, top = crop_parameters(rgb.size, output_size)
    h, w = output_size
    return rgb.resize((nw, nh), Image.Resampling.BICUBIC).crop(
        (left, top, left + w, top + h)
    )


def unproject(depth, intrinsic):
    y, x = np.indices(depth.shape, dtype=np.float64)
    return np.stack(
        (
            (x - intrinsic[0, 2]) * depth / intrinsic[0, 0],
            (y - intrinsic[1, 2]) * depth / intrinsic[1, 1],
            depth,
        ),
        axis=-1,
    )


def transform_points(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def valid_pool(world, valid, grid_size):
    """Adaptive average coordinates over valid pixels only; zero holes never vote."""
    clean = np.where(valid[..., None], world, 0).astype(np.float32)
    weights = torch.from_numpy(valid.astype(np.float32))[None, None]
    values = torch.from_numpy(clean).permute(2, 0, 1)[None]
    fraction = F.adaptive_avg_pool2d(weights, grid_size)
    sums = F.adaptive_avg_pool2d(values, grid_size)
    coords = (sums / fraction.clamp_min(1e-12))[0].permute(1, 2, 0).numpy()
    return coords, fraction[0, 0].numpy()


def register_geometry(
    scene, frame, depth, output_size=(480, 832), grid_size=(14, 14), min_coverage=0.1
):
    """Splat calibrated depth points into RGB pixels using a nearest-surface z-buffer.

    Coverage denotes the fraction of output pixels with an actual projected depth
    observation. We do not fill holes or invent depth across object boundaries.
    """
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1]")
    h, w = output_size
    points = unproject(depth, scene.depth_intrinsic)
    valid = np.isfinite(points).all(-1) & (depth > 0)
    points = points[valid]
    color = transform_points(points, scene.depth_to_color)
    visible = np.isfinite(color).all(-1) & (color[:, 2] > 0)
    points, color = points[visible], color[visible]
    k = scene.color_intrinsic
    u = k[0, 0] * color[:, 0] / color[:, 2] + k[0, 2]
    v = k[1, 1] * color[:, 1] / color[:, 2] + k[1, 2]
    nw, nh, left, top = crop_parameters(
        (scene.color_width, scene.color_height), output_size
    )
    # Pixel-center mapping agrees with the resize/crop applied to the RGB.
    u = (u + 0.5) * nw / scene.color_width - 0.5 - left
    v = (v + 0.5) * nh / scene.color_height - 0.5 - top
    ix, iy = np.rint(u).astype(int), np.rint(v).astype(int)
    inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    ix, iy, points, color = ix[inside], iy[inside], points[inside], color[inside]
    pixel_ids = iy * w + ix
    order = np.argsort(color[:, 2], kind="stable")
    _, first = np.unique(pixel_ids[order], return_index=True)
    keep = order[first]
    ids = pixel_ids[keep]
    world = np.zeros((h * w, 3), dtype=np.float32)
    mask = np.zeros(h * w, dtype=bool)
    world[ids] = transform_points(
        points[keep], scene.axis_alignment @ frame.pose
    ).astype(np.float32)
    mask[ids] = True
    world, mask = world.reshape(h, w, 3), mask.reshape(h, w)
    coords, fraction = valid_pool(world, mask, grid_size)
    token_valid = (fraction >= min_coverage) & np.isfinite(coords).all(-1)
    return coords, token_valid, fraction


def pair_geometry(xyz_i, xyz_j, valid_i, valid_j, voxel_size=0.1):
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    x, y = xyz_i.reshape(-1, 3), xyz_j.reshape(-1, 3)
    vi = valid_i.reshape(-1) & np.isfinite(x).all(-1)
    vj = valid_j.reshape(-1) & np.isfinite(y).all(-1)
    ids_i, ids_j = np.flatnonzero(vi), np.flatnonzero(vj)
    if not len(ids_i) or not len(ids_j):
        return {
            "ids_i": ids_i,
            "ids_j": ids_j,
            "shared": [],
            "vox_i": np.empty((len(ids_i), 3), int),
            "vox_j": np.empty((len(ids_j), 3), int),
            "xyz_i": x[ids_i],
            "xyz_j": y[ids_j],
        }
    origin = np.concatenate([x[ids_i], y[ids_j]]).min(0)
    vox_i = np.floor((x[ids_i] - origin) / voxel_size).astype(np.int64)
    vox_j = np.floor((y[ids_j] - origin) / voxel_size).astype(np.int64)
    groups = []
    for vox in [vox_i, vox_j]:
        group = {}
        for i, key in enumerate(map(tuple, vox)):
            group.setdefault(key, []).append(i)
        groups.append(group)
    shared = [
        (groups[0][key], groups[1][key])
        for key in sorted(groups[0].keys() & groups[1].keys())
    ]
    return {
        "ids_i": ids_i,
        "ids_j": ids_j,
        "vox_i": vox_i,
        "vox_j": vox_j,
        "xyz_i": x[ids_i],
        "xyz_j": y[ids_j],
        "shared": shared,
    }
