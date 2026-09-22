"""Reproduce fixed supervised256 from existing CALVIN caches, entirely offline.

Consumes plan.json, readout_inputs.npz and targets.npy; never fetches data or
updates encoders. This is a provenance/reproduction tool, not a training import.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.linalg as la

from .operators import digest

GROUPS = {
    "object_translation": [0, 1, 2, 9, 10, 11, 18, 19, 20],
    "object_rotation6d": list(range(3, 9)) + list(range(12, 18)) + list(range(21, 27)),
    "relative_position": [27, 28, 29, 33, 34, 35, 39, 40, 41],
    "relative_change": [30, 31, 32, 36, 37, 38, 42, 43, 44],
    "gripper_width": [54, 55],
    "robot_translation": [45, 46, 47],
    "robot_rotation6d": list(range(48, 54)),
}


def fit_projection(cache):
    cache = Path(cache)
    plan = json.loads((cache / "plan.json").read_text())
    fit = np.array([i for i, r in enumerate(plan["records"]) if r["split"] == "fit"])
    if len(fit) != 400:
        raise ValueError("Reference projection expects400 fit clips")
    with np.load(cache / "readout_inputs.npz", allow_pickle=False) as f:
        j, mean, std = [f[k].copy() for k in ("j", "j_mean", "j_std")]
    y = np.load(cache / "targets.npy", allow_pickle=False).astype(np.float64)
    if j.shape != (600, 8, 2048) or y.shape != (600, 112):
        raise ValueError("Unexpected descriptor/target shape")
    ym = y[fit].mean(0)
    scale = np.ones(112)
    for indices in GROUPS.values():
        ids = np.array(indices + [i + 56 for i in indices])
        rms = np.sqrt(np.mean((y[fit][:, ids] - ym[ids]) ** 2))
        scale[ids] = max(rms, 1e-3) * np.sqrt(len(ids))
    target = (y - ym) / scale
    x = j.reshape(len(j), -1)
    x = x - x[fit].mean(0)
    energy = max(float(np.mean(np.sum(x[fit] ** 2, axis=1))), 1e-12)
    kernel = x @ x[fit].T / energy
    x = x / np.sqrt(energy)
    gram = kernel[fit]
    ev, u = la.eigh((gram + gram.T) / 2, check_finite=False)
    coef = u @ ((u.T @ target[fit]) / (np.maximum(ev, 0)[:, None] + 0.001))
    weights = (
        (x[fit].T @ coef).reshape(8, 2048, 112).transpose(1, 0, 2).reshape(2048, -1)
    )
    basis, _, _ = la.svd(weights, full_matrices=False, check_finite=False)
    return mean, std, basis[:, :256].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reference = Path(__file__).with_name("artifacts")
    manifest = json.loads((reference / "manifest.json").read_text())
    if digest(args.cache / "plan.json") != manifest["calvin_plan_sha256"]:
        raise ValueError("Wrong CALVIN plan")
    mean, std, projection = fit_projection(args.cache)
    with np.load(reference / "jepa_supervised256.npz") as original:
        np.testing.assert_array_equal(mean, original["mean"])
        np.testing.assert_array_equal(std, original["std"])
        np.testing.assert_allclose(
            projection, original["projection"], atol=1e-9, rtol=1e-9
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        mean=mean,
        std=std,
        projection=projection,
        source="jepa_b17+jepa_finalnorm",
        plan_sha256=manifest["calvin_plan_sha256"],
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "reference_arrays_match": True,
                "fit_clips": 400,
                "beta": 0.001,
                "shape": list(projection.shape),
                "sha256": digest(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
