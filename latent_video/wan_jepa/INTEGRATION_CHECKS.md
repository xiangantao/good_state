# Mainline integration checks — 2026-09-22

Selected baseline: `ssv2_wan_stride8_supervised256_2gpu_100_20260922_035406`.
Integration evidence: workspace `runs/wan_jepa_mainline_integration_20260922/`.

## Passed

- Portable JEPA NPZ is byte-identical to the selected experiment, SHA256
  `227844cfa742554ee0bd7556e99af7758da5660584c48f63d56a0b1d16ced537`.
- Recomputed projection from existing CALVIN fit caches with the new offline
  reproduction tool. Arrays and resulting NPZ SHA256 exactly match the above.
- Wan portable NPZ contains only original mean/std/groupZCA arrays, preserving
  their values and float32 dtype; no recalibration or refitting.
- Preflight passes using the shared environment, SSv2 data, Wan checkpoint,
  JEPA checkpoint, fixed hidden-channel mask and packaged artifacts. No downloads.
- 47 CPU tests pass: five new representation tests plus existing data/fusion,
  training, checkpoint/resume, independent evaluation, execution and monitoring tests.
- Source syntax, shell syntax, Ruff checks and Git whitespace checks pass.
- Real SSv2 clip78687 on GPU4: same-process original vs migrated extractor outputs
  are bitwise equal for Wan `[896,8,16,16]` and JEPA `[256,8,16,16]`.
  This test shares loaded encoders between implementations and performs separate forwards.
- Official1152-wide attentive classifier accepts `[1,4096,1152]`, performs real
  FP16 forward/backward/optimizer update, saves and restores optimizer/head state.
  Evaluation checkpoint loading works; before/after restore logits are bitwise equal.
- The existing dirty `classification/encoder.py` was preserved byte-for-byte.
  Other preexisting workspace changes and diagnostic checkout files were not overwritten.

## Existing numerical limitation

First independent-process comparison: JEPA exact, Wan relativeRMS=0.0301163,
maxabs=0.892052 after fixed conditioning. A fresh run of the **unchanged original
extractor** also differs from its first run: Wan relativeRMS=0.0292543, JEPA exact.
Same-process original vs new implementations are exact for both branches.

This demonstrates an existing cross-process repeatability issue under this test
protocol; it does not establish its cause. Do not label it definitively BF16
rounding, cuDNN selection or a migration error. The upstream import enables
cuDNN benchmarking; no new determinism/precision policy was imposed by this merge.
The numerical records are `reference_repeat.json`, `same_process.json` and
`result.json` in the evidence directory. It remains a separate investigation.

## Scope and limits

- No new full eight-GPU training was started. The selected algorithm already has
  a completed two-GPU100-step experiment; mainline validation uses CPU tests and
  the bounded real-GPU check above, not a new full distributed performance benchmark.
- The GPU head check repeats the same real clip as two segments to validate
  dimensions/backward/checkpoint behavior. It is not an accuracy measurement.
- Ordinary mainline epoch checkpoints support same-world-size resume and separate
  multi-view evaluation. Old diagnostic short checkpoints are not converted.
- Default per-GPU training batch32, extraction2×16clips, not2×32clips; use the
  actually verified stride8 microbatch. DDP default8 gives global256 videos.
- CALVIN adds supervised proxy information; this representation is not a purely
  unsupervised PCA baseline. No closed-loop robot or validation-superiority claim.
- Integration validation itself did not commit or push. The user subsequently
  authorized a local Git commit, including the package and JEPA encoder dependency;
  no remote push or full training launch is part of that action.

## Exact integration file scope

New package: all files under `latent_video/wan_jepa/`, including four artifact files.
Changed existing files: `latent_video/classification/train.py` (injectable encoder
and classifier width), `latent_video/README.md` (promoted entry link).
Required previously-untracked dependency: `latent_video/classification/baselines/__init__.py`,
`latent_video/classification/baselines/vjepa2/__init__.py`, and
`latent_video/classification/baselines/vjepa2/encoder.py` (already present, reused unchanged).
The other untracked baseline training scripts are not required by this entry.
Do not blanket-add unrelated workspace changes or experiment outputs.
