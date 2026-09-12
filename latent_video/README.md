# Video Latent Experiments

Workspace for the video latent extraction and downstream classification experiments.
The extractor accepts already sampled RGB clips and returns pooled
features with protocol metadata. The [online classification runner](classification/README.md)
reuses V-JEPA2 video processing and its attentive classifier, extracts features on
every batch, and trains one head on the five concatenated candidates. It does not
cache latent features on disk.

## Working Protocol

The generic extractor retains the ScanNet-compatible 480x832 default. The SSv2
classification recipe explicitly uses 256x256 crops and encoder inputs, with two
extraction lanes per GPU. Each clip contains 16 sampled RGB frames. The video VAE temporarily
receives a repeated final frame to satisfy its 4n+1 chunking requirement. Crop the
posterior parameters back to 16 frames before latent sampling and diffusion noise
generation. The Transformer receives exactly 16 latent frames and retains its
native spatiotemporal attention and RoPE.

The 16-frame protocol is a new setting; equality with historical 25-frame features
is not assumed. A regression test checks the causal prefix of the VAE-only padding.

| Candidate | Channels | Actual timestep | Scheduler sigma | Shift |
| --- | ---: | ---: | ---: | ---: |
| L15H2 K | 128 | 57 | 0.05763683095574379 | 3 |
| L13H10 K | 128 | 57 | 0.05763683095574379 | 3 |
| L15H7 Q+K | 256 | 57 | 0.05763683095574379 | 3 |
| B15 selected block hidden | 256 | 299 | 0.299923837184906 | 5 |
| Additional L15H2 K | 128 | 299 | 0.299923837184906 | 5 |

Layer and head indices are zero-based. Q/K are captured after model normalization
and RoPE. Q+K means channel concatenation for classification; retain Q and K as
separate tensors in storage. It is distinct from the QK similarity operator used
in the existing matching reports. B15 hidden means the complete block output,
not the per-head attention output exposed as FeatureKind.HIDDEN_STATES.

Both noise branches start from the same sampled clean latent and share the same
Gaussian noise realization. The VAE runs once. Neither branch starts from the
other branch's output. Each branch performs one conditional Transformer forward,
without an unconditional forward, scheduler update, or VAE decoding.
Preserve the existing BF16 add_noise arithmetic, including sigma conversion to
the latent dtype. Record both scheduler sigma and the effective BF16 value.

Pool each frame from 30x52 to 14x14 using FP32 adaptive average pooling, then
return BF16 tensors. Select the fixed 256 hidden channels before pooling; this
preserves the per-channel pooling operation and avoids pooling unused channels.
Each output has shape [16,C,14,14]. Five candidates total 896 channels and
5.359375 MiB per clip, excluding metadata. Storing Q/K separately has the same
tensor payload as storing their concatenation.

## Early Stopping

`ClipConfig(stop_after_capture=True)` is the default. Each noise branch runs
blocks B0 through B15 and exits before B16 after all requested captures complete.
For the 30-block checkpoint this skips B16-B29 and the final output projection.
The stop happens after B15 self-attention, cross-attention, FFN, and residuals,
so it does not truncate the selected hidden feature. The t57 branch also stops
after the complete B15 block for a consistent boundary.

Set `ClipConfig(stop_after_capture=False)` to run every Transformer block while
capturing the same features. CPU regressions compare all six outputs bit for bit
and check that B16 is skipped only when the switch is enabled. This is a computation
switch, not a change to the feature definition. No speedup estimate is measured.

## Usage

Run from the HeFT root with the existing environment and local source paths:

```bash
cd /increase_kairos_vepfs/increase/liwenhao/agent/2337888765/heft
export PYTHONPATH="$PWD/src:$PWD/diffusers/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

Load one extractor per GPU worker and reuse it across clips. Loading requires an
existing local Wan2.1-T2V-1.3B checkpoint and CUDA; `local_files_only=True` prevents
model downloading. Use the fixed 256-channel table fitted jointly on all five
ScanNet calibration scenes. Every clip uses the same indices; no `held_out`
selection is needed:

```python
from pathlib import Path

from latent_video import ClipConfig, WanLatentExtractor

ROOT = Path("/increase_kairos_vepfs/increase/liwenhao/agent/2337888765")


def make_extractor() -> WanLatentExtractor:
    return WanLatentExtractor.from_local(
        ROOT / "models/Wan2.1-T2V-1.3B-Diffusers",
        channel_mask=(
            ROOT / "heft/reports/scannet_channels/73f4e810824b58cb/global_256/masks.json"
        ),
        mask_key="ablation_iterative_256",
        device="cuda:0",
        config=ClipConfig(stop_after_capture=True),
    )
```

The loader validates the 256-channel table, preserves its order, and records its
source, hash, and exact indices. No channels are reselected during extraction or
classification. The old `fold_masks.json` remains available only for reproducing
LOSO validation with an explicit `held_out`; those five masks are not deployment
choices. See the [fixed-mask report](../reports/scannet_channels/73f4e810824b58cb/global_256/report.md)
for fitting provenance and the distinction between calibration and held-out scores.

After creating `extractor`, call it with already sampled `frames`, a CPU uint8
RGB tensor of shape `[16,3,H,W]`. The existing preprocessing resizes to 480x832.
`frame_ids` and `clip_id` are optional identifiers supplied by the data reader:

```python
result = extractor.extract(frames, seed=42, frame_ids=frame_ids, clip_id=clip_id)
qk = result.candidate("l15h7_qk_t57")  # [16,256,14,14], concatenated Q then K
hidden = result.candidate("b15_hidden256_t299")  # [16,256,14,14]
```

The single-clip call defaults to CPU BF16 outputs. Online consumers can keep the
outputs on the model device and batch independent clips:

```python
results = extractor.extract_batch(
    [frames_a, frames_b],
    seeds=[42, 91],
    output_device=extractor.device,
)
```

`seeds`, `frame_ids`, and `clip_ids`, when supplied, must each contain one entry
per clip. Each clip uses its own CPU generator, sampling the cropped posterior
before drawing the shared diffusion noise. Batch order does not change these
random draws. Calls on an extractor instance must remain sequential because
it owns the VAE state and capture hooks. Grouping and selecting channels before
pooling can change floating-point rounding; metadata records the actual batch
size and sample index.
No features are persisted by either extraction method.

Each result contains these tensors on the requested output device:

| Tensor name | Shape |
| --- | --- |
| `l15h2_k_t57` | `[16,128,14,14]` |
| `l13h10_k_t57` | `[16,128,14,14]` |
| `l15h7_q_t57` | `[16,128,14,14]` |
| `l15h7_k_t57` | `[16,128,14,14]` |
| `b15_hidden256_t299` | `[16,256,14,14]` |
| `l15h2_k_t299` | `[16,128,14,14]` |

`result.metadata` is JSON-compatible. It records input hash and frame IDs, seed,
padding/cropping, channel indices, both noise schedules, effective sigma in the
latent dtype, captured shapes, early-stop settings, and dependency provenance.
Model identity records checkpoint file paths, sizes, and modification times;
it is not a full hash of the checkpoint weights. Source hashes cover the recorded
runtime files, not every transitive dependency. No cache lookup or file write is
performed by `extract`.

## Direct Reuse

Existing code is imported from the workspace. Runtime source identity is recorded
in extraction metadata for future cache validation. The local loader rejects a
pipeline or extraction helper imported from a different checkout.

| Existing source | Use |
| --- | --- |
| [Local WanPipeline](../diffusers/src/diffusers/pipelines/wan/pipeline_wan.py) | Local-only model loading, prompt encoding, and video preprocessing. Its prepare_video_latents implementation is the reference for normalization, sampling, and add_noise ordering. |
| [Local Wan VAE](../diffusers/src/diffusers/models/autoencoders/autoencoder_kl_wan.py) and [Wan Transformer](../diffusers/src/diffusers/models/transformers/transformer_wan.py) | Call the existing models with the existing weights, causal VAE convolutions, Q/K normalization, attention, and RoPE. |
| [AttentionFeatureCapture and CaptureSpec](../src/heft/attn_hook/capture.py) | Capture per-head Q/K. Begin a fresh capture session/chunk for each noise branch, since the Wan capture adapter tracks alternating conditional/unconditional calls. |
| [_prepare_input_video](../src/heft/extraction/worker.py) | Preserve the existing resize interpolation, tensor layout, and uint8-to-float conversion. This is a private helper, so include it in dependency identity and regression coverage. |
| [video_scheduler](../scripts/scannet/video_noise.py), including [selected_noise](../scripts/scannet/extract.py) | Resolve the original checkpoint step 49 and the legacy requested timestep 300 exactly. Reuse the scheduler's add_noise implementation; distinguish capture indices 49/0 from actual timesteps 57/299. |
| [pool_video_tokens](../scripts/scannet/video_extract.py) | Regression reference for frame/spatial layout and FP32 adaptive average pooling. The batched adapter preserves those operations on the model device. |
| [Fixed channel mask](../reports/scannet_channels/73f4e810824b58cb/global_256/masks.json) | Read the single ablation_iterative_256 table fitted on all five calibration scenes. Validate 256 unique indices, their bounds and order, and record its source/key/hash. All clips and datasets reuse this same table. |

The experiment package can call the existing scripts as modules when launched
from the HeFT repository root. Established experiments remain reference callers;
this first step does not require relocating their implementations.

## Reference Implementations To Adapt

- [WanVideoExtractor](../scripts/scannet/video_extract.py): retain the model/prompt
  lifecycle, complete-block forward hooks, shape checks, finite-value checks, and
  hook cleanup. Write a new coordinator for shared VAE encoding and two noise
  branches; calling this class twice would encode the video twice and would let
  padded frames participate in Transformer attention.
- [SafetensorsFeatureStorage](../src/heft/attn_hook/storage.py): reuse the safetensors
  dependency and temporary-file/atomic-replace pattern when adding persistence.
  Its uniform per-head grouping does not represent this heterogeneous feature
  bundle, which includes selected block channels and two noise protocols.
- [FeatureExtractionPool](../src/heft/extraction/pool.py): use its persistent
  per-GPU worker approach as a reference for a future extraction service. Its current
  output metadata describes a homogeneous, single-step tracking feature volume;
  integrating the new bundle needs an explicit metadata adapter or a dedicated
  batch entry point. It is not a drop-in executor replacement.
- [Video regression tests](../scripts/scannet/test_video.py): use the existing
  temporal-context, conditional-capture, pooling, and exact-noise checks as
  references for the new extractor's tests.

## New Code

| File | Responsibility |
| --- | --- |
| [config.py](config.py) | Clip settings, fixed noise/feature definitions, explicit channel-mask loading. |
| [extractor.py](extractor.py) | Local model lifecycle, single VAE encode, two noise branches, early stopping, pooled outputs and metadata. |
| [test_extractor.py](test_extractor.py) | CPU regressions with actual, small randomly initialized Wan models. |
| [__init__.py](__init__.py) | Public imports: `WanLatentExtractor`, `ClipConfig`, `ClipLatents`, `ChannelSelection`, `CANDIDATES`. |

The new implementation owns:

1. The VAE-only 16-to-17-frame adapter and posterior cropping before RNG use.
2. Shared clean latent/noise preparation and the two independently reset noise
   branches, with exact conditioning, sigma, and shift provenance.
3. A candidate-to-noise/layer/head/channel-table mapping and named feature outputs.
4. Validation of the selected channel table and metadata identifying it.
5. Targeted tests for the causal prefix, the 16-frame Transformer input, capture
   resets across branches, output shapes, and existing 25-frame reference behavior.
6. Independent clip batching, per-clip random streams, and CPU/GPU output selection.

## Verification

The focused checks use existing installed dependencies, CPU execution, and small
randomly initialized local Wan VAE/Transformer models. They do not load the large
checkpoint or any dataset, and are not GPU benchmarks:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m pytest \
  latent_video/test_extractor.py scripts/scannet/test_video.py -q
.venv/bin/ruff check latent_video
.venv/bin/ruff format --check latent_video
```

The 24 tests cover exact noise and FP32/BF16 add-noise arithmetic, channel order,
VAE causal-prefix preservation and cropping before RNG draws, a single VAE encode
with two 16-frame Transformer inputs, deterministic results without changing global
RNG state, temporal context, and hook cleanup after failures. They also check
early-stop/full-forward equality and, at 25 frames, equality with conditional
captures from the existing complete `WanVideoExtractor` pipeline at both noise
points. Model forward comparisons currently use FP32 compute with BF16 output;
large-checkpoint BF16 inference has not yet been exercised.

The [classification runner](classification/README.md) adds online video decoding,
multi-GPU single-head training, checkpoint resume, and evaluation. Its dataset
paths must be configured before a real run; its default mask is the fixed table.
Large-model training and evaluation have not yet been run. The implementation
and tests download no datasets, weights, or dependencies.
