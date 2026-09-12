# Online SSv2 Classification

This runner freezes the local Wan model, extracts all five candidates on every
batch, concatenates their channels, and trains one V-JEPA2 attentive classifier.
It never writes or reads a latent cache. Only the current batch's features are
materialized in memory. CSV manifests, predictions, metrics, and classifier
checkpoints are ordinary experiment outputs, not feature caches.

## Reuse

The existing V-JEPA2 checkout is imported directly; its files are not modified.
No alternate classifier implementation is maintained here.

| Source in V-JEPA2 | Use |
| --- | --- |
| `src/models/attentive_pooler.py` | The original `AttentiveClassifier`, its initialization, blocks, and activation checkpointing. |
| `evals/video_classification_frozen/eval.py:init_opt` | AdamW, learning-rate schedule, weight-decay schedule, and FP16 gradient scaler. Exactly one optimizer configuration is passed. |
| `evals/video_classification_frozen/eval.py:run_one_epoch` | Reference for cross-entropy training and averaging spatial-view probabilities. The local loop adds online Wan extraction, exact sample counts, Top-5, and prediction export. |
| `src/datasets/video_dataset.py:VideoDataset` | Decord decoding and the original temporal-segment sampling. A strict wrapper calls `get_item_video` so broken videos cannot be silently replaced by random videos. |
| `evals/video_classification_frozen/utils.py:make_transforms` | Original crop, RandAugment, erasing, and validation views, with horizontal flips disabled as in the SSv2 recipe. |

Original source files and local adapter files are hashed in the run protocol.
Model identity follows the extractor's file-size/mtime record, not a full weight
hash. Dependency installation and model downloads are never automatic.

## Protocol

- Fusion order: L15H2 K at t57; L13H10 K at t57; L15H7 Q then K at t57;
  B15 selected hidden256 at t299; L15H2 K at t299. Total width: 896.
- B15 uses one fixed 256-channel table fitted jointly on the five ScanNet
  calibration scenes. Training, validation, and every video use the same indices.
- Each segment contains 16 frames, jointly encoded at 256x256, pooled to 14x14.
  The two recorded noise settings and B15 early-stop switch remain unchanged.
- One VAE encoding and two noise-branch Transformer forwards per segment/view.
  `extract_batch_size` groups independent clips per extraction lane; it is separate from
  the classifier's video batch size. Features are selected, pooled, and fused on
  the encoder device without CPU round trips. Only the classifier participates
  in DDP or backpropagation.
- Two temporal segments yield `[B,6272,896]` per spatial view. The default probe
  has four blocks, sixteen internal attention heads, and 174 output classes.
  Sixteen attention heads do not mean sixteen separately trained classifiers.
- Training uses one spatial view per segment; validation uses three. Temporal
  tokens are concatenated before classification; the three spatial predictions
  are averaged after softmax, as in the official wrapper.
- Sampling, augmentations, and latent noise are seeded per video, phase, epoch,
  segment, and view. Training changes each epoch; validation fixes epoch zero.
  Resuming with the same world size preserves this sequence.

The official transform produces ImageNet-normalized square crops. The adapter
undoes normalization, rounds/clamps to RGB uint8, and passes them through the
existing Wan preprocessing. The default crop and Wan input are both 256x256;
there is no enlargement to 480x832. This includes RGB quantization and clipping of erased pixels; it is
recorded as an input adaptation, not claimed identical to V-JEPA2 preprocessing.
No extra positional encoding or trainable feature-fusion projection is added.

Wan extraction stays BF16 and executes outside classifier autocast. The default
classifier precision is explicitly `float16`, matching what the upstream
`use_bfloat16: true` flag actually does. `bfloat16` and `float32` classifier modes
are configurable and recorded as different optimization settings.

The supplied recipe uses `extract_lanes: 2`, `extract_batch_size: 32`, and
classifier `batch_size: 32` per GPU. Thus each rank's 32 videos produce 64 clips,
split between two independent extraction lanes. Eight ranks train one DDP head
with global video batch 256. Each lane owns its weights, VAE caches, scheduler,
hooks, fixed worker thread, and CUDA stream. New batch sizes warm serially on
their worker threads before concurrent execution. Tail batches are retained;
their smaller extraction groups also receive this warmup.

`compile_blocks: true` compiles only blocks 0-12 and 14, using fullgraph=True and
dynamic=False. Capture blocks 13 and 15 remain native. **VAE layout stays native;
channels_last_3d is not enabled.** Compilation has measured BF16 numerical
differences and is not claimed bitwise equivalent. These settings are recorded
in checkpoint compatibility metadata. Generic configs omitting these switches
retain one lane, extraction batch 1, and no compilation.

All clips retain their temporal sequence, posterior/noise RNG stream, and
metadata. Both noise branches share the same clean latent and noise for each
clip. VAE state is cleared between groups. No feature cache is written.

## Setup

Use the existing HeFT environment. In addition to its dependencies, the original
V-JEPA2 evaluation path requires `timm`, `decord`, and `pandas`. The prerequisite
check reports missing packages without installing them. The local checkout must
also be present at the configured `vjepa_root`.

The supplied config enables offline W&B monitoring for local logging;
configs that omit the mode also default to offline. Its additional
packages are pinned in `requirements-wandb.txt`. Set `wandb.enabled: false` to run
without the W&B SDK; the existing epoch metrics and checkpoints still work.
Set `wandb.entity` to your W&B username or team slug (the first path component
after `wandb.ai/`). The default `null` uses SDK defaults; set an explicit entity
to fix the destination. `wandb.project` names the project inside that account
or team, and `wandb sync --entity` can explicitly choose the upload destination.
The exact Linux/Python 3.12 wheel filenames, download URLs, and PyPI SHA256 values
are recorded in [wandb_wheels.json](wandb_wheels.json). These eleven wheels
supplement the current environment, including its existing protobuf 7.35.1.
The package files have not been downloaded or installed by this runner.

The local SSv2 upload uses `datasets/SSv2/videos/<shard>/<id>.webm`, where
`shard` is `int(id) % 256`, padded to three digits. Its original JSON annotations
are preserved under `datasets/SSv2/labels/`. The configured `train.csv` and
`validation.csv` are generated from those annotations with relative video paths
and the original class IDs. They contain 168,913 training and 24,777 validation
videos; the 27,157 test videos are also extracted.

Configure `configs/ssv2.yaml` before the first real run:

1. Set `data.train`, `data.val`, `data.videos`, and `data.labels` to the uploaded
   dataset. Official annotation JSON and space-delimited path/label CSV are
   supported. Labels must cover the original 174 classes. CSV paths may be quoted
   when they contain spaces; relative video paths resolve from their CSV file.
2. The default `channel_mask` points to the jointly fitted
   [fixed table](../../reports/scannet_channels/73f4e810824b58cb/global_256/masks.json).
   No `held_out` setting is needed. The five LOSO tables are retained for
   reproducing the earlier validation experiment, not choosing a deployment mask.
3. Choose a new `output_dir`. Relative config paths resolve from the HeFT root.
4. Set `extract_batch_size` independently of `optimization.batch_size`. Increasing
   it uses more backbone memory without changing the classifier's batch or LR.

The supplied optimization is one fixed choice from the official grid:
20 epochs, AdamW LR 0.0003, weight decay 0.1, and no warmup. Batch size is four
videos per rank, so eight GPUs give a global batch of 32. This differs from the
official 64-GPU global batch of 256 and is not a claim of identical training
budget or an optimized hyperparameter choice.

```bash
cd /increase_kairos_vepfs/increase/liwenhao/agent/2337888765/heft
export PYTHONPATH="$PWD/src:$PWD/diffusers/src:$PWD/../vjepa2"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# No model loading, video decoding, CUDA initialization, or installation.
.venv/bin/python -m latent_video.classification.train \
  --config latent_video/classification/configs/ssv2.yaml --check

# One classifier across eight GPUs, two frozen Wan instances per GPU.
bash latent_video/classification/launch.sh

# Resume at the next epoch with the same data, protocol, and world size.
bash latent_video/classification/launch.sh \
  --resume ../runs/ssv2_wan_fused_online/latest.pt

# Evaluate a retained checkpoint on separately available GPUs.
# CUDA_VISIBLE_DEVICES must select GPUs available to this evaluation process.
HEFT_NPROC_PER_NODE=1 bash latent_video/classification/launch.sh \
  --evaluate ../runs/ssv2_wan_fused_online/epoch_0001.pt
```

Fresh training checks video-file availability before loading the large model.
Decoding failures report the sample ID/path. Validation shards contain no repeated
padding examples, including when the number of GPUs does not divide the split.
Metrics sum correct predictions and sample counts before computing percentages.

The default `validate_every: 0` skips validation in the training loop. Every
completed epoch writes an immutable `epoch_NNNN.pt`; `latest.pt` is atomically
updated as a hard-link alias. Independent evaluators should use the epoch files,
whose contents will not change while they are being read. Set `validate_every`
to a positive interval to enable in-loop validation; only then can training
publish a `best.pt` alias. There is no automatic background evaluator or GPU
reservation. Evaluation output is isolated in `output_dir/evaluations/<checkpoint-stem>`.

These checkpoints contain one classifier, its optimizer/scaler/scheduler
state, per-rank RNG states, and protocol metadata. They contain no Wan weights or
latent tensors. Resume rejects changed features, data, optimization settings, or
world size. Epochs interrupted before checkpointing are replayed from their start.
Evaluation saves Top-1/Top-5 metrics and one `predictions_r<RANK>.jsonl` per rank.

The launcher defaults to eight local processes and offline operation. Optional
`HEFT_PYTHON`, `HEFT_NPROC_PER_NODE`, `HEFT_TMPDIR`, `HEFT_COMPILE_CACHE`,
`HEFT_TRITON_CACHE`, and `HEFT_CUDA_CACHE` select an existing environment and
cache directories. No dependencies are installed. `--model-path` and
`--output-dir` provide explicit path overrides for local staging or separate runs.
Use the same model path for training and evaluation because model provenance is
part of the checkpoint protocol. The temporary directory defaults to /dev/shm
for Unix socket compatibility; compiler caches default to the workspace cache.

## W&B Monitoring

The supplied recipe uses `wandb.mode: offline`. Training records metrics,
configuration, and feature protocol locally without a network connection,
proxy, or W&B login. Uploading these records later requires explicit
`wandb sync`; there is no automatic mode switching or synchronization.

Omitting the mode also selects offline. The run config overrides any stale
`WANDB_MODE` environment setting. Online mode remains an explicit configuration
option and requires credentials and connectivity when selected.
Only rank zero initializes the SDK and writes monitoring files. The SDK's default
system monitoring records available CPU/GPU utilization metrics. Code upload,
Git capture, and model upload remain disabled in both modes.

- Every `wandb.log_every` training batches, including the final partial window:
  sample-weighted loss, Top-1/Top-5, LR, weight decay, FP16 gradient scale,
  iteration time, videos/second, and maximum allocated CUDA memory across ranks.
- At each completed epoch: globally reduced training metrics; validation and best
  Top-1 appear only when in-loop validation is enabled. Separate evaluation runs
  record final evaluation metrics.
- Run configuration and feature protocol accompany the metrics. No latent tensors
  or model weights are logged. Monitoring does not change the classifier updates.

Step metrics reduce counts across all training ranks. Time and peak memory use
the maximum across ranks; the memory peak is measured since the epoch began.
The reported throughput includes online extraction and data loading in each
logging window. Validation reduces only after all local batches have finished,
so unequal validation shard lengths cannot introduce per-step collective hangs.

`output_dir/monitoring.jsonl` is flushed on every log event in both modes for live
local inspection. Online runs update the cloud dashboard during training and
store local files under `output_dir/wandb/run-*`. Offline runs are stored under
`output_dir/wandb/offline-run-*` and appear in the cloud only after explicit sync.
To upload an offline run, transfer its complete directory to a connected computer,
authenticate there, and explicitly select your account when syncing:

```bash
wandb login --relogin
wandb sync --entity YOUR_ENTITY --project heft-ssv2-latents /path/to/offline-run-...
```

These commands apply to previously recorded offline runs. The entity setting
specifies a destination; authentication uses the SDK's credentials. Keep API keys
out of the training YAML. Offline recording requires no API key on the server.
The runner does not transfer or sync previously recorded offline runs.
The connected computer needs a W&B installation for its own operating system;
the recorded Linux wheels target the training server.

Each launch, including resume, creates a new run. Runs share the configured
group (by default the output-directory name), and a resumed run records its source
checkpoint, starting epoch, and the actual training step. SDK run resumption is
not used; separate offline run directories must be synced separately.

## Verification And Comparison

CPU regressions cover manifest conversion, augmentation/RGB conversion, fresh
feature extraction, channel/time ordering, head-only gradients, exact validation
counts, and resumed optimization. A small randomly initialized real Wan model
checks that extracted inference features can be consumed by a trainable head.
The original attentive-head integration check requires `timm` and is explicitly
skipped when that dependency is absent. No test downloads data or weights.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m pytest latent_video/classification -q
```

Monitoring tests use a local SDK double to check both configured modes, rank-zero
ownership, flushed JSONL, metric aggregation, and unchanged classifier updates.
A local smoke check with W&B 0.28.0 also verified real offline run creation,
scalar logging, finalization, and nonempty `.wandb` and JSONL outputs. GPU system
metrics and online connectivity still need validation during an actual run.
No test connects to W&B, downloads data, or installs packages.

The 256x256, 16-frame, two-segment, global-batch-256, 20-epoch recipe aligns with
the repository's V-JEPA2 ViT-L SSv2 configuration. Validation uses three spatial
views and the same probability averaging. This does not align with its 384px,
64-frame ViT-g configuration. There are still differences in pretrained data,
backbone, feature width/token count, RGB adaptation, and numerical execution.
Wan repeats the final frame for a 17-frame VAE input and crops back to 16; it
does not observe an extra distinct frame. The fixed channels were selected on
ScanNet, not on SSv2 labels. Our fused input is [B,6272,896], whereas the ViT-L
two-segment input is [B,4096,1024]. Head architecture is reused, but parameter
count and input token count are different.

The V-JEPA2 public ViT-L probe result (73.7%) also includes a hyperparameter
search over multiple classifiers. This runner trains only one fixed head and
does not claim to reproduce that search budget or accuracy. A matched baseline
should use the same one-head training budget. This runner trains only our fused head;
V-JEPA2's existing inference entry can evaluate its supplied local checkpoint
separately once those weights are available. No baseline download is triggered.
