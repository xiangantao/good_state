# SSv2 online training integration and audit

Date: 2026-09-12. This report records the implemented recipe and bounded checks,
not a completed SSv2 training run or an accuracy comparison.

## Implemented recipe

- Eight local torchrun ranks; only one attentive classifier participates in DDP.
- Per-rank video batch 32, two temporal segments per video: global video batch 256.
- Two independent frozen extraction lanes per rank, each with microbatch 32.
  The usual 64 clips per rank are split into two simultaneous 32-clip forwards.
- 256x256 crops and encoder inputs; 16 retained frames; 14x14 pooled grid.
- Compile blocks 0-12 and 14 with fullgraph=True, dynamic=False. Blocks 13 and 15
  retain their feature-capture hooks. VAE memory layout remains native.
- Each lane has its own weights, prompt tensor, VAE state, scheduler, hooks,
  fixed thread, and CUDA stream. New microbatch sizes warm serially on their
  eventual worker threads. Outputs are restored to video/segment order and
  registered with the consuming CUDA stream before GPU fusion.
- Twenty epochs, one head, offline W&B, and no latent cache.
- In-loop validation disabled. Immutable epoch_NNNN.pt checkpoints support a
  separately launched evaluator; latest.pt is an atomic hard-link alias.
  Evaluation outputs live in a checkpoint-specific subdirectory.

The earlier 17% dual-lane throughput observation used channels-last VAE as part
of its tested combination. That number is not automatically transferable to
this native-VAE recipe. Partial compilation also has previously measured BF16
feature differences (roughly 1%-2.7% relative RMS); it is not labeled lossless.

## Ordered changes

1. 812b8ec: previously uncommitted batch extraction, device-resident features,
   dtype handling, lazy imports, and their regressions.
2. 2e4b87e: independent lanes, partial compilation, 256px eight-GPU recipe,
   retained epoch checkpoints, launcher, smoke tool, and integration tests.
3. The subsequent documentation commit records this audit and cleanup.

All new commits use author and committer xiangantao
<xiangantao@users.noreply.github.com>, without coauthor trailers. Unrelated
ScanNet/OVIS edits and pre-existing mode changes are outside these commits.

## Verification

The pre-integration suite passed 54 tests. The final suite passed 64 tests,
including ordered fusion across lane/tail layouts, thread ownership, real small
Wan replica equivalence and separate weight storage, classifier backward,
checkpoint alias safety, and train-only/restore/separate-evaluation workflows.
The latter workflow uses lightweight CPU components to exercise orchestration;
real Wan and the original attentive classifier are exercised by the GPU checks.
Ruff checks, launcher syntax, and git diff whitespace checks passed.

The standard heft/.venv prerequisite check returned ready=true with no issues.
The GPU checks used the already-staged local runtime and weights in /dev/shm.
No packages, models, or datasets were downloaded. Exact installed package
versions are retained in environment_versions.json.

### One GPU, full 2x32

The successful run used 32 distinct real training videos with original V-JEPA2
transforms. The decoded batch was reused for three bounded training iterations.
It compared concurrent features with serial execution on the same two warmed
worker threads before performing classifier forward/backward/AdamW updates.

| Check | Result |
| --- | --- |
| Per-lane extraction microbatch | 32 clips |
| Video batch | 32 |
| Concurrent vs same-lane serial output | Elementwise equal; max error 0 |
| Classifier updates | Three successful FP16-scaled updates |
| Frozen encoder | requires_grad=False, no encoder gradients |
| Peak allocated | 34.1187 GiB |
| Peak reserved | 46.7734 GiB |
| Iteration times | 16.5172, 15.7009, 13.4102 seconds |

These shared-GPU timings vary and exclude continuous data loading. They do not
establish an exclusive-GPU speedup or an eight-GPU completion-time estimate.
Reserve roughly 50 GiB of free memory per card for this tested implementation,
including allocator reservation and runtime overhead; an exclusive 80GB A800
has room, whereas a shared card with only 28-32 GiB free does not.

The first attempt was interrupted by a new competing allocation: at failure
other processes occupied about 46.9 GiB, our PyTorch allocation was 28.30 GiB,
and another 2.29 GiB was requested with only 1.14 GiB physically free. It is
recorded as an OOM attempt, not a successful full-batch check.

### Eight GPUs, DDP smoke

All eight ranks passed concurrent/serial equality. The test used two videos per
rank, two extraction lanes with microbatch two, and three training iterations.
NCCL gradient synchronization and distributed metric reduction executed;
the sampled classifier parameter agreed exactly across all ranks afterward.
The global sample total was 48, as expected for 16 videos over three iterations.

FP16 GradScaler reduced its scale from 65536 to 32768 on the second iteration
and skipped that optimizer update; the third iteration updated successfully.
This is retained in the raw step records. The test therefore demonstrates three
forward/backward iterations and two successful optimizer updates, not three.
The existing FP16 policy was retained to match the upstream classifier recipe.

Peak allocated memory was 8.54795 GiB and peak reserved 10.64648 GiB at this
small batch. The full eight-rank 2x32 configuration was not run because several
cards had less free memory than the measured single-rank requirement. A full
data-loader throughput run and a complete training epoch are still unmeasured.

## Fairness review

| Aspect | Assessment |
| --- | --- |
| Input resolution | 256x256 aligns with official V-JEPA2 ViT-L SSv2, not the 384px ViT-g recipe. |
| Sampling | 16 sampled frames, frame step 4, two segments, and three validation spatial views align. |
| Training budget | Global video batch 256 and 20 epochs align; only one fixed hyperparameter setting is trained. |
| Head implementation | Original attentive classifier, depth four and sixteen attention heads; its parameters differ because input width differs. |
| Representation | Wan [B,6272,896] versus ViT-L [B,4096,1024]; these are not identical token/parameter budgets. |
| Preprocessing | Upstream transforms reused, followed by Wan RGB inverse-normalization, quantization/clipping, and model-specific normalization. |
| Temporal adapter | Wan VAE receives a repeated final frame, then the posterior is cropped back to 16; no extra distinct frame is observed. |
| Position handling | Wan joint temporal attention and native RoPE retained; no extra position module added to the head. |
| Channel/noise selection | Fixed prior ScanNet-selected channels and recorded noise points; no SSv2-label-based channel search. |
| Evaluation | All validation samples counted once, probability averaging reused; standalone evaluation is isolated from training writes. |
| Numerical execution | BF16 encoder and optional partial compilation recorded; neither resolution changes nor compilation are claimed bitwise equivalent. |

The published ViT-L accuracy includes a multi-head hyperparameter search.
It is a contextual reference, not a matched one-head baseline. To claim matched
latent quality, train the V-JEPA2 head under the same single-setting budget and
report preprocessing, feature size, pretrained-data, and compute differences.

## Cleanup and reproduction

Cleanup removes the obsolete 16-video trial's best.pt/latest.pt (about 868.44
MiB combined), the failed full-batch attempt's temporary subset manifest, and
the intermediate 63-test log. Trial metrics/protocols, diagnostic logs, raw
results, model/data files, and reusable compile caches remain available.

From the HeFT repository, start the configured run with:

```bash
bash latent_video/classification/launch.sh
```

Use --model-path and HEFT_PYTHON only when selecting an existing staged local
model/runtime, and use the same model path again for checkpoint evaluation.
The bounded smoke entry is python -m latent_video.classification.smoke, with
--config, --model-path, --output-dir, --batch-size, and --extract-batch-size.
Launch that module with torchrun to repeat the DDP check. The recorded smoke
scores are repeated-batch diagnostics and must not be used as accuracy results.

Verification artifacts: single_gpu_2x32.json, ddp8.json,
environment_check.json, environment_versions.json, and tests.log.
