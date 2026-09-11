# B15 hidden channel scan

Multi-frame t=299: [noise comparison and Chinese technical report](../../scannet_video/f1d218e9976cd0b9/technical_report.md).

Five-scene exploration. Fit channel masks and PCA on four scenes, score the fifth.
All listed dimensions were fixed before scoring. A winning dimension selected from this table needs new-scene validation.
LOSO measures the selection procedure; the five fold masks are not one fixed deployment mask.

For downstream use, [one fixed 256-channel mask](global_256/report.md) is now fitted jointly on all five calibration scenes. All videos use that same table without a `held_out` choice. The held-out scores below remain the original LOSO results.

## Protocol

Cached pooled block outputs only; no model inference. Original geometry and evaluate_pair are unchanged.
Pair macro -> scene macro. Missing pairs are excluded, not zero-filled. Bidirectional top-1 searches all valid targets.
Cosine is normalized again after channel selection. Ties within 1e-6 receive fractional credit.
Deletion ranking uses training voxel-hit loss first, best-correct minus best-incorrect cosine margin second, then channel index.
Once: rank all original channels once. Iterative: rerank the current subset at each dimension in the schedule.
Combination scores are recomputed on training scenes after every pruning step; no held-out score controls pruning.
PCA uses scene-equal valid-token covariance, training-only centering, and no whitening. Centered-full separates centering from compression.
Random subsets use five fixed permutations, nested across dimensions; no data-dependent selection.
Seed standard deviation is across five scene-macro random runs; scene variation is separate in summary.csv. Neither is a confidence interval.

## Held-out results

| Method | Dimensions | Voxel hit (%) | 10cm hit (%) | Mean error (m) | Similarity gap | Voxel change (pp) |
|---|---:|---:|---:|---:|---:|---:|
| original | 1536 | 27.912 | 40.351 | 0.3115 | 0.0317 | +0.000 |
| centered_full | 1536 | 27.970 | 40.606 | 0.3076 | 0.3358 | +0.058 |
| ablation_once | 1024 | 31.795 | 45.617 | 0.2736 | 0.1386 | +3.884 |
| ablation_iterative | 1024 | 31.795 | 45.617 | 0.2736 | 0.1386 | +3.884 |
| pca | 1024 | 27.749 | 40.320 | 0.3098 | 0.3370 | -0.163 |
| random | 1024 | 27.810 +/- 0.882 | 40.224 | 0.3123 | 0.0720 | -0.101 |
| ablation_once | 768 | 32.002 | 45.517 | 0.2717 | 0.1509 | +4.090 |
| ablation_iterative | 768 | 33.412 | 47.349 | 0.2642 | 0.1444 | +5.500 |
| pca | 768 | 27.657 | 40.170 | 0.3113 | 0.3384 | -0.255 |
| random | 768 | 27.447 +/- 0.970 | 39.715 | 0.3156 | 0.0684 | -0.465 |
| ablation_once | 512 | 30.993 | 43.916 | 0.2792 | 0.1526 | +3.081 |
| ablation_iterative | 512 | 34.204 | 48.513 | 0.2559 | 0.1604 | +6.292 |
| pca | 512 | 27.368 | 39.773 | 0.3150 | 0.3408 | -0.543 |
| random | 512 | 26.968 +/- 0.740 | 39.134 | 0.3165 | 0.0885 | -0.944 |
| ablation_once | 384 | 29.662 | 42.487 | 0.2866 | 0.1572 | +1.751 |
| ablation_iterative | 384 | 34.504 | 48.971 | 0.2551 | 0.1885 | +6.592 |
| pca | 384 | 26.981 | 39.193 | 0.3189 | 0.3429 | -0.931 |
| random | 384 | 26.033 +/- 1.252 | 37.997 | 0.3240 | 0.0854 | -1.879 |
| ablation_once | 256 | 27.428 | 39.515 | 0.3095 | 0.1606 | -0.484 |
| ablation_iterative | 256 | 34.703 | 49.063 | 0.2549 | 0.2085 | +6.791 |
| pca | 256 | 26.213 | 38.011 | 0.3261 | 0.3462 | -1.699 |
| random | 256 | 25.536 +/- 1.464 | 37.391 | 0.3279 | 0.1367 | -2.376 |
| ablation_once | 128 | 22.143 | 32.836 | 0.3627 | 0.1662 | -5.769 |
| ablation_iterative | 128 | 32.686 | 46.800 | 0.2740 | 0.2228 | +4.774 |
| pca | 128 | 24.557 | 36.130 | 0.3375 | 0.3577 | -3.355 |
| random | 128 | 24.037 +/- 1.848 | 35.353 | 0.3430 | 0.1387 | -3.875 |

## Per-scene pruning results

| Scene | Original voxel (%) | Iterative 512 | Iterative 256 | Iterative 128 |
|---|---:|---:|---:|---:|
| scene0707_00 | 38.957 | 45.237 | 45.107 | 44.132 |
| scene0708_00 | 37.626 | 46.964 | 47.901 | 43.611 |
| scene0709_00 | 30.989 | 35.862 | 36.878 | 35.974 |
| scene0710_00 | 21.582 | 28.288 | 28.186 | 26.675 |
| scene0711_00 | 10.406 | 14.668 | 15.444 | 13.038 |

## Training versus held-out pruning

| Method | Dimensions | Training voxel (%) | Held-out voxel (%) |
|---|---:|---:|---:|
| ablation_once | 1024 | 32.285 | 31.795 |
| ablation_iterative | 1024 | 32.285 | 31.795 |
| ablation_once | 768 | 32.107 | 32.002 |
| ablation_iterative | 768 | 33.660 | 33.412 |
| ablation_once | 512 | 31.181 | 30.993 |
| ablation_iterative | 512 | 34.805 | 34.204 |
| ablation_once | 384 | 30.100 | 29.662 |
| ablation_iterative | 384 | 35.391 | 34.504 |
| ablation_once | 256 | 28.316 | 27.428 |
| ablation_iterative | 256 | 35.837 | 34.703 |
| ablation_once | 128 | 23.173 | 22.143 |
| ablation_iterative | 128 | 34.318 | 32.686 |

Training scores use float64 accelerated retrieval and overlap across folds; they are diagnostics, not additional independent observations.

![Dimension curves](dimension_curves.png)

## Artifacts

Results: `/increase_kairos_vepfs/increase/liwenhao/agent/2337888765/eval/scannet_channels/73f4e810824b58cb`

`baseline_check.json`: every archived pair checked before selection.
`folds/<scene>/split.json`, `masks.json`, `pca.npz`: fit provenance and reusable transforms.
`initial_ranking.npz`, `ranking_from_*.npz`, `training_trace.json`: deletion effects and combination checks.
All original metrics, including pair medians and coverage, are in the CSV files.
