# ScanNet / HEFT Multi-Frame 3D Probe

Implementation, alignment checks and the legacy comparison: [Chinese report](alignment_report.md).

Run: `0d6428dbc0f43571`. Chunk size: 25; last-frame tail padding.
WanPipeline: 50 steps, capture index 49, actual timestep 57, sigma 0.057636831.
Video VAE encoding and joint spatiotemporal Transformer attention use the HEFT video path.
Block outputs and conditional post-normalization/post-RoPE Q/K are pooled per frame.
Frames are uniformly sampled from each scene before chunking; original frame gaps and timestamps are retained.
This aligns video encoding, not sampling density or downstream tracking/semantic evaluation.
Within-chunk pairs share input context. All-pairs also includes chunk boundaries; padded frames are never scored.
Geometry, all-target retrieval, fractional ties, and pair-macro then scene-macro aggregation are unchanged.
Head QK is scaled dot product; KK and block hidden use cosine. RoPE is retained.
Channel masks fitted on the previous single-frame features have not been validated on these descriptors.

## within_chunk

| Feature | Block | Head | Mode | Scenes | Voxel hit (%) | 10cm hit (%) | Error (m) |
|---|---:|---:|---|---:|---:|---:|---:|
| hidden | 14 | -1 | cosine | 5 | 21.9769 | 32.3370 | 0.3727 |
| hidden | 15 | -1 | cosine | 5 | 21.5483 | 31.8578 | 0.3752 |
| head | 15 | 2 | kk | 5 | 27.2640 | 38.6442 | 0.3844 |
| head | 15 | 7 | kk | 5 | 27.1909 | 39.3717 | 0.3388 |
| head | 15 | 5 | kk | 5 | 25.9368 | 37.2661 | 0.3815 |
| head | 15 | 2 | qk | 5 | 25.3521 | 37.5463 | 0.3743 |
| head | 15 | 0 | kk | 5 | 24.8886 | 35.7609 | 0.3864 |
| head | 15 | 1 | qk | 5 | 24.2426 | 34.5392 | 0.4125 |
| head | 14 | 1 | kk | 5 | 23.9023 | 34.4709 | 0.3773 |
| head | 15 | 3 | kk | 5 | 23.4946 | 34.5253 | 0.3723 |
| head | 15 | 5 | qk | 5 | 22.8245 | 33.3908 | 0.3975 |
| head | 15 | 1 | kk | 5 | 22.5674 | 32.0617 | 0.4414 |
| head | 15 | 9 | kk | 5 | 22.1569 | 32.9402 | 0.3710 |
| head | 14 | 11 | kk | 5 | 22.1016 | 31.3749 | 0.4269 |
| head | 15 | 7 | qk | 5 | 21.2281 | 32.0229 | 0.3767 |
| head | 15 | 4 | qk | 5 | 20.9714 | 30.7779 | 0.4407 |
| head | 14 | 11 | qk | 5 | 20.8872 | 30.3318 | 0.4354 |
| head | 15 | 4 | kk | 5 | 20.6751 | 29.4270 | 0.4640 |
| head | 14 | 2 | kk | 5 | 20.4104 | 29.0452 | 0.4738 |
| head | 14 | 1 | qk | 5 | 19.4586 | 29.3802 | 0.4053 |
| head | 14 | 2 | qk | 5 | 19.0944 | 27.9274 | 0.4747 |
| head | 14 | 9 | kk | 5 | 18.6721 | 27.0943 | 0.4521 |
| head | 15 | 0 | qk | 5 | 18.4447 | 28.3538 | 0.4293 |
| head | 15 | 11 | kk | 5 | 17.9484 | 25.9507 | 0.4977 |
| head | 15 | 9 | qk | 5 | 16.7326 | 26.4085 | 0.4236 |
| head | 14 | 9 | qk | 5 | 16.5303 | 25.0109 | 0.4489 |
| head | 15 | 6 | kk | 5 | 15.4881 | 24.3665 | 0.4331 |
| head | 14 | 7 | kk | 5 | 15.3156 | 22.8645 | 0.4808 |
| head | 15 | 3 | qk | 5 | 15.1686 | 23.7508 | 0.4502 |
| head | 15 | 11 | qk | 5 | 13.4356 | 20.4175 | 0.5357 |
| head | 14 | 7 | qk | 5 | 11.6649 | 18.6380 | 0.5260 |
| head | 15 | 6 | qk | 5 | 10.4801 | 17.6285 | 0.5123 |
| head | 14 | 6 | kk | 5 | 10.2294 | 16.0261 | 0.5601 |
| head | 15 | 10 | kk | 5 | 8.1563 | 13.1723 | 0.6044 |
| head | 14 | 5 | kk | 5 | 7.4893 | 12.0491 | 0.6234 |
| head | 14 | 6 | qk | 5 | 6.7262 | 11.6321 | 0.6074 |
| head | 14 | 10 | kk | 5 | 6.6518 | 10.9765 | 0.6357 |
| head | 15 | 10 | qk | 5 | 6.5741 | 10.9212 | 0.6136 |
| head | 15 | 8 | kk | 5 | 5.9865 | 9.4807 | 0.6568 |
| head | 14 | 8 | kk | 5 | 5.1291 | 8.6224 | 0.6685 |
| head | 14 | 10 | qk | 5 | 4.7642 | 8.2935 | 0.6777 |
| head | 14 | 5 | qk | 5 | 4.0080 | 7.2818 | 0.6721 |
| head | 14 | 8 | qk | 5 | 3.6563 | 6.8200 | 0.6892 |
| head | 14 | 3 | kk | 5 | 3.4192 | 6.1539 | 0.6925 |
| head | 15 | 8 | qk | 5 | 3.2537 | 5.4402 | 0.7327 |
| head | 14 | 3 | qk | 5 | 3.1763 | 5.6690 | 0.6968 |
| head | 14 | 0 | kk | 5 | 2.7380 | 4.9144 | 0.7073 |
| head | 14 | 0 | qk | 5 | 2.6808 | 4.8005 | 0.7014 |
| head | 14 | 4 | kk | 5 | 2.6680 | 4.7873 | 0.7103 |
| head | 14 | 4 | qk | 5 | 2.3562 | 4.7848 | 0.7163 |

## all_pairs

| Feature | Block | Head | Mode | Scenes | Voxel hit (%) | 10cm hit (%) | Error (m) |
|---|---:|---:|---|---:|---:|---:|---:|
| hidden | 14 | -1 | cosine | 5 | 21.7391 | 31.9504 | 0.3760 |
| hidden | 15 | -1 | cosine | 5 | 21.3021 | 31.4778 | 0.3790 |
| head | 15 | 2 | kk | 5 | 26.9205 | 38.0987 | 0.3929 |
| head | 15 | 7 | kk | 5 | 26.8773 | 38.8744 | 0.3440 |
| head | 15 | 5 | kk | 5 | 25.5296 | 36.6312 | 0.3894 |
| head | 15 | 2 | qk | 5 | 24.9101 | 36.9122 | 0.3828 |
| head | 15 | 0 | kk | 5 | 24.4748 | 35.1588 | 0.3958 |
| head | 15 | 1 | qk | 5 | 23.6933 | 33.7983 | 0.4265 |
| head | 14 | 1 | kk | 5 | 23.6088 | 34.0050 | 0.3831 |
| head | 15 | 3 | kk | 5 | 23.0397 | 33.8531 | 0.3801 |
| head | 15 | 5 | qk | 5 | 22.3725 | 32.7490 | 0.4053 |
| head | 15 | 1 | kk | 5 | 22.1339 | 31.4615 | 0.4532 |
| head | 14 | 11 | kk | 5 | 21.7852 | 30.9748 | 0.4358 |
| head | 15 | 9 | kk | 5 | 21.7637 | 32.4405 | 0.3761 |
| head | 15 | 7 | qk | 5 | 20.8428 | 31.4643 | 0.3858 |
| head | 15 | 4 | qk | 5 | 20.4882 | 30.1489 | 0.4531 |
| head | 14 | 11 | qk | 5 | 20.4775 | 29.7906 | 0.4447 |
| head | 15 | 4 | kk | 5 | 20.2794 | 28.8761 | 0.4761 |
| head | 14 | 2 | kk | 5 | 19.9747 | 28.4338 | 0.4862 |
| head | 14 | 1 | qk | 5 | 19.1835 | 28.8930 | 0.4127 |
| head | 14 | 2 | qk | 5 | 18.7083 | 27.3888 | 0.4873 |
| head | 14 | 9 | kk | 5 | 18.3548 | 26.6458 | 0.4596 |
| head | 15 | 0 | qk | 5 | 18.0950 | 27.8502 | 0.4375 |
| head | 15 | 11 | kk | 5 | 17.5282 | 25.3656 | 0.5033 |
| head | 15 | 9 | qk | 5 | 16.4561 | 25.9815 | 0.4286 |
| head | 14 | 9 | qk | 5 | 16.2588 | 24.6292 | 0.4541 |
| head | 15 | 6 | kk | 5 | 15.2650 | 24.0545 | 0.4364 |
| head | 14 | 7 | kk | 5 | 15.0944 | 22.5590 | 0.4865 |
| head | 15 | 3 | qk | 5 | 14.8664 | 23.2775 | 0.4604 |
| head | 15 | 11 | qk | 5 | 13.1915 | 20.0455 | 0.5446 |
| head | 14 | 7 | qk | 5 | 11.5363 | 18.4497 | 0.5293 |
| head | 15 | 6 | qk | 5 | 10.3549 | 17.4776 | 0.5149 |
| head | 14 | 6 | kk | 5 | 10.1641 | 15.8838 | 0.5650 |
| head | 15 | 10 | kk | 5 | 7.9892 | 12.9229 | 0.6098 |
| head | 14 | 5 | kk | 5 | 7.3264 | 11.7944 | 0.6294 |
| head | 14 | 6 | qk | 5 | 6.6224 | 11.4501 | 0.6155 |
| head | 14 | 10 | kk | 5 | 6.4779 | 10.7076 | 0.6418 |
| head | 15 | 10 | qk | 5 | 6.4425 | 10.7350 | 0.6209 |
| head | 15 | 8 | kk | 5 | 5.8422 | 9.2790 | 0.6616 |
| head | 14 | 8 | kk | 5 | 5.0043 | 8.4422 | 0.6744 |
| head | 14 | 10 | qk | 5 | 4.6613 | 8.1231 | 0.6832 |
| head | 14 | 5 | qk | 5 | 3.9399 | 7.1822 | 0.6809 |
| head | 14 | 8 | qk | 5 | 3.5777 | 6.6780 | 0.6949 |
| head | 14 | 3 | kk | 5 | 3.3475 | 6.0319 | 0.6970 |
| head | 15 | 8 | qk | 5 | 3.2234 | 5.3745 | 0.7370 |
| head | 14 | 3 | qk | 5 | 3.0736 | 5.5323 | 0.7014 |
| head | 14 | 0 | kk | 5 | 2.6747 | 4.7905 | 0.7117 |
| head | 14 | 4 | kk | 5 | 2.6069 | 4.6631 | 0.7145 |
| head | 14 | 0 | qk | 5 | 2.6056 | 4.6673 | 0.7060 |
| head | 14 | 4 | qk | 5 | 2.2818 | 4.6600 | 0.7210 |

Full metrics: [summary.json](summary.json).

Raw pair/scene CSVs, configuration, chunk identities and forward-shape audits are in the experiment directory.
Five-scene results are exploratory; changes from the legacy k=300 run also include VAE and noise protocol changes.
