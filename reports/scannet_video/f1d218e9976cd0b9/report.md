# ScanNet / HEFT Multi-Frame 3D Probe

Noise control experiment and pruning results: [Chinese technical report](technical_report.md).

Run: `f1d218e9976cd0b9`. Chunk size: 25; last-frame tail padding.
WanPipeline: 1 scheduled steps, capture index 0, actual timestep 299, sigma 0.299923837.
Noise mode: legacy; shift 5. One active denoising step; features are captured before its scheduler update.
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
| hidden | 15 | -1 | cosine | 5 | 28.2677 | 40.8187 | 0.3084 |
| hidden | 14 | -1 | cosine | 5 | 27.8323 | 40.6149 | 0.3092 |
| head | 15 | 2 | kk | 5 | 36.3348 | 51.2068 | 0.2877 |
| head | 15 | 1 | qk | 5 | 34.6526 | 48.6240 | 0.2825 |
| head | 15 | 1 | kk | 5 | 34.4695 | 47.8235 | 0.3093 |
| head | 15 | 2 | qk | 5 | 34.1448 | 49.6943 | 0.2693 |
| head | 15 | 5 | kk | 5 | 33.7989 | 47.7550 | 0.2930 |
| head | 15 | 7 | kk | 5 | 33.3129 | 47.6464 | 0.2719 |
| head | 15 | 4 | kk | 5 | 33.0771 | 45.9514 | 0.3220 |
| head | 15 | 0 | kk | 5 | 31.5177 | 45.0069 | 0.2992 |
| head | 14 | 11 | kk | 5 | 31.3915 | 43.9965 | 0.3204 |
| head | 14 | 2 | kk | 5 | 30.8440 | 43.0787 | 0.3545 |
| head | 15 | 5 | qk | 5 | 30.3893 | 44.0467 | 0.3039 |
| head | 15 | 3 | kk | 5 | 29.6696 | 42.7363 | 0.2984 |
| head | 14 | 1 | kk | 5 | 29.2736 | 41.9971 | 0.3096 |
| head | 15 | 4 | qk | 5 | 29.1208 | 43.5222 | 0.3046 |
| head | 14 | 11 | qk | 5 | 29.1163 | 41.8182 | 0.3279 |
| head | 15 | 9 | kk | 5 | 28.4143 | 41.7758 | 0.3020 |
| head | 14 | 2 | qk | 5 | 27.5196 | 40.4462 | 0.3561 |
| head | 14 | 9 | kk | 5 | 25.3622 | 36.6122 | 0.3603 |
| head | 15 | 9 | qk | 5 | 23.7661 | 35.9445 | 0.3375 |
| head | 15 | 7 | qk | 5 | 22.2872 | 34.1627 | 0.3257 |
| head | 14 | 1 | qk | 5 | 22.2072 | 33.3446 | 0.3454 |
| head | 15 | 11 | kk | 5 | 22.1600 | 31.6550 | 0.4306 |
| head | 14 | 9 | qk | 5 | 21.1721 | 31.3671 | 0.3855 |
| head | 15 | 0 | qk | 5 | 20.9418 | 32.7579 | 0.3586 |
| head | 15 | 6 | kk | 5 | 19.5973 | 30.2383 | 0.3656 |
| head | 14 | 7 | kk | 5 | 18.6508 | 27.7656 | 0.4265 |
| head | 15 | 3 | qk | 5 | 17.4471 | 27.0174 | 0.3982 |
| head | 15 | 11 | qk | 5 | 14.8071 | 22.3865 | 0.5018 |
| head | 14 | 7 | qk | 5 | 12.8842 | 20.5652 | 0.4765 |
| head | 15 | 6 | qk | 5 | 12.0892 | 20.1672 | 0.4558 |
| head | 14 | 6 | kk | 5 | 10.8538 | 16.9850 | 0.5514 |
| head | 15 | 10 | kk | 5 | 9.8468 | 15.3763 | 0.5682 |
| head | 14 | 10 | kk | 5 | 8.8080 | 13.9343 | 0.5934 |
| head | 14 | 5 | kk | 5 | 7.8826 | 12.5963 | 0.6140 |
| head | 15 | 8 | kk | 5 | 7.6166 | 11.9860 | 0.6334 |
| head | 15 | 10 | qk | 5 | 6.1983 | 10.5968 | 0.6017 |
| head | 14 | 6 | qk | 5 | 5.9351 | 10.6755 | 0.6047 |
| head | 14 | 8 | kk | 5 | 5.5780 | 9.2144 | 0.6635 |
| head | 14 | 10 | qk | 5 | 4.4786 | 7.8682 | 0.6668 |
| head | 14 | 3 | kk | 5 | 3.5265 | 6.2250 | 0.6911 |
| head | 14 | 8 | qk | 5 | 3.2596 | 5.6946 | 0.6950 |
| head | 14 | 0 | kk | 5 | 2.8557 | 5.1812 | 0.7051 |
| head | 14 | 5 | qk | 5 | 2.8117 | 5.1143 | 0.7221 |
| head | 15 | 8 | qk | 5 | 2.7879 | 4.6314 | 0.7599 |
| head | 14 | 3 | qk | 5 | 2.7226 | 5.2600 | 0.6953 |
| head | 14 | 4 | kk | 5 | 2.6874 | 4.8289 | 0.7096 |
| head | 14 | 4 | qk | 5 | 2.0734 | 4.4989 | 0.7225 |
| head | 14 | 0 | qk | 5 | 1.9187 | 3.6999 | 0.7067 |

## all_pairs

| Feature | Block | Head | Mode | Scenes | Voxel hit (%) | 10cm hit (%) | Error (m) |
|---|---:|---:|---|---:|---:|---:|---:|
| hidden | 15 | -1 | cosine | 5 | 27.9119 | 40.3510 | 0.3115 |
| hidden | 14 | -1 | cosine | 5 | 27.4452 | 40.1012 | 0.3122 |
| head | 15 | 2 | kk | 5 | 35.8431 | 50.5331 | 0.2907 |
| head | 15 | 1 | qk | 5 | 34.0850 | 47.9226 | 0.2884 |
| head | 15 | 1 | kk | 5 | 33.9628 | 47.1886 | 0.3147 |
| head | 15 | 2 | qk | 5 | 33.5243 | 48.7980 | 0.2754 |
| head | 15 | 5 | kk | 5 | 33.3233 | 47.1287 | 0.2967 |
| head | 15 | 7 | kk | 5 | 32.8810 | 47.0399 | 0.2747 |
| head | 15 | 4 | kk | 5 | 32.5249 | 45.2668 | 0.3278 |
| head | 15 | 0 | kk | 5 | 30.9725 | 44.3180 | 0.3019 |
| head | 14 | 11 | kk | 5 | 30.9418 | 43.4152 | 0.3248 |
| head | 14 | 2 | kk | 5 | 30.2956 | 42.3397 | 0.3608 |
| head | 15 | 5 | qk | 5 | 29.8582 | 43.3966 | 0.3080 |
| head | 15 | 3 | kk | 5 | 29.2033 | 42.0432 | 0.3018 |
| head | 14 | 1 | kk | 5 | 28.8687 | 41.4391 | 0.3110 |
| head | 14 | 11 | qk | 5 | 28.5483 | 41.1363 | 0.3333 |
| head | 15 | 4 | qk | 5 | 28.5106 | 42.6688 | 0.3137 |
| head | 15 | 9 | kk | 5 | 27.8398 | 41.0015 | 0.3072 |
| head | 14 | 2 | qk | 5 | 27.1131 | 39.8734 | 0.3598 |
| head | 14 | 9 | kk | 5 | 24.7897 | 35.8597 | 0.3653 |
| head | 15 | 9 | qk | 5 | 23.4055 | 35.3615 | 0.3411 |
| head | 14 | 1 | qk | 5 | 21.9133 | 32.8848 | 0.3485 |
| head | 15 | 7 | qk | 5 | 21.9076 | 33.6122 | 0.3310 |
| head | 15 | 11 | kk | 5 | 21.6504 | 30.9526 | 0.4359 |
| head | 14 | 9 | qk | 5 | 20.7763 | 30.8972 | 0.3885 |
| head | 15 | 0 | qk | 5 | 20.6273 | 32.3039 | 0.3608 |
| head | 15 | 6 | kk | 5 | 19.2316 | 29.7187 | 0.3699 |
| head | 14 | 7 | kk | 5 | 18.3106 | 27.3461 | 0.4302 |
| head | 15 | 3 | qk | 5 | 17.1641 | 26.6031 | 0.4003 |
| head | 15 | 11 | qk | 5 | 14.5108 | 21.9745 | 0.5081 |
| head | 14 | 7 | qk | 5 | 12.8096 | 20.3596 | 0.4774 |
| head | 15 | 6 | qk | 5 | 11.8159 | 19.7506 | 0.4588 |
| head | 14 | 6 | kk | 5 | 10.6815 | 16.8205 | 0.5558 |
| head | 15 | 10 | kk | 5 | 9.6229 | 15.0514 | 0.5702 |
| head | 14 | 10 | kk | 5 | 8.6046 | 13.6625 | 0.5988 |
| head | 14 | 5 | kk | 5 | 7.7141 | 12.3656 | 0.6211 |
| head | 15 | 8 | kk | 5 | 7.4579 | 11.7563 | 0.6373 |
| head | 15 | 10 | qk | 5 | 6.0837 | 10.4941 | 0.6054 |
| head | 14 | 6 | qk | 5 | 5.8416 | 10.5214 | 0.6106 |
| head | 14 | 8 | kk | 5 | 5.4430 | 9.0069 | 0.6686 |
| head | 14 | 10 | qk | 5 | 4.3960 | 7.7502 | 0.6719 |
| head | 14 | 3 | kk | 5 | 3.4616 | 6.0945 | 0.6967 |
| head | 14 | 8 | qk | 5 | 3.1989 | 5.6040 | 0.7014 |
| head | 14 | 0 | kk | 5 | 2.7886 | 5.0534 | 0.7096 |
| head | 14 | 5 | qk | 5 | 2.7864 | 5.0974 | 0.7280 |
| head | 15 | 8 | qk | 5 | 2.7170 | 4.5401 | 0.7616 |
| head | 14 | 3 | qk | 5 | 2.6552 | 5.1385 | 0.7003 |
| head | 14 | 4 | kk | 5 | 2.6257 | 4.7032 | 0.7139 |
| head | 14 | 4 | qk | 5 | 2.0082 | 4.3946 | 0.7272 |
| head | 14 | 0 | qk | 5 | 1.8690 | 3.6229 | 0.7124 |

Full metrics: [summary.json](summary.json).

Raw pair/scene CSVs, configuration, chunk identities and forward-shape audits are in the experiment directory.
Five-scene results are exploratory. Legacy-noise mode matches the old noise point while retaining video VAE sampling and precision.
