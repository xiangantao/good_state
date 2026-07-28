# HeFT

HeFT tracks points with intermediate attention features from video diffusion models. It captures conditional, post-RoPE Q/K features at a selected denoising step, then keeps feature extraction, tracking, and dataset evaluation independent.

## Setup

The project uses Python 3.12, uv, PyTorch with CUDA 12.8, and the local `diffusers` source tree.

```bash
uv sync --group dev
```

## DAVIS workflow

Download the official TAP-Vid DAVIS pickle:

```bash
mkdir -p datasets
curl -L https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip -o /tmp/tapvid_davis.zip
unzip /tmp/tapvid_davis.zip -d datasets
```

This creates `datasets/tapvid_davis/tapvid_davis.pkl`.

Machine-specific paths live in the ignored `.env` file:

```dotenv
HEFT_DATASET_ROOT=/path/to/tapvid_davis
HEFT_MODEL_CACHE_DIR=/path/to/model_cache
HEFT_MODEL=cosmos2
HEFT_FEATURE_OUTPUT_ROOT=/path/to/features/{model}_davis
HEFT_EVALUATION_OUTPUT=/path/to/evaluations/{model}_davis.json
HEFT_GPU_IDS=0,1,2,3,4,5,6,7
HEFT_OVERWRITE=false
```

`HEFT_MODEL` accepts `wan`, `cosmos2`, or `cogvideox`.

Extract features:

```bash
uv run python scripts/extract_dataset_features.py
```

Track and evaluate:

```bash
uv run python scripts/evaluate_davis_features.py
```

## Python API

```python
from heft import (
    COSMOS_2,
    ExtractionConfig,
    ExtractionTask,
    TailFramePolicy,
    extract_feature,
)
from heft.attn_hook import CaptureSpec, FeatureKind

task = ExtractionTask(
    name="video-id",
    input_video=video,  # CPU uint8 [T, C, H, W]
    output_dir="features/video-id",
    seed=42,
)
config = ExtractionConfig(
    capture=CaptureSpec(
        step=34,
        layers=(18,),
        heads=(6,),
        features=(FeatureKind.QUERY, FeatureKind.KEY),
    ),
    chunk_size=25,
    tail_policy=TailFramePolicy.DISCARD,
)

result = extract_feature(
    task,
    model=COSMOS_2,
    config=config,
    gpu_ids=(0, 1),
    cache_dir="/path/to/model_cache",
)
```

Tracking is exposed through `track_feature()` and `track_features()`. Query generators include `ExplicitQueries`, `GridQueries`, `MaskGridQueries`, and `AnnotationQueries`. Dataset-backed results can be evaluated with `evaluate_dataset()`.

## Development

```bash
uv run pytest
uv run ruff check src scripts test
uv run pyright
```
