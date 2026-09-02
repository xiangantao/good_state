#!/usr/bin/env bash
# 所有 cache 重定向到 vePFS —— 根盘 overlay 只有 40G 且已 100% 满，
# triton/inductor/HF/tmp 任何一个落到 / 上都会把任务写爆。
# 用法: ./run.sh python scripts/extract_dataset_features.py
set -euo pipefail

PROJ=/KAIROS_vepfs-2/KAIROS_vepfs/liwenhao/2337888765

export TMPDIR="$PROJ/tmp"
export XDG_CACHE_HOME="$PROJ/cache/xdg"
export HF_HOME="$PROJ/cache/hf"
export TRITON_CACHE_DIR="$PROJ/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PROJ/cache/inductor"
export CUDA_CACHE_PATH="$PROJ/cache/nv"
export UV_CACHE_DIR="$PROJ/cache/uv"
export HF_HUB_OFFLINE=1        # 权重都在本地，禁止任何回源
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TRITON_CACHE_DIR" \
         "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"

cd "$PROJ/heft"
export PATH="$PROJ/heft/.venv/bin:$PATH"
exec "$@"
