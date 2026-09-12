#!/usr/bin/env bash
set -euo pipefail

repo=$(realpath -- "$(dirname -- "${BASH_SOURCE[0]}")/../..")
cd "$repo"
unset SLURM_LOCALID
export PYTHONPATH="$repo:$repo/src:$repo/diffusers/src:$repo/../vjepa2"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1
# Keep Unix sockets off GPFS and compiler artifacts off the small root overlay.
export TMPDIR="${HEFT_TMPDIR:-/dev/shm/heft-ssv2-$UID/tmp}"
export TORCHINDUCTOR_CACHE_DIR="${HEFT_COMPILE_CACHE:-$repo/../cache/ssv2/inductor}"
export TRITON_CACHE_DIR="${HEFT_TRITON_CACHE:-$repo/../cache/ssv2/triton}"
export CUDA_CACHE_PATH="${HEFT_CUDA_CACHE:-$repo/../cache/ssv2/cuda}"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
exec "${HEFT_PYTHON:-$repo/.venv/bin/python}" -m torch.distributed.run \
  --standalone --nproc-per-node="${HEFT_NPROC_PER_NODE:-8}" \
  --module latent_video.classification.train \
  --config latent_video/classification/configs/ssv2.yaml "$@"
