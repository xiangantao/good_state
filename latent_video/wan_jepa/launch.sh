#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$repo"
unset PYTHONHOME SLURM_LOCALID
export PYTHONPATH="$repo:$repo/src:$repo/diffusers/src:$repo/../vjepa2"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline WANDB_ERROR_REPORTING=false TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1
python_bin="${HEFT_PYTHON:-$repo/.venv/bin/python}"
for argument in "$@"; do
  if [[ "$argument" == --check || "$argument" == --help ]]; then
    exec "$python_bin" -m latent_video.wan_jepa.train "$@"
  fi
done
export TMPDIR="${HEFT_TMPDIR:-/dev/shm/heft-wan-jepa-$UID/tmp}"
export TORCHINDUCTOR_CACHE_DIR="${HEFT_COMPILE_CACHE:-$repo/../cache/ssv2/inductor}"
export TRITON_CACHE_DIR="${HEFT_TRITON_CACHE:-$repo/../cache/ssv2/triton}"
export CUDA_CACHE_PATH="${HEFT_CUDA_CACHE:-$repo/../cache/ssv2/cuda}"
mkdir -p -- "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
exec "$python_bin" -m torch.distributed.run --standalone \
  --nproc-per-node="${HEFT_NPROC_PER_NODE:-8}" --module latent_video.wan_jepa.train "$@"
