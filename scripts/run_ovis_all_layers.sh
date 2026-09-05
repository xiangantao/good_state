#!/usr/bin/env bash
# Sequentially extract, evaluate, and discard one Wan layer at a time.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORKSPACE=$(cd "$REPO_ROOT/.." && pwd)
PYTHON=${HEFT_PYTHON:-$REPO_ROOT/.venv/bin/python}
GPU_IDS=${HEFT_GPU_IDS:-0,1,2,3,4,5,6,7}
EXTRACT_WORKERS_PER_GPU=${HEFT_EXTRACT_WORKERS_PER_GPU:-3}
VIDEO_BATCH_SIZE=${HEFT_OVIS_VIDEO_BATCH_SIZE:-16}
MAX_PENDING_FEATURES=${HEFT_MAX_PENDING_FEATURES:-96}
EVAL_WORKERS=${HEFT_EVAL_WORKERS:-16}
EVAL_QUERY_BATCH_SIZE=${HEFT_EVAL_QUERY_BATCH_SIZE:-256}
EVAL_DEVICES=${HEFT_EVAL_DEVICES:-$(printf '%s' "$GPU_IDS" | sed 's/^/cuda:/; s/,/,cuda:/g')}
FIRST_LAYER=${HEFT_FIRST_LAYER:-0}
LAST_LAYER=${HEFT_LAST_LAYER:-29}
FEATURE_BASE=${HEFT_OVIS_FEATURE_BASE:-$WORKSPACE/features/ovis_wan/sequential_tmp}
LAYER_EVAL_DIR=${HEFT_OVIS_LAYER_EVAL_DIR:-$WORKSPACE/eval/ovis_wan_layers}
EVAL_OUTPUT=${HEFT_OVIS_EVAL_OUTPUT:-$WORKSPACE/eval/ovis_wan_all_layers_semantics.json}
EVAL_CSV=${HEFT_OVIS_EVAL_CSV:-$WORKSPACE/eval/ovis_wan_all_layers_semantics.csv}

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$FEATURE_BASE" "$LAYER_EVAL_DIR" "$(dirname "$EVAL_OUTPUT")"

printf '[%s] sequential scan: layers=%s..%s, GPUs=%s, workers/GPU=%s\n' \
  "$(date -u '+%F %T UTC')" "$FIRST_LAYER" "$LAST_LAYER" \
  "$GPU_IDS" "$EXTRACT_WORKERS_PER_GPU"

for ((layer=FIRST_LAYER; layer<=LAST_LAYER; layer++)); do
  layer_tag=$(printf '%03d' "$layer")
  feature_root="$FEATURE_BASE/layer_$layer_tag"
  layer_output="$LAYER_EVAL_DIR/layer_$layer_tag.json"

  if [[ -s "$layer_output" ]]; then
    printf '[%s] layer %s already evaluated; skip\n' \
      "$(date -u '+%F %T UTC')" "$layer_tag"
    rm -rf -- "$feature_root"
    "$PYTHON" scripts/merge_ovis_layer_reports.py \
      --input-dir "$LAYER_EVAL_DIR" \
      --output "$EVAL_OUTPUT" \
      --csv "$EVAL_CSV"
    continue
  fi

  # A result-less directory is an interrupted partial extraction; restart this layer.
  rm -rf -- "$feature_root"

  printf '[%s] layer %s/029 extraction start\n' \
    "$(date -u '+%F %T UTC')" "$layer_tag"
  "$PYTHON" -u scripts/extract_ovis_features.py \
    --layers "$layer" \
    --heads all \
    --gpu-ids "$GPU_IDS" \
    --workers-per-gpu "$EXTRACT_WORKERS_PER_GPU" \
    --video-batch-size "$VIDEO_BATCH_SIZE" \
    --max-pending-features "$MAX_PENDING_FEATURES" \
    --output-root "$feature_root"

  printf '[%s] layer %s evaluation start: devices=%s, workers=%s\n' \
    "$(date -u '+%F %T UTC')" "$layer_tag" "$EVAL_DEVICES" "$EVAL_WORKERS"
  "$PYTHON" -u scripts/evaluate_ovis_semantics.py \
    --feature-root "$feature_root" \
    --layers "$layer" \
    --heads all \
    --output "$layer_output" \
    --devices "$EVAL_DEVICES" \
    --workers "$EVAL_WORKERS" \
    --query-batch-size "$EVAL_QUERY_BATCH_SIZE"

  "$PYTHON" scripts/merge_ovis_layer_reports.py \
    --input-dir "$LAYER_EVAL_DIR" \
    --output "$EVAL_OUTPUT" \
    --csv "$EVAL_CSV"

  rm -rf -- "$feature_root"
  printf '[%s] layer %s complete; metrics saved and features deleted\n' \
    "$(date -u '+%F %T UTC')" "$layer_tag"
done

rmdir "$FEATURE_BASE" 2>/dev/null || true
printf '[%s] all layers complete: %s\n' \
  "$(date -u '+%F %T UTC')" "$EVAL_OUTPUT"
