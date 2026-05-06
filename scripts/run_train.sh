#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="configs/experiment/exp_magla_clip.yaml"
OVERRIDE=""
GPU="0"
DO_TEST="1"
EVAL_ONLY="0"
RESUME=""
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_train.sh [options]

Options:
  -c, --config PATH      Experiment config (default: configs/experiment/exp_magla_clip.yaml)
  -o, --override PATH    Override config YAML (optional)
  -g, --gpu ID           CUDA device id (default: 0)
      --resume PATH      Resume checkpoint
      --eval-only        Skip training, run evaluation only
      --no-test          Do not run --do-test after training
  -h, --help             Show help

Examples:
  ./scripts/run_train.sh
  ./scripts/run_train.sh --override configs/eval_sweeps/magla_k80_s8.yaml
  ./scripts/run_train.sh --override configs/eval_sweeps/magla_k90_s16.yaml --gpu 1
  ./scripts/run_train.sh --resume outputs/runs/magla_clip/checkpoints/best.pt
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config)
      CONFIG="$2"; shift 2 ;;
    -o|--override)
      OVERRIDE="$2"; shift 2 ;;
    -g|--gpu)
      GPU="$2"; shift 2 ;;
    --resume)
      RESUME="$2"; shift 2 ;;
    --eval-only)
      EVAL_ONLY="1"; shift ;;
    --no-test)
      DO_TEST="0"; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      EXTRA_ARGS+=("$1"); shift ;;
  esac
done

export CUDA_VISIBLE_DEVICES="$GPU"

CMD=(python src/scripts/train_fsl.py --config "$CONFIG")

if [[ -n "$OVERRIDE" ]]; then
  CMD+=(--override "$OVERRIDE")
fi

if [[ -n "$RESUME" ]]; then
  CMD+=(--resume "$RESUME")
fi

if [[ "$EVAL_ONLY" == "1" ]]; then
  CMD+=(--eval-only)
fi

if [[ "$DO_TEST" == "1" ]]; then
  CMD+=(--do-test)
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
