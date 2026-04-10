#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_full_experiment.sh [--config path] [--skip-prepare] [--prepare-only] [--dry-run]

What it does:
  1. rebuilds the merged Frieren full training dataset
  2. launches the SDXL DreamBooth-style LoRA training script

Notes:
  - This script assumes you are already inside the correct Python/accelerate environment.
  - Pass --dry-run to print the resolved training command without starting optimization.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_PATH="$REPO_ROOT/configs/train_sdxl_lora.yaml"
DRY_RUN="false"
SKIP_PREPARE="false"
PREPARE_ONLY="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --skip-prepare)
      SKIP_PREPARE="true"
      shift
      ;;
    --prepare-only)
      PREPARE_ONLY="true"
      shift
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "$SKIP_PREPARE" != "true" ]]; then
  python3 "$REPO_ROOT/scripts/prepare_frieren_full_dataset.py" --overwrite
fi

if [[ "$PREPARE_ONLY" == "true" ]]; then
  echo "Dataset preparation complete."
  exit 0
fi

TRAIN_ARGS=("$CONFIG_PATH")
if [[ "$DRY_RUN" == "true" ]]; then
  TRAIN_ARGS+=(--dry-run)
fi

bash "$REPO_ROOT/scripts/train_sdxl_lora.sh" "${TRAIN_ARGS[@]}"
