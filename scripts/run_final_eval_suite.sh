#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_final_eval_suite.sh [extra eval_prompt_bank.py args...]

Examples:
  scripts/run_final_eval_suite.sh
  scripts/run_final_eval_suite.sh --num-images-per-prompt 1
  scripts/run_final_eval_suite.sh --device cuda:0

The script always evaluates base SDXL and evaluates LoRA runs only when their
weights already exist under outputs/train.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROMPT_BANK="$REPO_ROOT/configs/prompt_banks/frieren_eval_v1.yaml"
OUTPUT_ROOT="$REPO_ROOT/outputs/experiments/frieren_eval_v1"

run_eval() {
  local label="$1"
  local config="$2"
  local lora_dir="${3:-}"
  shift 3

  if [[ -n "$lora_dir" ]]; then
    if [[ ! -f "$lora_dir/pytorch_lora_weights.safetensors" && ! -f "$lora_dir/pytorch_lora_weights.bin" ]]; then
      echo "Skipping $label because LoRA weights are not ready: $lora_dir"
      return 0
    fi
    python3 "$REPO_ROOT/scripts/eval_prompt_bank.py" \
      --config "$config" \
      --prompt-bank "$PROMPT_BANK" \
      --lora-dir "$lora_dir" \
      --model-label "$label" \
      --output-root "$OUTPUT_ROOT" \
      "$@"
  else
    python3 "$REPO_ROOT/scripts/eval_prompt_bank.py" \
      --config "$config" \
      --prompt-bank "$PROMPT_BANK" \
      --model-label "$label" \
      --output-root "$OUTPUT_ROOT" \
      "$@"
  fi
}

BASE_CONFIG="$REPO_ROOT/configs/experiments/train_l80_structured_existing.yaml"
L80_CONFIG="$REPO_ROOT/configs/experiments/train_l80_structured_existing.yaml"
L100_CONFIG="$REPO_ROOT/configs/experiments/train_l100_structured.yaml"
L80_SIMPLE_CONFIG="$REPO_ROOT/configs/experiments/train_l80_simple_caption.yaml"

run_eval "B0_base_sdxl" "$BASE_CONFIG" "" "$@"
run_eval "L80_structured" "$L80_CONFIG" "$REPO_ROOT/outputs/train/frieren_sdxl_lora_hd80_single_full_v1" "$@"
run_eval "L100_structured" "$L100_CONFIG" "$REPO_ROOT/outputs/train/frieren_sdxl_lora_hd100_full_structured_v1" "$@"
run_eval "L80_simple" "$L80_SIMPLE_CONFIG" "$REPO_ROOT/outputs/train/frieren_sdxl_lora_hd80_single_simple_caption_v1" "$@"
