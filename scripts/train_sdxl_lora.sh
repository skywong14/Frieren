#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/train_sdxl_lora.sh [config_path] [--dry-run]

Notes:
  - config_path defaults to configs/train_sdxl_lora.yaml
  - Set DIFFUSERS_TRAIN_SCRIPT to point at train_dreambooth_lora_sdxl.py, or
    vendor the script under third_party/diffusers/examples/dreambooth/
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_CONFIG="$REPO_ROOT/configs/train_sdxl_lora.yaml"
CONFIG_PATH="$DEFAULT_CONFIG"
DRY_RUN="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ "$CONFIG_PATH" != "$DEFAULT_CONFIG" ]]; then
        echo "Error: only one config path can be provided." >&2
        usage >&2
        exit 1
      fi
      CONFIG_PATH="$1"
      shift
      ;;
  esac
done

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: required command not found: $1" >&2
    exit 1
  fi
}

resolve_repo_path() {
  python3 - "$REPO_ROOT" "$1" <<'PY'
from pathlib import Path
import sys
repo_root = Path(sys.argv[1])
path_value = Path(sys.argv[2]).expanduser()
if not path_value.is_absolute():
    path_value = repo_root / path_value
print(path_value.resolve())
PY
}

resolve_path() {
  python3 - "$1" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
}

require_command python3
require_command accelerate

if ! python3 -c 'import yaml' >/dev/null 2>&1; then
  echo "Error: python module 'yaml' is required. Install PyYAML first." >&2
  exit 1
fi

CONFIG_PATH="$(resolve_path "$CONFIG_PATH")"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Error: config file not found: $CONFIG_PATH" >&2
  exit 1
fi

yaml_get() {
  python3 - "$CONFIG_PATH" "$1" <<'PY'
import json
import sys
import yaml

config_path, dotted_key = sys.argv[1], sys.argv[2]
with open(config_path, "r", encoding="utf-8") as handle:
    data = yaml.safe_load(handle)

value = data
for part in dotted_key.split("."):
    if not isinstance(value, dict) or part not in value:
        raise KeyError(f"Missing config key: {dotted_key}")
    value = value[part]

if value is None:
    print("")
elif isinstance(value, bool):
    print("true" if value else "false")
elif isinstance(value, (list, dict)):
    print(json.dumps(value))
else:
    print(value)
PY
}

add_bool_flag() {
  local enabled="$1"
  local flag_name="$2"
  if [[ "$enabled" == "true" ]]; then
    CMD+=("$flag_name")
  fi
}

PROJECT_RUN_NAME="$(yaml_get project.run_name)"
PROJECT_OUTPUT_ROOT="$(resolve_repo_path "$(yaml_get project.output_root)")"
PROJECT_SEED="$(yaml_get project.seed)"

MODEL_BASE="$(yaml_get model.base_model)"
MODEL_VAE="$(yaml_get model.vae_model)"

DATASET_ROOT="$(resolve_repo_path "$(yaml_get data.dataset_root)")"
IMAGES_DIR="$(resolve_repo_path "$(yaml_get data.images_dir)")"
METADATA_PATH="$(resolve_repo_path "$(yaml_get data.metadata_path)")"
IMAGE_COLUMN="$(yaml_get data.image_column)"
CAPTION_COLUMN="$(yaml_get data.caption_column)"
INSTANCE_PROMPT="$(yaml_get data.instance_prompt)"

TRAIN_RESOLUTION="$(yaml_get training.resolution)"
TRAIN_BATCH_SIZE="$(yaml_get training.train_batch_size)"
SAMPLE_BATCH_SIZE="$(yaml_get training.sample_batch_size)"
GRADIENT_ACCUMULATION_STEPS="$(yaml_get training.gradient_accumulation_steps)"
LEARNING_RATE="$(yaml_get training.learning_rate)"
LR_SCHEDULER="$(yaml_get training.lr_scheduler)"
LR_WARMUP_STEPS="$(yaml_get training.lr_warmup_steps)"
MAX_TRAIN_STEPS="$(yaml_get training.max_train_steps)"
CHECKPOINTING_STEPS="$(yaml_get training.checkpointing_steps)"
CHECKPOINTS_TOTAL_LIMIT="$(yaml_get training.checkpoints_total_limit)"
MIXED_PRECISION="$(yaml_get training.mixed_precision)"
RANK="$(yaml_get training.rank)"
TRAIN_TEXT_ENCODER="$(yaml_get training.train_text_encoder)"
GRADIENT_CHECKPOINTING="$(yaml_get training.gradient_checkpointing)"
USE_8BIT_ADAM="$(yaml_get training.use_8bit_adam)"
ENABLE_XFORMERS="$(yaml_get training.enable_xformers_memory_efficient_attention)"
RANDOM_FLIP="$(yaml_get training.random_flip)"
CENTER_CROP="$(yaml_get training.center_crop)"
DATALOADER_NUM_WORKERS="$(yaml_get training.dataloader_num_workers)"
REPORT_TO="$(yaml_get training.report_to)"
RESUME_FROM_CHECKPOINT="$(yaml_get training.resume_from_checkpoint)"

VALIDATION_PROMPT="$(yaml_get validation.prompt)"
VALIDATION_NUM_IMAGES="$(yaml_get validation.num_images)"
VALIDATION_EVERY_N_EPOCHS="$(yaml_get validation.every_n_epochs)"

OUTPUT_DIR="$PROJECT_OUTPUT_ROOT/$PROJECT_RUN_NAME"

if [[ -n "${DIFFUSERS_TRAIN_SCRIPT:-}" ]]; then
  TRAIN_SCRIPT="$(resolve_path "$DIFFUSERS_TRAIN_SCRIPT")"
else
  TRAIN_SCRIPT="$REPO_ROOT/third_party/diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py"
fi

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  cat >&2 <<EOF2
Error: could not find Diffusers training script.

Set DIFFUSERS_TRAIN_SCRIPT to the path of train_dreambooth_lora_sdxl.py,
or vendor it at:
  $REPO_ROOT/third_party/diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py
EOF2
  exit 1
fi

if [[ ! -f "$METADATA_PATH" ]]; then
  echo "Error: metadata file not found: $METADATA_PATH" >&2
  exit 1
fi

if [[ ! -d "$IMAGES_DIR" ]]; then
  echo "Error: images directory not found: $IMAGES_DIR" >&2
  exit 1
fi

IMAGE_COUNT="$(find "$IMAGES_DIR" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' -o -iname '*.bmp' \) | wc -l | tr -d ' ')"
if [[ "$IMAGE_COUNT" -lt 1 ]]; then
  echo "Error: no training images found under $IMAGES_DIR" >&2
  exit 1
fi

CMD=(
  accelerate launch "$TRAIN_SCRIPT"
  --pretrained_model_name_or_path "$MODEL_BASE"
  --pretrained_vae_model_name_or_path "$MODEL_VAE"
  --dataset_name "$DATASET_ROOT"
  --caption_column "$CAPTION_COLUMN"
  --image_column "$IMAGE_COLUMN"
  --instance_prompt "$INSTANCE_PROMPT"
  --output_dir "$OUTPUT_DIR"
  --seed "$PROJECT_SEED"
  --resolution "$TRAIN_RESOLUTION"
  --train_batch_size "$TRAIN_BATCH_SIZE"
  --sample_batch_size "$SAMPLE_BATCH_SIZE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --learning_rate "$LEARNING_RATE"
  --lr_scheduler "$LR_SCHEDULER"
  --lr_warmup_steps "$LR_WARMUP_STEPS"
  --max_train_steps "$MAX_TRAIN_STEPS"
  --checkpointing_steps "$CHECKPOINTING_STEPS"
  --checkpoints_total_limit "$CHECKPOINTS_TOTAL_LIMIT"
  --mixed_precision "$MIXED_PRECISION"
  --rank "$RANK"
  --validation_prompt "$VALIDATION_PROMPT"
  --num_validation_images "$VALIDATION_NUM_IMAGES"
  --validation_epochs "$VALIDATION_EVERY_N_EPOCHS"
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
  --report_to "$REPORT_TO"
)

add_bool_flag "$TRAIN_TEXT_ENCODER" --train_text_encoder
add_bool_flag "$GRADIENT_CHECKPOINTING" --gradient_checkpointing
add_bool_flag "$USE_8BIT_ADAM" --use_8bit_adam
add_bool_flag "$ENABLE_XFORMERS" --enable_xformers_memory_efficient_attention
add_bool_flag "$RANDOM_FLIP" --random_flip
add_bool_flag "$CENTER_CROP" --center_crop

if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  CMD+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi

printf 'Resolved training command:\n'
printf '  %q' "${CMD[@]}"
printf '\n'

if [[ "$DRY_RUN" == "true" ]]; then
  echo "Dry run complete."
  exit 0
fi

mkdir -p "$OUTPUT_DIR"
cp "$CONFIG_PATH" "$OUTPUT_DIR/config.snapshot.yaml"

printf 'Launching training run in %s\n' "$OUTPUT_DIR"
exec "${CMD[@]}"
