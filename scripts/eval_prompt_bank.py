#!/usr/bin/env python3
"""Generate fixed prompt-bank evaluation images for base SDXL or a Frieren LoRA."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional

import torch
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model with the fixed Frieren prompt bank.")
    parser.add_argument("--config", required=True, help="Training config used to resolve base model and seed.")
    parser.add_argument("--prompt-bank", required=True, help="YAML prompt bank.")
    parser.add_argument("--lora-dir", default=None, help="Directory containing LoRA weights. Omit for base SDXL.")
    parser.add_argument("--model-label", default=None, help="Output label. Defaults to base_sdxl or LoRA directory name.")
    parser.add_argument("--output-root", default="outputs/experiments/frieren_eval_v1")
    parser.add_argument("--device", default=None, help="Torch device, for example cuda, cuda:0, or cpu.")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--seed-base", type=int, default=None, help="Override prompt-bank/config seed.")
    parser.add_argument("--num-images-per-prompt", type=int, default=None)
    parser.add_argument("--lora-scales", type=float, nargs="+", default=None)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_repo_path(repo_root: Path, path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def resolve_device(requested_device: Optional[str]) -> str:
    if requested_device:
        return requested_device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(device: str, dtype_name: str) -> torch.dtype:
    if dtype_name == "fp16":
        return torch.float32 if device == "cpu" else torch.float16
    if dtype_name == "bf16":
        return torch.float32 if device == "cpu" else torch.bfloat16
    return torch.float32


def make_generator(device: str, seed: int) -> torch.Generator:
    generator_device = "cpu" if device in {"cpu", "mps"} else device
    return torch.Generator(device=generator_device).manual_seed(seed)


def safe_label(value: str) -> str:
    label = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    label = re.sub(r"_+", "_", label).strip("_")
    return label or "model"


def scale_label(scale: Optional[float]) -> str:
    if scale is None:
        return "base"
    return f"scale_{scale:.2f}".replace(".", "p")


def ensure_lora_weights_exist(lora_dir: Path) -> None:
    candidates = [
        lora_dir / "pytorch_lora_weights.safetensors",
        lora_dir / "pytorch_lora_weights.bin",
    ]
    if not any(candidate.exists() for candidate in candidates):
        raise FileNotFoundError(
            f"Could not find LoRA weights in {lora_dir}. "
            "Expected pytorch_lora_weights.safetensors or pytorch_lora_weights.bin."
        )


def load_pipeline(base_model: str, vae_model: Optional[str], device: str, dtype: torch.dtype) -> Any:
    try:
        from diffusers import AutoencoderKL, StableDiffusionXLPipeline
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("diffusers is required for prompt-bank evaluation.") from exc

    vae = None
    if vae_model:
        vae = AutoencoderKL.from_pretrained(vae_model, torch_dtype=dtype)

    pipe = StableDiffusionXLPipeline.from_pretrained(base_model, vae=vae, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    return pipe


def render_image(
    pipe: Any,
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    device: str,
    lora_scale: Optional[float],
) -> Any:
    kwargs: dict[str, Any] = {}
    if lora_scale is not None:
        kwargs["cross_attention_kwargs"] = {"scale": lora_scale}

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=make_generator(device, seed),
        **kwargs,
    )
    return result.images[0]


def resolve_lora_scales(args: argparse.Namespace, prompt_bank: dict[str, Any], has_lora: bool) -> list[Optional[float]]:
    if not has_lora:
        return [None]
    if args.lora_scales:
        return args.lora_scales
    return [float(value) for value in prompt_bank.get("lora_scales", [1.0])]


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    config_path = resolve_repo_path(repo_root, args.config)
    prompt_bank_path = resolve_repo_path(repo_root, args.prompt_bank)
    output_root = resolve_repo_path(repo_root, args.output_root)

    config = load_yaml(config_path)
    prompt_bank = load_yaml(prompt_bank_path)

    lora_dir = resolve_repo_path(repo_root, args.lora_dir) if args.lora_dir else None
    has_lora = lora_dir is not None
    if lora_dir:
        ensure_lora_weights_exist(lora_dir)

    model_label = args.model_label
    if not model_label:
        model_label = lora_dir.name if lora_dir else "base_sdxl"
    model_label = safe_label(model_label)

    device = resolve_device(args.device)
    dtype = resolve_dtype(device, args.dtype)
    seed_base = args.seed_base
    if seed_base is None:
        seed_base = int(prompt_bank.get("seed_base", config["project"]["seed"]))

    generation = prompt_bank["generation"]
    height = int(generation["height"])
    width = int(generation["width"])
    num_inference_steps = int(generation["num_inference_steps"])
    guidance_scale = float(generation["guidance_scale"])
    negative_prompt = generation["negative_prompt"]
    prompts = prompt_bank["prompts"]
    num_images = args.num_images_per_prompt or int(prompt_bank.get("num_images_per_prompt", 1))
    lora_scales = resolve_lora_scales(args, prompt_bank, has_lora)

    print(f"Loading pipeline on {device} with dtype {dtype}...")
    model_cfg = config["model"]
    pipe = load_pipeline(
        base_model=model_cfg["base_model"],
        vae_model=model_cfg.get("vae_model"),
        device=device,
        dtype=dtype,
    )

    if lora_dir:
        print(f"Loading LoRA weights from {lora_dir}...")
        pipe.load_lora_weights(str(lora_dir))

    records: list[dict[str, Any]] = []
    model_output_root = output_root / model_label
    model_output_root.mkdir(parents=True, exist_ok=True)

    for current_scale in lora_scales:
        current_scale_label = scale_label(current_scale)
        scale_output_root = model_output_root / current_scale_label
        scale_output_root.mkdir(parents=True, exist_ok=True)

        for prompt_index, prompt_record in enumerate(prompts):
            prompt_id = safe_label(prompt_record["id"])
            prompt_group = safe_label(prompt_record.get("group", "ungrouped"))
            prompt = prompt_record["prompt"]
            prompt_output_dir = scale_output_root / prompt_group
            prompt_output_dir.mkdir(parents=True, exist_ok=True)

            for sample_index in range(num_images):
                seed = seed_base + prompt_index * 1000 + sample_index
                image = render_image(
                    pipe=pipe,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    device=device,
                    lora_scale=current_scale,
                )
                output_name = f"{prompt_id}_sample{sample_index:02d}_seed{seed}.png"
                output_path = prompt_output_dir / output_name
                image.save(output_path)
                records.append(
                    {
                        "model_label": model_label,
                        "lora_dir": str(lora_dir) if lora_dir else None,
                        "lora_scale": current_scale,
                        "prompt_id": prompt_id,
                        "prompt_group": prompt_group,
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "sample_index": sample_index,
                        "seed": seed,
                        "image_path": str(output_path.relative_to(repo_root)),
                    }
                )

    manifest_path = model_output_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} images under {model_output_root}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

