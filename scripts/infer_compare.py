#!/usr/bin/env python3
"""Compare base SDXL outputs against a trained Frieren LoRA on a fixed prompt set."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate side-by-side comparisons for base SDXL and a trained LoRA.")
    parser.add_argument("--config", default="configs/train_sdxl_lora.yaml", help="Path to the project YAML config.")
    parser.add_argument(
        "--lora-dir",
        default=None,
        help="Directory containing the final LoRA weights. Defaults to outputs/train/<run_name>/.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory used for generated images. Defaults to outputs/infer/<run_name>/.",
    )
    parser.add_argument("--device", default=None, help="Torch device, for example cuda, cuda:0, or cpu.")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"], help="Inference dtype.")
    parser.add_argument("--seed", type=int, default=None, help="Base seed for reproducible generations.")
    parser.add_argument(
        "--num-images-per-prompt",
        type=int,
        default=1,
        help="How many images to sample for each prompt.",
    )
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Skip baseline generation and only render the LoRA outputs.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_repo_path(repo_root: Path, path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def slugify_prompt(prompt: str, index: int) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", prompt.lower()).strip("_")
    slug = re.sub(r"_+", "_", slug)
    if len(slug) > 72:
        slug = slug[:72].rstrip("_")
    return f"{index:02d}_{slug or 'prompt'}"


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
        if device == "cpu":
            print("fp16 is not supported on CPU, falling back to fp32.")
            return torch.float32
        return torch.float16
    if dtype_name == "bf16":
        if device == "cpu":
            print("bf16 is not supported on CPU for this script, falling back to fp32.")
            return torch.float32
        return torch.bfloat16
    return torch.float32


def make_generator(device: str, seed: int) -> torch.Generator:
    generator_device = "cpu" if device in {"cpu", "mps"} else device
    return torch.Generator(device=generator_device).manual_seed(seed)


def load_pipeline(base_model: str, vae_model: Optional[str], device: str, dtype: torch.dtype) -> Any:
    try:
        from diffusers import AutoencoderKL, StableDiffusionXLPipeline
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "diffusers is required for inference. Install it before running infer_compare.py."
        ) from exc

    vae = None
    if vae_model:
        vae = AutoencoderKL.from_pretrained(vae_model, torch_dtype=dtype)

    pipe = StableDiffusionXLPipeline.from_pretrained(base_model, vae=vae, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    return pipe


def ensure_lora_weights_exist(lora_dir: Path) -> None:
    candidates = [
        lora_dir / "pytorch_lora_weights.safetensors",
        lora_dir / "pytorch_lora_weights.bin",
    ]
    if not any(candidate.exists() for candidate in candidates):
        raise FileNotFoundError(
            "Could not find final LoRA weights in "
            f"{lora_dir}. Expected pytorch_lora_weights.safetensors or pytorch_lora_weights.bin."
        )


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
) -> Image.Image:
    generator = make_generator(device, seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    return result.images[0]


def save_labeled_grid(base_image: Image.Image, lora_image: Image.Image, output_path: Path) -> None:
    base_rgb = base_image.convert("RGB")
    lora_rgb = lora_image.convert("RGB")
    label_height = 32
    spacing = 12
    width, height = base_rgb.size
    grid = Image.new("RGB", (width * 2 + spacing, height + label_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default()

    draw.text((12, 8), "base", fill=(0, 0, 0), font=font)
    draw.text((width + spacing + 12, 8), "lora", fill=(0, 0, 0), font=font)
    grid.paste(base_rgb, (0, label_height))
    grid.paste(lora_rgb, (width + spacing, label_height))
    grid.save(output_path)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    config_path = resolve_repo_path(repo_root, args.config)
    config = load_config(config_path)

    run_name = config["project"]["run_name"]
    output_root = resolve_repo_path(repo_root, config["project"]["output_root"])
    default_lora_dir = output_root / run_name
    default_output_dir = repo_root / "outputs" / "infer" / run_name

    lora_dir = resolve_repo_path(repo_root, args.lora_dir) if args.lora_dir else default_lora_dir
    output_dir = resolve_repo_path(repo_root, args.output_dir) if args.output_dir else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    base_dir = output_dir / "base"
    lora_output_dir = output_dir / "lora"
    compare_dir = output_dir / "comparison"
    if not args.skip_base:
        base_dir.mkdir(parents=True, exist_ok=True)
        compare_dir.mkdir(parents=True, exist_ok=True)
    lora_output_dir.mkdir(parents=True, exist_ok=True)

    if not lora_dir.exists() or not lora_dir.is_dir():
        raise FileNotFoundError(f"LoRA directory does not exist: {lora_dir}")
    ensure_lora_weights_exist(lora_dir)

    device = resolve_device(args.device)
    dtype = resolve_dtype(device, args.dtype)
    seed = args.seed if args.seed is not None else int(config["project"]["seed"])

    model_cfg = config["model"]
    infer_cfg = config["inference"]
    prompts = infer_cfg["prompts"]
    negative_prompt = infer_cfg["negative_prompt"]
    num_inference_steps = int(infer_cfg["num_inference_steps"])
    guidance_scale = float(infer_cfg["guidance_scale"])
    height = int(infer_cfg["height"])
    width = int(infer_cfg["width"])
    lora_scale = float(infer_cfg["lora_scale"])

    pipe = load_pipeline(
        base_model=model_cfg["base_model"],
        vae_model=model_cfg.get("vae_model"),
        device=device,
        dtype=dtype,
    )

    base_paths: dict[tuple[int, int], Path] = {}
    if not args.skip_base:
        print(f"Generating base model outputs on {device} with dtype {dtype}...")
        for prompt_index, prompt in enumerate(prompts):
            prompt_slug = slugify_prompt(prompt, prompt_index)
            for sample_index in range(args.num_images_per_prompt):
                image_seed = seed + prompt_index * 1000 + sample_index
                image = render_image(
                    pipe=pipe,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=image_seed,
                    device=device,
                )
                image_path = base_dir / f"{prompt_slug}_sample{sample_index:02d}.png"
                image.save(image_path)
                base_paths[(prompt_index, sample_index)] = image_path

    print("Loading LoRA weights...")
    pipe.load_lora_weights(str(lora_dir))
    pipe.fuse_lora(lora_scale=lora_scale)

    print("Generating LoRA outputs...")
    for prompt_index, prompt in enumerate(prompts):
        prompt_slug = slugify_prompt(prompt, prompt_index)
        for sample_index in range(args.num_images_per_prompt):
            image_seed = seed + prompt_index * 1000 + sample_index
            lora_image = render_image(
                pipe=pipe,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=image_seed,
                device=device,
            )
            lora_image_path = lora_output_dir / f"{prompt_slug}_sample{sample_index:02d}.png"
            lora_image.save(lora_image_path)

            if not args.skip_base:
                base_image = Image.open(base_paths[(prompt_index, sample_index)])
                grid_path = compare_dir / f"{prompt_slug}_sample{sample_index:02d}_grid.png"
                save_labeled_grid(base_image=base_image, lora_image=lora_image, output_path=grid_path)
                base_image.close()

    print(f"Inference artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
