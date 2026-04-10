#!/usr/bin/env python3
"""Build an ImageFolder-compatible metadata.jsonl file for SDXL LoRA training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_PREPEND_TEXT = "sks_frieren"
DEFAULT_CAPTION = "elf girl, long silver hair, pointy ears, white and gold robe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan an image folder and write metadata.jsonl for Hugging Face ImageFolder datasets."
    )
    parser.add_argument(
        "--images-dir",
        default="data/processed/images",
        help="Directory containing processed training images.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/metadata.jsonl",
        help="Path to the metadata.jsonl file to write.",
    )
    parser.add_argument(
        "--default-caption",
        default=DEFAULT_CAPTION,
        help="Stable identity caption appended to every image.",
    )
    parser.add_argument(
        "--prepend-text",
        default=DEFAULT_PREPEND_TEXT,
        help="Text placed at the start of every caption, typically the trigger token.",
    )
    parser.add_argument(
        "--caption-ext",
        default=".txt",
        help="Sidecar caption file extension used to read per-image dynamic captions.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def normalize_caption(parts: Iterable[str]) -> str:
    """Join caption fragments while removing empty tokens and duplicate commas."""
    tokens: List[str] = []
    for part in parts:
        if not part:
            continue
        for token in part.split(","):
            cleaned = token.strip()
            if cleaned:
                tokens.append(cleaned)
    return ", ".join(tokens)


def iter_image_paths(images_dir: Path) -> list[Path]:
    image_paths = [path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(image_paths, key=lambda path: path.relative_to(images_dir).as_posix().lower())


def resolve_caption_path(image_path: Path, caption_ext: str) -> Path:
    if not caption_ext.startswith("."):
        caption_ext = f".{caption_ext}"
    return image_path.with_suffix(caption_ext)


def read_sidecar_caption(caption_path: Path) -> str:
    if not caption_path.exists():
        return ""
    return caption_path.read_text(encoding="utf-8").strip()


def build_records(images_dir: Path, output_path: Path, default_caption: str, prepend_text: str, caption_ext: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for image_path in iter_image_paths(images_dir):
        caption_path = resolve_caption_path(image_path, caption_ext)
        sidecar_text = read_sidecar_caption(caption_path)
        text = normalize_caption([prepend_text, default_caption, sidecar_text])

        # ImageFolder expects file_name to be relative to the dataset root, which is
        # usually the parent directory of metadata.jsonl.
        file_name = os.path.relpath(image_path, output_path.parent).replace(os.sep, "/")
        records.append({"file_name": file_name, "text": text})
    return records


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")
    if not images_dir.is_dir():
        raise NotADirectoryError(f"Images path is not a directory: {images_dir}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists, pass --overwrite to replace it: {output_path}")

    records = build_records(
        images_dir=images_dir,
        output_path=output_path,
        default_caption=args.default_caption,
        prepend_text=args.prepend_text,
        caption_ext=args.caption_ext,
    )
    if not records:
        raise RuntimeError(f"No supported image files were found under: {images_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
