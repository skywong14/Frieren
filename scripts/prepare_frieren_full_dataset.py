#!/usr/bin/env python3
"""Build the merged Frieren single-character dataset for the full LoRA experiment.

This script merges the existing 30 PNG screenshots under `data/raw/Frieren/Single`
with the newer 50 JPG screenshots under `data/raw/Frieren/Single2`, normalizes them to
RGB PNG, and writes a Diffusers-compatible `ImageFolder + metadata.jsonl` dataset.

Design goals:
- deterministic file naming and ordering
- preserve the manually refined captions already written for the first 30 images
- generate reproducible first-pass captions for the new JPG images
- keep review artifacts and manifests outside the training dataset root
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from PIL import Image, ImageStat

BASE_IDENTITY = "sks_frieren, elf girl, long silver hair, pointy ears, white and gold robe"
DEFAULT_SOURCES = [
    "data/raw/Frieren/Single",
    "data/raw/Frieren/Single2",
]
DEFAULT_DATASET_ROOT = "data/datasets/frieren_hd_single80_v1"
DEFAULT_ANNOTATIONS_ROOT = "data/annotations/frieren_hd_single80_v1"
LEGACY_CAPTIONS_DIR = "data/annotations/frieren_hd_v1/captions/single"
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass(frozen=True)
class Sample:
    source_group: str
    source_path: Path
    output_name: str
    caption: str
    caption_mode: str
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the merged Frieren single-image training dataset.")
    parser.add_argument(
        "--source-dir",
        action="append",
        default=None,
        help="Source directory to include. Can be passed multiple times. Defaults to Single + Single2.",
    )
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_DATASET_ROOT,
        help="Output dataset root. metadata.jsonl and images/ will be written here.",
    )
    parser.add_argument(
        "--annotations-root",
        default=DEFAULT_ANNOTATIONS_ROOT,
        help="Where to write captions and manifest files.",
    )
    parser.add_argument(
        "--legacy-captions-dir",
        default=LEGACY_CAPTIONS_DIR,
        help="Directory containing the refined captions for the original 30 Single images.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the existing dataset/annotation outputs before rebuilding them.",
    )
    return parser.parse_args()


def resolve_repo_path(repo_root: Path, path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def list_images(source_dir: Path) -> List[Path]:
    return sorted([path for path in source_dir.iterdir() if path.suffix.lower() in VALID_EXTENSIONS])


def load_legacy_captions(legacy_dir: Path) -> List[str]:
    if not legacy_dir.exists():
        return []
    captions: List[str] = []
    for path in sorted(legacy_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            captions.append(text)
    return captions


def composition_tag(width: int, height: int) -> str:
    if width / height > 1.25:
        return "landscape composition"
    if height / width > 1.25:
        return "portrait composition"
    return "balanced composition"


def lighting_tags(image: Image.Image) -> List[str]:
    sample = image.convert("RGB").resize((64, 64))
    red, green, blue = ImageStat.Stat(sample).mean
    brightness = (red + green + blue) / 3.0

    if brightness < 78:
        brightness_tag = "low light"
    elif brightness > 190:
        brightness_tag = "bright lighting"
    else:
        brightness_tag = "soft lighting"

    if red > blue * 1.12 and red > green * 1.05:
        color_tag = "warm lighting"
    elif blue > red * 1.12:
        color_tag = "cool lighting"
    else:
        color_tag = "neutral lighting"

    return [color_tag, brightness_tag]


def build_auto_caption(image: Image.Image, width: int, height: int) -> str:
    tags = [
        BASE_IDENTITY,
        "solo character",
        "anime screenshot",
        composition_tag(width, height),
        *lighting_tags(image),
    ]
    return ", ".join(tags)


def collect_samples(
    repo_root: Path,
    source_dirs: Iterable[Path],
    dataset_root: Path,
    annotations_root: Path,
    legacy_captions: List[str],
) -> List[Sample]:
    samples: List[Sample] = []
    images_dir = dataset_root / "images"
    captions_dir = annotations_root / "captions"
    images_dir.mkdir(parents=True, exist_ok=True)
    captions_dir.mkdir(parents=True, exist_ok=True)

    global_index = 1
    legacy_index = 0

    for source_dir in source_dirs:
        group_name = source_dir.name.lower()
        files = list_images(source_dir)
        for path in files:
            output_name = f"frieren_hd_full_{global_index:04d}.png"
            output_path = images_dir / output_name
            caption_path = captions_dir / f"frieren_hd_full_{global_index:04d}.txt"

            with Image.open(path) as image:
                width, height = image.size
                rgb_image = image.convert("RGB")
                rgb_image.save(output_path, format="PNG", optimize=True)
                if group_name == "single" and legacy_index < len(legacy_captions):
                    caption = legacy_captions[legacy_index]
                    caption_mode = "legacy_manual"
                    legacy_index += 1
                else:
                    caption = build_auto_caption(rgb_image, width, height)
                    caption_mode = "heuristic_auto"

            caption_path.write_text(caption + "\n", encoding="utf-8")
            samples.append(
                Sample(
                    source_group=group_name,
                    source_path=path.resolve(),
                    output_name=output_name,
                    caption=caption,
                    caption_mode=caption_mode,
                    width=width,
                    height=height,
                )
            )
            global_index += 1

    return samples


def write_metadata(dataset_root: Path, samples: Iterable[Sample]) -> None:
    metadata_path = dataset_root / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(
                json.dumps(
                    {
                        "file_name": f"images/{sample.output_name}",
                        "text": sample.caption,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_manifest(repo_root: Path, annotations_root: Path, dataset_root: Path, samples: Iterable[Sample]) -> None:
    manifest_path = annotations_root / "dataset_manifest.tsv"
    rows = []
    for index, sample in enumerate(samples, start=1):
        rows.append(
            {
                "index": index,
                "output_name": sample.output_name,
                "source_group": sample.source_group,
                "source_path": str(sample.source_path.relative_to(repo_root)),
                "dataset_root": str(dataset_root.relative_to(repo_root)),
                "caption_mode": sample.caption_mode,
                "width": sample.width,
                "height": sample.height,
                "caption": sample.caption,
            }
        )

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_review_notes(repo_root: Path, annotations_root: Path, dataset_root: Path, samples: List[Sample]) -> None:
    by_group = {}
    by_caption_mode = {}
    for sample in samples:
        by_group[sample.source_group] = by_group.get(sample.source_group, 0) + 1
        by_caption_mode[sample.caption_mode] = by_caption_mode.get(sample.caption_mode, 0) + 1

    review_path = annotations_root / "review_notes.md"
    lines = [
        "# Frieren HD Single Full Dataset Review",
        "",
        f"- Total merged images: {len(samples)}",
        f"- Dataset root: `{dataset_root.relative_to(repo_root)}`",
        f"- Source breakdown: {json.dumps(by_group, ensure_ascii=False)}",
        f"- Caption breakdown: {json.dumps(by_caption_mode, ensure_ascii=False)}",
        "- Intended use: main single-character training set for the full LoRA experiment",
        "",
        "Notes:",
        "- Images are normalized to RGB PNG with deterministic names.",
        "- The original 30 Single captions are reused from the manually refined set.",
        "- The new 50 JPG images use reproducible heuristic captions and can be refined later without changing the file naming scheme.",
        "- Diffusers compatibility requirement: keep only `images/` and `metadata.jsonl` under the dataset root.",
        "",
    ]
    review_path.write_text("\n".join(lines), encoding="utf-8")


def clear_output(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    source_dir_values = args.source_dir or DEFAULT_SOURCES
    source_dirs = [resolve_repo_path(repo_root, value) for value in source_dir_values]
    dataset_root = resolve_repo_path(repo_root, args.dataset_root)
    annotations_root = resolve_repo_path(repo_root, args.annotations_root)
    legacy_captions_dir = resolve_repo_path(repo_root, args.legacy_captions_dir)

    for source_dir in source_dirs:
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")

    if args.overwrite:
        clear_output(dataset_root)
        clear_output(annotations_root)

    dataset_root.mkdir(parents=True, exist_ok=True)
    annotations_root.mkdir(parents=True, exist_ok=True)

    legacy_captions = load_legacy_captions(legacy_captions_dir)
    samples = collect_samples(repo_root, source_dirs, dataset_root, annotations_root, legacy_captions)
    if not samples:
        raise RuntimeError("No images were collected from the requested source directories.")

    write_metadata(dataset_root, samples)
    write_manifest(repo_root, annotations_root, dataset_root, samples)
    write_review_notes(repo_root, annotations_root, dataset_root, samples)

    print(f"Built merged dataset with {len(samples)} images.")
    print(f"Dataset root: {dataset_root}")
    print(f"Annotations: {annotations_root}")


if __name__ == "__main__":
    main()
