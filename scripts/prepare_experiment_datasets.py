#!/usr/bin/env python3
"""Prepare derived datasets used by the final Frieren LoRA experiments."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable

SIMPLE_CAPTION = "sks_frieren, elf girl, long silver hair, pointy ears, white and gold robe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build L100 and L80-simple experiment datasets.")
    parser.add_argument("--single80-root", default="data/datasets/frieren_hd_single80_v1")
    parser.add_argument("--multi20-root", default="data/datasets/frieren_hd_multi_candidates_v1")
    parser.add_argument("--l100-root", default="data/datasets/frieren_hd_all100_v1")
    parser.add_argument("--l80-simple-root", default="data/datasets/frieren_hd_single80_simple_caption_v1")
    parser.add_argument("--annotations-root", default="data/annotations/experiments")
    parser.add_argument("--overwrite", action="store_true", help="Delete derived outputs before rebuilding.")
    return parser.parse_args()


def resolve_repo_path(repo_root: Path, path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def read_jsonl(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            record = json.loads(text)
            if "file_name" not in record or "text" not in record:
                raise ValueError(f"Malformed metadata line {line_number} in {path}")
            records.append(record)
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def reset_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_record_image(source_root: Path, dest_root: Path, source_file_name: str, dest_file_name: str) -> None:
    source_path = source_root / source_file_name
    dest_path = dest_root / dest_file_name
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source image referenced by metadata: {source_path}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, dest_path)


def build_l80_simple(single80_root: Path, output_root: Path, overwrite: bool) -> list[dict[str, str]]:
    reset_dir(output_root, overwrite)
    records = read_jsonl(single80_root / "metadata.jsonl")
    derived_records: list[dict[str, str]] = []

    for record in records:
        file_name = record["file_name"]
        copy_record_image(single80_root, output_root, file_name, file_name)
        derived_records.append({"file_name": file_name, "text": SIMPLE_CAPTION})

    write_jsonl(output_root / "metadata.jsonl", derived_records)
    return derived_records


def prefixed_file_name(prefix: str, file_name: str) -> str:
    path = Path(file_name)
    return str(path.parent / f"{prefix}_{path.name}").replace("\\", "/")


def build_l100(single80_root: Path, multi20_root: Path, output_root: Path, overwrite: bool) -> list[dict[str, str]]:
    reset_dir(output_root, overwrite)
    combined_records: list[dict[str, str]] = []

    for prefix, source_root in (("single80", single80_root), ("multi20", multi20_root)):
        for record in read_jsonl(source_root / "metadata.jsonl"):
            source_file_name = record["file_name"]
            dest_file_name = prefixed_file_name(prefix, source_file_name)
            copy_record_image(source_root, output_root, source_file_name, dest_file_name)
            combined_records.append({"file_name": dest_file_name, "text": record["text"]})

    write_jsonl(output_root / "metadata.jsonl", combined_records)
    return combined_records


def write_review(
    repo_root: Path,
    annotations_root: Path,
    dataset_root: Path,
    records: list[dict[str, str]],
    title: str,
    notes: list[str],
) -> None:
    annotations_root.mkdir(parents=True, exist_ok=True)
    review_path = annotations_root / "review_notes.md"
    lines = [
        f"# {title}",
        "",
        f"- Dataset root: `{dataset_root.relative_to(repo_root)}`",
        f"- Total images: {len(records)}",
        f"- Metadata: `{(dataset_root / 'metadata.jsonl').relative_to(repo_root)}`",
        "",
        "Notes:",
    ]
    lines.extend([f"- {note}" for note in notes])
    review_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    single80_root = resolve_repo_path(repo_root, args.single80_root)
    multi20_root = resolve_repo_path(repo_root, args.multi20_root)
    l100_root = resolve_repo_path(repo_root, args.l100_root)
    l80_simple_root = resolve_repo_path(repo_root, args.l80_simple_root)
    annotations_root = resolve_repo_path(repo_root, args.annotations_root)

    for required_root in (single80_root, multi20_root):
        if not (required_root / "metadata.jsonl").exists():
            raise FileNotFoundError(f"Missing source metadata: {required_root / 'metadata.jsonl'}")

    l80_simple_records = build_l80_simple(single80_root, l80_simple_root, args.overwrite)
    l100_records = build_l100(single80_root, multi20_root, l100_root, args.overwrite)

    write_review(
        repo_root=repo_root,
        annotations_root=annotations_root / "frieren_hd_single80_simple_caption_v1",
        dataset_root=l80_simple_root,
        records=l80_simple_records,
        title="Frieren L80 Simple Caption Dataset",
        notes=[
            "Images are copied from the existing 80-image single-character dataset.",
            f"Every sample uses the same simple caption: `{SIMPLE_CAPTION}`.",
            "This dataset is used only for the caption-organization comparison.",
        ],
    )
    write_review(
        repo_root=repo_root,
        annotations_root=annotations_root / "frieren_hd_all100_v1",
        dataset_root=l100_root,
        records=l100_records,
        title="Frieren L100 Structured Dataset",
        notes=[
            "The dataset combines the 80-image single-character set with the 20-image multi/complex candidate set.",
            "Source file names are prefixed with `single80_` or `multi20_` to keep provenance visible.",
            "Captions are inherited from the source datasets.",
        ],
    )

    print(f"Built L80-simple dataset with {len(l80_simple_records)} images: {l80_simple_root}")
    print(f"Built L100 dataset with {len(l100_records)} images: {l100_root}")


if __name__ == "__main__":
    main()

