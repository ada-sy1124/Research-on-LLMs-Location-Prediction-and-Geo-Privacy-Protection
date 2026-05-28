#!/usr/bin/env python3
"""One-off patch helper for data/YES_NEW_afterSAM.

This script lives outside code/ on purpose. It is not imported by the project.

Goal:
  Add stable sample_id/source_index metadata to an existing YES_NEW_afterSAM
  dataset that was produced before those columns existed.

Important:
  Hugging Face datasets are Arrow-backed and immutable in practice. Adding
  physical columns requires writing a new dataset directory. For the current
  19GB dataset, this needs enough free disk space for a second copy.

Modes:
  1. Full patch:
     python patch_yes_new_aftersam_ids.py --input data/YES_NEW_afterSAM --output data/YES_NEW_afterSAM_with_ids

  2. Sidecar only, explicit metadata report only:
     python patch_yes_new_aftersam_ids.py --input data/YES_NEW_afterSAM --sidecar data/YES_NEW_afterSAM_id_sidecar.jsonl --sidecar-only

The sidecar JSONL does NOT modify the dataset and is NOT consumed by the
main pipeline. It is only a row-to-id report.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/YES_NEW_afterSAM")
    parser.add_argument("--output", default="data/YES_NEW_afterSAM_with_ids")
    parser.add_argument("--sidecar", default="data/YES_NEW_afterSAM_id_sidecar.jsonl")
    parser.add_argument("--id-prefix", default="after_sam")
    parser.add_argument("--source-index-start", type=int, default=0)
    parser.add_argument("--sidecar-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_dataset_info(dataset_path: Path) -> dict:
    info_path = dataset_path / "dataset_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing dataset_info.json: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def get_num_rows(dataset_path: Path) -> int:
    info = load_dataset_info(dataset_path)
    splits = info.get("splits") or {}
    train = splits.get("train") or {}
    num_examples = train.get("num_examples")
    if not isinstance(num_examples, int):
        raise ValueError(f"Could not read train.num_examples from {dataset_path / 'dataset_info.json'}")
    return num_examples


def get_dataset_size_bytes(dataset_path: Path) -> int:
    info = load_dataset_info(dataset_path)
    size = info.get("size_in_bytes") or info.get("dataset_size")
    if isinstance(size, int):
        return size
    return sum(path.stat().st_size for path in dataset_path.glob("*.arrow"))


def write_sidecar(sidecar_path: Path, row_count: int, id_prefix: str, source_index_start: int) -> None:
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    with sidecar_path.open("w", encoding="utf-8") as f:
        for row_index in range(row_count):
            source_index = source_index_start + row_index
            row = {
                "row_index": row_index,
                "sample_id": f"{id_prefix}_{source_index:06d}",
                "source_index": source_index,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_enough_space(input_path: Path, output_path: Path) -> None:
    required = get_dataset_size_bytes(input_path)
    free = shutil.disk_usage(output_path.parent).free
    # Leave a small margin for Arrow metadata and filesystem overhead.
    margin = 512 * 1024 * 1024
    if free < required + margin:
        raise RuntimeError(
            "Not enough free disk space for a full physical patch.\n"
            f"Dataset size: {required / (1024 ** 3):.2f} GiB\n"
            f"Free space:   {free / (1024 ** 3):.2f} GiB\n"
            "Free enough space and rerun the full patch. "
            "--sidecar-only only writes a report and does not patch the dataset."
        )


def full_patch(input_path: Path, output_path: Path, id_prefix: str, source_index_start: int, overwrite: bool) -> None:
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")
        shutil.rmtree(output_path)

    ensure_enough_space(input_path, output_path)

    from datasets import Value, load_from_disk

    dataset = load_from_disk(str(input_path))
    row_count = len(dataset)
    sample_ids = [f"{id_prefix}_{source_index_start + i:06d}" for i in range(row_count)]
    source_indices = [source_index_start + i for i in range(row_count)]

    if "sample_id" in dataset.column_names:
        dataset = dataset.remove_columns(["sample_id"])
    if "source_index" in dataset.column_names:
        dataset = dataset.remove_columns(["source_index"])

    dataset = dataset.add_column("sample_id", sample_ids).cast_column("sample_id", Value("string"))
    dataset = dataset.add_column("source_index", source_indices).cast_column("source_index", Value("int64"))
    dataset.save_to_disk(str(output_path))


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    sidecar_path = Path(args.sidecar)

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    if args.sidecar_only:
        row_count = get_num_rows(input_path)
        write_sidecar(sidecar_path, row_count, args.id_prefix, args.source_index_start)
        print(f"Wrote sidecar report: {sidecar_path}")
        print(f"Rows: {row_count}")
        print("Sidecar-only mode finished. The Arrow dataset itself was not rewritten.")
        return

    full_patch(input_path, output_path, args.id_prefix, args.source_index_start, args.overwrite)
    print(f"Wrote full patched dataset: {output_path}")


if __name__ == "__main__":
    main()
