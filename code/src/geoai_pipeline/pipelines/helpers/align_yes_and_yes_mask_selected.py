from collections import defaultdict, deque
from json import JSONDecodeError
import os
from pathlib import Path
import shutil

import datasets
from datasets import Dataset, DatasetInfo, Features, Image, Value, concatenate_datasets, load_from_disk


YES_INPUT_PATH = "./data/YES"
YES_MASK_SELECTED_INPUT_PATH = "./data/YES_Mask_selected"
YES_OUTPUT_PATH = "./data/YES_aligned_with_id"
YES_MASK_SELECTED_OUTPUT_PATH = "./data/YES_Mask_selected_aligned_with_id"

ALIGNMENT_ID_COLUMN = "alignment_id"


def get_path(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        value = default
    return str(Path(value).expanduser())


def get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def prepare_output_path(output_dataset_path: str, overwrite: bool) -> None:
    output_path = Path(output_dataset_path).expanduser()
    if not output_path.exists():
        return
    if not overwrite:
        raise FileExistsError(
            f"Output path already exists: {output_dataset_path}. "
            "Set ALIGN_DATASETS_OVERWRITE=1 to overwrite it."
        )

    resolved = output_path.resolve()
    if resolved.parent == resolved or resolved.name in {"", ".", ".."}:
        raise ValueError(f"Refusing to overwrite unsafe output path: {output_dataset_path}")

    if output_path.is_dir():
        shutil.rmtree(output_path)
    else:
        output_path.unlink()


def build_yes_mask_selected_features() -> Features:
    return Features(
        {
            "image": Image(),
            "latitude_true": Value("float64"),
            "longitude_true": Value("float64"),
            "d_original": Value("float64"),
            "ablated_class": datasets.Sequence(Value("string")),
            "q_ratio": datasets.Sequence(Value("float64")),
            "d_prime": datasets.Sequence(Value("float64")),
            "d_diff": datasets.Sequence(Value("float64")),
            "sample_id": Value("string"),
            "target_mask_category": Value("string"),
            "label": Value("string"),
            "target_mask_category_index": Value("int64"),
            "final_utility": Value("float64"),
            "final_privacy": Value("float64"),
            "selection_rule": Value("string"),
        }
    )


def load_yes_mask_selected(dataset_path: str) -> Dataset:
    try:
        return load_from_disk(dataset_path)
    except JSONDecodeError:
        root = Path(dataset_path)
        arrow_files = sorted(root.glob("data-*.arrow"))
        if not arrow_files:
            raise FileNotFoundError(f"No data-*.arrow files found in {dataset_path}")

        info = DatasetInfo(features=build_yes_mask_selected_features())
        datasets_from_files = [Dataset.from_file(str(path), info=info) for path in arrow_files]
        if len(datasets_from_files) == 1:
            return datasets_from_files[0]
        return concatenate_datasets(datasets_from_files)


def normalize_classes(classes) -> tuple[str, ...]:
    if classes is None:
        return ()
    return tuple(str(value) for value in classes)


def yes_key(sample: dict) -> tuple:
    return (
        float(sample["latitude"]),
        float(sample["longitude"]),
        float(sample["d"]),
        normalize_classes(sample["reason_class"]),
    )


def yes_mask_selected_key(sample: dict) -> tuple:
    return (
        float(sample["latitude_true"]),
        float(sample["longitude_true"]),
        float(sample["d_original"]),
        normalize_classes(sample["ablated_class"]),
    )


def validate_columns(dataset: Dataset, required_columns: set[str], dataset_name: str) -> None:
    missing_columns = sorted(required_columns - set(dataset.column_names))
    if missing_columns:
        raise KeyError(f"{dataset_name} is missing required columns: {missing_columns}")


def build_matched_indices(yes_dataset: Dataset, yes_mask_selected_dataset: Dataset) -> list[int]:
    selected_buckets = defaultdict(deque)
    for selected_index in range(len(yes_mask_selected_dataset)):
        selected_buckets[yes_mask_selected_key(yes_mask_selected_dataset[selected_index])].append(selected_index)

    matched_indices = []
    unmatched_yes_indices = []
    for yes_index in range(len(yes_dataset)):
        key = yes_key(yes_dataset[yes_index])
        if selected_buckets[key]:
            matched_indices.append(selected_buckets[key].popleft())
        else:
            unmatched_yes_indices.append(yes_index)

    if unmatched_yes_indices:
        preview = unmatched_yes_indices[:20]
        raise ValueError(f"Failed to match {len(unmatched_yes_indices)} YES rows. First unmatched indices: {preview}")

    return matched_indices


def add_alignment_id(dataset: Dataset, alignment_ids: list[str]) -> Dataset:
    if ALIGNMENT_ID_COLUMN in dataset.column_names:
        dataset = dataset.remove_columns([ALIGNMENT_ID_COLUMN])
    return dataset.add_column(ALIGNMENT_ID_COLUMN, alignment_ids)


def run():
    yes_input_path = get_path("YES_ALIGN_INPUT_PATH", YES_INPUT_PATH)
    selected_input_path = get_path("YES_MASK_SELECTED_ALIGN_INPUT_PATH", YES_MASK_SELECTED_INPUT_PATH)
    yes_output_path = get_path("YES_ALIGN_OUTPUT_PATH", YES_OUTPUT_PATH)
    selected_output_path = get_path("YES_MASK_SELECTED_ALIGN_OUTPUT_PATH", YES_MASK_SELECTED_OUTPUT_PATH)
    overwrite_output = get_bool("ALIGN_DATASETS_OVERWRITE", True)

    yes_dataset = load_from_disk(yes_input_path).cast_column("image", Image(decode=False))
    selected_dataset = load_yes_mask_selected(selected_input_path).cast_column("image", Image(decode=False))

    validate_columns(
        yes_dataset,
        {"image", "latitude", "longitude", "d", "reason_class"},
        "YES",
    )
    validate_columns(
        selected_dataset,
        {"image", "latitude_true", "longitude_true", "d_original", "ablated_class"},
        "YES_Mask_selected",
    )

    matched_selected_indices = build_matched_indices(yes_dataset, selected_dataset)
    alignment_ids = [f"yes_{index:06d}" for index in range(len(yes_dataset))]

    yes_with_id = add_alignment_id(yes_dataset, alignment_ids).cast_column("image", Image())
    selected_aligned = selected_dataset.select(matched_selected_indices)
    selected_aligned = add_alignment_id(selected_aligned, alignment_ids).cast_column("image", Image())

    prepare_output_path(yes_output_path, overwrite_output)
    prepare_output_path(selected_output_path, overwrite_output)

    yes_with_id.save_to_disk(yes_output_path)
    selected_aligned.save_to_disk(selected_output_path)

    print(f"Saved YES with IDs to: {yes_output_path}")
    print(f"Saved aligned YES_Mask_selected with IDs to: {selected_output_path}")
    print(f"YES rows: {len(yes_with_id)}")
    print(f"Aligned YES_Mask_selected rows: {len(selected_aligned)}")
    print(f"Dropped YES_Mask_selected rows: {len(selected_dataset) - len(selected_aligned)}")
    print(f"Alignment column: {ALIGNMENT_ID_COLUMN}")


if __name__ == "__main__":
    run()
