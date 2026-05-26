from json import JSONDecodeError
import os
from pathlib import Path
import shutil

import datasets
from datasets import Dataset, DatasetInfo, Features, Image, Value, concatenate_datasets, load_from_disk


INPUT_DATASET_PATH = "./data/YES_Mask_selected_id"
OUTPUT_DATASET_PATH = "./data/YES_Mask_selected_flat"

REQUIRED_COLUMNS = {
    "alignment_id",
    "image",
    "target_mask_category",
    "target_mask_category_index",
    "q_ratio",
    "d_diff",
}


def build_input_features() -> Features:
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
            "alignment_id": Value("string"),
            "target_mask_category": Value("string"),
            "label": Value("string"),
            "target_mask_category_index": Value("int64"),
            "final_utility": Value("float64"),
            "final_privacy": Value("float64"),
            "selection_rule": Value("string"),
        }
    )


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
            "Set YES_MASK_SELECTED_FLAT_OVERWRITE=1 to overwrite it."
        )

    resolved = output_path.resolve()
    if resolved.parent == resolved or resolved.name in {"", ".", ".."}:
        raise ValueError(f"Refusing to overwrite unsafe output path: {output_dataset_path}")

    if output_path.is_dir():
        shutil.rmtree(output_path)
    else:
        output_path.unlink()


def load_dataset_from_arrow_files(dataset_path: str) -> Dataset:
    root = Path(dataset_path)
    arrow_files = sorted(root.glob("data-*.arrow"))
    if not arrow_files:
        raise FileNotFoundError(f"No data-*.arrow files found in {dataset_path}")

    info = DatasetInfo(features=build_input_features())
    datasets_from_files = [Dataset.from_file(str(path), info=info) for path in arrow_files]
    if len(datasets_from_files) == 1:
        return datasets_from_files[0]
    return concatenate_datasets(datasets_from_files)


def load_yes_mask_selected(dataset_path: str) -> Dataset:
    try:
        return load_from_disk(dataset_path)
    except JSONDecodeError:
        return load_dataset_from_arrow_files(dataset_path)


def get_selected_value(values, selected_index: int, column_name: str, row_index: int) -> float:
    if values is None:
        raise ValueError(f"Row {row_index}: {column_name} is missing")
    if not isinstance(values, list):
        return float(values)
    if selected_index < 0 or selected_index >= len(values):
        raise IndexError(
            f"Row {row_index}: target_mask_category_index={selected_index} is out of range for {column_name}"
        )
    return float(values[selected_index])


def flatten_batch(batch: dict, row_indices: list[int]) -> dict:
    q_ratios = []
    d_diffs = []

    for batch_index, row_index in enumerate(row_indices):
        selected_index = int(batch["target_mask_category_index"][batch_index])
        q_ratios.append(get_selected_value(batch["q_ratio"][batch_index], selected_index, "q_ratio", row_index))
        d_diffs.append(get_selected_value(batch["d_diff"][batch_index], selected_index, "d_diff", row_index))

    return {
        "target_mask_category": [str(value) for value in batch["target_mask_category"]],
        "q_ratio": q_ratios,
        "d_diff": d_diffs,
    }


def run():
    input_dataset_path = get_path(
        "YES_MASK_SELECTED_INPUT_DATASET_PATH",
        INPUT_DATASET_PATH,
    )
    output_dataset_path = get_path(
        "YES_MASK_SELECTED_FLAT_OUTPUT_PATH",
        OUTPUT_DATASET_PATH,
    )
    overwrite_output = get_bool("YES_MASK_SELECTED_FLAT_OVERWRITE", True)

    dataset = load_yes_mask_selected(input_dataset_path)
    missing_columns = sorted(REQUIRED_COLUMNS - set(dataset.column_names))
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    output_features = Features(
        {
            "image": Image(),
            "alignment_id": Value("string"),
            "target_mask_category": Value("string"),
            "q_ratio": Value("float64"),
            "d_diff": Value("float64"),
        }
    )

    dataset = dataset.cast_column("image", Image(decode=False))
    flat_dataset = dataset.map(
        flatten_batch,
        batched=True,
        with_indices=True,
        remove_columns=[column for column in dataset.column_names if column not in {"image", "alignment_id"}],
        load_from_cache_file=False,
        desc="Flattening selected mask fields",
    )
    flat_dataset = flat_dataset.select_columns(["image", "alignment_id", "target_mask_category", "q_ratio", "d_diff"])
    flat_dataset = flat_dataset.cast(output_features)
    prepare_output_path(output_dataset_path, overwrite_output)
    flat_dataset.save_to_disk(output_dataset_path)

    print(f"Saved flat dataset to: {output_dataset_path}")
    print(f"Rows: {len(flat_dataset)}")
    print(f"Columns: {flat_dataset.column_names}")


if __name__ == "__main__":
    run()
