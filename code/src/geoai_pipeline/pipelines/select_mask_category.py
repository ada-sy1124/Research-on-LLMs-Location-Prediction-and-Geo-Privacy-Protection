import json
from pathlib import Path

import numpy as np

from geoai_pipeline.config import get_bool, get_float, get_path
from geoai_pipeline.tools.dataset_io import load_chunks_or_dataset


def select_optimal_category(utility_scores, privacy_scores, tau=0.7):
    u_arr = np.array(utility_scores, dtype=float)
    p_arr = np.array(privacy_scores, dtype=float)

    if len(u_arr) == 0 or len(u_arr) != len(p_arr):
        return -1, "Invalid: empty or mismatched score list"

    valid_mask = u_arr >= tau
    if np.any(valid_mask):
        masked_privacy = np.where(valid_mask, p_arr, -np.inf)
        return int(np.argmax(masked_privacy)), "Condition A: utility qualified, maximize privacy"

    return int(np.argmax(u_arr)), "Condition B: no qualified utility, maximize utility"


def build_scores(q_ratio, d_diff, privacy_cap_km):
    q_arr = np.array(q_ratio, dtype=float)
    d_arr = np.array(d_diff, dtype=float)

    utility_scores = np.clip(1.0 - q_arr, 0.0, 1.0)
    privacy_gain = np.nan_to_num(d_arr, nan=0.0, posinf=privacy_cap_km, neginf=0.0)
    privacy_gain = np.clip(privacy_gain, 0.0, privacy_cap_km)
    privacy_scores = privacy_gain / privacy_cap_km

    return utility_scores.tolist(), privacy_scores.tolist()


def select_for_sample(sample, sample_index, tau, privacy_cap_km):
    sample_id = sample.get("sample_id") or f"yes_mask_{sample_index:04d}"
    ablated_class = sample.get("ablated_class") or []
    q_ratio = sample.get("q_ratio") or []
    d_diff = sample.get("d_diff") or []

    if not ablated_class or len(ablated_class) != len(q_ratio) or len(ablated_class) != len(d_diff):
        return {
            "sample_id": sample_id,
            "target_mask_category": "Nothing",
            "label": "Nothing",
            "target_mask_category_index": -1,
            "final_utility": 1.0,
            "final_privacy": 0.0,
            "selection_rule": "Invalid: missing or mismatched ablated_class/q_ratio/d_diff",
        }

    utility_scores, privacy_scores = build_scores(q_ratio, d_diff, privacy_cap_km)
    best_idx, rule = select_optimal_category(utility_scores, privacy_scores, tau=tau)

    if best_idx < 0:
        target_category = "Nothing"
        final_utility = 1.0
        final_privacy = 0.0
    else:
        target_category = ablated_class[best_idx]
        final_utility = utility_scores[best_idx]
        final_privacy = privacy_scores[best_idx]

    return {
        "sample_id": sample_id,
        "target_mask_category": target_category,
        "label": target_category,
        "target_mask_category_index": best_idx,
        "final_utility": float(final_utility),
        "final_privacy": float(final_privacy),
        "selection_rule": rule,
    }


def _prepare_output_path(path: Path, overwrite: bool) -> None:
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(f"Output path already exists: {path}. Set SELECT_MASK_OVERWRITE=1 to overwrite it.")
    if path.resolve().parent == path.resolve() or path.name in {"", ".", ".."}:
        raise ValueError(f"Refusing to overwrite unsafe output path: {path}")

    import shutil

    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def run():
    input_path = Path(get_path("YES_MASK_INPUT_PATH", "./data/YES_Mask")).expanduser()
    output_path = Path(get_path("YES_MASK_SELECTED_OUTPUT_PATH", "./data/YES_Mask_selected")).expanduser()
    json_path = Path(get_path("SELECT_MASK_JSON_OUTPUT_PATH", "./data/qwen_sft_alignment_dataset.json")).expanduser()
    tau = get_float("SELECT_MASK_TAU", 0.6)
    privacy_cap_km = get_float("SELECT_MASK_PRIVACY_CAP_KM", 100.0)
    overwrite = get_bool("SELECT_MASK_OVERWRITE", False)

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    dataset = load_chunks_or_dataset(str(input_path))
    print(f"Loaded {len(dataset)} YES_Mask rows from: {input_path}")

    selections = [
        select_for_sample(sample, idx, tau=tau, privacy_cap_km=privacy_cap_km)
        for idx, sample in enumerate(dataset)
    ]

    annotated = dataset
    columns_to_update = [
        "sample_id",
        "target_mask_category",
        "label",
        "target_mask_category_index",
        "final_utility",
        "final_privacy",
        "selection_rule",
    ]
    for column in columns_to_update:
        if column in annotated.column_names:
            annotated = annotated.remove_columns(column)
        annotated = annotated.add_column(column, [row[column] for row in selections])

    _prepare_output_path(output_path, overwrite)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save_to_disk(str(output_path))

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "sample_id": row["sample_id"],
                    "target_mask_category": row["target_mask_category"],
                    "target_mask_category_index": row["target_mask_category_index"],
                }
                for row in selections
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )

    rules = {}
    cat_counts = {}
    for row in selections:
        rules[row["selection_rule"]] = rules.get(row["selection_rule"], 0) + 1
        cat_counts[row["target_mask_category"]] = cat_counts.get(row["target_mask_category"], 0) + 1

    mean_utility = float(np.mean([row["final_utility"] for row in selections])) if selections else 0.0
    mean_privacy = float(np.mean([row["final_privacy"] for row in selections])) if selections else 0.0

    print("Selection audit:")
    for rule, count in sorted(rules.items()):
        print(f"  - {rule}: {count}")
    print(f"Average utility retention: {mean_utility * 100:.2f}%")
    print(f"Average privacy score: {mean_privacy * 100:.2f}%")
    print("Category distribution:")
    for cat, count in sorted(cat_counts.items(), key=lambda item: item[1], reverse=True):
        percentage = (count / len(selections)) * 100 if selections else 0.0
        print(f"  - {cat:<20}: {count:>5} ({percentage:.2f}%)")
    print(f"Saved selected dataset to: {output_path}")
    print(f"Saved compact JSON to: {json_path}")


if __name__ == "__main__":
    run()
