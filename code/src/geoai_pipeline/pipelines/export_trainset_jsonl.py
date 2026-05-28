import json
from pathlib import Path

from tqdm import tqdm

from geoai_pipeline.config import get_env, get_path
from geoai_pipeline.tools.dataset_io import load_chunks_or_dataset


YES_DATASET_PATH = "./data/YES"
SELECTED_DATASET_PATH = "./data/YES_Mask_selected"
IMAGE_SAVE_DIR = "./data/sft_images"
JSONL_OUTPUT_PATH = "./data/train_cot.jsonl"

PROMPT = """<image>
You are an advanced privacy-preserving visual agent. Your objective is to defend this image against precise geolocation tracking by large multimodal models, while minimizing the destruction of original image pixels (utility).

To achieve optimal privacy-utility trade-off, you must adopt an "Attack-Simulation-then-Defend" strategy:

STAGE 1 (Simulated Attacker's Scan): 
Perform a vulnerability scan. Identify and group all concrete, physical objects that an adversary could use to pinpoint this location. Use predefined classes: "Road Markings", "Signage & Text", "Vehicles", "Architecture", "Vegetation", "Infrastructure".

STAGE 2 (Defender's Decision): 
Based on the vulnerability scan, select the SINGLE most efficient macro-category to mask. You may also select "Nothing" if masking any category would destroy image utility without significant privacy gains.

CRITICAL INSTRUCTIONS FOR OUTPUT FORMAT & CONTENT:

[GLOBAL STRUCTURE RULES]
1. The <VULNERABILITY_SCAN> block MUST be generated first to simulate the attack.
2. The <DEFENSE_DECISION> block MUST be the absolute final element. 
3. Format strictly as: ClassName: obj1, obj2; NextClass: obj3; ... 
4. List ONLY physical, visible objects. Every object must be a concrete visible instance. You MUST include a quantifier (e.g., a number, "a/an", or "some") and at least one descriptive adjective before the noun (e.g., "a blue street name sign", "some white lane lines").

Output strictly using the following XML format:
<VULNERABILITY_SCAN>
Class1: obj1, obj2; Class2: obj3; ...
</VULNERABILITY_SCAN>

<DEFENSE_DECISION>
[Optimal Category Name]
</DEFENSE_DECISION>"""


def sample_id_for_index(index: int) -> str:
    return f"yes_{index:06d}"


def get_sample_id(sample: dict, index: int) -> str:
    sample_id = sample.get("sample_id")
    if sample_id is None or sample_id == "":
        return sample_id_for_index(index)
    return str(sample_id)


def build_reason_by_sample_id(yes_dataset) -> dict[str, str]:
    if "reason" not in yes_dataset.column_names:
        raise KeyError(f"YES dataset must contain column 'reason'. Current columns: {yes_dataset.column_names}")

    reason_by_sample_id = {}
    duplicate_sample_ids = []

    for index, sample in enumerate(yes_dataset):
        sample_id = get_sample_id(sample, index)
        if sample_id in reason_by_sample_id:
            duplicate_sample_ids.append(sample_id)
            continue
        reason_by_sample_id[sample_id] = str(sample.get("reason") or "").strip()

    if duplicate_sample_ids:
        raise ValueError(f"Duplicate sample_id values in YES dataset: {duplicate_sample_ids[:20]}")

    return reason_by_sample_id


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


def selected_key(sample: dict) -> tuple:
    return (
        float(sample["latitude_true"]),
        float(sample["longitude_true"]),
        float(sample["d_original"]),
        normalize_classes(sample["ablated_class"]),
    )


def build_reason_by_selected_index(yes_dataset, selected_dataset) -> dict[int, tuple[str, str]]:
    selected_buckets = {}
    for selected_index, sample in enumerate(selected_dataset):
        selected_buckets.setdefault(selected_key(sample), []).append(selected_index)

    reason_by_selected_index = {}
    for yes_index, yes_sample in enumerate(yes_dataset):
        bucket = selected_buckets.get(yes_key(yes_sample), [])
        if not bucket:
            continue
        selected_index = bucket.pop(0)
        sample_id = get_sample_id(yes_sample, yes_index)
        reason_by_selected_index[selected_index] = (sample_id, str(yes_sample.get("reason") or "").strip())

    missing_count = len(selected_dataset) - len(reason_by_selected_index)
    if missing_count:
        raise ValueError(f"{missing_count} selected samples could not be matched to YES rows.")

    return reason_by_selected_index


def validate_selected_dataset(selected_dataset, use_sample_id: bool) -> None:
    required_columns = {"image", "target_mask_category"}
    if not use_sample_id:
        required_columns.update({"latitude_true", "longitude_true", "d_original", "ablated_class"})
    missing_columns = sorted(required_columns - set(selected_dataset.column_names))
    if missing_columns:
        raise KeyError(f"Selected dataset is missing required columns: {missing_columns}")


def build_assistant_content(vulnerability_scan: str, decision: str) -> str:
    if not vulnerability_scan:
        vulnerability_scan = "Nothing: none"
    if not decision:
        decision = "Nothing"

    return (
        "<VULNERABILITY_SCAN>\n"
        f"{vulnerability_scan}\n"
        "</VULNERABILITY_SCAN>\n\n"
        "<DEFENSE_DECISION>\n"
        f"{decision}\n"
        "</DEFENSE_DECISION>"
    )


def run():
    yes_dataset_path = get_path("YES_REASON_DATASET_PATH", YES_DATASET_PATH)
    selected_default_path = get_env("YES_MASK_SELECTED_FLAT_DATASET_PATH", SELECTED_DATASET_PATH)
    selected_dataset_path = get_path("YES_MASK_SELECTED_DATASET_PATH", selected_default_path)
    image_save_dir = get_path("TRAINSET_IMAGE_SAVE_DIR", IMAGE_SAVE_DIR)
    jsonl_output_path = get_path("TRAINSET_JSONL_OUTPUT_PATH", JSONL_OUTPUT_PATH)

    yes_dataset = load_chunks_or_dataset(yes_dataset_path)
    selected_dataset = load_chunks_or_dataset(selected_dataset_path)
    use_sample_id = "sample_id" in selected_dataset.column_names and "sample_id" in yes_dataset.column_names
    validate_selected_dataset(selected_dataset, use_sample_id)
    if use_sample_id:
        reason_by_sample_id = build_reason_by_sample_id(yes_dataset)
        reason_by_selected_index = None
    else:
        reason_by_sample_id = None
        reason_by_selected_index = build_reason_by_selected_index(yes_dataset, selected_dataset)

    image_save_path = Path(image_save_dir)
    jsonl_path = Path(jsonl_output_path)
    image_save_path.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    missing_ids = []
    exported_sample_ids = set()
    with jsonl_path.open("w", encoding="utf-8") as f:
        for index, sample in enumerate(tqdm(selected_dataset, desc="Exporting trainset JSONL")):
            if use_sample_id:
                sample_id = get_sample_id(sample, index)
                vulnerability_scan = reason_by_sample_id.get(sample_id)
                if vulnerability_scan is None:
                    missing_ids.append(sample_id)
                    continue
            else:
                sample_id, vulnerability_scan = reason_by_selected_index[index]

            if sample_id in exported_sample_ids:
                raise ValueError(f"Duplicate sample_id in selected dataset: {sample_id}")
            exported_sample_ids.add(sample_id)

            image_filename = f"{sample_id}.jpg"
            image_path = image_save_path / image_filename
            sample["image"].save(image_path)

            assistant_content = build_assistant_content(
                vulnerability_scan=vulnerability_scan,
                decision=str(sample.get("target_mask_category") or "Nothing").strip(),
            )

            record = {
                "messages": [
                    {"role": "user", "content": PROMPT},
                    {"role": "assistant", "content": assistant_content},
                ],
                "images": [str(image_path.resolve())],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if missing_ids:
        raise ValueError(f"{len(missing_ids)} selected samples could not be matched by sample_id: {missing_ids[:20]}")

    print(f"Saved SFT JSONL: {jsonl_path}")
    print(f"Saved images to: {image_save_path}")
    print(f"YES rows: {len(yes_dataset)}")
    print(f"Selected rows: {len(selected_dataset)}")
    print(f"Exported rows: {len(selected_dataset)}")


if __name__ == "__main__":
    run()
