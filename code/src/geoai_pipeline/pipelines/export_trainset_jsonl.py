import json
import os
from pathlib import Path

from datasets import load_from_disk
from tqdm import tqdm


YES_DATASET_PATH = "./data/YES_id"
FLAT_DATASET_PATH = "./data/YES_Mask_selected_id_flat"
IMAGE_SAVE_DIR = "./data/sft_images"
JSONL_OUTPUT_PATH = "./data/train_cot.jsonl"

PROMPT = """<image>
You are an advanced privacy-preserving visual agent. Your objective is to defend this image against precise geolocation tracking by large multimodal models, while minimizing the destruction of original image pixels (utility).

To achieve optimal privacy-utility trade-off, you must adopt an "Attack-Simulation-then-Defend" strategy:

STAGE 1 (Simulated Attacker's Scan): 
Perform a vulnerability scan. Identify and group all concrete, physical objects that an adversary could use to pinpoint this location. Use predefined classes: "Road Markings", "Signage & Text", "Vehicles", "Architecture", "Vegetation", "Infrastructure".

STAGE 2 (Defender's Decision): 
Based on the vulnerability scan, select the SINGLE most efficient macro-category to mask. You may also select "Nothing" if masking any category would destroy image utility without significant privacy gains.

Output strictly using the following XML format:
<VULNERABILITY_SCAN>
Class1: obj1, obj2; Class2: obj3; ...
</VULNERABILITY_SCAN>

<DEFENSE_DECISION>
[Optimal Category Name]
</DEFENSE_DECISION>

Example:
<VULNERABILITY_SCAN>
Signage & Text: street name sign #1, warning sign #1, parking sign #1; Road Markings: lane line #1, zebra crossing #1; Vehicles: bus #1; Architecture: house #1, house #2; Vegetation: tree #1, tree #2, tree #3; Infrastructure: bollard #1, bollard #1
</VULNERABILITY_SCAN>

<DEFENSE_DECISION>
Signage & Text
</DEFENSE_DECISION>"""


def get_path(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or value == "":
        value = default
    return str(Path(value).expanduser())


def alignment_id_for_index(index: int) -> str:
    return f"yes_{index:06d}"


def get_alignment_id(sample: dict, index: int) -> str:
    alignment_id = sample.get("alignment_id")
    if alignment_id is None or alignment_id == "":
        return alignment_id_for_index(index)
    return str(alignment_id)


def build_reason_by_id(yes_dataset) -> dict[str, str]:
    if "reason" not in yes_dataset.column_names:
        raise KeyError(f"YES dataset must contain column 'reason'. Current columns: {yes_dataset.column_names}")

    reason_by_id = {}
    duplicate_ids = []

    for index, sample in enumerate(yes_dataset):
        alignment_id = get_alignment_id(sample, index)
        if alignment_id in reason_by_id:
            duplicate_ids.append(alignment_id)
            continue
        reason_by_id[alignment_id] = str(sample.get("reason") or "").strip()

    if duplicate_ids:
        raise ValueError(f"Duplicate alignment_id values in YES dataset: {duplicate_ids[:20]}")

    return reason_by_id


def validate_flat_dataset(flat_dataset) -> None:
    required_columns = {"image", "alignment_id", "target_mask_category"}
    missing_columns = sorted(required_columns - set(flat_dataset.column_names))
    if missing_columns:
        raise KeyError(f"Flat dataset is missing required columns: {missing_columns}")


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
    flat_dataset_path = get_path("YES_MASK_SELECTED_FLAT_DATASET_PATH", FLAT_DATASET_PATH)
    image_save_dir = get_path("TRAINSET_IMAGE_SAVE_DIR", IMAGE_SAVE_DIR)
    jsonl_output_path = get_path("TRAINSET_JSONL_OUTPUT_PATH", JSONL_OUTPUT_PATH)

    yes_dataset = load_from_disk(yes_dataset_path)
    flat_dataset = load_from_disk(flat_dataset_path)
    validate_flat_dataset(flat_dataset)

    reason_by_id = build_reason_by_id(yes_dataset)

    image_save_path = Path(image_save_dir)
    jsonl_path = Path(jsonl_output_path)
    image_save_path.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    missing_ids = []
    with jsonl_path.open("w", encoding="utf-8") as f:
        for index, sample in enumerate(tqdm(flat_dataset, desc="Exporting trainset JSONL")):
            alignment_id = get_alignment_id(sample, index)
            vulnerability_scan = reason_by_id.get(alignment_id)
            if vulnerability_scan is None:
                missing_ids.append(alignment_id)
                continue

            image_filename = f"{alignment_id}.jpg"
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
        raise ValueError(f"{len(missing_ids)} flat samples could not be matched by alignment_id: {missing_ids[:20]}")

    print(f"Saved SFT JSONL: {jsonl_path}")
    print(f"Saved images to: {image_save_path}")
    print(f"YES rows: {len(yes_dataset)}")
    print(f"Flat rows: {len(flat_dataset)}")
    print(f"Exported rows: {len(flat_dataset)}")


if __name__ == "__main__":
    run()
