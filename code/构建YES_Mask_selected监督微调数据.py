import json
import os
from pathlib import Path

from datasets import load_from_disk


DATASET_PATH = "./data/YES_Mask_selected"
IMAGE_SAVE_DIR = "./data/sft_images"
JSONL_OUTPUT_PATH = "./data/yes_mask_selected_sft.jsonl"

IMAGE_COLUMN = "image"
LABEL_COLUMN = "target_mask_category"

PROMPT = (
    "<image>Analyze the image and output the target category from 'Architecture', "
    "'Infrastructure', 'Road Markings', 'Signage & Text', 'Vegetation', "
    "'Vehicles', 'Nothing' that, when ignored or masked, maximizes the "
    "prevention of precise geolocation prediction by an LLM while minimizing "
    "information destruction.\n\n"
    "Output ONLY ONE category name."
)


def main():
    dataset = load_from_disk(DATASET_PATH)

    image_save_dir = Path(IMAGE_SAVE_DIR)
    jsonl_output_path = Path(JSONL_OUTPUT_PATH)
    image_save_dir.mkdir(parents=True, exist_ok=True)
    jsonl_output_path.parent.mkdir(parents=True, exist_ok=True)

    if IMAGE_COLUMN not in dataset.column_names:
        raise KeyError(f"找不到图片列 {IMAGE_COLUMN}，现有列为: {dataset.column_names}")
    if LABEL_COLUMN not in dataset.column_names:
        raise KeyError(f"找不到标签列 {LABEL_COLUMN}，现有列为: {dataset.column_names}")

    with jsonl_output_path.open("w", encoding="utf-8") as f:
        for i, sample in enumerate(dataset):
            image_path = image_save_dir / f"yes_mask_selected_{i:05d}.jpg"
            sample[IMAGE_COLUMN].save(image_path)

            record = {
                "messages": [
                    {"role": "user", "content": PROMPT},
                    {"role": "assistant", "content": str(sample[LABEL_COLUMN])},
                ],
                "images": [os.path.abspath(image_path)],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"已保存 SFT JSONL: {jsonl_output_path}")
    print(f"已保存图片目录: {image_save_dir}")
    print(f"样本数: {len(dataset)}")


if __name__ == "__main__":
    main()

# python ./code/构建YES_Mask_selected监督微调数据.py