from pathlib import Path

from datasets import load_from_disk
from PIL import Image


DATASET_PATH = "./data/YES_Mask"
OUTPUT_DIR = "./data/preview_data"
START = 0
LIMIT = 2
COLUMNS = ["ablated_class", "d_diff"]


def safe_name(value):
    return str(value).replace("/", "_").replace("\\", "_").replace(" ", "_")


def save_image(image, path):
    if isinstance(image, Image.Image):
        image.save(path)
    elif isinstance(image, dict) and "bytes" in image:
        path.write_bytes(image["bytes"])
    else:
        image.save(path)


def write_value(text_file, sample_index, column, value):
    text_file.write(f"\n===== sample {sample_index} | {column} =====\n")
    text_file.write(repr(value))
    text_file.write("\n")


def run():
    dataset = load_from_disk(DATASET_PATH)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    end = min(START + LIMIT, len(dataset))
    text_path = output_dir / "preview_text.txt"

    with text_path.open("w", encoding="utf-8") as text_file:
        text_file.write(f"DATASET_PATH = {DATASET_PATH}\n")
        text_file.write(f"ROWS = {len(dataset)}\n")
        text_file.write(f"COLUMNS = {dataset.column_names}\n")

        for i in range(START, end):
            sample = dataset[i]
            for column in COLUMNS:
                if column not in sample:
                    write_value(text_file, i, column, "COLUMN NOT FOUND")
                    continue

                value = sample[column]

                if isinstance(value, Image.Image):
                    save_path = output_dir / f"sample_{i:05d}_{safe_name(column)}.jpg"
                    save_image(value, save_path)
                    print(save_path)
                elif isinstance(value, list) and value and isinstance(value[0], Image.Image):
                    for j, image in enumerate(value):
                        suffix = ""
                        if column == "masked_image" and "ablated_class" in sample and j < len(sample["ablated_class"]):
                            suffix = f"_{safe_name(sample['ablated_class'][j])}"
                        save_path = output_dir / f"sample_{i:05d}_{safe_name(column)}_{j:02d}{suffix}.jpg"
                        save_image(image, save_path)
                        print(save_path)
                else:
                    write_value(text_file, i, column, value)

    print(text_path)


if __name__ == "__main__":
    run()

# python .\code\辅助功能\查看数据.py
