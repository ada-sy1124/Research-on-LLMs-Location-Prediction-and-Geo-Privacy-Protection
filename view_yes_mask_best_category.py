from pathlib import Path

import numpy as np
from datasets import load_from_disk
from PIL import Image, ImageDraw


# Change these values, then run: python view_yes_mask_best_category.py
YES_MASK_PATH = "./data/YES_Mask"
AFTER_SAM_PATH = "./data/YES_NEW_afterSAM"
OUTPUT_DIR = "./yes_mask_best_category_preview"
START_INDEX = 0
END_INDEX = 10



TAU = 0.6
PRIVACY_CAP_KM = 100.0


def safe_name(value):
    text = str(value)
    for old, new in (("/", "_"), ("\\", "_"), (" ", "_"), (":", "_")):
        text = text.replace(old, new)
    return text


def select_best_category(sample):
    classes = sample.get("ablated_class") or []
    q_ratio = sample.get("q_ratio") or []
    d_diff = sample.get("d_diff") or []

    if not classes or len(classes) != len(q_ratio) or len(classes) != len(d_diff):
        return -1, "Nothing", 1.0, 0.0, "invalid columns"

    utility = np.clip(1.0 - np.array(q_ratio, dtype=float), 0.0, 1.0)
    privacy_gain = np.nan_to_num(np.array(d_diff, dtype=float), nan=0.0, posinf=PRIVACY_CAP_KM, neginf=0.0)
    privacy = np.clip(privacy_gain, 0.0, PRIVACY_CAP_KM) / PRIVACY_CAP_KM

    valid = utility >= TAU
    if np.any(valid):
        scores = np.where(valid, privacy, -np.inf)
        best_idx = int(np.argmax(scores))
        rule = "utility >= TAU, max privacy"
    else:
        best_idx = int(np.argmax(utility))
        rule = "no utility-qualified class, max utility"

    return best_idx, classes[best_idx], float(utility[best_idx]), float(privacy[best_idx]), rule


def save_image(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)


def make_compare(before, after, title, path):
    before = before.convert("RGB")
    after = after.convert("RGB")

    width = max(before.width, after.width)
    label_height = 44
    height = max(before.height, after.height) + label_height
    canvas = Image.new("RGB", (width * 2, height), "white")
    draw = ImageDraw.Draw(canvas)

    canvas.paste(before, (0, label_height))
    canvas.paste(after, (width, label_height))
    draw.text((10, 12), "before", fill=(0, 0, 0))
    draw.text((width + 10, 12), f"after: {title}", fill=(0, 0, 0))
    canvas.save(path)


def get_masked_image(after_sam_sample, best_idx, best_category):
    masked_images = after_sam_sample.get("masked_image") or []
    classes = after_sam_sample.get("ablated_class") or []

    if 0 <= best_idx < len(masked_images):
        return masked_images[best_idx]

    if best_category in classes:
        idx = classes.index(best_category)
        if idx < len(masked_images):
            return masked_images[idx]

    return None


def main():
    yes_mask = load_from_disk(YES_MASK_PATH)
    after_sam = load_from_disk(AFTER_SAM_PATH)

    output_dir = Path(OUTPUT_DIR)
    image_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    start = max(0, START_INDEX)
    end = min(END_INDEX, len(yes_mask))

    lines = [
        f"YES_MASK_PATH = {YES_MASK_PATH}",
        f"AFTER_SAM_PATH = {AFTER_SAM_PATH}",
        f"RANGE = [{start}, {end})",
        f"TAU = {TAU}",
        f"PRIVACY_CAP_KM = {PRIVACY_CAP_KM}",
        "",
    ]

    for i in range(start, end):
        sample = yes_mask[i]
        best_idx, best_category, utility, privacy, rule = select_best_category(sample)

        before = sample["image"]
        after = None
        if i < len(after_sam):
            after = get_masked_image(after_sam[i], best_idx, best_category)

        prefix = f"sample_{i:06d}_{safe_name(best_category)}"
        before_path = image_dir / f"{prefix}_before.jpg"
        save_image(before, before_path)

        after_path = ""
        compare_path = ""
        if after is not None:
            after_path_obj = image_dir / f"{prefix}_after.jpg"
            compare_path_obj = image_dir / f"{prefix}_compare.jpg"
            save_image(after, after_path_obj)
            make_compare(before, after, best_category, compare_path_obj)
            after_path = str(after_path_obj)
            compare_path = str(compare_path_obj)

        lines.extend(
            [
                f"sample_index: {i}",
                f"best_category: {best_category}",
                f"best_category_index: {best_idx}",
                f"utility: {utility:.6f}",
                f"privacy: {privacy:.6f}",
                f"rule: {rule}",
                f"ablated_class: {sample.get('ablated_class')}",
                f"q_ratio: {sample.get('q_ratio')}",
                f"d_diff: {sample.get('d_diff')}",
                f"before_image: {before_path}",
                f"after_image: {after_path or 'NOT FOUND'}",
                f"compare_image: {compare_path or 'NOT FOUND'}",
                "",
            ]
        )

        print(f"{i}: {best_category} | before={before_path} | after={after_path or 'NOT FOUND'}")

    report_path = output_dir / "report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved report: {report_path.resolve()}")
    print(f"Saved images: {image_dir.resolve()}")


if __name__ == "__main__":
    main()


# python view_yes_mask_best_category.py