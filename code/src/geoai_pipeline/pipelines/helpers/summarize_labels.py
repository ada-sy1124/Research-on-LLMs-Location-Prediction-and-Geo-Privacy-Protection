from collections import Counter

from datasets import load_from_disk

from geoai_pipeline.config import get_path


def run():
    train_set_path = get_path(
        "TRAIN_SET_PATH",
        "./data/train_set+no",
    )

    dataset = load_from_disk(train_set_path)

    label_key = None
    for key in ("Label", "label", "class"):
        if key in dataset.column_names:
            label_key = key
            break

    if label_key is None:
        raise KeyError(f"train_set 中未找到 Label/label/class 字段，当前字段有: {dataset.column_names}")

    label_counts = Counter()
    for sample in dataset:
        label = sample.get(label_key)
        if label is None:
            continue
        label_counts[str(label)] += 1

    sorted_counts = sorted(label_counts.items(), key=lambda x: (-x[1], x[0]))

    print(f"统计字段: {label_key}")
    print("Label 各类型样本数（按数量降序）：")
    for label, count in sorted_counts:
        print(f"{label}: {count}")
    print(f"\n总样本数: {sum(label_counts.values())}")
    print(f"Label 类型数: {len(label_counts)}")


if __name__ == "__main__":
    run()
