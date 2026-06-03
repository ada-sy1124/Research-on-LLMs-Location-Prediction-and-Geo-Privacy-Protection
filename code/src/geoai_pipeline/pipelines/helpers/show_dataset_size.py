from datasets import load_from_disk

from geoai_pipeline.config import get_path


def run():
    yes_dir = get_path(
        "YES_DIR",
        "./data/YES",
    )

    dataset = load_from_disk(yes_dir)
    total_samples = len(dataset)

    print("数据集加载成功！")
    print(f"该数据集共有: {total_samples} 个样本")
    print(f"包含的列名: {dataset.column_names}")


if __name__ == "__main__":
    run()
