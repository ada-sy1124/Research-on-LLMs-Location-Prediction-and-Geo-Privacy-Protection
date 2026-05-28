#!/usr/bin/env python3
from pathlib import Path
from datasets import concatenate_datasets, load_from_disk

# ==========================================
# 🎯 在这里手动修改你想查看的参数
# ==========================================
DATASET_PATH = "/root/autodl-tmp/data/NO/chunk_4"   # 替换为你的数据集绝对或相对路径
TARGET_COLUMN = "d"       # 你想查看的列名 (例如: "image", "latitude", "reason")
# ==========================================

def load_dataset(path: Path):
    chunks = sorted(p for p in path.iterdir() if p.is_dir() and p.name.startswith("chunk_")) if path.is_dir() else []
    if chunks:
        return concatenate_datasets([load_from_disk(str(p)) for p in chunks])
    return load_from_disk(str(path))

def main() -> None:
    path = Path(DATASET_PATH).expanduser()
    dataset = load_dataset(path)
    
    # 遍历并顺着打印该字段的所有内容，去掉了所有多余的格式
    for item in dataset:
        print(item[TARGET_COLUMN])

if __name__ == "__main__":
    main()


# python ./查看数据.py