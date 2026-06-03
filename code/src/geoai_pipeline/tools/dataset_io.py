import os
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_from_disk


def save_chunk(buffer_list: list, out_dir: str, chunk_id: str | int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    save_path = f"{out_dir}/chunk_{chunk_id}"
    Dataset.from_list(buffer_list).save_to_disk(save_path)
    print(f"💾 已保存块 {chunk_id} 至 {save_path} | 数量: {len(buffer_list)}")


def list_chunk_paths(chunks_dir: str) -> list[str]:
    root = Path(chunks_dir)
    if not root.exists():
        return []
    return sorted([str(p) for p in root.iterdir() if p.is_dir() and p.name.startswith("chunk_")])


def load_chunks_or_dataset(path: str):
    chunk_paths = list_chunk_paths(path)
    if chunk_paths:
        return concatenate_datasets([load_from_disk(p) for p in chunk_paths])
    return load_from_disk(path)
