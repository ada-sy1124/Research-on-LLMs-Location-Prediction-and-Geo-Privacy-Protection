from datasets import concatenate_datasets, load_from_disk

from geoai_pipeline.config import get_path
from geoai_pipeline.tools.dataset_io import list_chunk_paths


def run():
    chunks_dir = get_path("CHUNKS_DIR", "./data/YES_mask")
    output_path = get_path("MERGED_OUTPUT_PATH", "./data/Yes_Mask1")

    chunk_paths = list_chunk_paths(chunks_dir)
    print(f"发现 {len(chunk_paths)} 个 chunk:")
    for p in chunk_paths:
        print(" -", p)

    datasets = [load_from_disk(p) for p in chunk_paths]
    merged = concatenate_datasets(datasets)
    merged.save_to_disk(output_path)

    print("✅ 合并完成！已保存至：", output_path)
    print("📦 合并后样本数量：", len(merged))


if __name__ == "__main__":
    run()
