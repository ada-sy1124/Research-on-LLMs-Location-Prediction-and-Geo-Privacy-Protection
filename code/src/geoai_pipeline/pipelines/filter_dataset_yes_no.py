import gc
import os
import time

import io
from PIL import Image

from datasets import load_dataset
from tqdm import tqdm

from geoai_pipeline.config import get_env, get_float, get_int, get_path
from geoai_pipeline.constants import GEO_PROMPT
from geoai_pipeline.tools.dataset_io import save_chunk
from geoai_pipeline.tools.geo import haversine_km
from geoai_pipeline.tools.model_backend import create_inference_model, predict_latlon_and_reason


def run():
    model_backend = get_env("MODEL_BACKEND", "local")
    dataset_name = get_env("DATASET_NAME", "stochastic/random_streetview_images_pano_v0.0.2")

    start_index = get_int("FILTER_START_INDEX", 0)
    end_index = get_int("FILTER_END_INDEX", 11054)
    batch_size = get_int("FILTER_BATCH_SIZE", 30)

    yes_dir = get_path("YES_DIR", "./data/YES")
    no_dir = get_path("NO_DIR", "./data/NO")

    buffer_size = get_int("FILTER_BUFFER_SIZE", 30)
    dist_threshold_km = get_float("FILTER_DIST_THRESHOLD_KM", 5.0)
    sleep_seconds = get_int("FILTER_SLEEP_SECONDS", 0)

    yes_chunk_start_id = get_int("FILTER_YES_CHUNK_START_ID", 8)
    no_chunk_start_id = get_int("FILTER_NO_CHUNK_START_ID", 28)

    print(
        "🔧 当前生效参数: "
        f"FILTER_START_INDEX={start_index}, FILTER_END_INDEX={end_index}, FILTER_BATCH_SIZE={batch_size}, "
        f"FILTER_BUFFER_SIZE={buffer_size}, FILTER_SLEEP_SECONDS={sleep_seconds}, "
        f"FILTER_DIST_THRESHOLD_KM={dist_threshold_km}, MODEL_BACKEND={model_backend}, "
        f"YES_DIR={yes_dir}, NO_DIR={no_dir}, DATASET={dataset_name}"
    )

    model = create_inference_model()

    os.makedirs(yes_dir, exist_ok=True)
    os.makedirs(no_dir, exist_ok=True)

    yes_buffer, no_buffer = [], []
    yes_chunk_id, no_chunk_id = yes_chunk_start_id, no_chunk_start_id

    for batch_start in range(start_index, end_index, batch_size):
        batch_end = min(batch_start + batch_size, end_index)
        current_split = f"train[{batch_start}:{batch_end}]"

        print(f"\n 开始处理批次: {batch_start} 到 {batch_end} (Split: {current_split})")
        
        if os.path.isdir(dataset_name):
            dataset = load_dataset("parquet", data_dir=dataset_name, split=current_split)
        else:
            dataset = load_dataset(dataset_name, split=current_split)

        for local_index, item in enumerate(tqdm(dataset, desc=f"Batch {batch_start}-{batch_end}")):
            source_index = batch_start + local_index
            sample_id = str(item.get("sample_id") or item.get("id") or f"source_{source_index:06d}")
            lat_true = float(item["latitude"])
            lon_true = float(item["longitude"])

            # --- 坚不可摧的图片解析逻辑 ---
            raw_image = item["image"]
            
            if isinstance(raw_image, dict) and "bytes" in raw_image:
                img_obj = Image.open(io.BytesIO(raw_image["bytes"])).convert("RGB")
            elif hasattr(raw_image, "convert"):
                img_obj = raw_image.convert("RGB")
            else:
                img_obj = raw_image
            # -------------------------------

            lat_pred, lon_pred, reason_text, reason_classes, q = predict_latlon_and_reason(model, img_obj, GEO_PROMPT)

            if lat_pred == 0.0 and lon_pred == 0.0 and q == 0:
                dist = 99999.0
            else:
                dist = haversine_km(lat_pred, lon_pred, lat_true, lon_true)

            label = "YES" if dist <= dist_threshold_km else "NO"

            item_out = {
                "sample_id": sample_id,
                "source_index": source_index,
                "image": img_obj,  # 确保后续保存阶段拿到的是真正的图片对象
                "latitude_pred": lat_pred,
                "longitude_pred": lon_pred,
                "latitude": lat_true,
                "longitude": lon_true,
                "d": dist,
                "reason": reason_text,
                "reason_class": reason_classes,
                "q": q,
            }

            if label == "YES":
                yes_buffer.append(item_out)
                if len(yes_buffer) >= buffer_size:
                    save_chunk(yes_buffer, yes_dir, yes_chunk_id)
                    yes_buffer.clear()
                    yes_chunk_id += 1
            else:
                no_buffer.append(item_out)
                if len(no_buffer) >= buffer_size:
                    save_chunk(no_buffer, no_dir, no_chunk_id)
                    no_buffer.clear()
                    no_chunk_id += 1

            time.sleep(sleep_seconds)

        del dataset
        gc.collect()
        print(f"✅ 批次 {batch_start}-{batch_end} 完成，内存已清理。")

    if yes_buffer:
        save_chunk(yes_buffer, yes_dir, yes_chunk_id)
    if no_buffer:
        save_chunk(no_buffer, no_dir, no_chunk_id)

    print("🎉 所有批次处理完成!")


if __name__ == "__main__":
    run()