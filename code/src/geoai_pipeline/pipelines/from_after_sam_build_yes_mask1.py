import io
import os
import time

import numpy as np
from datasets import load_from_disk
from PIL import Image
from tqdm import tqdm

from geoai_pipeline.config import get_env, get_float, get_int, get_path
from geoai_pipeline.constants import GEO_PROMPT
from geoai_pipeline.tools.dataset_io import save_chunk
from geoai_pipeline.tools.genai_client import create_genai_client
from geoai_pipeline.tools.gemini import gemini_predict_latlon
from geoai_pipeline.tools.geo import haversine_km


def run():
    gemini_model = get_env("GEMINI_MODEL", "gemini-3-flash-preview")

    input_dataset_path = get_path(
        "MASK1_AFTER_SAM_INPUT_DATASET_PATH",
        get_env("AFTER_SAM_INPUT_DATASET_PATH", "./data/YES_NEW_afterSAM"),
    )
    output_dir = get_path(
        "MASK1_YES_MASK_OUTPUT_DIR",
        get_env("YES_MASK_OUTPUT_DIR", "./data/YES_Mask1"),
    )

    start_index = get_int("MASK1_START_INDEX", get_int("START_INDEX", 0))
    end_index = get_int("MASK1_END_INDEX", get_int("END_INDEX", 1030))

    buffer_size = get_int("MASK1_BUFFER_SIZE", get_int("BUFFER_SIZE", 30))
    sleep_seconds = get_int("MASK1_SLEEP_SECONDS", get_int("SLEEP_SECONDS", 15))
    temperature = get_float("MASK1_GEMINI_TEMPERATURE", get_float("GEMINI_TEMPERATURE", 0.0))

    print(
        "🔧 当前生效参数: "
        f"MASK1_START_INDEX={start_index}, MASK1_END_INDEX={end_index}, MASK1_BUFFER_SIZE={buffer_size}, "
        f"MASK1_SLEEP_SECONDS={sleep_seconds}, GEMINI_MODEL={gemini_model}, "
        f"MASK1_AFTER_SAM_INPUT_DATASET_PATH={input_dataset_path}, MASK1_YES_MASK_OUTPUT_DIR={output_dir}"
    )

    client = create_genai_client("MASK1_GEMINI_API_KEY")

    print("正在加载已打码数据集...")
    if not os.path.exists(input_dataset_path):
        print(f"❌ 找不到输入路径: {input_dataset_path}")
        return

    dataset = load_from_disk(input_dataset_path)

    total_len = len(dataset)
    start = start_index if start_index is not None else 0
    end = end_index if end_index is not None else total_len
    end = min(end, total_len)

    if start >= end:
        print(f"❌ 范围设置有误：START_INDEX ({start}) 大于等于 END_INDEX ({end})")
        return

    dataset = dataset.select(range(start, end))
    print(f"✂️ 数据集已截取范围: [{start}:{end}]")
    print(f"🚀 开始基于物理遮挡进行 Gemini 坐标重测，当前待处理数据量: {len(dataset)} 条...")

    buffer = []
    chunk_id = 24 #########################

    for item in tqdm(dataset, desc="Evaluating Masked Images"):
        image_obj = item["image_original"]
        lat_true = item["latitude"]
        lon_true = item["longitude"]
        d_orig = item["d_original"]

        ablated_class = item.get("ablated_class", [])
        masked_image = item.get("masked_image", [])
        q_ratio = item.get("q_ratio", [])

        if not masked_image or len(masked_image) != len(ablated_class):
            print("\n[脏数据拦截] ID 或索引异常！")
            print(
                f" -> masked_image 类型: {type(masked_image)}, 长度: "
                f"{len(masked_image) if isinstance(masked_image, (list, tuple, str)) else 'N/A'}"
            )
            print(f" -> ablated_class 类型: {type(ablated_class)}, 长度/值: {len(ablated_class)} / {ablated_class}")
            continue

        d_prime_list = []
        d_diff_list = []

        for masked_img in masked_image:
            if isinstance(masked_img, dict) and "bytes" in masked_img:
                masked_img = Image.open(io.BytesIO(masked_img["bytes"]))
            elif not isinstance(masked_img, Image.Image):
                masked_img = Image.fromarray(np.array(masked_img))

            lat_pred, lon_pred = gemini_predict_latlon(
                client=client,
                model=gemini_model,
                image_obj=masked_img,
                prompt=GEO_PROMPT,
                temperature=temperature,
            )

            if lat_pred == 0.0 and lon_pred == 0.0:
                d_prime = 99999.0
                d_diff = 99999.0
            else:
                d_prime = haversine_km(lat_pred, lon_pred, lat_true, lon_true)
                d_diff = d_prime - d_orig

            d_prime_list.append(d_prime)
            d_diff_list.append(d_diff)

            time.sleep(sleep_seconds)

        new_item = {
            "image": image_obj,
            "latitude_true": lat_true,
            "longitude_true": lon_true,
            "d_original": d_orig,
            "ablated_class": ablated_class,
            "q_ratio": q_ratio,
            "d_prime": d_prime_list,
            "d_diff": d_diff_list,
        }

        buffer.append(new_item)

        if len(buffer) >= buffer_size:
            save_chunk(buffer, output_dir, f"{chunk_id}_part_{start}_to_{end}")
            buffer.clear()
            chunk_id += 1

    if buffer:
        save_chunk(buffer, output_dir, f"{chunk_id}_part_{start}_to_{end}")

    print("🎉 所有物理遮挡重测完成，终极数据集构建完毕！")


if __name__ == "__main__":
    run()
