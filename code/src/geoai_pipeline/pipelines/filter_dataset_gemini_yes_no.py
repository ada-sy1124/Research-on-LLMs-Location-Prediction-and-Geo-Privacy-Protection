import gc
import os
import re
import time

from datasets import load_dataset
from google.genai import types
from tqdm import tqdm

from geoai_pipeline.config import get_env, get_float, get_int, get_path
from geoai_pipeline.constants import GEO_PROMPT
from geoai_pipeline.tools.dataset_io import save_chunk
from geoai_pipeline.tools.genai_client import create_genai_client
from geoai_pipeline.tools.geo import haversine_km


def gemini_predict_latlon_and_reason(client, model, image_obj, prompt, temperature: float):
    """保持原脚本行为：返回 lat, lon, reason_text, reason_classes, q。"""
    max_retries = 5
    base_wait_time = 5

    text = ""
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[image_obj, prompt],
                config=types.GenerateContentConfig(temperature=temperature),
            )
            text = (resp.text or "").strip()
            if text:
                break
        except Exception as e:  # noqa: BLE001
            error_msg = str(e)
            if "503" in error_msg or "429" in error_msg:
                wait_time = base_wait_time * (attempt + 1)
                print(f"⚠️ 服务器繁忙 (503/429)，正在进行第 {attempt + 1}/{max_retries} 次重试，等待 {wait_time} 秒...")
                time.sleep(wait_time)
            else:
                print(f"❌ API 致命错误: {e}")
                return 0.0, 0.0, "", [], 0

    if not text:
        print("❌ 多次重试失败，跳过此图")
        return 0.0, 0.0, "", [], 0

    lines = [l.strip() for l in text.splitlines() if l.strip()]

    lat_val, lon_val = 0.0, 0.0
    reason_text = ""
    classes = []
    q = 0

    for line in lines:
        if "COORDINATES:" in line.upper():
            try:
                nums = re.findall(r"-?\d+\.?\d*", line)
                valid_nums = []
                for n in nums:
                    try:
                        valid_nums.append(float(n))
                    except ValueError:
                        continue
                if len(valid_nums) >= 2:
                    lat_val = valid_nums[0]
                    lon_val = valid_nums[1]
            except Exception as e:  # noqa: BLE001
                print(f"坐标解析异常: {line} | Error: {e}")
        elif line.startswith("REASONING:"):
            try:
                parts = line.split("REASONING:", 1)
                if len(parts) > 1:
                    reason_text = parts[1].strip()
                    for seg in reason_text.split(";"):
                        seg = seg.strip()
                        if ":" in seg:
                            cls, objs_text = seg.split(":", 1)
                            cls = cls.strip()
                            if cls:
                                classes.append(cls)
                            objs = [o.strip() for o in objs_text.split(",") if o.strip()]
                            q += len(objs)
            except Exception as e:  # noqa: BLE001
                print(f"Reasoning解析异常: {line} | Error: {e}")

    return lat_val, lon_val, reason_text, classes, q


def run():
    gemini_model = get_env("GEMINI_MODEL", "gemini-3-flash-preview")
    dataset_name = get_env("DATASET_NAME", "stochastic/random_streetview_images_pano_v0.0.2")

    start_index = get_int("FILTER_START_INDEX", 1080)
    end_index = get_int("FILTER_END_INDEX", 5527)
    batch_size = get_int("FILTER_BATCH_SIZE", 0)

    yes_dir = get_path("YES_DIR", "./data/YES")
    no_dir = get_path("NO_DIR", "./data/NO")

    buffer_size = get_int("FILTER_BUFFER_SIZE", 30)
    dist_threshold_km = get_float("FILTER_DIST_THRESHOLD_KM", 5.0)
    sleep_seconds = get_int("FILTER_SLEEP_SECONDS", 20)
    temperature = get_float("GEMINI_TEMPERATURE", 0.0)

    yes_chunk_start_id = get_int("FILTER_YES_CHUNK_START_ID", 8)
    no_chunk_start_id = get_int("FILTER_NO_CHUNK_START_ID", 28)

    print(
        "🔧 当前生效参数: "
        f"FILTER_START_INDEX={start_index}, FILTER_END_INDEX={end_index}, FILTER_BATCH_SIZE={batch_size}, "
        f"FILTER_BUFFER_SIZE={buffer_size}, FILTER_SLEEP_SECONDS={sleep_seconds}, "
        f"FILTER_DIST_THRESHOLD_KM={dist_threshold_km}, GEMINI_MODEL={gemini_model}, "
        f"YES_DIR={yes_dir}, NO_DIR={no_dir}"
    )

    client = create_genai_client("GEMINI_API_KEY")

    os.makedirs(yes_dir, exist_ok=True)
    os.makedirs(no_dir, exist_ok=True)

    yes_buffer, no_buffer = [], []
    yes_chunk_id, no_chunk_id = yes_chunk_start_id, no_chunk_start_id

    total_count = max(0, end_index - start_index)
    use_load_batches = batch_size > 0 and batch_size < total_count

    if use_load_batches:
        split_ranges = [
            (batch_start, min(batch_start + batch_size, end_index))
            for batch_start in range(start_index, end_index, batch_size)
        ]
    else:
        split_ranges = [(start_index, end_index)]

    for range_start, range_end in split_ranges:
        current_split = f"train[{range_start}:{range_end}]"
        print(f"\n开始处理数据范围: {range_start} 到 {range_end} (Split: {current_split})")
        dataset = load_dataset(dataset_name, split=current_split)

        for offset, item in enumerate(tqdm(dataset, desc=f"Range {range_start}-{range_end}")):
            global_index = range_start + offset
            lat_true = float(item["latitude"])
            lon_true = float(item["longitude"])

            lat_pred, lon_pred, reason_text, reason_classes, q = gemini_predict_latlon_and_reason(
                client, gemini_model, item["image"], GEO_PROMPT, temperature
            )

            if lat_pred == 0.0 and lon_pred == 0.0 and q == 0:
                dist = 99999.0
            else:
                dist = haversine_km(lat_pred, lon_pred, lat_true, lon_true)

            label = "YES" if dist <= dist_threshold_km else "NO"

            item_out = {
                "image": item["image"],
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

            if global_index % 10 == 0:
                tqdm.write(f"已处理到全局索引: {global_index}")

            time.sleep(sleep_seconds)

        del dataset
        gc.collect()
        print(f"✅ 数据范围 {range_start}-{range_end} 完成，内存已清理。")

    if yes_buffer:
        save_chunk(yes_buffer, yes_dir, yes_chunk_id)
    if no_buffer:
        save_chunk(no_buffer, no_dir, no_chunk_id)

    print("🎉 所有数据处理完成!")


if __name__ == "__main__":
    run()
