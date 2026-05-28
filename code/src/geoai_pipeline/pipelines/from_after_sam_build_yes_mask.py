import io
import os
import time

import numpy as np
from datasets import load_from_disk
from PIL import Image
from tqdm import tqdm

from geoai_pipeline.config import get_env, get_int, get_path
from geoai_pipeline.constants import GEO_PROMPT
from geoai_pipeline.tools.dataset_io import save_chunk
from geoai_pipeline.tools.geo import haversine_km
from geoai_pipeline.tools.model_backend import create_inference_model, predict_latlon


def _key(prefix: str, name: str) -> str:
    return f"{prefix}_{name}" if prefix else name


def _get_path(prefix: str, name: str, fallback_name: str, default: str) -> str:
    if prefix:
        return get_path(_key(prefix, name), get_env(fallback_name, default))
    return get_path(fallback_name, default)


def _get_int(prefix: str, name: str, fallback_name: str, default: int) -> int:
    if prefix:
        return get_int(_key(prefix, name), get_int(fallback_name, default))
    return get_int(fallback_name, default)


def _to_pil_image(masked_img):
    if isinstance(masked_img, dict) and "bytes" in masked_img:
        return Image.open(io.BytesIO(masked_img["bytes"]))
    if isinstance(masked_img, Image.Image):
        return masked_img
    return Image.fromarray(np.array(masked_img))


def run(env_prefix: str = ""):
    prefix = env_prefix.strip().upper()
    label = prefix or "YES_MASK"

    model_backend = get_env("MODEL_BACKEND", "local")
    input_dataset_path = _get_path(
        prefix,
        "AFTER_SAM_INPUT_DATASET_PATH",
        "AFTER_SAM_INPUT_DATASET_PATH",
        "./data/YES_NEW_afterSAM",
    )
    output_dir = _get_path(
        prefix,
        "YES_MASK_OUTPUT_DIR",
        "YES_MASK_OUTPUT_DIR",
        "./data/YES_Mask",
    )

    start_index = _get_int(prefix, "START_INDEX", "START_INDEX", 0)
    end_index = _get_int(prefix, "END_INDEX", "END_INDEX", 1030)
    buffer_size = _get_int(prefix, "BUFFER_SIZE", "BUFFER_SIZE", 30)
    sleep_seconds = _get_int(prefix, "SLEEP_SECONDS", "SLEEP_SECONDS", 0)
    chunk_start_id = _get_int(prefix, "CHUNK_START_ID", "CHUNK_START_ID", 0)

    print(
        "Current params: "
        f"{label}_START_INDEX={start_index}, {label}_END_INDEX={end_index}, "
        f"{label}_BUFFER_SIZE={buffer_size}, {label}_SLEEP_SECONDS={sleep_seconds}, "
        f"{label}_CHUNK_START_ID={chunk_start_id}, MODEL_BACKEND={model_backend}, "
        f"AFTER_SAM_INPUT_DATASET_PATH={input_dataset_path}, YES_MASK_OUTPUT_DIR={output_dir}"
    )

    if not os.path.exists(input_dataset_path):
        print(f"Input path not found: {input_dataset_path}")
        return

    model = create_inference_model()
    dataset = load_from_disk(input_dataset_path)

    total_len = len(dataset)
    start = max(start_index, 0)
    end = min(end_index if end_index is not None else total_len, total_len)

    if start >= end:
        print(f"Invalid range: START_INDEX ({start}) >= END_INDEX ({end})")
        return

    dataset = dataset.select(range(start, end))
    print(f"Selected dataset range: [{start}:{end}], rows={len(dataset)}")

    buffer = []
    chunk_id = chunk_start_id

    for local_index, item in enumerate(tqdm(dataset, desc="Evaluating masked images")):
        sample_id = str(item.get("sample_id") or f"after_sam_{start + local_index:06d}")
        source_index_value = item.get("source_index")
        source_index = int(source_index_value) if source_index_value is not None else -1
        image_obj = item["image_original"]
        lat_true = item["latitude"]
        lon_true = item["longitude"]
        d_orig = item["d_original"]

        ablated_class = item.get("ablated_class", [])
        masked_image = item.get("masked_image", [])
        q_ratio = item.get("q_ratio", [])

        if not masked_image or len(masked_image) != len(ablated_class):
            print("\n[dirty row skipped] masked_image and ablated_class length mismatch.")
            continue

        d_prime_list = []
        d_diff_list = []

        for masked_img in masked_image:
            lat_pred, lon_pred = predict_latlon(model, _to_pil_image(masked_img), GEO_PROMPT)

            if lat_pred == 0.0 and lon_pred == 0.0:
                d_prime = 99999.0
                d_diff = 99999.0
            else:
                d_prime = haversine_km(lat_pred, lon_pred, lat_true, lon_true)
                d_diff = d_prime - d_orig

            d_prime_list.append(d_prime)
            d_diff_list.append(d_diff)
            time.sleep(sleep_seconds)

        buffer.append(
            {
                "sample_id": sample_id,
                "source_index": source_index,
                "image": image_obj,
                "latitude_true": lat_true,
                "longitude_true": lon_true,
                "d_original": d_orig,
                "ablated_class": ablated_class,
                "q_ratio": q_ratio,
                "d_prime": d_prime_list,
                "d_diff": d_diff_list,
            }
        )

        if len(buffer) >= buffer_size:
            save_chunk(buffer, output_dir, f"{chunk_id}_part_{start}_to_{end}")
            buffer.clear()
            chunk_id += 1

    if buffer:
        save_chunk(buffer, output_dir, f"{chunk_id}_part_{start}_to_{end}")

    print("Masked-image reevaluation completed.")


if __name__ == "__main__":
    run()
