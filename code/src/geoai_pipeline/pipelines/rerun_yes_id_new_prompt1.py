import os
import time
from pathlib import Path

from datasets import load_from_disk
from tqdm import tqdm

from geoai_pipeline.config import get_env, get_float, get_int, get_path
from geoai_pipeline.constants import YES_ID_RERUN_PROMPT
from geoai_pipeline.tools.dataset_io import save_chunk
from geoai_pipeline.tools.gemini import gemini_predict_latlon_with_output
from geoai_pipeline.tools.genai_client import create_genai_client
from geoai_pipeline.tools.geo import haversine_km


REPO_ROOT = Path(__file__).resolve().parents[4]


def _extract_reasoning(raw_response: str) -> str:
    if not raw_response:
        return ""
    marker = "REASONING:"
    upper_text = raw_response.upper()
    marker_index = upper_text.find(marker)
    if marker_index == -1:
        return raw_response.strip()
    return raw_response[marker_index + len(marker) :].strip()


def _get_end_index(total_len: int) -> int:
    end_value = get_env("YES_ID_RERUN1_END_INDEX", "")
    if not end_value:
        return total_len
    try:
        return min(int(end_value), total_len)
    except ValueError:
        return total_len


def run():
    gemini_model = get_env("GEMINI_MODEL", "gemini-3-flash-preview")

    input_dataset_path = get_path("YES_ID_RERUN1_INPUT_DATASET_PATH", str(REPO_ROOT / "data" / "YES_id"))
    output_dir = get_path("YES_ID_RERUN1_OUTPUT_DIR", str(REPO_ROOT / "data" / "YES_id_new_prompt1"))

    start_index = get_int("YES_ID_RERUN1_START_INDEX", 0)
    buffer_size = get_int("YES_ID_RERUN1_BUFFER_SIZE", 30)
    sleep_seconds = get_int("YES_ID_RERUN1_SLEEP_SECONDS", 15)
    temperature = get_float("YES_ID_RERUN1_GEMINI_TEMPERATURE", 0.0)
    chunk_start_id = get_int("YES_ID_RERUN1_CHUNK_START_ID", 0)

    print(
        "Current params: "
        f"YES_ID_RERUN1_START_INDEX={start_index}, YES_ID_RERUN1_BUFFER_SIZE={buffer_size}, "
        f"YES_ID_RERUN1_SLEEP_SECONDS={sleep_seconds}, GEMINI_MODEL={gemini_model}, "
        f"YES_ID_RERUN1_INPUT_DATASET_PATH={input_dataset_path}, YES_ID_RERUN1_OUTPUT_DIR={output_dir}"
    )

    if not os.path.exists(input_dataset_path):
        print(f"Input dataset not found: {input_dataset_path}")
        return

    client = create_genai_client("YES_ID_RERUN1_GEMINI_API_KEY")
    dataset = load_from_disk(input_dataset_path)

    total_len = len(dataset)
    end_index = _get_end_index(total_len)
    start_index = max(start_index, 0)

    if start_index >= end_index:
        print(f"Invalid range: start_index={start_index}, end_index={end_index}")
        return

    dataset = dataset.select(range(start_index, end_index))
    print(f"Loaded YES_id range [{start_index}:{end_index}], rows={len(dataset)}")

    buffer = []
    chunk_id = chunk_start_id

    for local_index, item in enumerate(tqdm(dataset, desc="Rerun YES_id with new prompt")):
        global_index = start_index + local_index
        sample_id = item.get("alignment_id") or f"yes_id_{global_index:06d}"
        lat_true = float(item["latitude"])
        lon_true = float(item["longitude"])

        lat_pred, lon_pred, raw_response = gemini_predict_latlon_with_output(
            client=client,
            model=gemini_model,
            image_obj=item["image"],
            prompt=YES_ID_RERUN_PROMPT,
            temperature=temperature,
        )

        if lat_pred == 0.0 and lon_pred == 0.0:
            dist = 99999.0
        else:
            dist = haversine_km(lat_pred, lon_pred, lat_true, lon_true)

        model_reasoning = _extract_reasoning(raw_response)

        buffer.append(
            {
                "sample_id": sample_id,
                "image": item["image"],
                "latitude_true": lat_true,
                "longitude_true": lon_true,
                "latitude_pred": lat_pred,
                "longitude_pred": lon_pred,
                "d": dist,
                "model_chain_of_thought": model_reasoning,
                "model_reasoning": model_reasoning,
                "model_raw_output": raw_response,
            }
        )

        if len(buffer) >= buffer_size:
            save_chunk(buffer, output_dir, f"{chunk_id}_part_{start_index}_to_{end_index}")
            buffer.clear()
            chunk_id += 1

        time.sleep(sleep_seconds)

    if buffer:
        save_chunk(buffer, output_dir, f"{chunk_id}_part_{start_index}_to_{end_index}")

    print("YES_id rerun completed.")


if __name__ == "__main__":
    run()

# python ./code/rerun_yes_id_new_prompt_1.py
