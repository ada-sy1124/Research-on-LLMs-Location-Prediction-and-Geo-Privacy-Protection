#!/usr/bin/env python3
"""
Step 4: Run deletion evaluation on original and hard-masked heatmap images.
"""

import csv
import json
import math
import os
import re
from datetime import datetime

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


# ================= 0. PATHS AND PARAMETERS =================
RENDERED_OUTPUTS_JSON = "/Applications/Documents/geoai/Research-on-LLMs-Location-Prediction-and-Geo-Privacy-Protection/data/gtr_heatmap/3_rendered_outputs.json"
OUTPUT_DIR = "/Applications/Documents/geoai/Research-on-LLMs-Location-Prediction-and-Geo-Privacy-Protection/data/gtr_heatmap"

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
DEVICE = "auto"  # "auto", "cuda", "mps", or "cpu"
MODEL_DTYPE = "auto"  # "auto", "float16", "bfloat16", or "float32"
MAX_NEW_TOKENS = 64
DO_SAMPLE = False

GEOLOCATION_PROMPT = (
    "Return the most likely location of this image as coordinates in the format:\n"
    "LAT=+DD.DDDD;LON=+DDD.DDDD\n"
    "Only output the coordinate string."
)

OUTPUT_JSON = f"{OUTPUT_DIR}/4_deletion_evaluation.json"
OUTPUT_CSV = f"{OUTPUT_DIR}/4_deletion_evaluation.csv"

# ================= 1. CODE =================

EARTH_RADIUS_KM = 6371.0088
COORD_RE = re.compile(
    r"LAT\s*=\s*([+-]?\d+(?:\.\d+)?)\s*;\s*LON\s*=\s*([+-]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def choose_device() -> str:
    if DEVICE != "auto":
        return DEVICE
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def choose_dtype(device: str):
    if MODEL_DTYPE == "float16":
        return torch.float16
    if MODEL_DTYPE == "bfloat16":
        return torch.bfloat16
    if MODEL_DTYPE == "float32":
        return torch.float32
    if device == "cuda":
        return torch.float16
    return torch.float32


def build_messages(prompt: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def parse_coordinate(text: str):
    match = COORD_RE.search(text)
    if not match:
        return None
    lat = float(match.group(1))
    lon = float(match.group(2))
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    return {"lat": lat, "lon": lon}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def generate_coordinate(model, processor, image_path: str, device: str) -> tuple[str, dict | None]:
    image = Image.open(image_path).convert("RGB")
    text = processor.apply_chat_template(build_messages(GEOLOCATION_PROMPT), tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
        )
    prompt_len = inputs["input_ids"].shape[1]
    answer = processor.batch_decode(generated_ids[:, prompt_len:], skip_special_tokens=True)[0].strip()
    return answer, parse_coordinate(answer)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RENDERED_OUTPUTS_JSON, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    reference = manifest.get("reference")
    if not reference:
        raise ValueError("No reference coordinate found in rendered output manifest.")
    ref_lat = float(reference["lat"])
    ref_lon = float(reference["lon"])

    device = choose_device()
    dtype = choose_dtype(device)
    print(f"[4] Loading model: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForVision2Seq.from_pretrained(MODEL_ID, torch_dtype=dtype)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    eval_items = [
        {
            "name": "original_resized",
            "top_area_ratio": 0.0,
            "image_path": f"{OUTPUT_DIR}/3_original_resized.png",
        }
    ]
    for item in manifest["top_area_outputs"]:
        eval_items.append(
            {
                "name": f"gtr_top_{int(round(item['top_area_ratio'] * 100)):02d}",
                "top_area_ratio": item["top_area_ratio"],
                "image_path": item["hard_masked_image_path"],
            }
        )

    rows = []
    for item in eval_items:
        print(f"[4] Evaluating: {item['name']}")
        answer, parsed = generate_coordinate(model, processor, item["image_path"], device)
        distance = None
        if parsed is not None:
            distance = haversine_km(ref_lat, ref_lon, parsed["lat"], parsed["lon"])
        row = {
            "name": item["name"],
            "top_area_ratio": item["top_area_ratio"],
            "image_path": item["image_path"],
            "raw_answer": answer,
            "parsed_lat": None if parsed is None else parsed["lat"],
            "parsed_lon": None if parsed is None else parsed["lon"],
            "distance_from_reference_km": distance,
        }
        rows.append(row)
        print(f"[4] {item['name']} answer={answer} distance_km={distance}")

    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_id": MODEL_ID,
        "prompt": GEOLOCATION_PROMPT,
        "reference": reference,
        "rows": rows,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[4] Saved: {OUTPUT_JSON}")
    print(f"[4] Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
