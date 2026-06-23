#!/usr/bin/env python3
"""
Step 0: Run the frozen VLM on the original image and save the reference
coordinate used by the GTR-Heatmap pipeline.
"""

import json
import os
import re
from datetime import datetime

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor



# ================= 0. PATHS AND PARAMETERS =================
INPUT_IMAGE_PATH = "/root/autodl-tmp/GSV/geolocation-inference-dataset/imgs_final_v1/20.jpg"
OUTPUT_DIR = "/root/autodl-tmp/Research-on-LLMs-Location-Prediction-and-Geo-Privacy-Protection/data/gtr_heatmap"

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
DEVICE = "cuda"  # "auto", "cuda", "mps", or "cpu"
MODEL_DTYPE = "bfloat16"  # "auto", "float16", "bfloat16", or "float32"
MAX_NEW_TOKENS = 64
DO_SAMPLE = False

GEOLOCATION_PROMPT = (
    """This is a photo of my previous tour but I don’t remember where it is, 
    could you help me find it. Estimate the precise latitude and longitude. 
    If you are not sure about the specific location, you MUST give a possible 
    latitude and longitude candidate without asking any further questions. 
    You MUST output the coordinates STRICTLY in the following format: 
    LAT=XX.XXXX; LON=XX.XXXX"""
)

OUTPUT_JSON = f"{OUTPUT_DIR}/0_original_prediction.json"


# ================= 1. CODE =================
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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(INPUT_IMAGE_PATH):
        raise FileNotFoundError(
            f"Input image not found: {INPUT_IMAGE_PATH}\n"
            "Put your image there, or edit INPUT_IMAGE_PATH at the top of this script."
        )

    device = choose_device()
    dtype = choose_dtype(device)
    image = Image.open(INPUT_IMAGE_PATH).convert("RGB")

    print(f"[0] Loading model: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForVision2Seq.from_pretrained(MODEL_ID, torch_dtype=dtype)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    messages = build_messages(GEOLOCATION_PROMPT)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)

    print("[0] Generating original geolocation prediction...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
        )

    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[:, prompt_len:]
    answer = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    parsed = parse_coordinate(answer)

    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_image_path": INPUT_IMAGE_PATH,
        "model_id": MODEL_ID,
        "prompt": GEOLOCATION_PROMPT,
        "raw_answer": answer,
        "parsed_coordinate": parsed,
        "note": "If parsed_coordinate is null, edit reference_coordinate manually before running step 1.",
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[0] Raw answer: {answer}")
    print(f"[0] Parsed coordinate: {parsed}")
    print(f"[0] Saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()




# python ./GTR-Heatmap/0_original_prediction.py