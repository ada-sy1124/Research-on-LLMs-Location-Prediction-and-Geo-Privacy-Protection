#!/usr/bin/env python3
"""
Step 2: Optimize a continuous GTR-driven geolocation heatmap.

This MVP is written for HuggingFace VLMs with dense image tensors shaped like
[B, 3, H, W], such as LLaVA-style models. That keeps the gradient path:
A -> S -> rendered image -> pixel_values -> VLM logits -> GTR loss.
"""

import csv
import json
import math
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


# ================= 0. PATHS AND PARAMETERS =================
INPUT_IMAGE_PATH = "/Applications/Documents/geoai/Research-on-LLMs-Location-Prediction-and-Geo-Privacy-Protection/data/input/example.jpg"
COORDINATE_BANK_JSON = "/Applications/Documents/geoai/Research-on-LLMs-Location-Prediction-and-Geo-Privacy-Protection/data/gtr_heatmap/1_coordinate_bank.json"
OUTPUT_DIR = "/Applications/Documents/geoai/Research-on-LLMs-Location-Prediction-and-Geo-Privacy-Protection/data/gtr_heatmap"

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
DEVICE = "auto"  # "auto", "cuda", "mps", or "cpu"
MODEL_DTYPE = "auto"  # "auto", "float16", "bfloat16", or "float32"

GEOLOCATION_PROMPT = (
    "Return the most likely location of this image as coordinates in the format:\n"
    "LAT=+DD.DDDD;LON=+DDD.DDDD\n"
    "Only output the coordinate string."
)

IMAGE_SIZE = 336
MASK_LOW_RES_H = 21
MASK_LOW_RES_W = 21
INITIAL_MASK_LOGIT = -4.0
OPTIMIZATION_STEPS = 100
LEARNING_RATE = 0.05
CANDIDATE_BATCH_SIZE = 4
SOFTMAX_TEMPERATURE = 1.0
LAMBDA_AREA = 0.10
LAMBDA_TV = 0.03
USE_BANK_LOSS = False
BANK_LOSS_BETA = 0.001
BLUR_KERNEL_SIZE = 31
SAVE_EVERY_STEPS = 10
RANDOM_SEED = 2026

OUTPUT_PT = f"{OUTPUT_DIR}/2_heatmap_result.pt"
OUTPUT_LOG_CSV = f"{OUTPUT_DIR}/2_optimization_log.csv"
OUTPUT_HEATMAP_PNG = f"{OUTPUT_DIR}/2_heatmap.png"
OUTPUT_OVERLAY_PNG = f"{OUTPUT_DIR}/2_heatmap_overlay.png"
OUTPUT_SOFT_MASKED_PNG = f"{OUTPUT_DIR}/2_soft_masked_image.png"

# ================= 1. CODE =================

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


def build_messages(prompt: str, target: str | None = None):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": target})
    return messages


def load_image_tensor(path: str, size: int, device: str) -> tuple[Image.Image, torch.Tensor]:
    pil = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    raw = torch.ByteTensor(torch.ByteStorage.from_buffer(pil.tobytes()))
    tensor = raw.view(size, size, 3).permute(2, 0, 1).float().div(255.0)
    return pil, tensor.unsqueeze(0).to(device)


def blur_image_tensor(image: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size % 2 == 0:
        raise ValueError("BLUR_KERNEL_SIZE must be odd.")
    pad = kernel_size // 2
    return F.avg_pool2d(F.pad(image, (pad, pad, pad, pad), mode="reflect"), kernel_size, stride=1)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    x = image.detach().float().cpu().clamp(0, 1)[0]
    x = (x.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(x)


def colorize_heatmap(mask: torch.Tensor) -> Image.Image:
    m = mask.detach().float().cpu().clamp(0, 1)
    if m.ndim == 4:
        m = m[0, 0]
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    red = m
    green = 1.0 - (m - 0.5).abs() * 2.0
    blue = 1.0 - m
    rgb = torch.stack([red, green.clamp(0, 1), blue], dim=-1)
    arr = (rgb.numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(arr)


def save_overlay(image: torch.Tensor, mask: torch.Tensor, out_path: str):
    base = tensor_to_pil(image).convert("RGBA")
    heat = colorize_heatmap(mask).resize(base.size).convert("RGBA")
    alpha = mask.detach().float().cpu().clamp(0, 1)[0, 0]
    alpha = (alpha / (alpha.max() + 1e-8) * 150.0).round().byte().numpy()
    heat.putalpha(Image.fromarray(alpha))
    blended = Image.alpha_composite(base, heat)
    blended.convert("RGB").save(out_path)


def get_image_mean_std(processor, device: str):
    image_processor = getattr(processor, "image_processor", None)
    mean = getattr(image_processor, "image_mean", [0.48145466, 0.4578275, 0.40821073])
    std = getattr(image_processor, "image_std", [0.26862954, 0.26130258, 0.27577711])
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)
    return mean_t, std_t


def make_model_pixel_values(image_0_1: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, dtype):
    return ((image_0_1 - mean) / std).to(dtype=dtype)


def total_variation(mask: torch.Tensor) -> torch.Tensor:
    dy = torch.abs(mask[:, :, 1:, :] - mask[:, :, :-1, :]).mean()
    dx = torch.abs(mask[:, :, :, 1:] - mask[:, :, :, :-1]).mean()
    return dx + dy


def load_bank():
    with open(COORDINATE_BANK_JSON, "r", encoding="utf-8") as f:
        bank = json.load(f)
    texts = [row["coord_text"] for row in bank["candidates"]]
    distances = torch.tensor([float(row["distance_norm"]) for row in bank["candidates"]])
    return bank, texts, distances


def sequence_logprobs_batch(model, processor, pil_image, pixel_values, prompt, targets, device):
    prefix_text = processor.apply_chat_template(
        build_messages(prompt),
        tokenize=False,
        add_generation_prompt=True,
    )
    full_texts = [
        processor.apply_chat_template(
            build_messages(prompt, target),
            tokenize=False,
            add_generation_prompt=False,
        )
        for target in targets
    ]
    tokenizer = getattr(processor, "tokenizer", processor)
    prefix_len = len(tokenizer(prefix_text, add_special_tokens=False)["input_ids"])

    inputs = processor(text=full_texts, images=[pil_image] * len(targets), padding=True, return_tensors="pt").to(device)
    if "pixel_values" not in inputs:
        raise RuntimeError("The selected processor did not return pixel_values.")
    if inputs["pixel_values"].ndim != 4:
        raise RuntimeError(
            f"This MVP expects dense pixel_values [B,3,H,W], got shape {tuple(inputs['pixel_values'].shape)}. "
            "Use a LLaVA-style model, or adapt the visual preprocessing for this VLM."
        )

    inputs["pixel_values"] = pixel_values.repeat(len(targets), 1, 1, 1)
    outputs = model(**inputs)
    logits = outputs.logits
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

    shift_logits = logits[:, :-1, :]
    shift_ids = input_ids[:, 1:]
    shift_attention = attention_mask[:, 1:].bool()
    token_logp = torch.log_softmax(shift_logits, dim=-1).gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)

    target_mask = torch.zeros_like(shift_attention)
    start = max(prefix_len - 1, 0)
    target_mask[:, start:] = True
    target_mask = target_mask & shift_attention
    return (token_logp * target_mask).sum(dim=1)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(RANDOM_SEED)

    device = choose_device()
    dtype = choose_dtype(device)
    pil_image, image = load_image_tensor(INPUT_IMAGE_PATH, IMAGE_SIZE, device)
    blurred = blur_image_tensor(image, BLUR_KERNEL_SIZE)
    bank, coord_texts, distance_norm = load_bank()
    distance_norm = distance_norm.to(device=device, dtype=torch.float32)

    print(f"[2] Loading model: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForVision2Seq.from_pretrained(MODEL_ID, torch_dtype=dtype)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    mean, std = get_image_mean_std(processor, device)
    mean = mean.to(dtype=torch.float32)
    std = std.to(dtype=torch.float32)

    mask_logits = torch.nn.Parameter(
        torch.full((1, 1, MASK_LOW_RES_H, MASK_LOW_RES_W), INITIAL_MASK_LOGIT, device=device, dtype=torch.float32)
    )
    optimizer = torch.optim.Adam([mask_logits], lr=LEARNING_RATE)
    log_rows = []

    print(f"[2] Optimizing with {len(coord_texts)} coordinate candidates for {OPTIMIZATION_STEPS} steps...")
    for step in range(OPTIMIZATION_STEPS):
        optimizer.zero_grad(set_to_none=True)
        soft_mask = torch.sigmoid(mask_logits)
        mask = F.interpolate(soft_mask, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        rendered = image * (1.0 - mask) + blurred * mask
        pixel_values = make_model_pixel_values(rendered, mean, std, dtype)

        logp_chunks = []
        for start in range(0, len(coord_texts), CANDIDATE_BATCH_SIZE):
            batch_targets = coord_texts[start : start + CANDIDATE_BATCH_SIZE]
            batch_logps = sequence_logprobs_batch(
                model=model,
                processor=processor,
                pil_image=pil_image,
                pixel_values=pixel_values,
                prompt=GEOLOCATION_PROMPT,
                targets=batch_targets,
                device=device,
            )
            logp_chunks.append(batch_logps)

        logps = torch.cat(logp_chunks, dim=0)
        q = torch.softmax(logps / SOFTMAX_TEMPERATURE, dim=0)
        risk = (q * distance_norm).sum()
        area_loss = mask.mean()
        tv_loss = total_variation(mask)
        loss = -risk + LAMBDA_AREA * area_loss + LAMBDA_TV * tv_loss
        bank_loss_value = torch.tensor(0.0, device=device)
        if USE_BANK_LOSS:
            bank_loss_value = -torch.logsumexp(logps, dim=0)
            loss = loss + BANK_LOSS_BETA * bank_loss_value

        loss.backward()
        optimizer.step()

        row = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "risk": float(risk.detach().cpu()),
            "area": float(area_loss.detach().cpu()),
            "tv": float(tv_loss.detach().cpu()),
            "bank_loss": float(bank_loss_value.detach().cpu()),
            "mask_min": float(mask.detach().min().cpu()),
            "mask_max": float(mask.detach().max().cpu()),
            "mask_mean": float(mask.detach().mean().cpu()),
        }
        log_rows.append(row)
        if step % SAVE_EVERY_STEPS == 0 or step == OPTIMIZATION_STEPS - 1:
            print(
                f"[2] step={step:03d} loss={row['loss']:.4f} risk={row['risk']:.4f} "
                f"area={row['area']:.4f} tv={row['tv']:.4f} mask_max={row['mask_max']:.4f}"
            )

    with torch.no_grad():
        final_soft_mask = torch.sigmoid(mask_logits)
        final_mask = F.interpolate(final_soft_mask, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        final_rendered = image * (1.0 - final_mask) + blurred * final_mask

    with open(OUTPUT_LOG_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    torch.save(
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "input_image_path": INPUT_IMAGE_PATH,
            "model_id": MODEL_ID,
            "prompt": GEOLOCATION_PROMPT,
            "coordinate_bank_path": COORDINATE_BANK_JSON,
            "reference": bank.get("reference"),
            "config": {
                "image_size": IMAGE_SIZE,
                "mask_low_res_h": MASK_LOW_RES_H,
                "mask_low_res_w": MASK_LOW_RES_W,
                "optimization_steps": OPTIMIZATION_STEPS,
                "learning_rate": LEARNING_RATE,
                "softmax_temperature": SOFTMAX_TEMPERATURE,
                "lambda_area": LAMBDA_AREA,
                "lambda_tv": LAMBDA_TV,
                "use_bank_loss": USE_BANK_LOSS,
                "bank_loss_beta": BANK_LOSS_BETA,
                "blur_kernel_size": BLUR_KERNEL_SIZE,
            },
            "mask": final_mask.detach().cpu(),
            "mask_low_res": final_soft_mask.detach().cpu(),
            "original_image": image.detach().cpu(),
            "blurred_image": blurred.detach().cpu(),
            "soft_masked_image": final_rendered.detach().cpu(),
            "optimization_log": log_rows,
        },
        OUTPUT_PT,
    )

    colorize_heatmap(final_mask).save(OUTPUT_HEATMAP_PNG)
    save_overlay(image, final_mask, OUTPUT_OVERLAY_PNG)
    tensor_to_pil(final_rendered).save(OUTPUT_SOFT_MASKED_PNG)

    print(f"[2] Saved: {OUTPUT_PT}")
    print(f"[2] Saved: {OUTPUT_LOG_CSV}")
    print(f"[2] Saved: {OUTPUT_HEATMAP_PNG}")
    print(f"[2] Saved: {OUTPUT_OVERLAY_PNG}")
    print(f"[2] Saved: {OUTPUT_SOFT_MASKED_PNG}")


if __name__ == "__main__":
    main()
