#!/usr/bin/env python3
"""
Step 3: Render the optimized heatmap into inspectable images and hard deletion
masks with fixed area budgets.
"""

import json
import os
from datetime import datetime

import torch
from PIL import Image


# ================= 0. PATHS AND PARAMETERS =================
HEATMAP_RESULT_PT = "/root/autodl-tmp/Research-on-LLMs-Location-Prediction-and-Geo-Privacy-Protection/data/gtr_heatmap/2_heatmap_result.pt"
OUTPUT_DIR = "/root/autodl-tmp/Research-on-LLMs-Location-Prediction-and-Geo-Privacy-Protection/data/gtr_heatmap"

TOP_AREA_RATIOS = [0.05, 0.10, 0.20]
USE_BLUR_FOR_HARD_MASK = True

OUTPUT_MANIFEST_JSON = f"{OUTPUT_DIR}/3_rendered_outputs.json"

# ================= 1. CODE =================

def safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    x = image.detach().float().cpu().clamp(0, 1)
    if x.ndim == 4:
        x = x[0]
    x = (x.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(x)


def mask_to_pil(mask: torch.Tensor) -> Image.Image:
    m = mask.detach().float().cpu().clamp(0, 1)
    if m.ndim == 4:
        m = m[0, 0]
    arr = (m.numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(arr)


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
    alpha = mask.detach().float().cpu().clamp(0, 1)
    if alpha.ndim == 4:
        alpha = alpha[0, 0]
    alpha = (alpha / (alpha.max() + 1e-8) * 150.0).round().byte().numpy()
    heat.putalpha(Image.fromarray(alpha))
    Image.alpha_composite(base, heat).convert("RGB").save(out_path)


def top_area_binary_mask(mask: torch.Tensor, ratio: float) -> torch.Tensor:
    m = mask.detach().float().cpu()
    if m.ndim == 4:
        m = m[0, 0]
    flat = m.flatten()
    keep = max(1, int(round(flat.numel() * ratio)))
    threshold = torch.topk(flat, keep).values.min()
    binary = (m >= threshold).float()
    return binary.view(1, 1, m.shape[0], m.shape[1])


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = safe_torch_load(HEATMAP_RESULT_PT)
    mask = data["mask"].float()
    original = data["original_image"].float()
    blurred = data["blurred_image"].float()
    replacement = blurred if USE_BLUR_FOR_HARD_MASK else torch.zeros_like(original)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "heatmap_result_pt": HEATMAP_RESULT_PT,
        "input_image_path": data.get("input_image_path"),
        "model_id": data.get("model_id"),
        "reference": data.get("reference"),
        "top_area_outputs": [],
    }

    tensor_to_pil(original).save(f"{OUTPUT_DIR}/3_original_resized.png")
    colorize_heatmap(mask).save(f"{OUTPUT_DIR}/3_heatmap_color.png")
    mask_to_pil(mask).save(f"{OUTPUT_DIR}/3_heatmap_gray.png")
    save_overlay(original, mask, f"{OUTPUT_DIR}/3_heatmap_overlay.png")

    for ratio in TOP_AREA_RATIOS:
        binary = top_area_binary_mask(mask, ratio)
        hard_masked = original * (1.0 - binary) + replacement * binary
        pct = int(round(ratio * 100))
        mask_path = f"{OUTPUT_DIR}/3_top_{pct:02d}_mask.png"
        overlay_path = f"{OUTPUT_DIR}/3_top_{pct:02d}_overlay.png"
        image_path = f"{OUTPUT_DIR}/3_top_{pct:02d}_hard_masked.png"
        mask_to_pil(binary).save(mask_path)
        save_overlay(original, binary, overlay_path)
        tensor_to_pil(hard_masked).save(image_path)
        manifest["top_area_outputs"].append(
            {
                "top_area_ratio": ratio,
                "mask_path": mask_path,
                "overlay_path": overlay_path,
                "hard_masked_image_path": image_path,
                "fill": "blur" if USE_BLUR_FOR_HARD_MASK else "black",
            }
        )

    with open(OUTPUT_MANIFEST_JSON, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[3] Saved manifest: {OUTPUT_MANIFEST_JSON}")
    for item in manifest["top_area_outputs"]:
        print(f"[3] top={item['top_area_ratio']:.0%} image={item['hard_masked_image_path']}")


if __name__ == "__main__":
    main()





# python ./GTR-Heatmap/3_render_heatmap_outputs.py