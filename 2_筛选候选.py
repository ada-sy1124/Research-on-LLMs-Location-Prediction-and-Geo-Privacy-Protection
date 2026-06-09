import json
import os
import re
from collections import defaultdict

import numpy as np
import torch
from PIL import Image, ImageDraw


# ================= 0. 路径和参数设置 =================
IMAGE_PATH = "/root/autodl-tmp/data/im2gps3ktest/im2gps3ktest/167345423_ba2914f302_59_79954435@N00.jpg"
RAW_MASKS_FILE = "/root/autodl-tmp/DCSD分层提取凸包/中间件/02_raw_masks/raw_masks_dict.pt"
RAW_REGIONS_FILE = "/root/autodl-tmp/DCSD分层提取凸包/中间件/02_raw_masks/raw_regions.json"
OUTPUT_DIR = "/root/autodl-tmp/DCSD分层提取凸包/中间件/03_balanced_masks"
BALANCED_MASKS_FILE = f"{OUTPUT_DIR}/masks_dict.pt"
BALANCED_REGIONS_FILE = f"{OUTPUT_DIR}/balanced_regions.json"

# 小于该面积比例的原始 mask 不单独作为候选，优先合并。
SMALL_AREA_RATIO = 0.015
# 大于该面积比例的候选保留为大区域候选。
LARGE_AREA_RATIO = 0.18
# 过度重叠去重阈值。
DEDUP_IOU = 0.85
# 小目标抱团后如果仍低于该面积，继续并入对应层级 bundle。
MIN_GROUP_AREA_RATIO = 0.012


def normalize_base_noun(phrase):
    words = re.findall(r"[a-zA-Z]+", phrase.lower())
    stop_modifiers = {
        "red", "black", "white", "gray", "grey", "blue", "green", "yellow",
        "silver", "glass", "concrete", "wooden", "metal", "tall", "short",
        "large", "small", "front", "rear", "side", "overhead", "distant",
        "nearby", "central", "left", "right", "wet", "dry",
    }
    words = [word for word in words if word not in stop_modifiers]
    if not words:
        return phrase.lower().strip()
    if words[-1].endswith("ies"):
        words[-1] = words[-1][:-3] + "y"
    elif words[-1].endswith("s") and len(words[-1]) > 3:
        words[-1] = words[-1][:-1]
    return words[-1]


def mask_iou(a, b):
    inter = (a.bool() & b.bool()).sum().item()
    union = (a.bool() | b.bool()).sum().item()
    return inter / union if union else 0.0


def merge_masks(items, raw_masks):
    merged = None
    for item in items:
        mask = raw_masks[item["key"]].bool()
        merged = mask.clone() if merged is None else (merged | mask)
    return merged.float()


def area_ratio(mask, image_area):
    return float(mask.sum().item()) / float(image_area)


def deduplicate_candidates(candidates, image_area):
    kept = []
    for candidate in sorted(candidates, key=lambda item: item["area_ratio"], reverse=True):
        duplicate = False
        for old in kept:
            if mask_iou(candidate["mask"], old["mask"]) >= DEDUP_IOU:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return sorted(kept, key=lambda item: item["area_ratio"], reverse=True)


def draw_overlay(image, candidates, output_path):
    arr = np.asarray(image).astype("float32")
    overlay = arr.copy()
    rng = np.random.default_rng(2026)
    draw_img = Image.fromarray(arr.astype("uint8"))
    draw = ImageDraw.Draw(draw_img)

    for index, candidate in enumerate(candidates):
        color = rng.integers(40, 255, size=3).astype("float32")
        mask = candidate["mask"].bool().numpy()
        overlay[mask] = overlay[mask] * 0.45 + color * 0.55
        yx = np.argwhere(mask)
        if yx.size:
            y1, x1 = yx.min(axis=0)
            y2, x2 = yx.max(axis=0)
            draw.rectangle([int(x1), int(y1), int(x2), int(y2)], outline=tuple(color.astype("uint8")), width=2)
            draw.text((int(x1), int(y1)), str(index), fill=tuple(color.astype("uint8")))

    overlay_img = Image.fromarray(overlay.clip(0, 255).astype("uint8"))
    overlay_img.save(output_path)


def blackout_all(image, candidates, output_path):
    arr = np.asarray(image).astype("float32")
    merged = None
    for candidate in candidates:
        mask = candidate["mask"].bool()
        merged = mask.clone() if merged is None else (merged | mask)
    if merged is not None:
        arr[merged.numpy()] = 0
    Image.fromarray(arr.clip(0, 255).astype("uint8")).save(output_path)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image = Image.open(IMAGE_PATH).convert("RGB")
    image_area = image.width * image.height
    raw_masks = torch.load(RAW_MASKS_FILE, map_location="cpu")
    with open(RAW_REGIONS_FILE, "r", encoding="utf-8") as f:
        raw_regions = json.load(f)

    detections = raw_regions["detections"]
    by_key = {item["key"]: item for item in detections}

    large_or_mid = []
    small = []
    for item in detections:
        ratio = float(item["area_ratio"])
        if ratio < SMALL_AREA_RATIO:
            small.append(item)
        else:
            large_or_mid.append(item)

    candidates = []
    for item in large_or_mid:
        mask = raw_masks[item["key"]].float()
        candidates.append(
            {
                "name": item["key"],
                "source_keys": [item["key"]],
                "level": item["level"],
                "phrase": item["phrase"],
                "merge_type": "single_large" if item["area_ratio"] >= LARGE_AREA_RATIO else "single_mid",
                "mask": mask,
                "area_ratio": area_ratio(mask, image_area),
            }
        )

    small_by_base = defaultdict(list)
    for item in small:
        small_by_base[normalize_base_noun(item["phrase"])].append(item)

    leftovers_by_level = defaultdict(list)
    for base, items in small_by_base.items():
        merged = merge_masks(items, raw_masks)
        merged_ratio = area_ratio(merged, image_area)
        if len(items) >= 2 or merged_ratio >= MIN_GROUP_AREA_RATIO:
            level_names = sorted({item["level"] for item in items})
            candidates.append(
                {
                    "name": f"group_{base}",
                    "source_keys": [item["key"] for item in items],
                    "level": "+".join(level_names),
                    "phrase": f"grouped {base}",
                    "merge_type": "small_same_base_group",
                    "mask": merged,
                    "area_ratio": merged_ratio,
                }
            )
        else:
            leftovers_by_level[items[0]["level"]].extend(items)

    for level, items in leftovers_by_level.items():
        if not items:
            continue
        merged = merge_masks(items, raw_masks)
        candidates.append(
            {
                "name": f"{level}_small_bundle",
                "source_keys": [item["key"] for item in items],
                "level": level,
                "phrase": f"{level} small bundled anchors",
                "merge_type": "small_level_bundle",
                "mask": merged,
                "area_ratio": area_ratio(merged, image_area),
            }
        )

    candidates = deduplicate_candidates(candidates, image_area)

    masks_dict = {candidate["name"]: candidate["mask"].cpu() for candidate in candidates}
    torch.save(masks_dict, BALANCED_MASKS_FILE)

    report_candidates = []
    for index, candidate in enumerate(candidates):
        report_candidates.append(
            {
                "index": index,
                "name": candidate["name"],
                "phrase": candidate["phrase"],
                "level": candidate["level"],
                "merge_type": candidate["merge_type"],
                "area_ratio": candidate["area_ratio"],
                "source_keys": candidate["source_keys"],
                "source_phrases": [by_key[key]["phrase"] for key in candidate["source_keys"] if key in by_key],
            }
        )

    report = {
        "image_path": IMAGE_PATH,
        "raw_masks_file": RAW_MASKS_FILE,
        "balanced_masks_file": BALANCED_MASKS_FILE,
        "small_area_ratio": SMALL_AREA_RATIO,
        "large_area_ratio": LARGE_AREA_RATIO,
        "min_group_area_ratio": MIN_GROUP_AREA_RATIO,
        "dedup_iou": DEDUP_IOU,
        "num_raw_detections": len(detections),
        "num_balanced_candidates": len(candidates),
        "candidates": report_candidates,
    }
    with open(BALANCED_REGIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    draw_overlay(image, candidates, os.path.join(OUTPUT_DIR, "balanced_candidates_overlay.jpg"))
    blackout_all(image, candidates, os.path.join(OUTPUT_DIR, "balanced_candidates_blackout_all.jpg"))

    print(f"原始候选数量: {len(detections)}")
    print(f"平衡候选数量: {len(candidates)}")
    print(f"已保存 masks: {BALANCED_MASKS_FILE}")
    print(f"已保存报告: {BALANCED_REGIONS_FILE}")
    print(f"已保存候选预览: {os.path.join(OUTPUT_DIR, 'balanced_candidates_overlay.jpg')}")


if __name__ == "__main__":
    main()



# python ./2_筛选候选.py