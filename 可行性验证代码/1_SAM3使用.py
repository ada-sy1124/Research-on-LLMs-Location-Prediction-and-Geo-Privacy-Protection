import json
import os
import re

import torch
import torchvision.transforms.functional as F
from PIL import Image
from transformers import Sam3Model, Sam3Processor


# ================= 0. 路径和参数设置 =================
IMAGE_PATH = "/root/autodl-tmp/data/im2gps3ktest/im2gps3ktest/167345423_ba2914f302_59_79954435@N00.jpg"
FEATURES_FILE = "/root/autodl-tmp/DCSD分层提取凸包/中间件/01_qwen_geolocation.json"
OUTPUT_DIR = "/root/autodl-tmp/DCSD分层提取凸包/中间件/02_raw_masks"
MASKS_FILE = f"{OUTPUT_DIR}/raw_masks_dict.pt"
REGIONS_FILE = f"{OUTPUT_DIR}/raw_regions.json"
SEGMENTED_FILE = "/root/autodl-tmp/DCSD分层提取凸包/中间件/02_segmented_features.json"

SAM3_MODEL_ID = "facebook/sam3"
USE_LOCAL_MODEL = True
USE_BBOX_MASK = False
USE_CONVEX_HULL = True
DILATION_KERNEL_SIZE = 15
DILATION_ITERATIONS = 1
MIN_COMPONENT_AREA = 20
THRESHOLD = 0.5
MASK_THRESHOLD = 0.5

DISALLOWED_TERMS = {
    "sky",
    "cloud",
    "clouds",
    "weather",
    "lighting",
    "sunlight",
    "shadow",
    "shadows",
    "atmosphere",
}

device = "cuda" if torch.cuda.is_available() else "cpu"


def safe_name(text):
    text = re.sub(r"[\\/:*?\"<>|]+", "_", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text[:120]


def is_allowed_phrase(phrase):
    words = re.findall(r"[a-zA-Z]+", phrase.lower())
    return not any(word in DISALLOWED_TERMS for word in words)


def mask_to_box(mask):
    indices = torch.nonzero(mask, as_tuple=False)
    if indices.numel() == 0:
        return None
    y1, x1 = indices.min(dim=0)[0]
    y2, x2 = indices.max(dim=0)[0]
    return [int(x1), int(y1), int(x2), int(y2)]


def box_to_mask(box, reference_mask):
    x1, y1, x2, y2 = box
    box_mask = torch.zeros_like(reference_mask, dtype=torch.bool)
    box_mask[y1:y2 + 1, x1:x2 + 1] = True
    return box_mask


def convex_hull_and_dilate(mask):
    if not USE_CONVEX_HULL and DILATION_KERNEL_SIZE <= 1:
        return mask.bool()

    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "当前设置需要 OpenCV。请先安装 opencv-python，或把 USE_CONVEX_HULL=False 且 DILATION_KERNEL_SIZE=1。"
        ) from exc

    mask_np = mask.detach().cpu().numpy().astype("uint8")
    if USE_CONVEX_HULL:
        num_labels, labels = cv2.connectedComponents(mask_np)
        hull_mask = np.zeros_like(mask_np, dtype="uint8")
        for label in range(1, num_labels):
            component = (labels == label).astype("uint8")
            if int(component.sum()) < MIN_COMPONENT_AREA:
                continue
            contours, _ = cv2.findContours(
                component,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            for contour in contours:
                if len(contour) < 3:
                    hull_mask = np.maximum(hull_mask, component)
                    continue
                hull = cv2.convexHull(contour)
                cv2.drawContours(hull_mask, [hull], -1, color=1, thickness=-1)
        if int(hull_mask.sum()) == 0:
            hull_mask = mask_np
        mask_np = hull_mask

    if DILATION_KERNEL_SIZE > 1 and DILATION_ITERATIONS > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (DILATION_KERNEL_SIZE, DILATION_KERNEL_SIZE),
        )
        mask_np = cv2.dilate(mask_np, kernel, iterations=DILATION_ITERATIONS)

    return torch.from_numpy(mask_np.astype("bool")).to(mask.device)


def load_layered_phrases(data):
    hierarchy = data.get("hierarchy", {})
    items = []
    for level in ["macro_environment", "meso_landmarks", "micro_anchors"]:
        for phrase in hierarchy.get(level, []):
            phrase = str(phrase).strip()
            if not phrase or not is_allowed_phrase(phrase):
                continue
            if phrase not in [item["phrase"] for item in items]:
                items.append({"phrase": phrase, "level": level})
    return items


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(FEATURES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = load_layered_phrases(data)
    if not items:
        raise RuntimeError(f"{FEATURES_FILE} 中没有可用分层目标词，请先运行 0.py")

    print(f"加载 SAM3 模型: {SAM3_MODEL_ID}, local_files_only={USE_LOCAL_MODEL}")
    model = Sam3Model.from_pretrained(
        SAM3_MODEL_ID,
        local_files_only=USE_LOCAL_MODEL,
    ).to(device)
    processor = Sam3Processor.from_pretrained(
        SAM3_MODEL_ID,
        local_files_only=USE_LOCAL_MODEL,
    )

    image = Image.open(IMAGE_PATH).convert("RGB")
    img_tensor = F.to_tensor(image).to(device)
    original_size = [image.size[::-1]]
    height, width = image.height, image.width
    image_area = float(height * width)

    masks_dict = {}
    detections = []
    missing = []

    for item in items:
        phrase = item["phrase"]
        level = item["level"]
        print(f"分割目标 [{level}]: {phrase}")
        inputs = processor(images=image, text=phrase, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=THRESHOLD,
            mask_threshold=MASK_THRESHOLD,
            target_sizes=original_size,
        )[0]

        if len(results["masks"]) == 0:
            missing.append(item)
            continue

        contour_mask = (results["masks"] > MASK_THRESHOLD).any(dim=0)
        morphed_mask = convex_hull_and_dilate(contour_mask)
        box = mask_to_box(morphed_mask)
        if box is None:
            missing.append(item)
            continue

        final_mask = box_to_mask(box, morphed_mask) if USE_BBOX_MASK else morphed_mask
        area_px = int(final_mask.sum().item())
        if area_px <= 0:
            missing.append(item)
            continue

        key = f"{level}:{phrase}"
        masks_dict[key] = final_mask.cpu()

        isolated_img_tensor = img_tensor * final_mask.unsqueeze(0)
        isolated_img = F.to_pil_image(isolated_img_tensor)
        base = safe_name(f"{level}_{phrase}")
        isolated_path = os.path.join(OUTPUT_DIR, f"isolated_{base}.png")
        mask_path = os.path.join(OUTPUT_DIR, f"mask_{base}.pt")
        isolated_img.save(isolated_path)
        torch.save(final_mask.cpu(), mask_path)

        detections.append(
            {
                "key": key,
                "phrase": phrase,
                "level": level,
                "box": box,
                "area_px": area_px,
                "area_ratio": area_px / image_area,
                "mask_path": mask_path,
                "isolated_path": isolated_path,
                "mask_type": "bbox" if USE_BBOX_MASK else "convex_hull_dilated",
                "use_convex_hull": USE_CONVEX_HULL,
                "dilation_kernel_size": DILATION_KERNEL_SIZE,
                "dilation_iterations": DILATION_ITERATIONS,
            }
        )

    if not masks_dict:
        raise RuntimeError("SAM3 没有成功分割任何分层目标，请检查 0.py 输出。")

    torch.save(masks_dict, MASKS_FILE)

    report = {
        "image_path": IMAGE_PATH,
        "image_size": [height, width],
        "raw_masks_file": MASKS_FILE,
        "mask_processing": {
            "use_bbox_mask": USE_BBOX_MASK,
            "use_convex_hull": USE_CONVEX_HULL,
            "dilation_kernel_size": DILATION_KERNEL_SIZE,
            "dilation_iterations": DILATION_ITERATIONS,
            "min_component_area": MIN_COMPONENT_AREA,
        },
        "detections": detections,
        "missing": missing,
    }
    with open(REGIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    segmented = {
        "source_features_file": FEATURES_FILE,
        "raw_regions_file": REGIONS_FILE,
        "anchors": [item["key"] for item in detections],
        "missing_anchors": [item["phrase"] for item in missing],
    }
    with open(SEGMENTED_FILE, "w", encoding="utf-8") as f:
        json.dump(segmented, f, indent=2, ensure_ascii=False)

    print(f"成功分割数量: {len(detections)}")
    if missing:
        print(f"未成功分割: {[item['phrase'] for item in missing]}")
    print(f"已保存 raw masks: {MASKS_FILE}")
    print(f"已保存 raw regions: {REGIONS_FILE}")


if __name__ == "__main__":
    main()



# python ./1_SAM3使用.py