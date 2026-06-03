import os

import datasets
import numpy as np
import torch
from datasets import Dataset, load_from_disk
from PIL import Image

from geoai_pipeline.config import get_env, get_path
from geoai_pipeline.constants import SAM_PROMPT_MAPPING


def get_masked_images_and_ratios(image, categories, processor):
    image_rgb = image.convert("RGB")
    img_array_base = np.array(image_rgb)

    total_pixels = img_array_base.shape[0] * img_array_base.shape[1]
    inference_state = processor.set_image(image_rgb)

    masked_images_list = []
    mask_ratios_list = []

    for cat in categories:
        prompts_to_search = SAM_PROMPT_MAPPING.get(cat, [cat])
        cat_combined_mask = None

        for prompt_text in prompts_to_search:
            output = processor.set_text_prompt(state=inference_state, prompt=prompt_text)
            masks = output.get("masks")

            if masks is None or len(masks) == 0:
                continue

            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()

            while len(masks.shape) > 2:
                masks = np.any(masks, axis=0)

            current_mask = masks

            if cat_combined_mask is None:
                cat_combined_mask = current_mask
            else:
                cat_combined_mask = cat_combined_mask | current_mask

        img_array_masked = img_array_base.copy()

        if cat_combined_mask is not None:
            img_array_masked[cat_combined_mask] = [0, 0, 0]
            mask_ratio = float(np.sum(cat_combined_mask) / total_pixels)
        else:
            mask_ratio = 0.0

        masked_pil = Image.fromarray(img_array_masked)
        masked_images_list.append(masked_pil)
        mask_ratios_list.append(mask_ratio)

    return masked_images_list, mask_ratios_list


def run():
    hf_cache = get_env("HF_DATASETS_CACHE", "./data/cache/hf_datasets")
    os.environ["HF_DATASETS_CACHE"] = hf_cache

    input_dataset_path = get_path(
        "YES_INPUT_DATASET_PATH",
        "./data/YES",
    )
    output_dir = get_path(
        "YES_AFTER_SAM_OUTPUT_DIR",
        "./data/YES_NEW_afterSAM",
    )

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model().to(device)
    processor = Sam3Processor(model)

    print("正在加载输入数据集...")
    dataset = load_from_disk(input_dataset_path)

    features = datasets.Features(
        {
            "image_original": datasets.Image(),
            "latitude": datasets.Value("float64"),
            "longitude": datasets.Value("float64"),
            "d_original": datasets.Value("float64"),
            "ablated_class": datasets.Sequence(datasets.Value("string")),
            "masked_image": datasets.Sequence(datasets.Image()),
            "q_ratio": datasets.Sequence(datasets.Value("float64")),
        }
    )

    def data_generator():
        for item in dataset:
            image_obj = item["image"]
            lat = item["latitude"]
            lon = item["longitude"]
            d_orig = item["d"]
            reason_classes = item["reason_class"]

            if not reason_classes:
                continue

            masked_images, q_ratios = get_masked_images_and_ratios(image_obj, reason_classes, processor)

            yield {
                "image_original": image_obj,
                "latitude": lat,
                "longitude": lon,
                "d_original": d_orig,
                "ablated_class": reason_classes,
                "masked_image": masked_images,
                "q_ratio": q_ratios,
            }

    print(f"开始执行像素级全黑物理遮挡，并实时写入硬盘，共计 {len(dataset)} 条数据...")

    final_dataset = Dataset.from_generator(data_generator, features=features)

    os.makedirs(output_dir, exist_ok=True)
    final_dataset.save_to_disk(output_dir)

    print("✅ 处理完成！")
    print(f"✅ 保存路径: {output_dir}")


def main():
    with torch.no_grad():
        run()


if __name__ == "__main__":
    main()
