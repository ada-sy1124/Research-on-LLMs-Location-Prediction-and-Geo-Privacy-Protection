import json
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image

from args_geo_pg import init_geo_pg_args
from geo_pg_eval import deletion_heatmap_from_preservation_mask, evaluate_deletion
from geo_pg_score import generate_reference_coordinate
from methods_geo_pg import geo_pg_igos_mask_optimization
from model_adapters import get_model_adapter
from utils import get_initial, get_kernel_size, save_heatmaps, save_images


def load_data(args):
    if not args.data_path.endswith("jsonl"):
        raise ValueError("Geo-PG expects a jsonl file with fields: id, image_path")
    rows = [json.loads(line) for line in open(args.data_path, "r", encoding="utf-8")]
    for row in rows:
        if "id" not in row or "image_path" not in row:
            raise ValueError(f"bad row, expected id and image_path: {row}")
    return rows


def load_image(args, row):
    image_path = os.path.join(args.image_folder, row["image_path"])
    return Image.open(image_path).convert("RGB")


def prepare_image_tensors(args, adapter, model, image_processor, image):
    kernel_size = get_kernel_size(image.size)
    blur = cv2.GaussianBlur(
        np.asarray(image), (kernel_size, kernel_size), sigmaX=kernel_size - 1
    )
    blur = Image.fromarray(blur.astype(np.uint8))
    image_tensor, blur_tensor, prompt_inputs = adapter.prepare_inputs(
        model=model,
        image_processor=image_processor,
        image=image,
        baseline_image=blur,
        prompt=args.geo_prompt,
    )
    return image_tensor, blur_tensor, prompt_inputs


def run_geo_pg(args, adapter, model, tokenizer, image_processor, data):
    output_dir = os.path.join(
        args.output_dir,
        f"GeoPG_L1_{args.L1}_L2_{args.L2}_L3_{args.L3}_"
        f"N_{args.geo_rollouts}_T_{args.geo_temperature}_iter_{args.iterations}",
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    for index, row in enumerate(data):
        image = load_image(args, row)
        image_tensor, blur_tensor, prompt_inputs = prepare_image_tensors(
            args, adapter, model, image_processor, image
        )
        reference_coord, reference_text, reference_answer_ids, _ = generate_reference_coordinate(
            args, adapter, model, tokenizer, prompt_inputs, image_tensor
        )
        print(f"reference {index}: {reference_text}")

        pred_data = {
            "labels": reference_answer_ids.detach(),
            "boxes": np.array([[0, 0, args.input_size, args.input_size]]),
            "pred_text": reference_text,
        }
        pred_data = get_initial(
            pred_data,
            args.diverse_k,
            args.init_posi,
            args.init_val,
            args.input_size,
            args.size,
        )

        mask, history = geo_pg_igos_mask_optimization(
            args=args,
            adapter=adapter,
            model=model,
            tokenizer=tokenizer,
            init_mask=pred_data["init_masks"][0],
            image=image_tensor,
            baseline=blur_tensor,
            prompt_inputs=prompt_inputs,
            reference_coord=reference_coord,
        )

        deletion_heatmap = deletion_heatmap_from_preservation_mask(mask)
        deletion_eval = evaluate_deletion(
            args=args,
            adapter=adapter,
            model=model,
            tokenizer=tokenizer,
            prompt_inputs=prompt_inputs,
            image=image_tensor,
            baseline=blur_tensor,
            reference_coord=reference_coord,
            preservation_mask=mask,
        )

        sid = str(row["id"]).replace(os.sep, "_")
        sample_dir = os.path.join(output_dir, f"{index}_{sid}")
        os.makedirs(sample_dir, exist_ok=True)
        mask_path = os.path.join(sample_dir, "geo_pg_mask.pt")
        heatmap_path = os.path.join(sample_dir, f"{index}_0_heatmap.jpg")

        torch.save(
            {
                "preservation_mask": mask.cpu(),
                "deletion_heatmap": deletion_heatmap.cpu(),
            },
            mask_path,
        )
        save_heatmaps(
            mask,
            image_tensor,
            args.size,
            index,
            0,
            sample_dir,
            args.model,
            None,
            None,
            reference_answer_ids,
        )
        save_images(
            image_tensor,
            index,
            0,
            sample_dir,
            None,
            reference_answer_ids,
            pred_data,
            text=args.geo_prompt,
        )

        result = {
            "image_id": sid,
            "raw_reference_answer": reference_text,
            "reference_coord": [reference_coord.lat, reference_coord.lon],
            "heatmap_path": heatmap_path,
            "mask_tensor_path": mask_path,
            "optimization_log": history,
            "deletion_eval": deletion_eval,
        }
        with open(os.path.join(sample_dir, "geo_pg_result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print(f"finished {index}: reference={reference_text}")


if __name__ == "__main__":
    args = init_geo_pg_args()
    torch.manual_seed(args.manual_seed)
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)

    adapter = get_model_adapter(args.model)
    tokenizer, model, image_processor, _ = adapter.load_model(args.model_path, args.torch_dtype)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    data = load_data(args)
    run_geo_pg(args, adapter, model, tokenizer, image_processor, data)
