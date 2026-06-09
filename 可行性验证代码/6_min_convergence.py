"""Minimal DCSD convergence test with Qwen3-VL and precomputed token costs.

This script tests the differentiable path:
mask weights -> masked visual inputs -> Qwen next-token logits -> coordinate loss.
Only the mask logits alpha are trainable; the VLM is frozen.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dcsd_coordinate_table import (
    load_coordinate_cost_table,
    normalize_target_coord,
)


# ================= 0. 路径和参数设置 =================
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
USE_LOCAL_MODEL = True
IMAGE_PATH = "/root/autodl-tmp/data/im2gps3ktest/im2gps3ktest/167345423_ba2914f302_59_79954435@N00.jpg"
INTERMEDIATE_DIR = "/root/autodl-tmp/DCSD分层提取凸包/中间件"
FEATURES_FILE = f"{INTERMEDIATE_DIR}/01_qwen_geolocation.json"
MASKS_FILE = f"{INTERMEDIATE_DIR}/03_balanced_masks/masks_dict.pt"
OUTPUT_DIR = f"{INTERMEDIATE_DIR}/dcsd_min_test"
COST_TABLE_FILE = f"{OUTPUT_DIR}/05_coordinate_cost_table.pt"

# 留空则自动使用 0.py 保存的 Location；如果你有真实 GT 坐标，就填这里。
TARGET_COORD = ""

# 优化 alpha 的迭代步数；一般 30~50 步先看能否收敛。
STEPS = 50
# Adam 学习率；越大 alpha 变化越快，也越容易震荡。
LR = 0.25
# 稀疏正则权重；现阶段先设为 0，不惩罚选择目标数量，避免遮挡面积太小。
LAMBDA_SPARSE = 0.0
# 物理距离损失权重；越大越强调把坐标推远。
GAMMA = 50.0
# 距离归一化尺度，单位 km；1000 表示超过 1000km 的错误按最大距离奖励处理。
DISTANCE_NORM_KM = 10000.0
# 最多使用多少个 mask；0 表示使用 SAM3 成功分割出的全部 mask。
MAX_MASKS = 0
# 最终二值化阈值；p_i 大于该值的目标会被判定为核心因果目标。
THRESHOLD = 0.5


# 这里不要和 0.py 的 CoT/Anchors prompt 保持一致。
# 4.py 做的是坐标 Teacher Forcing，只需要模型处在“输出经纬度”的任务语境里。
GEO_COORD_PROMPT = (
    "This is a photo of my previous tour but I don't remember where it is. "
    "Estimate the latitude and longitude. Output only latitude and longitude, "
    "separated by a comma. Do not output any other text."
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--image", default=IMAGE_PATH, help="Original image path.")
    parser.add_argument("--masks-file", default=MASKS_FILE, help="SAM/SAM3 masks_dict.pt path.")
    parser.add_argument("--features-file", default=FEATURES_FILE)
    parser.add_argument("--cost-table-file", default=COST_TABLE_FILE)
    parser.add_argument(
        "--target-coord",
        default=TARGET_COORD,
        help="Fixed-format target coordinate, e.g. +51.4983,-000.1748",
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--lambda-sparse", type=float, default=LAMBDA_SPARSE)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--distance-norm-km", type=float, default=DISTANCE_NORM_KM)
    parser.add_argument("--max-masks", type=int, default=MAX_MASKS, help="0 means use all masks.")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--prompt", default=GEO_COORD_PROMPT)
    return parser.parse_args()


def load_target_coord(args):
    if args.target_coord:
        return normalize_target_coord(args.target_coord)
    with open(args.features_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    location = data.get("location", "")
    if not location:
        raise RuntimeError(
            "没有提供 TARGET_COORD，且 0.py 的中间件里没有 location。"
        )
    return normalize_target_coord(location)


def load_masks(mask_path: str, image_size, max_masks: int):
    import torch
    import torch.nn.functional as torch_f

    data = torch.load(mask_path, map_location="cpu")
    if not isinstance(data, dict):
        raise TypeError("Expected masks_dict.pt to contain a dict: noun -> mask tensor")

    names = list(data.keys())
    if max_masks > 0:
        names = names[:max_masks]

    masks = []
    height, width = image_size
    for name in names:
        mask = data[name]
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        mask = mask.bool().float()
        if tuple(mask.shape) != (height, width):
            mask = torch_f.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=(height, width),
                mode="nearest",
            ).squeeze(0).squeeze(0)
        masks.append(mask)

    if not masks:
        raise ValueError(f"No masks loaded from {mask_path}")
    return names, torch.stack(masks, dim=0)


def image_to_tensor(image):
    import numpy as np
    import torch

    arr = np.asarray(image).astype("float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def blackout_image(image, mask):
    import numpy as np
    from PIL import Image

    arr = np.asarray(image).astype("float32")
    mask_np = mask.detach().cpu().numpy().astype("float32")
    arr = arr * (1.0 - mask_np[..., None])
    return Image.fromarray(arr.clip(0, 255).astype("uint8"))


def qwen_patchify_image(image_tensor, image_grid_thw, image_processor):
    import torch
    import torch.nn.functional as torch_f

    patch_size = int(image_processor.patch_size)
    temporal_patch_size = int(image_processor.temporal_patch_size)
    merge_size = int(image_processor.merge_size)

    grid_t, grid_h, grid_w = [int(x) for x in image_grid_thw[0].tolist()]
    if grid_t != 1:
        raise RuntimeError(f"当前脚本只处理单图输入，但 image_grid_thw[0,0]={grid_t}")

    resized_h = grid_h * patch_size
    resized_w = grid_w * patch_size

    image = torch_f.interpolate(
        image_tensor.unsqueeze(0),
        size=(resized_h, resized_w),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).squeeze(0).clamp(0, 1)

    mean = torch.tensor(
        image_processor.image_mean,
        dtype=image.dtype,
        device=image.device,
    ).view(3, 1, 1)
    std = torch.tensor(
        image_processor.image_std,
        dtype=image.dtype,
        device=image.device,
    ).view(3, 1, 1)
    patches = (image - mean) / std

    patches = patches.unsqueeze(0)
    batch_size, channel = patches.shape[:2]
    patches = patches.reshape(
        batch_size,
        channel,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
    flatten_patches = (
        patches.unsqueeze(6)
        .expand(-1, -1, -1, -1, -1, -1, temporal_patch_size, -1, -1)
        .reshape(
            batch_size,
            grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size,
        )
    )
    return flatten_patches.reshape(
        grid_h * grid_w,
        channel * temporal_patch_size * patch_size * patch_size,
    )


def build_soft_masked_pixel_values(image_tensor, masks, probs, image_grid_thw, image_processor):
    weighted_mask = (probs[:, None, None] * masks).sum(dim=0).clamp(0, 1)
    soft_image = image_tensor * (1.0 - weighted_mask.unsqueeze(0))
    return qwen_patchify_image(soft_image, image_grid_thw, image_processor)


def prepare_inputs(processor, image, prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )


def move_batch(batch, device):
    import torch

    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def save_selected_blackout(image, masks, names, probs, threshold, output_path):
    import torch

    selected = probs.detach().cpu() > threshold
    if selected.any():
        merged = masks.detach().cpu()[selected].bool().any(dim=0).float()
    else:
        merged = torch.zeros_like(masks[0])
    out = blackout_image(image, merged)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)
    return [name for name, keep in zip(names, selected.tolist()) if keep]


def build_static_inputs(base_inputs, target_ids_tensor, prompt_len):
    import torch

    static_inputs = {}
    extended_keys = []
    for key, value in base_inputs.items():
        if key in {"input_ids", "attention_mask", "pixel_values"}:
            continue
        if (
            torch.is_tensor(value)
            and value.dim() == 2
            and value.shape[0] == 1
            and value.shape[1] == prompt_len
        ):
            target_pad = torch.zeros(
                (1, target_ids_tensor.shape[1]),
                dtype=value.dtype,
                device=value.device,
            )
            static_inputs[key] = torch.cat([value, target_pad], dim=1)
            extended_keys.append(key)
        else:
            static_inputs[key] = value

    if extended_keys:
        print(f"已同步扩展序列字段: {extended_keys}")
    return static_inputs


def align_cost_table_to_model_vocab(valid_mask, distance_table, model_vocab_size):
    import torch

    table_vocab_size = valid_mask.shape[1]
    if table_vocab_size == model_vocab_size:
        return valid_mask, distance_table

    if table_vocab_size > model_vocab_size:
        raise RuntimeError(
            "查询表词表维度大于模型 logits 维度，不能安全截断。\n"
            f"table vocab: {table_vocab_size}, model vocab: {model_vocab_size}"
        )

    pad_cols = model_vocab_size - table_vocab_size
    valid_pad = torch.zeros(
        valid_mask.shape[0],
        pad_cols,
        dtype=valid_mask.dtype,
        device=valid_mask.device,
    )
    dist_pad = torch.zeros(
        distance_table.shape[0],
        pad_cols,
        dtype=distance_table.dtype,
        device=distance_table.device,
    )
    print(
        "查询表 vocab 维度小于模型 logits 维度，已补齐并将额外 token 全部设为非法: "
        f"{table_vocab_size} -> {model_vocab_size}"
    )
    return torch.cat([valid_mask, valid_pad], dim=1), torch.cat([distance_table, dist_pad], dim=1)


def main():
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    import torch
    import torch.nn.functional as torch_f
    from modelscope import AutoProcessor, Qwen3VLForConditionalGeneration
    from PIL import Image

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model_id}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=USE_LOCAL_MODEL,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        local_files_only=USE_LOCAL_MODEL,
    )

    image = Image.open(args.image).convert("RGB")
    width, height = image.size
    mask_names, raw_masks = load_masks(args.masks_file, (height, width), args.max_masks)
    print(f"Loaded masks: {mask_names}")

    base_inputs = prepare_inputs(processor, image, args.prompt)
    prompt_len = base_inputs["input_ids"].shape[1]

    target_coord = load_target_coord(args)
    cost_table_path = Path(args.cost_table_file)
    if not cost_table_path.exists():
        raise RuntimeError(
            f"没有找到坐标距离查询表: {cost_table_path}\n"
            "请先运行 DCSD/3_build_distance_table.py。"
        )
    cost_table = load_coordinate_cost_table(cost_table_path)
    if cost_table["target_coord"] != target_coord:
        raise RuntimeError(
            "查询表坐标和当前目标坐标不一致。\n"
            f"table: {cost_table['target_coord']}\n"
            f"current: {target_coord}\n"
            "请重新运行 DCSD/3_build_distance_table.py。"
        )
    if abs(float(cost_table["distance_norm_km"]) - float(args.distance_norm_km)) > 1e-8:
        raise RuntimeError(
            "查询表 distance_norm_km 和当前参数不一致，请重新运行 DCSD/3_build_distance_table.py。"
        )

    target_ids = cost_table["target_ids"]
    valid_mask = cost_table["valid_mask"]
    distance_table = cost_table["distance_table"]
    spans = cost_table["spans"]
    output_embeddings = model.get_output_embeddings()
    model_vocab_size = getattr(output_embeddings, "out_features", None)
    if model_vocab_size is None:
        model_vocab_size = int(model.config.vocab_size)
    valid_mask, distance_table = align_cost_table_to_model_vocab(
        valid_mask,
        distance_table,
        model_vocab_size,
    )
    print(f"Target coord: {target_coord}")
    print(f"Target token count: {len(target_ids)}")
    valid_counts = valid_mask.sum(dim=1).tolist()
    print(f"Valid candidate tokens per target position: {valid_counts}")
    print(f"Token spans: {spans}")

    base_inputs = move_batch(base_inputs, model.device)
    base_pixel = base_inputs["pixel_values"].detach()
    image_tensor = image_to_tensor(image).to(model.device)
    raw_masks = raw_masks.to(model.device)

    with torch.no_grad():
        reconstructed_base_pixel = qwen_patchify_image(
            image_tensor,
            base_inputs["image_grid_thw"],
            processor.image_processor,
        )
        if reconstructed_base_pixel.shape != base_pixel.shape:
            raise RuntimeError(
                "torch 版 Qwen 图像预处理输出形状和 processor 不一致。\n"
                f"torch: {tuple(reconstructed_base_pixel.shape)}\n"
                f"processor: {tuple(base_pixel.shape)}"
            )
        diff = (reconstructed_base_pixel - base_pixel).abs()
        print(
            "torch 版 Qwen 图像预处理校验: "
            f"mean_abs_diff={diff.mean().item():.6f}, max_abs_diff={diff.max().item():.6f}"
        )

    target_ids_tensor = torch.tensor(target_ids, dtype=torch.long, device=model.device).unsqueeze(0)
    full_input_ids = torch.cat([base_inputs["input_ids"], target_ids_tensor], dim=1)
    target_attention = torch.ones_like(target_ids_tensor, device=model.device)
    full_attention_mask = torch.cat([base_inputs["attention_mask"], target_attention], dim=1)

    valid_mask = valid_mask.to(model.device)
    distance_table = distance_table.to(model.device)
    target_positions = torch.arange(len(target_ids), device=model.device)

    alpha = torch.zeros(len(mask_names), device=model.device, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([alpha], lr=args.lr)
    history = []

    static_inputs = build_static_inputs(base_inputs, target_ids_tensor, prompt_len)

    print("Starting alpha optimization...")
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        probs = torch.sigmoid(alpha)
        mixed_pixel = build_soft_masked_pixel_values(
            image_tensor=image_tensor,
            masks=raw_masks,
            probs=probs,
            image_grid_thw=base_inputs["image_grid_thw"],
            image_processor=processor.image_processor,
        )

        model_inputs = {
            **static_inputs,
            "input_ids": full_input_ids,
            "attention_mask": full_attention_mask,
            "pixel_values": mixed_pixel,
            "use_cache": False,
        }
        outputs = model(**model_inputs)
        logits = outputs.logits[0, prompt_len - 1 : prompt_len + len(target_ids) - 1, :]
        masked_logits = logits.masked_fill(~valid_mask, -1e9)
        log_probs = torch_f.log_softmax(masked_logits, dim=-1)
        token_probs = torch_f.softmax(masked_logits, dim=-1)

        ce = -log_probs[target_positions, target_ids_tensor.squeeze(0)].sum()
        dist = (token_probs * distance_table).sum(dim=-1).sum()
        sparse = probs.sum()
        loss = args.lambda_sparse * sparse - (ce + args.gamma * dist)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            true_prob = token_probs[target_positions, target_ids_tensor.squeeze(0)].mean()
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "ce": float(ce.detach().cpu()),
                "dist": float(dist.detach().cpu()),
                "sparse": float(sparse.detach().cpu()),
                "true_prob_mean": float(true_prob.detach().cpu()),
                "probs": [float(x) for x in probs.detach().cpu()],
            }
            history.append(row)

        if step % 5 == 0 or step == args.steps - 1:
            prob_text = ", ".join(
                f"{name}={value:.3f}"
                for name, value in zip(mask_names, probs.detach().float().cpu().tolist())
            )
            print(
                f"step={step:03d} loss={row['loss']:.4f} ce={row['ce']:.4f} "
                f"dist={row['dist']:.4f} sparse={row['sparse']:.4f} "
                f"true_p={row['true_prob_mean']:.4f} | {prob_text}"
            )

    final_probs = torch.sigmoid(alpha).detach().cpu()
    selected = save_selected_blackout(
        image=image,
        masks=raw_masks,
        names=mask_names,
        probs=final_probs,
        threshold=args.threshold,
        output_path=output_dir / "selected_blackout.jpg",
    )

    result = {
        "image": args.image,
        "masks_file": args.masks_file,
        "target_coord": target_coord,
        "mask_names": mask_names,
        "final_probs": [float(x) for x in final_probs],
        "selected": selected,
        "threshold": args.threshold,
        "history": history,
    }
    with (output_dir / "convergence_result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Saved result: {output_dir / 'convergence_result.json'}")
    print(f"Saved selected blackout preview: {output_dir / 'selected_blackout.jpg'}")
    print(f"Selected masks: {selected}")


if __name__ == "__main__":
    main()




# python ./6_min_convergence.py