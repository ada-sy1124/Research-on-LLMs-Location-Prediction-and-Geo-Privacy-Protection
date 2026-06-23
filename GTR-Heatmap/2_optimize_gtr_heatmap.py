#!/usr/bin/env python3
"""
Step 2: Optimize a continuous MSGRA / DSDP-G geolocation heatmap.

This implementation keeps the gradient path:

mask logits
 -> sigmoid mask
 -> rendered image
 -> pixel_values
 -> VLM logits
 -> teacher-forced coordinate log-probabilities
 -> weighted spatial distribution q
 -> KL / geodesic risk / near-far contrast
 -> mask logits

No generate(), no argmax, no dynamic candidate sampling inside optimization.
"""

import csv
import json
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


# ================= 0. PATHS AND PARAMETERS =================

INPUT_IMAGE_PATH = "/root/autodl-tmp/GSV/geolocation-inference-dataset/imgs_final_v1/20.jpg"
COORDINATE_BANK_JSON = "/root/autodl-tmp/Research-on-LLMs-Location-Prediction-and-Geo-Privacy-Protection/data/gtr_heatmap/1_coordinate_bank.json"
COORDINATE_BANK_PT = "/root/autodl-tmp/Research-on-LLMs-Location-Prediction-and-Geo-Privacy-Protection/data/gtr_heatmap/1_coordinate_bank.pt"
OUTPUT_DIR = "/root/autodl-tmp/Research-on-LLMs-Location-Prediction-and-Geo-Privacy-Protection/data/gtr_heatmap"

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
DEVICE = "cuda"  # "auto", "cuda", "mps", or "cpu"
MODEL_DTYPE = "bfloat16"  # "auto", "float16", "bfloat16", or "float32"

GEOLOCATION_PROMPT = (
    """This is a photo of my previous tour but I don’t remember where it is, 
    could you help me find it. Estimate the precise latitude and longitude. 
    If you are not sure about the specific location, you MUST give a possible 
    latitude and longitude candidate without asking any further questions. 
    You MUST output the coordinates STRICTLY in the following format: 
    LAT=XX.XXXX; LON=XX.XXXX"""
)

IMAGE_SIZE = 336
MASK_LOW_RES_H = 21
MASK_LOW_RES_W = 21

INITIAL_MASK_LOGIT = -4.0
OPTIMIZATION_STEPS = 100
LEARNING_RATE = 0.05
CANDIDATE_BATCH_SIZE = 8

SOFTMAX_TEMPERATURE = 1.0
ALPHA_KL = 0.25
BETA_GEO = 1.0
GAMMA_NF = 0.25
LOGPROB_REDUCTION = "sum"  # "sum" for joint probability, "mean" for length-normalized ablation.

LAMBDA_AREA = 0.10
LAMBDA_TV = 0.03

EPS = 1e-8
BLUR_KERNEL_SIZE = 31
SAVE_EVERY_STEPS = 10
RANDOM_SEED = 2026

DEBUG_TARGET_MASK = True
FINAL_TOPK_ANCHORS = 10

OUTPUT_PT = f"{OUTPUT_DIR}/2_heatmap_result.pt"
OUTPUT_LOG_CSV = f"{OUTPUT_DIR}/2_optimization_log.csv"
OUTPUT_HEATMAP_PNG = f"{OUTPUT_DIR}/2_heatmap.png"
OUTPUT_OVERLAY_PNG = f"{OUTPUT_DIR}/2_heatmap_overlay.png"
OUTPUT_SOFT_MASKED_PNG = f"{OUTPUT_DIR}/2_soft_masked_image.png"
OUTPUT_TOP_ANCHORS_JSON = f"{OUTPUT_DIR}/2_final_top_anchors.json"


# ================= 1. BASIC UTILS =================

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


def safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


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
    x = image.detach().float().cpu().clamp(0, 1)
    if x.ndim == 4:
        x = x[0]
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
    alpha = mask.detach().float().cpu().clamp(0, 1)
    if alpha.ndim == 4:
        alpha = alpha[0, 0]
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


# ================= 2. BANK LOADING =================

def load_bank():
    if os.path.exists(COORDINATE_BANK_PT):
        bank = safe_torch_load(COORDINATE_BANK_PT)
        return {
            "metadata": bank,
            "texts": list(bank["coord_texts"]),
            "D_norm": bank["D_norm"].float(),
            "weights": bank["weights"].float(),
            "soft_target": bank["soft_target"].float(),
            "near_mask": bank["near_mask"].bool(),
            "far_mask": bank["far_mask"].bool(),
            "reference": bank.get("reference"),
            "distance_km": bank.get("distance_km"),
            "anchors": bank.get("anchors"),
            "group_name": bank.get("group_name", None),
            "candidates": bank.get("candidates", None),
        }

    with open(COORDINATE_BANK_JSON, "r", encoding="utf-8") as f:
        bank = json.load(f)

    candidates = bank["candidates"]

    return {
        "metadata": bank,
        "texts": [row["coord_text"] for row in candidates],
        "D_norm": torch.tensor([float(row.get("D_norm", row["distance_norm"])) for row in candidates]),
        "weights": torch.tensor([float(row.get("weight", 1.0)) for row in candidates]),
        "soft_target": torch.tensor([float(row.get("soft_target", 0.0)) for row in candidates]),
        "near_mask": torch.tensor(
            [bool(row.get("is_near", float(row["distance_km"]) <= 100.0)) for row in candidates]
        ),
        "far_mask": torch.tensor(
            [bool(row.get("is_far", float(row["distance_km"]) >= 5000.0)) for row in candidates]
        ),
        "reference": bank.get("reference"),
        "distance_km": torch.tensor([float(row["distance_km"]) for row in candidates]),
        "anchors": torch.tensor([[float(row["lat"]), float(row["lon"])] for row in candidates]),
        "group_name": [row.get("group_name", row.get("shell_name", "unknown")) for row in candidates],
        "candidates": candidates,
    }


def normalize_bank_tensors(bank: dict, device: str):
    weights = bank["weights"].to(device=device, dtype=torch.float32)
    weights = weights / weights.sum().clamp_min(EPS)

    soft_target = bank["soft_target"].to(device=device, dtype=torch.float32)
    if float(soft_target.sum().detach().cpu()) <= 0:
        soft_target = weights.clone()
    else:
        soft_target = soft_target / soft_target.sum().clamp_min(EPS)

    distance_norm = bank["D_norm"].to(device=device, dtype=torch.float32)
    near_mask = bank["near_mask"].to(device=device)
    far_mask = bank["far_mask"].to(device=device)

    if not bool(near_mask.any().detach().cpu()):
        raise ValueError("Coordinate bank has no near candidates for near-far contrast.")
    if not bool(far_mask.any().detach().cpu()):
        raise ValueError("Coordinate bank has no far candidates for near-far contrast.")

    return distance_norm, weights, soft_target, near_mask, far_mask


# ================= 3. TEACHER FORCING =================

def find_subsequence(sequence: list[int], pattern: list[int]):
    if not pattern or len(pattern) > len(sequence):
        return None
    for start in range(0, len(sequence) - len(pattern) + 1):
        if sequence[start : start + len(pattern)] == pattern:
            return start
    return None


def find_last_subsequence(sequence: list[int], pattern: list[int]):
    if not pattern or len(pattern) > len(sequence):
        return None
    for start in range(len(sequence) - len(pattern), -1, -1):
        if sequence[start : start + len(pattern)] == pattern:
            return start
    return None


def sequence_logprobs_batch(
    model,
    processor,
    pil_image,
    pixel_values,
    prompt: str,
    targets: list[str],
    device: str,
    debug: bool = False,
):
    """
    Returns teacher-forced log-probability for each target sequence.

    Important:
    - Does not use generate().
    - Does not use argmax.
    - Keeps gradient attached to pixel_values during optimization.
    """
    full_texts = [
        processor.apply_chat_template(
            build_messages(prompt, target),
            tokenize=False,
            add_generation_prompt=False,
        )
        for target in targets
    ]

    tokenizer = getattr(processor, "tokenizer", processor)

    inputs = processor(
        text=full_texts,
        images=[pil_image] * len(targets),
        padding=True,
        return_tensors="pt",
    ).to(device)

    if "pixel_values" not in inputs:
        raise RuntimeError("The selected processor did not return pixel_values.")

    if inputs["pixel_values"].ndim != 4:
        raise RuntimeError(
            f"This MVP expects dense pixel_values [B,3,H,W], got shape {tuple(inputs['pixel_values'].shape)}. "
            "Use a LLaVA-style model, or adapt visual preprocessing for this VLM."
        )

    # Replace processor-generated pixel values with differentiable rendered image tensor.
    inputs["pixel_values"] = pixel_values.repeat(len(targets), 1, 1, 1)

    outputs = model(**inputs)
    logits = outputs.logits
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))

    shift_logits = logits[:, :-1, :]
    shift_ids = input_ids[:, 1:]
    shift_attention = attention_mask[:, 1:].bool()

    token_logp = torch.log_softmax(shift_logits, dim=-1).gather(
        -1, shift_ids.unsqueeze(-1)
    ).squeeze(-1)

    target_mask = torch.zeros_like(shift_attention)

    for row_idx in range(input_ids.shape[0]):
        target_ids = tokenizer(targets[row_idx], add_special_tokens=False)["input_ids"]
        if not target_ids:
            continue

        valid_positions = attention_mask[row_idx].nonzero(as_tuple=False).flatten()
        valid_ids = input_ids[row_idx, valid_positions].detach().cpu().tolist()

        # Image placeholders can expand into many visual tokens inside processor
        # output, so prefix matching is fragile. The coordinate answer is at the
        # tail; search for a target suffix and map it back to original positions.
        search_pattern = target_ids[1:] if len(target_ids) > 1 else target_ids
        match_idx = find_last_subsequence(valid_ids, search_pattern)
        if match_idx is not None:
            offset = 1 if len(target_ids) > 1 else 0
            compressed_start = max(match_idx - offset, 0)
        else:
            compressed_start = max(len(valid_ids) - len(target_ids), 0)

        compressed_end = min(compressed_start + len(target_ids), len(valid_positions))
        if compressed_end <= compressed_start:
            continue

        original_start = int(valid_positions[compressed_start].detach().cpu())
        original_end = int(valid_positions[compressed_end - 1].detach().cpu()) + 1

        shifted_start = max(original_start - 1, 0)
        shifted_end = max(original_end - 1, shifted_start + 1)
        target_mask[row_idx, shifted_start:shifted_end] = True

    target_mask = target_mask & shift_attention
    target_lengths = target_mask.sum(dim=1).clamp_min(1)

    seq_logp = (token_logp * target_mask).sum(dim=1)
    if LOGPROB_REDUCTION == "mean":
        seq_logp = seq_logp / target_lengths
    elif LOGPROB_REDUCTION != "sum":
        raise ValueError("LOGPROB_REDUCTION must be 'sum' or 'mean'.")

    if debug and len(targets) > 0:
        row_idx = 0
        masked_token_ids = shift_ids[row_idx][target_mask[row_idx]].detach().cpu().tolist()
        try:
            decoded_masked = tokenizer.decode(masked_token_ids, skip_special_tokens=False)
        except Exception:
            decoded_masked = str(masked_token_ids)

        print("\n[debug] Teacher-forcing target mask sanity check")
        print("[debug] target text:", targets[0])
        print("[debug] decoded masked tokens:", repr(decoded_masked))
        print("[debug] target token count:", int(target_lengths[row_idx].detach().cpu()))
        print()

    return seq_logp


def compute_all_logps(
    model,
    processor,
    pil_image,
    pixel_values,
    prompt: str,
    coord_texts: list[str],
    device: str,
    candidate_batch_size: int,
    debug_first_batch: bool = False,
):
    chunks = []

    for start in range(0, len(coord_texts), candidate_batch_size):
        batch_targets = coord_texts[start : start + candidate_batch_size]
        batch_logps = sequence_logprobs_batch(
            model=model,
            processor=processor,
            pil_image=pil_image,
            pixel_values=pixel_values,
            prompt=prompt,
            targets=batch_targets,
            device=device,
            debug=(debug_first_batch and start == 0),
        )
        chunks.append(batch_logps)

    return torch.cat(chunks, dim=0)


# ================= 4. MSGRA OBJECTIVE =================

def compute_msgra_terms(
    logps: torch.Tensor,
    distance_norm: torch.Tensor,
    weights: torch.Tensor,
    soft_target: torch.Tensor,
    near_mask: torch.Tensor,
    far_mask: torch.Tensor,
):
    """
    Computes:
    - weighted q
    - KL(K || q)
    - expected geodesic risk
    - group-normalized near-far contrast
    """
    log_weights = torch.log(weights.clamp_min(EPS))
    log_weighted = logps / SOFTMAX_TEMPERATURE + log_weights

    q = torch.softmax(log_weighted, dim=0)

    risk = (q * distance_norm).sum()

    kl = (
        soft_target
        * (torch.log(soft_target.clamp_min(EPS)) - torch.log(q.clamp_min(EPS)))
    ).sum()

    # Group-normalized logsumexp:
    # log( sum_j w_j exp(logp_j/tau) / sum_j w_j )
    l_near = (
        torch.logsumexp(log_weighted[near_mask], dim=0)
        - torch.log(weights[near_mask].sum().clamp_min(EPS))
    )
    l_far = (
        torch.logsumexp(log_weighted[far_mask], dim=0)
        - torch.log(weights[far_mask].sum().clamp_min(EPS))
    )
    near_far = l_far - l_near

    msgra_score = ALPHA_KL * kl + BETA_GEO * risk + GAMMA_NF * near_far

    return {
        "q": q,
        "risk": risk,
        "kl": kl,
        "l_near": l_near,
        "l_far": l_far,
        "near_far": near_far,
        "msgra_score": msgra_score,
    }


def summarize_top_anchors(
    coord_texts: list[str],
    q: torch.Tensor,
    logps: torch.Tensor,
    distance_norm: torch.Tensor,
    weights: torch.Tensor,
    group_name,
    topk: int,
):
    q_cpu = q.detach().float().cpu()
    logps_cpu = logps.detach().float().cpu()
    distance_cpu = distance_norm.detach().float().cpu()
    weights_cpu = weights.detach().float().cpu()

    k = min(topk, len(coord_texts))
    top = torch.topk(q_cpu, k=k)

    rows = []
    for rank, idx in enumerate(top.indices.tolist(), start=1):
        group = None
        if group_name is not None and idx < len(group_name):
            group = group_name[idx]

        rows.append(
            {
                "rank": rank,
                "index": idx,
                "coord_text": coord_texts[idx],
                "q": float(q_cpu[idx]),
                "logp": float(logps_cpu[idx]),
                "distance_norm": float(distance_cpu[idx]),
                "weight": float(weights_cpu[idx]),
                "group_name": group,
            }
        )

    return rows


# ================= 5. MAIN =================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(RANDOM_SEED)

    device = choose_device()
    dtype = choose_dtype(device)

    pil_image, image = load_image_tensor(INPUT_IMAGE_PATH, IMAGE_SIZE, device)
    blurred = blur_image_tensor(image, BLUR_KERNEL_SIZE)

    bank = load_bank()
    coord_texts = bank["texts"]
    distance_norm, weights, soft_target, near_mask, far_mask = normalize_bank_tensors(bank, device)

    print(f"[2] Candidate count: {len(coord_texts)}")
    print(f"[2] Near candidates: {int(near_mask.sum().detach().cpu())}")
    print(f"[2] Far candidates: {int(far_mask.sum().detach().cpu())}")
    print(f"[2] Weight sum: {float(weights.sum().detach().cpu()):.6f}")
    print(f"[2] Soft target sum: {float(soft_target.sum().detach().cpu()):.6f}")

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
        torch.full(
            (1, 1, MASK_LOW_RES_H, MASK_LOW_RES_W),
            INITIAL_MASK_LOGIT,
            device=device,
            dtype=torch.float32,
        )
    )
    optimizer = torch.optim.Adam([mask_logits], lr=LEARNING_RATE)

    log_rows = []

    print(f"[2] Optimizing MSGRA for {OPTIMIZATION_STEPS} steps...")
    for step in range(OPTIMIZATION_STEPS):
        optimizer.zero_grad(set_to_none=True)

        soft_mask_low = torch.sigmoid(mask_logits)
        mask = F.interpolate(
            soft_mask_low,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        rendered = image * (1.0 - mask) + blurred * mask
        pixel_values = make_model_pixel_values(rendered, mean, std, dtype)

        # Stage 1: compute scalar objective on detached log-probabilities.
        # This gives d(loss)/d(logp_j) without keeping every VLM batch graph.
        with torch.no_grad():
            logps_values = compute_all_logps(
                model=model,
                processor=processor,
                pil_image=pil_image,
                pixel_values=pixel_values,
                prompt=GEOLOCATION_PROMPT,
                coord_texts=coord_texts,
                device=device,
                candidate_batch_size=CANDIDATE_BATCH_SIZE,
                debug_first_batch=(DEBUG_TARGET_MASK and step == 0),
            )

        logps = logps_values.detach().requires_grad_(True)

        terms = compute_msgra_terms(
            logps=logps,
            distance_norm=distance_norm,
            weights=weights,
            soft_target=soft_target,
            near_mask=near_mask,
            far_mask=far_mask,
        )

        area_loss = mask.mean()
        tv_loss = total_variation(mask)

        loss = (
            -terms["msgra_score"]
            + LAMBDA_AREA * area_loss
            + LAMBDA_TV * tv_loss
        )

        loss.backward(retain_graph=True)
        d_logps = logps.grad.detach()

        # Stage 2: replay VLM forward in small chunks and accumulate only the
        # visual gradient. Each chunk graph is freed immediately after backward.
        pixel_values_var = pixel_values.detach().requires_grad_(True)
        for start in range(0, len(coord_texts), CANDIDATE_BATCH_SIZE):
            batch_targets = coord_texts[start : start + CANDIDATE_BATCH_SIZE]
            batch_logps = sequence_logprobs_batch(
                model=model,
                processor=processor,
                pil_image=pil_image,
                pixel_values=pixel_values_var,
                prompt=GEOLOCATION_PROMPT,
                targets=batch_targets,
                device=device,
                debug=False,
            )
            batch_logps.backward(d_logps[start : start + len(batch_targets)])

        if pixel_values_var.grad is None:
            raise RuntimeError("No visual gradient was produced by the VLM forward pass.")
        pixel_values.backward(pixel_values_var.grad)
        optimizer.step()

        row = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "msgra_score": float(terms["msgra_score"].detach().cpu()),
            "risk": float(terms["risk"].detach().cpu()),
            "kl": float(terms["kl"].detach().cpu()),
            "near_far": float(terms["near_far"].detach().cpu()),
            "l_near": float(terms["l_near"].detach().cpu()),
            "l_far": float(terms["l_far"].detach().cpu()),
            "area": float(area_loss.detach().cpu()),
            "tv": float(tv_loss.detach().cpu()),
            "mask_min": float(mask.detach().min().cpu()),
            "mask_max": float(mask.detach().max().cpu()),
            "mask_mean": float(mask.detach().mean().cpu()),
        }
        log_rows.append(row)

        if step % SAVE_EVERY_STEPS == 0 or step == OPTIMIZATION_STEPS - 1:
            print(
                f"[2] step={step:03d} "
                f"loss={row['loss']:.4f} "
                f"score={row['msgra_score']:.4f} "
                f"risk={row['risk']:.4f} "
                f"kl={row['kl']:.4f} "
                f"nf={row['near_far']:.4f} "
                f"area={row['area']:.4f} "
                f"tv={row['tv']:.4f} "
                f"mask_max={row['mask_max']:.4f}"
            )

    # Final rendering and final bank distribution recomputation.
    with torch.no_grad():
        final_soft_mask_low = torch.sigmoid(mask_logits)
        final_mask = F.interpolate(
            final_soft_mask_low,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        final_rendered = image * (1.0 - final_mask) + blurred * final_mask
        final_pixel_values = make_model_pixel_values(final_rendered, mean, std, dtype)

        final_logps = compute_all_logps(
            model=model,
            processor=processor,
            pil_image=pil_image,
            pixel_values=final_pixel_values,
            prompt=GEOLOCATION_PROMPT,
            coord_texts=coord_texts,
            device=device,
            candidate_batch_size=CANDIDATE_BATCH_SIZE,
            debug_first_batch=False,
        )

        final_terms = compute_msgra_terms(
            logps=final_logps,
            distance_norm=distance_norm,
            weights=weights,
            soft_target=soft_target,
            near_mask=near_mask,
            far_mask=far_mask,
        )

        # Optional original/no-mask distribution for analysis.
        original_pixel_values = make_model_pixel_values(image, mean, std, dtype)
        original_logps = compute_all_logps(
            model=model,
            processor=processor,
            pil_image=pil_image,
            pixel_values=original_pixel_values,
            prompt=GEOLOCATION_PROMPT,
            coord_texts=coord_texts,
            device=device,
            candidate_batch_size=CANDIDATE_BATCH_SIZE,
            debug_first_batch=False,
        )
        original_terms = compute_msgra_terms(
            logps=original_logps,
            distance_norm=distance_norm,
            weights=weights,
            soft_target=soft_target,
            near_mask=near_mask,
            far_mask=far_mask,
        )

    final_top_anchors = summarize_top_anchors(
        coord_texts=coord_texts,
        q=final_terms["q"],
        logps=final_logps,
        distance_norm=distance_norm,
        weights=weights,
        group_name=bank.get("group_name"),
        topk=FINAL_TOPK_ANCHORS,
    )

    original_top_anchors = summarize_top_anchors(
        coord_texts=coord_texts,
        q=original_terms["q"],
        logps=original_logps,
        distance_norm=distance_norm,
        weights=weights,
        group_name=bank.get("group_name"),
        topk=FINAL_TOPK_ANCHORS,
    )

    # Save optimization log.
    with open(OUTPUT_LOG_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    # Save top anchors as JSON for inspection.
    top_anchor_manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_id": MODEL_ID,
        "prompt": GEOLOCATION_PROMPT,
        "reference": bank.get("reference"),
        "original_top_anchors": original_top_anchors,
        "final_top_anchors": final_top_anchors,
        "original_metrics": {
            "risk": float(original_terms["risk"].detach().cpu()),
            "kl": float(original_terms["kl"].detach().cpu()),
            "near_far": float(original_terms["near_far"].detach().cpu()),
            "l_near": float(original_terms["l_near"].detach().cpu()),
            "l_far": float(original_terms["l_far"].detach().cpu()),
        },
        "final_metrics": {
            "risk": float(final_terms["risk"].detach().cpu()),
            "kl": float(final_terms["kl"].detach().cpu()),
            "near_far": float(final_terms["near_far"].detach().cpu()),
            "l_near": float(final_terms["l_near"].detach().cpu()),
            "l_far": float(final_terms["l_far"].detach().cpu()),
        },
    }

    with open(OUTPUT_TOP_ANCHORS_JSON, "w", encoding="utf-8") as f:
        json.dump(top_anchor_manifest, f, ensure_ascii=False, indent=2)

    # Save main result.
    torch.save(
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "input_image_path": INPUT_IMAGE_PATH,
            "model_id": MODEL_ID,
            "prompt": GEOLOCATION_PROMPT,
            "coordinate_bank_json": COORDINATE_BANK_JSON,
            "coordinate_bank_pt": COORDINATE_BANK_PT,
            "reference": bank.get("reference"),
            "config": {
                "method": "MSGRA_DSDP_G_v2_semantic_shells_groupnorm_nf",
                "image_size": IMAGE_SIZE,
                "mask_low_res_h": MASK_LOW_RES_H,
                "mask_low_res_w": MASK_LOW_RES_W,
                "optimization_steps": OPTIMIZATION_STEPS,
                "learning_rate": LEARNING_RATE,
                "candidate_batch_size": CANDIDATE_BATCH_SIZE,
                "softmax_temperature": SOFTMAX_TEMPERATURE,
                "alpha_kl": ALPHA_KL,
                "beta_geo": BETA_GEO,
                "gamma_nf": GAMMA_NF,
                "logprob_reduction": LOGPROB_REDUCTION,
                "lambda_area": LAMBDA_AREA,
                "lambda_tv": LAMBDA_TV,
                "blur_kernel_size": BLUR_KERNEL_SIZE,
            },
            "mask": final_mask.detach().cpu(),
            "mask_low_res": final_soft_mask_low.detach().cpu(),
            "original_image": image.detach().cpu(),
            "blurred_image": blurred.detach().cpu(),
            "soft_masked_image": final_rendered.detach().cpu(),
            "optimization_log": log_rows,
            "coord_texts": coord_texts,
            "distance_norm": distance_norm.detach().cpu(),
            "weights": weights.detach().cpu(),
            "soft_target": soft_target.detach().cpu(),
            "near_mask": near_mask.detach().cpu(),
            "far_mask": far_mask.detach().cpu(),
            "original_logps": original_logps.detach().cpu(),
            "original_q": original_terms["q"].detach().cpu(),
            "final_logps": final_logps.detach().cpu(),
            "final_q": final_terms["q"].detach().cpu(),
            "original_top_anchors": original_top_anchors,
            "final_top_anchors": final_top_anchors,
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
    print(f"[2] Saved: {OUTPUT_TOP_ANCHORS_JSON}")

    print("[2] Original metrics:", top_anchor_manifest["original_metrics"])
    print("[2] Final metrics:", top_anchor_manifest["final_metrics"])


if __name__ == "__main__":
    main()






# python ./GTR-Heatmap/2_optimize_gtr_heatmap.py



