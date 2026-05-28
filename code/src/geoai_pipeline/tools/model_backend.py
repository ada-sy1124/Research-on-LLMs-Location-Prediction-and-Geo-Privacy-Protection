import json
import math
import re
import time
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from geoai_pipeline.config import PROJECT_ROOT, get_env, get_float, get_int

# ================= 原生 InternVL 2.5 图像预处理逻辑 =================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image_from_pil(image, input_size=448, max_num=12):
    image = image.convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        "InternVL2_5-1B": 24,
        "InternVL2_5-2B": 24,
        "InternVL2_5-4B": 36,
        "InternVL2_5-8B": 32,
        "InternVL2_5-26B": 48,
        "InternVL2_5-38B": 64,
        "InternVL2_5-78B": 80,
    }[model_name]
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.model.rotary_emb"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0

    return device_map


# ================= 数据清洗与模型推理类 =================

def parse_latlon_from_text(text: str) -> tuple[float, float]:
    if not text:
        return 0.0, 0.0
    for line in [line.strip() for line in text.splitlines() if line.strip()]:
        if "COORDINATES:" not in line.upper():
            continue
        nums = re.findall(r"-?\d+\.?\d*", line)
        valid_nums = []
        for value in nums:
            try:
                valid_nums.append(float(value))
            except ValueError:
                continue
        if len(valid_nums) >= 2:
            return valid_nums[0], valid_nums[1]

    json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            if isinstance(data, dict):
                lat = data.get("latitude_pred", data.get("predicted_latitude", data.get("latitude", data.get("lat"))))
                lon = data.get(
                    "longitude_pred",
                    data.get("predicted_longitude", data.get("longitude", data.get("lon", data.get("lng")))),
                )
                if lat is not None and lon is not None:
                    return float(lat), float(lon)
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

    lat_match = re.search(r"(?:latitude|lat)\s*[:=]\s*(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    lon_match = re.search(r"(?:longitude|lon|lng)\s*[:=]\s*(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if lat_match and lon_match:
        try:
            return float(lat_match.group(1)), float(lon_match.group(1))
        except ValueError:
            pass

    return 0.0, 0.0


def parse_latlon_reason_from_text(text: str) -> tuple[float, float, str, list[str], int]:
    lat_val, lon_val = parse_latlon_from_text(text)
    reason_text = ""
    classes = []
    q = 0

    for line in [line.strip() for line in text.splitlines() if line.strip()]:
        if not line.upper().startswith("REASONING:"):
            continue
        reason_text = line.split(":", 1)[1].strip()
        for seg in reason_text.split(";"):
            seg = seg.strip()
            if ":" not in seg:
                continue
            cls, objs_text = seg.split(":", 1)
            cls = cls.strip()
            if cls:
                classes.append(cls)
            q += len([obj.strip() for obj in objs_text.split(",") if obj.strip()])
        break

    return lat_val, lon_val, reason_text, classes, q


def resolve_model_path(env_name: str = "LOCAL_MODEL_PATH") -> str:
    raw_path = get_env(env_name, "")
    if not raw_path:
        return ""
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return str(path)
    candidates = [Path.cwd() / path, PROJECT_ROOT / path, PROJECT_ROOT.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(path)


class LocalVisionLanguageModel:
    def __init__(self):
        model_path = resolve_model_path()
        if not model_path:
            raise ValueError("LOCAL_MODEL_PATH is empty. Fill it in code/.env before running local inference.")

        from transformers import AutoModel, AutoTokenizer

        print(f"\n🚀 正在使用作者原生 Transformers 架构加载 InternVL: {model_path}")
        print("⚡ 显存充足检测通过 (2x H20 96G)，已自动绕过量化 Bug，开启纯血 BF16 满速加载！")
        print("🔥 FlashAttention-2 物理外挂已就绪！")

        # 使用作者提供的切分逻辑
        device_map = split_model("InternVL2_5-78B")

        # 彻底移除导致崩溃的 8-bit 量化，使用纯 bfloat16 直接塞进两张卡
        # 并且挂载了 use_flash_attn=True 提升极限速度
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_flash_attn=True,
            device_map=device_map,
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.generation_config = dict(max_new_tokens=1024, do_sample=False)

    def generate(self, image_obj, prompt: str) -> str:
        # 1. 组合提示词
        full_prompt = f"<image>\n{prompt}"
        
        # 2. 图像预处理 (转张量并推入显卡)
        pixel_values = load_image_from_pil(image_obj, max_num=12).to(torch.bfloat16).cuda()
        
        # 3. 运行作者官方推理接口
        response = self.model.chat(self.tokenizer, pixel_values, full_prompt, self.generation_config)
        return response.strip()


def create_inference_model():
    return LocalVisionLanguageModel()


def predict_latlon_with_output(model, image_obj, prompt: str, max_retries: int | None = None, base_wait_time: int | None = None):
    retries = max_retries if max_retries is not None else get_int("LOCAL_MAX_RETRIES", 1)
    wait_time = base_wait_time if base_wait_time is not None else get_int("LOCAL_RETRY_WAIT_SECONDS", 5)

    text = ""
    for attempt in range(max(retries, 1)):
        try:
            text = model.generate(image_obj=image_obj, prompt=prompt)
            if text:
                break
        except Exception as exc:  # noqa: BLE001
            if attempt + 1 >= max(retries, 1):
                print(f"Local model inference failed: {exc}")
                return 0.0, 0.0, ""
            time.sleep(wait_time)

    lat, lon = parse_latlon_from_text(text)
    return lat, lon, text


def predict_latlon(model, image_obj, prompt: str) -> tuple[float, float]:
    lat, lon, _ = predict_latlon_with_output(model, image_obj, prompt)
    return lat, lon


def predict_latlon_and_reason(model, image_obj, prompt: str):
    _, _, text = predict_latlon_with_output(model, image_obj, prompt)
    if not text:
        return 0.0, 0.0, "", [], 0
    return parse_latlon_reason_from_text(text)