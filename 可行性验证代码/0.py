import json
import os
import re

import torch
from modelscope import AutoProcessor, Qwen3VLForConditionalGeneration


# ================= 0. 路径和参数设置 =================
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
USE_LOCAL_MODEL = True
IMAGE_PATH = "/root/autodl-tmp/DCSD分层提取凸包/中间件/dcsd_min_test/selected_blackout.jpg"
INTERMEDIATE_DIR = "/root/autodl-tmp/DCSD分层提取凸包/中间件"
OUTPUT_FILE = os.path.join(INTERMEDIATE_DIR, "01_qwen_geolocation.json")

MAX_NEW_TOKENS_LOCATION = 128
MAX_NEW_TOKENS_HIERARCHY = 1024
TORCH_DTYPE = torch.bfloat16

os.environ["OMP_NUM_THREADS"] = "1"


LOCATION_PROMPT = """
Estimate the latitude and longitude of this image using visible evidence.
Output exactly one XML block and nothing else:
<Location>
latitude, longitude
</Location>
"""


HIERARCHY_PROMPT = """
角色设定：
你是一个顶级的 geoguesser 与视觉语义分析专家。你的任务是深度解析输入图像，并提取出所有能够用于推断该照片拍摄地经纬度的物理实体线索。

提取规则：
请打破常规的平面视觉描述，采用“自顶向下（Top-Down）”的空间层级，严格按照以下三个尺度提取物理实体。提取的名词短语必须具体、包含可见视觉特征，并且能够直接指导下游图像分割模型。

层级 1：宏观环境 macro_environment
定义：预计占据画面 30% 以上面积，决定整体地理风貌、气候环境或城市基调的广域物理区域。
例子：“高层玻璃住宅楼群”、“潮湿沥青主干道”、“连续道路绿化带”。

层级 2：中观地标 meso_landmarks
定义：预计占据画面 5% 到 30% 面积，具有明确空间独立性和地理特异性的核心结构实体。
例子：“绿色悬臂交通标志”、“红色双层公交车”、“道路中央隔离花坛”。

层级 3：微观锚点 micro_anchors
定义：预计占据画面 5% 以下面积，不起眼但包含较高地理信息熵的细微人造物、标识、纹理或局部实体。
例子：“蓝色路牌”、“黄色交通信号箱”、“白色车道箭头”。

重要约束：
- 三个列表里的实体短语必须使用英文输出，JSON 键名保持不变。
- 不要输出天空、云、天气、光照、阴影、气氛。
- 不要输出城市名、国家名、街道名、地标名、品牌名、车牌号、路线号或可读文字内容。
- 不要输出抽象词，例如 infrastructure、urban scene、architecture、transportation。
- 优先输出可被分割模型定位的物理名词短语。
- 面积大的对象可以更细分，例如不同颜色车辆、建筑立面、道路区域、绿化带。
- 面积很小且同类重复的对象应合并成复数基础名词，例如 cars、traffic lights、street signs。
- 每个层级输出 3 到 10 个短语。不要无限枚举细碎零件。

输出格式要求：
请严格以 JSON 格式输出，不要包含任何额外解释性文字、Markdown 代码块标记或思维过程。JSON 结构必须严格如下：
{
  "macro_environment": ["实体1", "实体2"],
  "meso_landmarks": ["实体1", "实体2"],
  "micro_anchors": ["实体1", "实体2"]
}
"""


def extract_xml_block(text, tag):
    closed_pattern = rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>"
    closed_match = re.search(closed_pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if closed_match:
        return closed_match.group(1).strip()

    open_pattern = rf"<{re.escape(tag)}>\s*"
    open_match = re.search(open_pattern, text, flags=re.IGNORECASE)
    if not open_match:
        return ""
    start = open_match.end()
    next_tag = re.search(r"\n\s*<[^/][^>]*>\s*", text[start:], flags=re.IGNORECASE)
    end = start + next_tag.start() if next_tag else len(text)
    return text[start:end].strip()


def strip_json_text(text):
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end >= start:
        cleaned = cleaned[start:end + 1]
    return cleaned


def parse_hierarchy(raw_text):
    try:
        data = json.loads(strip_json_text(raw_text))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"分层 JSON 解析失败，请检查 raw_hierarchy_output。\n{exc}") from exc

    result = {}
    for key in ["macro_environment", "meso_landmarks", "micro_anchors"]:
        values = data.get(key, [])
        if not isinstance(values, list):
            values = []
        cleaned = []
        for item in values:
            item = str(item).strip()
            if item and item not in cleaned:
                cleaned.append(item)
        result[key] = cleaned
    return result


def run_generation(model, processor, prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": IMAGE_PATH},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    max_new_tokens = (
        MAX_NEW_TOKENS_LOCATION
        if prompt == LOCATION_PROMPT
        else MAX_NEW_TOKENS_HIERARCHY
    )
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def flatten_hierarchy(hierarchy):
    anchors = []
    levels = []
    for level in ["macro_environment", "meso_landmarks", "micro_anchors"]:
        for item in hierarchy[level]:
            if item not in anchors:
                anchors.append(item)
                levels.append(level)
    return anchors, levels


def main():
    print(f"正在加载 {MODEL_ID} 模型...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=TORCH_DTYPE,
        device_map="auto",
        local_files_only=USE_LOCAL_MODEL,
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        local_files_only=USE_LOCAL_MODEL,
    )
    print("模型与 Processor 加载完成")

    print(f"正在处理图像: {IMAGE_PATH}")
    print("正在生成坐标...")
    raw_location = run_generation(model, processor, LOCATION_PROMPT)
    location = extract_xml_block(raw_location, "Location")
    if not location:
        raise RuntimeError("坐标解析失败，请检查 raw_location_output。")

    print("正在生成分层目标词 JSON...")
    raw_hierarchy = run_generation(model, processor, HIERARCHY_PROMPT)
    hierarchy = parse_hierarchy(raw_hierarchy)
    anchors, anchor_levels = flatten_hierarchy(hierarchy)

    output = {
        "image_path": IMAGE_PATH,
        "model_id": MODEL_ID,
        "location": location,
        "hierarchy": hierarchy,
        "anchors": anchors,
        "anchor_levels": anchor_levels,
        "raw_location_output": raw_location,
        "raw_hierarchy_output": raw_hierarchy,
    }

    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("=" * 50)
    print(raw_location)
    print(raw_hierarchy)
    print("=" * 50)
    print(f"已保存分层中间件: {OUTPUT_FILE}")
    print(f"Location: {location}")
    print(f"Macro: {hierarchy['macro_environment']}")
    print(f"Meso: {hierarchy['meso_landmarks']}")
    print(f"Micro: {hierarchy['micro_anchors']}")


if __name__ == "__main__":
    main()



# python ./0.py