import json
import os

from modelscope import AutoProcessor

from dcsd_coordinate_table import (
    coordinate_char_weights,
    normalize_target_coord,
    token_spans,
)


# ================= 0. 路径和参数设置 =================
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
USE_LOCAL_MODEL = True
INTERMEDIATE_DIR = "/root/autodl-tmp/DCSD分层提取凸包/中间件"
FEATURES_FILE = f"{INTERMEDIATE_DIR}/01_qwen_geolocation.json"
OUTPUT_DIR = f"{INTERMEDIATE_DIR}/dcsd_min_test"
OUTPUT_FILE = f"{OUTPUT_DIR}/03_coordinate_tokens.json"

# 留空则自动使用 0.py 保存的 Location；如果你有真实 GT 坐标，就填这里。
TARGET_COORD = ""

os.environ.setdefault("OMP_NUM_THREADS", "1")


def load_target_coord():
    if TARGET_COORD:
        return normalize_target_coord(TARGET_COORD)
    with open(FEATURES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    location = data.get("location", "")
    if not location:
        raise RuntimeError(
            "没有填写 TARGET_COORD，且 0.py 的中间件里没有 location。"
        )
    return normalize_target_coord(location)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    target_coord = load_target_coord()
    print(f"加载 tokenizer: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        local_files_only=USE_LOCAL_MODEL,
    )
    tokenizer = processor.tokenizer

    target_ids = tokenizer(target_coord, add_special_tokens=False)["input_ids"]
    spans = token_spans(tokenizer, target_coord, target_ids)
    metadata = coordinate_char_weights(target_coord)

    token_rows = []
    for index, token_id in enumerate(target_ids):
        start, end = spans[index]
        token_rows.append(
            {
                "position": index,
                "token_id": int(token_id),
                "token_text": tokenizer.decode(
                    [token_id],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ),
                "span": [start, end],
                "target_piece": target_coord[start:end],
            }
        )

    output = {
        "model_id": MODEL_ID,
        "features_file": FEATURES_FILE,
        "target_coord": target_coord,
        "target_ids": [int(x) for x in target_ids],
        "spans": [[int(a), int(b)] for a, b in spans],
        "char_metadata": metadata,
        "tokens": token_rows,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"已保存坐标 token 化结果: {OUTPUT_FILE}")
    print(f"target_coord: {target_coord}")
    print(f"target token 数量: {len(target_ids)}")
    for row in token_rows:
        print(
            f"pos={row['position']:02d} id={row['token_id']} "
            f"text={row['token_text']!r} span={row['span']} piece={row['target_piece']!r}"
        )


if __name__ == "__main__":
    main()



# python ./3_prepare_coordinate_tokens.py