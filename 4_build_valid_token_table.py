import json
import os
from pathlib import Path

import torch
from modelscope import AutoProcessor

from dcsd_coordinate_table import decoded_vocab_tokens


# ================= 0. 路径和参数设置 =================
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
USE_LOCAL_MODEL = True
INTERMEDIATE_DIR = "/root/autodl-tmp/DCSD分层提取凸包/中间件"
OUTPUT_DIR = f"{INTERMEDIATE_DIR}/dcsd_min_test"
COORD_TOKENS_FILE = f"{OUTPUT_DIR}/03_coordinate_tokens.json"
DECODED_VOCAB_FILE = f"{OUTPUT_DIR}/decoded_vocab_tokens.json"
VALID_TABLE_FILE = f"{OUTPUT_DIR}/04_valid_token_table.pt"
REPORT_FILE = f"{OUTPUT_DIR}/04_valid_token_table_report.json"

MAX_EXAMPLES_PER_POSITION = 30

os.environ.setdefault("OMP_NUM_THREADS", "1")


def is_token_valid_for_span(piece, target_piece, metadata, start):
    if len(piece) != len(target_piece):
        return False
    for offset, new_ch in enumerate(piece):
        info = metadata[start + offset]
        if info["kind"] == "struct":
            if new_ch != info["char"]:
                return False
        elif new_ch not in "0123456789":
            return False
    return True


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(COORD_TOKENS_FILE, "r", encoding="utf-8") as f:
        spec = json.load(f)

    print(f"加载 tokenizer: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        local_files_only=USE_LOCAL_MODEL,
    )
    tokenizer = processor.tokenizer
    decoded = decoded_vocab_tokens(tokenizer, Path(DECODED_VOCAB_FILE))

    vocab_size = len(tokenizer)
    target_ids = spec["target_ids"]
    spans = spec["spans"]
    metadata = spec["char_metadata"]
    target_coord = spec["target_coord"]

    valid_mask = torch.zeros(len(target_ids), vocab_size, dtype=torch.bool)
    report_positions = []

    for position, (start, end) in enumerate(spans):
        target_piece = target_coord[start:end]
        examples = []
        for token_id, piece in decoded.items():
            if not is_token_valid_for_span(piece, target_piece, metadata, start):
                continue
            valid_mask[position, token_id] = True
            if len(examples) < MAX_EXAMPLES_PER_POSITION:
                examples.append({"token_id": int(token_id), "token": piece})

        true_id = target_ids[position]
        valid_mask[position, true_id] = True
        report_positions.append(
            {
                "position": position,
                "span": [start, end],
                "target_piece": target_piece,
                "target_token_id": int(true_id),
                "valid_token_count": int(valid_mask[position].sum()),
                "examples": examples,
            }
        )

    table = {
        "target_coord": target_coord,
        "target_ids": target_ids,
        "spans": spans,
        "char_metadata": metadata,
        "valid_mask": valid_mask,
    }
    torch.save(table, VALID_TABLE_FILE)

    report = {
        "target_coord": target_coord,
        "target_token_count": len(target_ids),
        "valid_token_counts": [int(x) for x in valid_mask.sum(dim=1)],
        "positions": report_positions,
    }
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"已保存合法 token 表: {VALID_TABLE_FILE}")
    print(f"已保存可读报告: {REPORT_FILE}")
    print(f"每个位置合法 token 数量: {report['valid_token_counts']}")


if __name__ == "__main__":
    main()


# python ./4_build_valid_token_table.py