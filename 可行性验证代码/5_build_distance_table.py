import json
import math
import os

import torch

from dcsd_coordinate_table import parse_target_coord


# ================= 0. 路径和参数设置 =================
INTERMEDIATE_DIR = "/root/autodl-tmp/DCSD分层提取凸包/中间件"
OUTPUT_DIR = f"{INTERMEDIATE_DIR}/dcsd_min_test"
VALID_TABLE_FILE = f"{OUTPUT_DIR}/04_valid_token_table.pt"
DECODED_VOCAB_FILE = f"{OUTPUT_DIR}/decoded_vocab_tokens.json"
COST_TABLE_FILE = f"{OUTPUT_DIR}/05_coordinate_cost_table.pt"
REPORT_FILE = f"{OUTPUT_DIR}/05_coordinate_cost_table_report.json"

DISTANCE_NORM_KM = 10000.0
TOP_K = 12

os.environ.setdefault("OMP_NUM_THREADS", "1")


def token_delta_km(piece, target_piece, metadata, start, cos_lat):
    if len(piece) != len(target_piece):
        return None

    delta_lat = 0.0
    delta_lon = 0.0
    for offset, new_ch in enumerate(piece):
        info = metadata[start + offset]
        if info["kind"] == "struct":
            if new_ch != info["char"]:
                return None
            continue
        if new_ch not in "0123456789":
            return None

        delta_digit = int(new_ch) - info["digit"]
        delta_degree = info["sign"] * delta_digit * info["place"]
        if info["kind"] == "lat":
            delta_lat += delta_degree
        elif info["kind"] == "lon":
            delta_lon += delta_degree

    return math.sqrt((111.32 * delta_lat) ** 2 + (111.32 * cos_lat * delta_lon) ** 2)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    valid_table = torch.load(VALID_TABLE_FILE, map_location="cpu")
    with open(DECODED_VOCAB_FILE, "r", encoding="utf-8") as f:
        decoded = {int(k): v for k, v in json.load(f).items()}

    target_coord = valid_table["target_coord"]
    target_ids = valid_table["target_ids"]
    spans = valid_table["spans"]
    metadata = valid_table["char_metadata"]
    valid_mask = valid_table["valid_mask"]

    true_lat, _ = parse_target_coord(target_coord)
    cos_lat = math.cos(math.radians(true_lat))
    distance_table = torch.zeros_like(valid_mask, dtype=torch.float32)

    report_positions = []
    for position, (start, end) in enumerate(spans):
        target_piece = target_coord[start:end]
        valid_ids = valid_mask[position].nonzero(as_tuple=False).flatten()
        for token_id_tensor in valid_ids:
            token_id = int(token_id_tensor)
            km = token_delta_km(decoded[token_id], target_piece, metadata, start, cos_lat)
            if km is None:
                continue
            distance_table[position, token_id] = min(km / DISTANCE_NORM_KM, 1.0)

        true_id = int(target_ids[position])
        distance_table[position, true_id] = 0.0

        valid_ids = valid_mask[position].nonzero(as_tuple=False).flatten()
        valid_dist = distance_table[position, valid_ids]
        order = valid_dist.argsort(descending=True)[:TOP_K]
        candidates = []
        for item in order.tolist():
            cand_id = int(valid_ids[item])
            candidates.append(
                {
                    "token_id": cand_id,
                    "token": decoded[cand_id],
                    "distance_norm": float(distance_table[position, cand_id]),
                }
            )
        report_positions.append(
            {
                "position": position,
                "span": [start, end],
                "target_piece": target_piece,
                "target_token_id": true_id,
                "valid_token_count": int(valid_mask[position].sum()),
                "top_distance_candidates": candidates,
            }
        )

    cost_table = {
        "target_coord": target_coord,
        "target_ids": target_ids,
        "spans": spans,
        "valid_mask": valid_mask,
        "distance_table": distance_table,
        "distance_norm_km": float(DISTANCE_NORM_KM),
    }
    torch.save(cost_table, COST_TABLE_FILE)

    report = {
        "target_coord": target_coord,
        "distance_norm_km": DISTANCE_NORM_KM,
        "target_token_count": len(target_ids),
        "valid_token_counts": [int(x) for x in valid_mask.sum(dim=1)],
        "positions": report_positions,
    }
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"已保存距离查询表: {COST_TABLE_FILE}")
    print(f"已保存可读报告: {REPORT_FILE}")


if __name__ == "__main__":
    main()



# python ./5_build_distance_table.py