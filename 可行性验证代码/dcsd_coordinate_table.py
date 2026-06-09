import json
import math
import re
from pathlib import Path


def parse_target_coord(text: str):
    match = re.search(
        r"([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)",
        text,
    )
    if not match:
        raise ValueError(
            "target_coord must be comma-separated, for example +51.4983,-000.1748"
        )
    return float(match.group(1)), float(match.group(2))


def normalize_target_coord(text: str) -> str:
    lat, lon = parse_target_coord(text)
    return f"{lat:+08.4f},{lon:+09.4f}"


def coordinate_char_weights(coord: str):
    comma = coord.index(",")
    metadata = [None for _ in coord]

    def fill_part(start: int, part: str, kind: str):
        sign = -1.0 if part.startswith("-") else 1.0
        dot_local = part.index(".")
        for local_idx, ch in enumerate(part):
            global_idx = start + local_idx
            if ch not in "0123456789":
                metadata[global_idx] = {"kind": "struct", "char": ch}
                continue
            if local_idx < dot_local:
                place = 10 ** (dot_local - local_idx - 1)
            else:
                place = 10 ** (-(local_idx - dot_local))
            metadata[global_idx] = {
                "kind": kind,
                "place": float(place),
                "sign": sign,
                "digit": int(ch),
            }

    fill_part(0, coord[:comma], "lat")
    metadata[comma] = {"kind": "struct", "char": ","}
    fill_part(comma + 1, coord[comma + 1 :], "lon")
    return metadata


def token_spans(tokenizer, target_coord: str, target_ids: list[int]):
    try:
        encoded = tokenizer(
            target_coord,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        if list(encoded["input_ids"]) == target_ids and "offset_mapping" in encoded:
            return [tuple(item) for item in encoded["offset_mapping"]]
    except Exception:
        pass

    spans = []
    cursor = 0
    for token_id in target_ids:
        piece = tokenizer.decode(
            [token_id],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not target_coord.startswith(piece, cursor):
            raise RuntimeError(
                "Could not recover tokenizer spans for target coordinate. "
                f"Failed at token {token_id!r}, decoded as {piece!r}."
            )
        spans.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    if cursor != len(target_coord):
        raise RuntimeError("Tokenizer span recovery did not consume the whole coordinate.")
    return spans


def decoded_vocab_tokens(tokenizer, cache_path: Path):
    cache_path = Path(cache_path)
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}

    vocab_size = len(tokenizer)
    decoded = {}
    for token_id in range(vocab_size):
        try:
            text = tokenizer.decode(
                [token_id],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except Exception:
            text = ""
        decoded[token_id] = text

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in decoded.items()}, f, ensure_ascii=False)
    return decoded


def build_coordinate_cost_table(
    tokenizer,
    target_coord: str,
    distance_norm_km: float,
    cache_dir,
):
    import torch

    cache_dir = Path(cache_dir)
    target_coord = normalize_target_coord(target_coord)
    target_ids = tokenizer(target_coord, add_special_tokens=False)["input_ids"]
    if not target_ids:
        raise ValueError("Target coordinate produced no target tokens.")

    true_lat, _ = parse_target_coord(target_coord)
    cos_lat = math.cos(math.radians(true_lat))
    metadata = coordinate_char_weights(target_coord)
    spans = token_spans(tokenizer, target_coord, target_ids)
    decoded = decoded_vocab_tokens(tokenizer, cache_dir / "decoded_vocab_tokens.json")

    vocab_size = len(tokenizer)
    valid_mask = torch.zeros(len(target_ids), vocab_size, dtype=torch.bool)
    distance_table = torch.zeros(len(target_ids), vocab_size, dtype=torch.float32)

    for t, (start, end) in enumerate(spans):
        true_piece = target_coord[start:end]
        for token_id, piece in decoded.items():
            if len(piece) != len(true_piece):
                continue

            delta_lat = 0.0
            delta_lon = 0.0
            ok = True
            for offset, new_ch in enumerate(piece):
                pos = start + offset
                info = metadata[pos]
                if info["kind"] == "struct":
                    if new_ch != info["char"]:
                        ok = False
                        break
                    continue
                if new_ch not in "0123456789":
                    ok = False
                    break

                delta_digit = int(new_ch) - info["digit"]
                delta_degree = info["sign"] * delta_digit * info["place"]
                if info["kind"] == "lat":
                    delta_lat += delta_degree
                elif info["kind"] == "lon":
                    delta_lon += delta_degree

            if not ok:
                continue

            km = math.sqrt((111.32 * delta_lat) ** 2 + (111.32 * cos_lat * delta_lon) ** 2)
            valid_mask[t, token_id] = True
            distance_table[t, token_id] = min(km / distance_norm_km, 1.0)

        true_id = target_ids[t]
        valid_mask[t, true_id] = True
        distance_table[t, true_id] = 0.0

    return {
        "target_coord": target_coord,
        "target_ids": target_ids,
        "spans": spans,
        "valid_mask": valid_mask,
        "distance_table": distance_table,
        "distance_norm_km": float(distance_norm_km),
    }


def save_coordinate_cost_table(table: dict, output_file):
    import torch

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(table, output_file)


def load_coordinate_cost_table(output_file):
    import torch

    return torch.load(output_file, map_location="cpu")


def write_table_report(tokenizer, table: dict, report_file, top_k: int = 12):
    report_file = Path(report_file)
    report_file.parent.mkdir(parents=True, exist_ok=True)

    target_ids = table["target_ids"]
    spans = table["spans"]
    valid_mask = table["valid_mask"]
    distance_table = table["distance_table"]
    target_coord = table["target_coord"]

    rows = []
    for idx, token_id in enumerate(target_ids):
        valid_ids = valid_mask[idx].nonzero(as_tuple=False).flatten()
        valid_dist = distance_table[idx, valid_ids]
        order = valid_dist.argsort(descending=True)[:top_k]
        candidates = []
        for item in order.tolist():
            cand_id = int(valid_ids[item])
            candidates.append(
                {
                    "token_id": cand_id,
                    "token": tokenizer.decode(
                        [cand_id],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ),
                    "distance_norm": float(distance_table[idx, cand_id]),
                }
            )
        start, end = spans[idx]
        rows.append(
            {
                "position": idx,
                "span": [start, end],
                "target_piece": target_coord[start:end],
                "target_token_id": int(token_id),
                "valid_token_count": int(valid_mask[idx].sum()),
                "top_distance_candidates": candidates,
            }
        )

    report = {
        "target_coord": target_coord,
        "distance_norm_km": table["distance_norm_km"],
        "target_token_count": len(target_ids),
        "valid_token_counts": [int(x) for x in valid_mask.sum(dim=1)],
        "positions": rows,
    }
    with report_file.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report
