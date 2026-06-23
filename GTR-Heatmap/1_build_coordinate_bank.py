#!/usr/bin/env python3
"""
Step 1: Build a fixed multi-scale geodesic anchor bank for MSGRA / DSDP-G.

Key design:
- Use the model's original prediction g0 as the explanation center.
- Use semantic shell groups, not one-shell-per-radius.
- Assign priors at the semantic group level.
- Generate local/regional/far/global/antipodal anchors.
- Precompute Haversine distances, normalized distances, shell weights,
  Gaussian soft target K, near/far masks, and save both JSON/CSV/PT.
"""

import csv
import json
import math
import os
from datetime import datetime

import torch


# ================= 0. PATHS AND PARAMETERS =================

REFERENCE_JSON = "/root/autodl-tmp/Research-on-LLMs-Location-Prediction-and-Geo-Privacy-Protection/data/gtr_heatmap/0_original_prediction.json"
OUTPUT_DIR = "/root/autodl-tmp/Research-on-LLMs-Location-Prediction-and-Geo-Privacy-Protection/data/gtr_heatmap"

# If Step 0 cannot parse the VLM answer, fill these manually.
MANUAL_REFERENCE_LAT = None
MANUAL_REFERENCE_LON = None

# Direction resolution. Use 8 for debugging, 16 for paper-grade runs.
BEARINGS_DEG = [i * 22.5 for i in range(16)]

# Semantic shell groups.
# These groups, not individual radii, receive the top-level prior weights.
SHELL_GROUPS = {
    "reference": [0],
    "local": [1, 5, 10, 25, 50],
    "regional": [100, 200, 500],
    "far": [750, 1500, 2500, 5000],
    "global": [7500, 10000, 15000],
    "antipodal": [19000],
}

# Group priors should sum to 1 after normalization.
# These are intentionally not all equal, because a single reference point should
# not automatically receive the same total mass as a whole multi-radius shell.
GROUP_PRIORS = {
    "reference": 0.10,
    "local": 0.25,
    "regional": 0.15,
    "far": 0.20,
    "global": 0.20,
    "antipodal": 0.10,
}

D_MAX_KM = 20000.0
SOFT_TARGET_SIGMA_KM = 50.0

# Near and far definitions for near-far contrast in Step 2.
NEAR_MAX_KM = 100.0
FAR_MIN_KM = 5000.0

OUTPUT_JSON = f"{OUTPUT_DIR}/1_coordinate_bank.json"
OUTPUT_CSV = f"{OUTPUT_DIR}/1_coordinate_bank.csv"
OUTPUT_PT = f"{OUTPUT_DIR}/1_coordinate_bank.pt"


# ================= 1. GEO UTILS =================

EARTH_RADIUS_KM = 6371.0088


def clamp_lat(lat: float) -> float:
    return max(-90.0, min(90.0, lat))


def wrap_lon(lon: float) -> float:
    return ((lon + 180.0) % 360.0) - 180.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(max(0.0, min(1.0, a))))


def destination_point(lat: float, lon: float, distance_km: float, bearing_deg: float):
    """
    Great-circle destination from (lat, lon), distance in km, bearing in degrees.
    """
    phi1 = math.radians(lat)
    lam1 = math.radians(lon)
    theta = math.radians(bearing_deg)
    delta = distance_km / EARTH_RADIUS_KM

    sin_phi2 = (
        math.sin(phi1) * math.cos(delta)
        + math.cos(phi1) * math.sin(delta) * math.cos(theta)
    )
    phi2 = math.asin(max(-1.0, min(1.0, sin_phi2)))

    y = math.sin(theta) * math.sin(delta) * math.cos(phi1)
    x = math.cos(delta) - math.sin(phi1) * math.sin(phi2)
    lam2 = lam1 + math.atan2(y, x)

    return clamp_lat(math.degrees(phi2)), wrap_lon(math.degrees(lam2))


def antipodal_point(lat: float, lon: float):
    return clamp_lat(-lat), wrap_lon(lon + 180.0)


def format_coord(lat: float, lon: float) -> str:
    # Fixed-width formatting reduces token-length variation.
    return f"LAT={lat:+08.4f};LON={lon:+09.4f}"


def load_reference():
    if MANUAL_REFERENCE_LAT is not None and MANUAL_REFERENCE_LON is not None:
        return float(MANUAL_REFERENCE_LAT), float(MANUAL_REFERENCE_LON), "manual"

    with open(REFERENCE_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    coord = data.get("parsed_coordinate")
    if not coord:
        raise ValueError(
            "No parsed coordinate found in Step 0 output. "
            "Set MANUAL_REFERENCE_LAT and MANUAL_REFERENCE_LON at the top of this script."
        )

    return float(coord["lat"]), float(coord["lon"]), "step0"


# ================= 2. BANK BUILDING =================

def normalize_group_priors(group_priors: dict[str, float]) -> dict[str, float]:
    total = sum(float(v) for v in group_priors.values())
    if total <= 0:
        raise ValueError("GROUP_PRIORS must have positive total mass.")
    return {k: float(v) / total for k, v in group_priors.items()}


def add_candidate(
    rows: list[dict],
    seen: set[str],
    lat: float,
    lon: float,
    ref_lat: float,
    ref_lon: float,
    group_id: int,
    group_name: str,
    radius_km: float,
    source: str,
):
    lat = round(clamp_lat(lat), 4)
    lon = round(wrap_lon(lon), 4)
    text = format_coord(lat, lon)

    if text in seen:
        return

    distance = haversine_km(ref_lat, ref_lon, lat, lon)

    rows.append(
        {
            "id": len(rows),
            "coord_text": text,
            "lat": lat,
            "lon": lon,
            "distance_km": distance,
            "D_norm": min(distance, D_MAX_KM) / D_MAX_KM,
            "distance_norm": min(distance, D_MAX_KM) / D_MAX_KM,
            "group_id": group_id,
            "group_name": group_name,
            "shell_id": group_id,
            "shell_name": group_name,
            "shell_radius_km": radius_km,
            "weight": None,
            "group_prior": None,
            "shell_weight": None,
            "is_near": distance <= NEAR_MAX_KM,
            "is_far": distance >= FAR_MIN_KM,
            "soft_target": None,
            "source": source,
        }
    )
    seen.add(text)


def build_rows(ref_lat: float, ref_lon: float) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()

    group_names = list(SHELL_GROUPS.keys())

    for group_id, group_name in enumerate(group_names):
        radii = SHELL_GROUPS[group_name]

        for radius in radii:
            if radius == 0:
                add_candidate(
                    rows=rows,
                    seen=seen,
                    lat=ref_lat,
                    lon=ref_lon,
                    ref_lat=ref_lat,
                    ref_lon=ref_lon,
                    group_id=group_id,
                    group_name=group_name,
                    radius_km=radius,
                    source="reference",
                )
                continue

            for bearing in BEARINGS_DEG:
                lat, lon = destination_point(ref_lat, ref_lon, radius, bearing)
                add_candidate(
                    rows=rows,
                    seen=seen,
                    lat=lat,
                    lon=lon,
                    ref_lat=ref_lat,
                    ref_lon=ref_lon,
                    group_id=group_id,
                    group_name=group_name,
                    radius_km=radius,
                    source=f"{group_name}_radius_{radius}_bearing_{bearing}",
                )

    # Exact antipode belongs to the same semantic group "antipodal".
    if "antipodal" in group_names:
        antipodal_group_id = group_names.index("antipodal")
    else:
        antipodal_group_id = len(group_names)
        group_names.append("antipodal")

    anti_lat, anti_lon = antipodal_point(ref_lat, ref_lon)
    anti_distance = haversine_km(ref_lat, ref_lon, anti_lat, anti_lon)

    add_candidate(
        rows=rows,
        seen=seen,
        lat=anti_lat,
        lon=anti_lon,
        ref_lat=ref_lat,
        ref_lon=ref_lon,
        group_id=antipodal_group_id,
        group_name="antipodal",
        radius_km=anti_distance,
        source="antipodal_exact",
    )

    return rows


def add_weights_and_soft_target(rows: list[dict]):
    group_priors = normalize_group_priors(GROUP_PRIORS)

    group_counts: dict[str, int] = {}
    for row in rows:
        group_counts[row["group_name"]] = group_counts.get(row["group_name"], 0) + 1

    # Per-anchor weights: group prior divided equally within group.
    unnormalized_k: list[float] = []
    for row in rows:
        group_name = row["group_name"]
        prior = group_priors.get(group_name, 0.0)

        if prior <= 0:
            raise ValueError(f"Missing or non-positive prior for group: {group_name}")

        weight = prior / group_counts[group_name]
        row["group_prior"] = prior
        row["shell_weight"] = prior
        row["weight"] = weight

        distance = float(row["distance_km"])
        k_value = weight * math.exp(-(distance ** 2) / (2.0 * SOFT_TARGET_SIGMA_KM ** 2))
        unnormalized_k.append(k_value)

    k_sum = sum(unnormalized_k)
    if k_sum <= 0:
        raise ValueError("Soft target normalization failed; K sum is non-positive.")

    for row, k_value in zip(rows, unnormalized_k):
        row["soft_target"] = k_value / k_sum


def sanity_check_bank(rows: list[dict]):
    if not rows:
        raise ValueError("Empty coordinate bank.")

    near_count = sum(1 for r in rows if r["is_near"])
    far_count = sum(1 for r in rows if r["is_far"])

    if near_count == 0:
        raise ValueError("No near candidates. Increase NEAR_MAX_KM or adjust radii.")

    if far_count == 0:
        raise ValueError("No far candidates. Decrease FAR_MIN_KM or add global radii.")

    weight_sum = sum(float(r["weight"]) for r in rows)
    soft_sum = sum(float(r["soft_target"]) for r in rows)

    if abs(weight_sum - 1.0) > 1e-5:
        raise ValueError(f"Anchor weights do not sum to 1. Got {weight_sum}")

    if abs(soft_sum - 1.0) > 1e-5:
        raise ValueError(f"Soft target does not sum to 1. Got {soft_sum}")


def save_outputs(rows: list[dict], ref_lat: float, ref_lon: float, ref_source: str):
    created_at = datetime.now().isoformat(timespec="seconds")

    result = {
        "created_at": created_at,
        "method": "MSGRA_DSDP_G_bank_v2_semantic_shells",
        "reference": {"lat": ref_lat, "lon": ref_lon, "source": ref_source},
        "d_max_km": D_MAX_KM,
        "soft_target_sigma_km": SOFT_TARGET_SIGMA_KM,
        "near_max_km": NEAR_MAX_KM,
        "far_min_km": FAR_MIN_KM,
        "shell_groups": SHELL_GROUPS,
        "group_priors": normalize_group_priors(GROUP_PRIORS),
        "bearings_deg": BEARINGS_DEG,
        "candidate_count": len(rows),
        "group_count": len({row["group_name"] for row in rows}),
        "near_count": sum(1 for row in rows if row["is_near"]),
        "far_count": sum(1 for row in rows if row["is_far"]),
        "candidates": rows,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    torch.save(
        {
            "created_at": created_at,
            "method": result["method"],
            "reference": result["reference"],
            "d_max_km": D_MAX_KM,
            "soft_target_sigma_km": SOFT_TARGET_SIGMA_KM,
            "near_max_km": NEAR_MAX_KM,
            "far_min_km": FAR_MIN_KM,
            "shell_groups": SHELL_GROUPS,
            "group_priors": result["group_priors"],
            "bearings_deg": BEARINGS_DEG,
            "coord_texts": [row["coord_text"] for row in rows],
            "anchors": torch.tensor([[float(row["lat"]), float(row["lon"])] for row in rows], dtype=torch.float32),
            "distance_km": torch.tensor([float(row["distance_km"]) for row in rows], dtype=torch.float32),
            "D_norm": torch.tensor([float(row["D_norm"]) for row in rows], dtype=torch.float32),
            "weights": torch.tensor([float(row["weight"]) for row in rows], dtype=torch.float32),
            "soft_target": torch.tensor([float(row["soft_target"]) for row in rows], dtype=torch.float32),
            "near_mask": torch.tensor([bool(row["is_near"]) for row in rows], dtype=torch.bool),
            "far_mask": torch.tensor([bool(row["is_far"]) for row in rows], dtype=torch.bool),
            "group_id": torch.tensor([int(row["group_id"]) for row in rows], dtype=torch.long),
            "shell_id": torch.tensor([int(row["shell_id"]) for row in rows], dtype=torch.long),
            "group_name": [row["group_name"] for row in rows],
            "shell_name": [row["shell_name"] for row in rows],
            "candidates": rows,
        },
        OUTPUT_PT,
    )

    print(f"[1] Saved: {OUTPUT_JSON}")
    print(f"[1] Saved: {OUTPUT_CSV}")
    print(f"[1] Saved: {OUTPUT_PT}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ref_lat, ref_lon, ref_source = load_reference()

    rows = build_rows(ref_lat, ref_lon)
    add_weights_and_soft_target(rows)
    sanity_check_bank(rows)
    save_outputs(rows, ref_lat, ref_lon, ref_source)

    print(f"[1] Reference: lat={ref_lat:.4f}, lon={ref_lon:.4f} ({ref_source})")
    print(f"[1] Candidate count: {len(rows)}")
    print(f"[1] Semantic groups: {sorted({row['group_name'] for row in rows})}")
    print(f"[1] Near count: {sum(1 for row in rows if row['is_near'])}")
    print(f"[1] Far count: {sum(1 for row in rows if row['is_far'])}")


if __name__ == "__main__":
    main()




# python ./GTR-Heatmap/1_build_coordinate_bank.py