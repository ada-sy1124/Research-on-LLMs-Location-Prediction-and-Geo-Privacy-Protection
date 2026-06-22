#!/usr/bin/env python3
"""
Step 1: Build a legal coordinate candidate bank around the reference coordinate.
"""

import csv
import json
import math
import os
import random
from datetime import datetime



# ================= 0. PATHS AND PARAMETERS =================
REFERENCE_JSON = "/Applications/Documents/geoai/Research-on-LLMs-Location-Prediction-and-Geo-Privacy-Protection/data/gtr_heatmap/0_original_prediction.json"
OUTPUT_DIR = "/Applications/Documents/geoai/Research-on-LLMs-Location-Prediction-and-Geo-Privacy-Protection/data/gtr_heatmap"

# If step 0 cannot parse the VLM answer, fill these manually and rerun this script.
MANUAL_REFERENCE_LAT = None
MANUAL_REFERENCE_LON = None

LOCAL_RADII_KM = [0, 1, 5, 25, 100, 500, 1500, 5000]
BEARINGS_DEG = [0, 45, 90, 135, 180, 225, 270, 315]
RANDOM_FAR_POINTS = 32
RANDOM_SEED = 2026
MAX_DISTANCE_KM_FOR_NORMALIZATION = 5000.0

OUTPUT_JSON = f"{OUTPUT_DIR}/1_coordinate_bank.json"
OUTPUT_CSV = f"{OUTPUT_DIR}/1_coordinate_bank.csv"

# ================= 1. CODE =================

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
    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def destination_point(lat: float, lon: float, distance_km: float, bearing_deg: float):
    phi1 = math.radians(lat)
    lam1 = math.radians(lon)
    theta = math.radians(bearing_deg)
    delta = distance_km / EARTH_RADIUS_KM

    sin_phi2 = math.sin(phi1) * math.cos(delta) + math.cos(phi1) * math.sin(delta) * math.cos(theta)
    phi2 = math.asin(max(-1.0, min(1.0, sin_phi2)))
    y = math.sin(theta) * math.sin(delta) * math.cos(phi1)
    x = math.cos(delta) - math.sin(phi1) * math.sin(phi2)
    lam2 = lam1 + math.atan2(y, x)
    return clamp_lat(math.degrees(phi2)), wrap_lon(math.degrees(lam2))


def format_coord(lat: float, lon: float) -> str:
    return f"LAT={lat:+08.4f};LON={lon:+09.4f}"


def load_reference():
    if MANUAL_REFERENCE_LAT is not None and MANUAL_REFERENCE_LON is not None:
        return float(MANUAL_REFERENCE_LAT), float(MANUAL_REFERENCE_LON), "manual"

    with open(REFERENCE_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    coord = data.get("parsed_coordinate")
    if not coord:
        raise ValueError(
            "No parsed coordinate found in step 0 output. "
            "Set MANUAL_REFERENCE_LAT and MANUAL_REFERENCE_LON at the top of this script."
        )
    return float(coord["lat"]), float(coord["lon"]), "step0"


def add_candidate(rows, seen, lat, lon, ref_lat, ref_lon, source):
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
            "distance_norm": min(distance, MAX_DISTANCE_KM_FOR_NORMALIZATION) / MAX_DISTANCE_KM_FOR_NORMALIZATION,
            "source": source,
        }
    )
    seen.add(text)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ref_lat, ref_lon, ref_source = load_reference()
    rng = random.Random(RANDOM_SEED)

    rows = []
    seen = set()
    for radius in LOCAL_RADII_KM:
        if radius == 0:
            add_candidate(rows, seen, ref_lat, ref_lon, ref_lat, ref_lon, "reference")
            continue
        for bearing in BEARINGS_DEG:
            lat, lon = destination_point(ref_lat, ref_lon, radius, bearing)
            add_candidate(rows, seen, lat, lon, ref_lat, ref_lon, f"radius_{radius}_bearing_{bearing}")

    for _ in range(RANDOM_FAR_POINTS):
        lat = rng.uniform(-70.0, 70.0)
        lon = rng.uniform(-180.0, 180.0)
        add_candidate(rows, seen, lat, lon, ref_lat, ref_lon, "random_far")

    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "reference": {"lat": ref_lat, "lon": ref_lon, "source": ref_source},
        "max_distance_km_for_normalization": MAX_DISTANCE_KM_FOR_NORMALIZATION,
        "candidate_count": len(rows),
        "candidates": rows,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[1] Reference: lat={ref_lat:.4f}, lon={ref_lon:.4f} ({ref_source})")
    print(f"[1] Candidate count: {len(rows)}")
    print(f"[1] Saved: {OUTPUT_JSON}")
    print(f"[1] Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
