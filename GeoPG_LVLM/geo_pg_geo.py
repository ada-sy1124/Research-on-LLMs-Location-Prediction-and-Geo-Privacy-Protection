import math
import re
from dataclasses import dataclass

import torch


EARTH_RADIUS_KM = 6371.0088
COORD_RE = re.compile(
    r"^\s*(?P<lat>[+-]?\d+(?:\.\d+)?)\s*,\s*(?P<lon>[+-]?\d+(?:\.\d+)?)\s*$"
)


@dataclass(frozen=True)
class Coordinate:
    lat: float
    lon: float
    text: str


def parse_coordinate(text):
    match = COORD_RE.match(text)
    if match is None:
        raise ValueError(f"not a strict coordinate answer: {text!r}")

    lat = float(match.group("lat"))
    lon = float(match.group("lon"))
    if lat < -90.0 or lat > 90.0:
        raise ValueError(f"latitude out of range: {text!r}")
    if lon < -180.0 or lon > 180.0:
        raise ValueError(f"longitude out of range: {text!r}")
    return Coordinate(lat=lat, lon=lon, text=text.strip())


def haversine_km(coord, reference):
    lat = math.radians(float(coord.lat))
    lon = math.radians(float(coord.lon))
    lat0 = math.radians(float(reference.lat))
    lon0 = math.radians(float(reference.lon))
    d_lat = lat - lat0
    d_lon = lon - lon0
    hav = (
        math.sin(d_lat / 2.0) ** 2
        + math.cos(lat0) * math.cos(lat) * math.sin(d_lon / 2.0) ** 2
    )
    hav = min(max(hav, 0.0), 1.0)
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(hav))


def clipped_haversine_reward(coord, reference, d_max_km):
    distance = haversine_km(coord, reference)
    return min(distance, float(d_max_km)) / float(d_max_km), distance


def reward_to_advantage(rewards, device, eps, clip_adv, min_std):
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    std = rewards.std(unbiased=False)
    if float(std.detach().cpu()) < float(min_std):
        return rewards, torch.zeros_like(rewards), std
    adv = (rewards - rewards.mean()) / (std + float(eps))
    return rewards, adv.clamp(-float(clip_adv), float(clip_adv)).detach(), std
