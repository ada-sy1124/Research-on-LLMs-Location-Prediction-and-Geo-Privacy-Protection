import json
import re
import time

from google.genai import types


def parse_latlon_from_text(text: str) -> tuple[float, float]:
    """Parse latitude/longitude from Gemini text output."""
    if not text:
        return 0.0, 0.0

    for line in [l.strip() for l in text.splitlines() if l.strip()]:
        if "COORDINATES:" not in line.upper():
            continue
        nums = re.findall(r"-?\d+\.?\d*", line)
        valid_nums = []
        for n in nums:
            try:
                valid_nums.append(float(n))
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


def gemini_predict_latlon_with_output(
    client,
    model: str,
    image_obj,
    prompt: str,
    temperature: float = 0.0,
    max_retries: int = 5,
    base_wait_time: int = 5,
):
    text = ""

    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[image_obj, prompt],
                config=types.GenerateContentConfig(temperature=temperature),
            )
            text = (resp.text or "").strip()
            if text:
                break
        except Exception as e:  # noqa: BLE001
            error_msg = str(e)
            if "503" in error_msg or "429" in error_msg:
                wait_time = base_wait_time * (attempt + 1)
                print(f"API busy (503/429), retry {attempt + 1}/{max_retries}, wait {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Fatal API error: {e}")
                return 0.0, 0.0, ""

    if not text:
        return 0.0, 0.0, ""

    lat, lon = parse_latlon_from_text(text)
    return lat, lon, text


def gemini_predict_latlon(client, model: str, image_obj, prompt: str, temperature: float = 0.0, max_retries: int = 5, base_wait_time: int = 5):
    text = ""

    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[image_obj, prompt],
                config=types.GenerateContentConfig(temperature=temperature),
            )
            text = (resp.text or "").strip()
            if text:
                break
        except Exception as e:  # noqa: BLE001
            error_msg = str(e)
            if "503" in error_msg or "429" in error_msg:
                wait_time = base_wait_time * (attempt + 1)
                print(f"⚠️ API 繁忙 (503/429)，第 {attempt + 1}/{max_retries} 次重试，等待 {wait_time} 秒...")
                time.sleep(wait_time)
            else:
                print(f"❌ API 致命错误: {e}")
                return 0.0, 0.0

    if not text:
        return 0.0, 0.0

    return parse_latlon_from_text(text)
