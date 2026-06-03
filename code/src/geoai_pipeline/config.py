import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

# 优先读取 code/.env（如存在）
load_dotenv(dotenv_path=ENV_FILE if ENV_FILE.exists() else None)


def get_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def get_int(name: str, default: int) -> int:
    value = get_env(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_float(name: str, default: float) -> float:
    value = get_env(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def get_path(name: str, default: str) -> str:
    return str(Path(get_env(name, default)).expanduser())
