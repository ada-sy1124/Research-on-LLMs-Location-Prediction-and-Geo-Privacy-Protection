import os

from google import genai
from google.genai.types import HttpOptions

from geoai_pipeline.config import get_env


def _first_non_empty(*values: str | None) -> str:
    for value in values:
        if value:
            return value
    return ""


def create_genai_client(primary_api_key_env: str = "GEMINI_API_KEY"):
    """
    Create a google-genai client for either AI Studio or Vertex AI API key mode.

    Env:
    - GEMINI_AUTH_MODE: auto | ai_studio | vertex_api_key
    - GEMINI_API_KEY / MASK*_GEMINI_API_KEY
    - GOOGLE_CLOUD_PROJECT (optional but recommended for Vertex)
    - GOOGLE_CLOUD_LOCATION (optional, default: global)
    """
    auth_mode = (get_env("GEMINI_AUTH_MODE", "auto") or "auto").strip().lower()

    primary_key = get_env(primary_api_key_env, "")
    default_key = get_env("GEMINI_API_KEY", "")
    api_key = _first_non_empty(primary_key, default_key)

    if not api_key:
        raise ValueError(
            f"缺少 {primary_api_key_env} 或 GEMINI_API_KEY，请在 code/.env 中配置。"
        )

    if auth_mode in {"vertex", "vertex_api_key"}:
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
        os.environ["GOOGLE_CLOUD_PROJECT"] = get_env("GOOGLE_CLOUD_PROJECT", "") or ""
        os.environ["GOOGLE_CLOUD_LOCATION"] = get_env("GOOGLE_CLOUD_LOCATION", "global") or "global"
        return genai.Client(api_key=api_key, http_options=HttpOptions(api_version="v1"))

    if auth_mode in {"ai_studio", "aistudio", "studio"}:
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"
        return genai.Client(api_key=api_key)

    # auto mode:
    # - Key starts with "AIza" -> prefer AI Studio by default (current project behavior)
    # - Otherwise try Vertex API key mode
    if api_key.startswith("AIza"):
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"
        return genai.Client(api_key=api_key)

    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
    os.environ["GOOGLE_CLOUD_PROJECT"] = get_env("GOOGLE_CLOUD_PROJECT", "") or ""
    os.environ["GOOGLE_CLOUD_LOCATION"] = get_env("GOOGLE_CLOUD_LOCATION", "global") or "global"
    return genai.Client(api_key=api_key, http_options=HttpOptions(api_version="v1"))
