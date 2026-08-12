"""Environment-backed settings for the VAD → car-STC component webtest."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEMO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    host: str = os.getenv("WEBTEST_HOST", "127.0.0.1")
    port: int = int(os.getenv("WEBTEST_PORT", "8010"))
    model_mode: str = os.getenv("WEBTEST_MODEL_MODE", "local")
    model_dir: str = os.getenv(
        "WEBTEST_MODEL_DIR",
        str(WORKSPACE_ROOT / "stc/outputs/models/route_v1_best_step1250_hf"),
    )
    device: str = os.getenv("WEBTEST_DEVICE", "auto")
    dtype: str = os.getenv("WEBTEST_DTYPE", "bfloat16")
    allow_cpu_model: bool = _flag("WEBTEST_ALLOW_CPU_MODEL")
    eager_model: bool = _flag("WEBTEST_EAGER_MODEL")
    max_new_tokens: int = int(os.getenv("WEBTEST_MAX_NEW_TOKENS", "256"))
    vllm_base_url: str = os.getenv(
        "WEBTEST_VLLM_BASE_URL", "http://127.0.0.1:8100/v1"
    ).rstrip("/")
    vllm_model: str = os.getenv("WEBTEST_VLLM_MODEL", "command-asr")
    vllm_api_key: str | None = os.getenv("WEBTEST_VLLM_API_KEY") or None
    vllm_timeout_s: float = float(os.getenv("WEBTEST_VLLM_TIMEOUT_S", "180"))
    default_vad: str = os.getenv("WEBTEST_VAD", "omnivad")
    firered_model_dir: str = os.getenv(
        "WEBTEST_FIRERED_MODEL_DIR",
        str(REPO_ROOT / "outputs/vad/models/FireRedVAD/Stream-VAD"),
    )
    frontend_dir: str = str(DEMO_ROOT / "frontend")
    capture_dir: str = os.getenv(
        "WEBTEST_CAPTURE_DIR",
        str(REPO_ROOT / "outputs/demo/captures"),
    )


def get_settings() -> Settings:
    return Settings()
