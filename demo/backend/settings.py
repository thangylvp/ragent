"""Environment-backed settings for the end-to-end RAGENT demo."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEMO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
CLOUD_PROVIDER = os.getenv("DEMO_CLOUD_PROVIDER", "gemini").strip().lower()


def _load_env(path: Path) -> None:
    """Load ordinary KEY=VALUE entries without overriding the process env."""

    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value[:1] == value[-1:] and value[:1] in {'"', "'"}:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


for _env_path in (
    WORKSPACE_ROOT / ".env",
    WORKSPACE_ROOT / "stc/.env",  # local compatibility with the earlier car demo
    REPO_ROOT / ".env",
    DEMO_ROOT / ".env",
):
    _load_env(_env_path)


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    host: str = os.getenv("WEBTEST_HOST", "127.0.0.1")
    port: int = int(os.getenv("WEBTEST_PORT", "8010"))
    model_mode: str = os.getenv("WEBTEST_MODEL_MODE", "vllm")
    model_dir: str = os.getenv(
        "WEBTEST_MODEL_DIR",
        os.getenv("RAGENT_MODEL_DIR", str(REPO_ROOT / "outputs/models/stcc")),
    )
    max_new_tokens: int = int(os.getenv("WEBTEST_MAX_NEW_TOKENS", "256"))
    vllm_base_url: str = os.getenv(
        "WEBTEST_VLLM_BASE_URL", "http://127.0.0.1:8100/v1"
    ).rstrip("/")
    vllm_model: str = os.getenv("WEBTEST_VLLM_MODEL", "stcc")
    vllm_api_key: str | None = os.getenv("WEBTEST_VLLM_API_KEY") or None
    vllm_timeout_s: float = float(os.getenv("WEBTEST_VLLM_TIMEOUT_S", "180"))
    default_vad: str = os.getenv("WEBTEST_VAD", "omnivad")
    default_enhancer: str = os.getenv("WEBTEST_ENHANCER", "fastenhancer_s")
    fastenhancer_s_model: str = os.getenv(
        "WEBTEST_FASTENHANCER_S_MODEL",
        str(REPO_ROOT / "outputs/denoise/models/fastenhancer_s_dns.onnx"),
    )
    firered_model_dir: str = os.getenv(
        "WEBTEST_FIRERED_MODEL_DIR",
        str(REPO_ROOT / "outputs/vad/models/FireRedVAD/Stream-VAD"),
    )
    frontend_dir: str = str(DEMO_ROOT / "frontend")
    capture_dir: str = os.getenv(
        "WEBTEST_CAPTURE_DIR", str(REPO_ROOT / "outputs/demo/captures")
    )
    voice_cache_dir: str = os.getenv(
        "DEMO_VOICE_CACHE_DIR", str(REPO_ROOT / "outputs/demo/voice")
    )
    static_audio_manifest: str = os.getenv(
        "DEMO_STATIC_AUDIO_MANIFEST",
        str(REPO_ROOT / "outputs/demo/voice/manifest.json"),
    )
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    gemini_api_key: str | None = (
        os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or None
    )
    gemini_base_url: str = os.getenv(
        "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
    ).rstrip("/")
    cloud_enabled: bool = _flag("DEMO_CLOUD_ENABLED", "1")
    cloud_provider: str = CLOUD_PROVIDER
    cloud_model: str = os.getenv(
        "DEMO_CLOUD_MODEL",
        "gemini-3.6-flash" if CLOUD_PROVIDER == "gemini" else "gpt-5.4",
    )
    gemini_thinking_level: str = os.getenv(
        "DEMO_GEMINI_THINKING_LEVEL", "low"
    ).strip().lower()
    cloud_timeout_s: float = float(os.getenv("DEMO_CLOUD_TIMEOUT_S", "60"))
    cloud_history_turns: int = int(os.getenv("DEMO_CLOUD_HISTORY_TURNS", "8"))
    tts_enabled: bool = _flag("DEMO_TTS_ENABLED", "1")
    tts_provider: str = os.getenv("DEMO_TTS_PROVIDER", "omnivoice")
    tts_model: str = os.getenv("DEMO_TTS_MODEL", "gpt-4o-mini-tts")
    tts_voice: str = os.getenv("DEMO_TTS_VOICE", "coral")
    edge_tts_voice: str = os.getenv("DEMO_EDGE_TTS_VOICE", "vi-VN-HoaiMyNeural")
    omnivoice_base_url: str = os.getenv(
        "DEMO_OMNIVOICE_BASE_URL", "http://127.0.0.1:8120"
    ).rstrip("/")
    omnivoice_checkpoint: str = os.getenv(
        "DEMO_OMNIVOICE_CHECKPOINT", "splendor1811/omnivoice-vietnamese"
    )
    omnivoice_voice_id: str = os.getenv(
        "DEMO_OMNIVOICE_VOICE_ID", "female_north_1"
    )
    omnivoice_ref_audio: str = os.getenv(
        "DEMO_OMNIVOICE_REF_AUDIO",
        str(REPO_ROOT / "outputs/demo/reference_voice/reference.mp3"),
    )
    omnivoice_ref_text: str = os.getenv(
        "DEMO_OMNIVOICE_REF_TEXT",
        str(REPO_ROOT / "outputs/demo/reference_voice/reference.txt"),
    )
    omnivoice_num_steps: int = int(os.getenv("DEMO_OMNIVOICE_NUM_STEPS", "32"))
    omnivoice_speed: float = float(os.getenv("DEMO_OMNIVOICE_SPEED", "0.80"))
    omnivoice_timeout_s: float = float(os.getenv("DEMO_OMNIVOICE_TIMEOUT_S", "180"))
    omnivoice_force_synthesis: bool = _flag("DEMO_OMNIVOICE_FORCE_SYNTHESIS")
    tts_timeout_s: float = float(os.getenv("DEMO_TTS_TIMEOUT_S", "60"))
    tts_instructions: str = os.getenv(
        "DEMO_TTS_INSTRUCTIONS",
        "Nói tiếng Việt tự nhiên, thân thiện, rõ ràng và ngắn gọn.",
    )


def get_settings() -> Settings:
    return Settings()
