"""Vietnamese response library and persistent speech cache for the demo."""

from __future__ import annotations

import hashlib
import json
import os
import asyncio
import re
import struct
import subprocess
import tempfile
import threading
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .responses import ResponseLibrary, ResponseTemplate


@dataclass(frozen=True, slots=True)
class AudioClip:
    id: str
    text: str
    path: str | None
    synthesis_ms: float
    cache_hit: bool

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["url"] = f"/api/voice/{self.id}.wav" if self.path else None
        value.pop("path")
        return value


class ManifestAudioStore:
    """Read-only fixed-response audio bundle deployed with the edge app."""

    def __init__(self, manifest_path: str | Path, *, verify: bool = True):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"static response manifest not found: {self.manifest_path}; "
                "run scripts/prewarm_demo_audio.py first"
            )
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported static response manifest version")
        clips = payload.get("clips")
        if not isinstance(clips, dict) or not clips:
            raise ValueError("static response manifest contains no clips")
        self._clips: dict[str, AudioClip] = {}
        root = self.manifest_path.parent.resolve()
        for key, item in clips.items():
            if not isinstance(key, str) or not isinstance(item, dict):
                raise ValueError("invalid clip entry in static response manifest")
            clip_id = item.get("id")
            filename = item.get("file")
            text = item.get("text")
            if not isinstance(clip_id, str) or not re.fullmatch(r"[0-9a-f]{32}", clip_id):
                raise ValueError(f"invalid clip id for {key}")
            if filename != f"{clip_id}.wav" or not isinstance(text, str):
                raise ValueError(f"invalid clip metadata for {key}")
            path = (root / filename).resolve()
            if path.parent != root or not path.is_file():
                raise FileNotFoundError(f"static response audio missing for {key}: {path}")
            if verify:
                data = path.read_bytes()
                if not data.startswith(b"RIFF") or len(data) <= 44:
                    raise ValueError(f"invalid WAV file for {key}: {path}")
                expected_hash = item.get("sha256")
                if expected_hash and hashlib.sha256(data).hexdigest() != expected_hash:
                    raise ValueError(f"checksum mismatch for {key}: {path}")
            self._clips[key] = AudioClip(clip_id, text, str(path), 0.0, True)
        self._metadata = payload.get("generator") or {}

    def resolve(self, template: ResponseTemplate) -> AudioClip:
        try:
            clip = self._clips[template.key]
        except KeyError as exc:
            raise KeyError(f"static audio key is not packaged: {template.key}") from exc
        if clip.text != template.text:
            raise ValueError(
                f"static audio text is stale for {template.key}; regenerate the audio bundle"
            )
        return clip

    @property
    def info(self) -> dict[str, Any]:
        return {
            "ready": True,
            "mode": "read-only-manifest",
            "manifest": str(self.manifest_path),
            "clips": len(self._clips),
            "generator": self._metadata,
        }


class SynthesizingAudioStore:
    """Test/prewarm adapter; production construction never uses this class."""

    def __init__(self, synthesizer):
        self.synthesizer = synthesizer

    def resolve(self, template: ResponseTemplate) -> AudioClip:
        return self.synthesizer.synthesize(template.text)

    @property
    def info(self) -> dict[str, Any]:
        return {"ready": True, "mode": "synthesizing-test-adapter"}


class DisabledTTS:
    enabled = False

    def synthesize(self, text: str) -> AudioClip:
        return AudioClip("disabled", text, None, 0.0, True)

    @property
    def info(self) -> dict[str, Any]:
        return {"enabled": False, "model": "disabled", "voice": None}


class ToneTTS:
    """Deterministic WAV generator for integration tests, never production UI."""

    enabled = True

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def synthesize(self, text: str) -> AudioClip:
        clip_id = hashlib.sha256(text.encode()).hexdigest()[:32]
        path = self.cache_dir / f"{clip_id}.wav"
        hit = path.is_file()
        started = time.perf_counter()
        if not hit:
            samples = [int(2200 * ((i // 80) % 2 * 2 - 1)) for i in range(3200)]
            with wave.open(str(path), "wb") as output:
                output.setparams((1, 2, 16_000, 0, "NONE", "not compressed"))
                output.writeframes(struct.pack(f"<{len(samples)}h", *samples))
        return AudioClip(
            clip_id,
            text,
            str(path),
            round((time.perf_counter() - started) * 1000, 3),
            hit,
        )

    @property
    def info(self) -> dict[str, Any]:
        return {"enabled": True, "model": "test-tone", "voice": "synthetic"}


class OpenAITTS:
    enabled = True

    def __init__(self, settings):
        import httpx

        self.model = settings.tts_model
        self.voice = settings.tts_voice
        self.instructions = settings.tts_instructions
        self.cache_dir = Path(settings.voice_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._url = f"{settings.openai_base_url}/audio/speech"
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            timeout=settings.tts_timeout_s,
            trust_env=True,
        )
        self._lock = threading.RLock()

    def _id(self, text: str) -> str:
        identity = "\0".join((self.model, self.voice, self.instructions, text))
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]

    def synthesize(self, text: str) -> AudioClip:
        clip_id = self._id(text)
        path = self.cache_dir / f"{clip_id}.wav"
        started = time.perf_counter()
        with self._lock:
            if path.is_file() and path.stat().st_size > 44:
                return AudioClip(clip_id, text, str(path), 0.0, True)
            response = self._client.post(
                self._url,
                json={
                    "model": self.model,
                    "voice": self.voice,
                    "input": text,
                    "instructions": self.instructions,
                    "response_format": "wav",
                },
            )
            response.raise_for_status()
            data = response.content
            if not data.startswith(b"RIFF") or len(data) <= 44:
                raise RuntimeError("TTS response is not a valid WAV file")
            temporary = path.with_suffix(".wav.part")
            temporary.write_bytes(data)
            os.replace(temporary, path)
        return AudioClip(
            clip_id,
            text,
            str(path),
            round((time.perf_counter() - started) * 1000, 3),
            False,
        )

    @property
    def info(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "provider": "openai",
            "model": self.model,
            "voice": self.voice,
        }


class EdgeTTS:
    """Vietnamese neural voice rendered once and served thereafter as WAV."""

    enabled = True

    def __init__(self, settings):
        import edge_tts  # noqa: F401

        self.voice = settings.edge_tts_voice
        self.cache_dir = Path(settings.voice_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._guard = threading.Lock()
        self._clip_locks: dict[str, threading.Lock] = {}

    def _id(self, text: str) -> str:
        return hashlib.sha256(f"edge\0{self.voice}\0{text}".encode("utf-8")).hexdigest()[:32]

    def synthesize(self, text: str) -> AudioClip:
        import edge_tts

        clip_id = self._id(text)
        path = self.cache_dir / f"{clip_id}.wav"
        started = time.perf_counter()
        with self._guard:
            clip_lock = self._clip_locks.setdefault(clip_id, threading.Lock())
        with clip_lock:
            if path.is_file() and path.stat().st_size > 44:
                return AudioClip(clip_id, text, str(path), 0.0, True)
            with tempfile.TemporaryDirectory(
                prefix="edge_tts_", dir=self.cache_dir
            ) as temporary_dir:
                mp3_path = Path(temporary_dir) / "speech.mp3"
                wav_path = Path(temporary_dir) / "speech.wav"
                asyncio.run(
                    edge_tts.Communicate(
                        text,
                        self.voice,
                        connect_timeout=10,
                        receive_timeout=20,
                    ).save(str(mp3_path))
                )
                subprocess.run(
                    [
                        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", str(mp3_path), "-ac", "1", "-ar", "24000", str(wav_path),
                    ],
                    check=True,
                )
                data = wav_path.read_bytes()
                if not data.startswith(b"RIFF") or len(data) <= 44:
                    raise RuntimeError("Edge TTS conversion did not produce a valid WAV")
                staging = path.with_suffix(".wav.part")
                staging.write_bytes(data)
                os.replace(staging, path)
        return AudioClip(
            clip_id,
            text,
            str(path),
            round((time.perf_counter() - started) * 1000, 3),
            False,
        )

    @property
    def info(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "provider": "edge",
            "model": "edge-neural-tts",
            "voice": self.voice,
        }


class OmniVoiceTTS:
    """Network client for the laptop-hosted, single-voice OmniVoice worker.

    The worker owns the model and reference voice. The edge client needs only
    its URL and a stable voice identity; it must not require laptop file paths.
    """

    enabled = True

    def __init__(self, settings):
        import httpx

        self.model = settings.omnivoice_checkpoint
        self.voice_id = getattr(settings, "omnivoice_voice_id", "female_north_1")
        self.num_steps = settings.omnivoice_num_steps
        self.speed = settings.omnivoice_speed
        self.force_synthesis = getattr(settings, "omnivoice_force_synthesis", False)
        if not 0.5 <= self.speed <= 1.5:
            raise ValueError("DEMO_OMNIVOICE_SPEED must be between 0.5 and 1.5")
        self.cache_dir = Path(settings.voice_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._base_url = settings.omnivoice_base_url
        self._url = f"{self._base_url}/synthesize"
        self._health_url = f"{self._base_url}/health"
        self._client = httpx.Client(
            timeout=settings.omnivoice_timeout_s,
            trust_env=False,
        )
        self._guard = threading.Lock()
        self._clip_locks: dict[str, threading.Lock] = {}
        self._ready: bool | None = None
        self._error: str | None = None
        self._server_info: dict[str, Any] = {}

    def _id(self, text: str) -> str:
        identity = "\0".join(
            (
                "omnivoice",
                self.model,
                self.voice_id,
                str(self.num_steps),
                str(self.speed),
                text,
            )
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]

    def synthesize(self, text: str) -> AudioClip:
        clip_id = self._id(text)
        path = self.cache_dir / f"{clip_id}.wav"
        started = time.perf_counter()
        with self._guard:
            clip_lock = self._clip_locks.setdefault(clip_id, threading.Lock())
        with clip_lock:
            if (
                not self.force_synthesis
                and path.is_file()
                and path.stat().st_size > 44
            ):
                return AudioClip(clip_id, text, str(path), 0.0, True)
            try:
                response = self._client.post(
                    self._url,
                    json={"text": text, "speed": self.speed},
                )
                response.raise_for_status()
                data = response.content
                if not data.startswith(b"RIFF") or len(data) <= 44:
                    raise RuntimeError("OmniVoice worker did not return a valid WAV")
                staging = path.with_suffix(".wav.part")
                staging.write_bytes(data)
                os.replace(staging, path)
                self._ready = True
                self._error = None
            except Exception as exc:
                self._ready = False
                self._error = f"{type(exc).__name__}: {exc}"
                raise
        return AudioClip(
            clip_id,
            text,
            str(path),
            round((time.perf_counter() - started) * 1000, 3),
            False,
        )

    @property
    def info(self) -> dict[str, Any]:
        if self._ready is None:
            try:
                # Configuration/health pages must remain responsive when the
                # laptop TTS service is offline. Synthesis keeps the longer
                # generation timeout, but readiness probing is fail-fast.
                response = self._client.get(self._health_url, timeout=3.0)
                response.raise_for_status()
                payload = response.json()
                self._ready = bool(payload.get("ready"))
                self._error = payload.get("error")
                self._server_info = {
                    key: payload.get(key)
                    for key in ("model", "voice_id", "device", "num_steps", "sample_rate")
                    if payload.get(key) is not None
                }
            except Exception as exc:
                self._ready = False
                self._error = f"{type(exc).__name__}: {exc}"
        return {
            "enabled": True,
            "ready": self._ready,
            "provider": "omnivoice",
            "model": self.model,
            "voice": self.voice_id,
            "num_steps": self.num_steps,
            "speed": self.speed,
            "force_synthesis": self.force_synthesis,
            "base_url": self._base_url,
            "server": self._server_info,
            "error": self._error,
        }


def build_tts(settings):
    if not settings.tts_enabled or settings.tts_provider == "disabled":
        return DisabledTTS()
    provider = settings.tts_provider.strip().lower()
    if provider == "edge":
        return EdgeTTS(settings)
    if provider == "omnivoice":
        return OmniVoiceTTS(settings)
    if provider == "openai":
        if not settings.openai_api_key:
            return DisabledTTS()
        return OpenAITTS(settings)
    raise ValueError(
        "DEMO_TTS_PROVIDER must be omnivoice, edge, openai, or disabled"
    )


def build_static_audio(settings) -> ManifestAudioStore:
    return ManifestAudioStore(settings.static_audio_manifest)
