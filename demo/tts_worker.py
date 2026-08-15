#!/usr/bin/env python3
"""Resident, single-reference OmniVoice worker for the RAGENT demo."""

from __future__ import annotations

import io
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = REPO_ROOT / "outputs/demo/reference_voice"
CHECKPOINT = os.getenv("DEMO_OMNIVOICE_CHECKPOINT", "splendor1811/omnivoice-vietnamese")
REF_AUDIO = Path(
    os.getenv(
        "DEMO_OMNIVOICE_REF_AUDIO",
        str(REFERENCE_ROOT / "reference.mp3"),
    )
)
REF_TEXT_PATH = Path(
    os.getenv(
        "DEMO_OMNIVOICE_REF_TEXT",
        str(REFERENCE_ROOT / "reference.txt"),
    )
)
DEVICE = os.getenv("DEMO_OMNIVOICE_DEVICE", "cuda:0")
NUM_STEPS = int(os.getenv("DEMO_OMNIVOICE_NUM_STEPS", "32"))
VOICE_ID = os.getenv("DEMO_OMNIVOICE_VOICE_ID", "female_north_1")
SAMPLE_RATE = 24_000


class SynthesisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2_000)
    speed: float | None = Field(default=None, ge=0.5, le=1.5)


class Runtime:
    def __init__(self):
        self.model = None
        self.voice_prompt = None
        self.lock = threading.Lock()
        self.error: str | None = None

    def load(self) -> None:
        if not REF_AUDIO.is_file():
            raise FileNotFoundError(f"reference audio not found: {REF_AUDIO}")
        if not REF_TEXT_PATH.is_file():
            raise FileNotFoundError(f"reference transcript not found: {REF_TEXT_PATH}")
        from omnivoice import OmniVoice

        dtype = torch.float16 if DEVICE.startswith("cuda") else torch.float32
        self.model = OmniVoice.from_pretrained(
            CHECKPOINT,
            device_map=DEVICE,
            dtype=dtype,
        )
        # Prompt creation uses the tokenizer briefly, then inference owns it on GPU.
        if DEVICE.startswith("cuda"):
            self.model.audio_tokenizer.to("cpu")
        self.voice_prompt = self.model.create_voice_clone_prompt(
            ref_audio=str(REF_AUDIO),
            ref_text=REF_TEXT_PATH.read_text(encoding="utf-8").strip(),
        )
        if DEVICE.startswith("cuda"):
            self.model.audio_tokenizer.to(DEVICE)
            torch.cuda.empty_cache()
        self.error = None

    def synthesize(self, text: str, *, speed: float | None = None) -> bytes:
        if self.model is None or self.voice_prompt is None:
            raise RuntimeError(self.error or "OmniVoice is not loaded")
        with self.lock, torch.inference_mode():
            audio = self.model.generate(
                text=text,
                language="Vietnamese",
                voice_clone_prompt=self.voice_prompt,
                speed=speed,
                num_step=NUM_STEPS,
            )[0]
        buffer = io.BytesIO()
        sf.write(buffer, np.asarray(audio, dtype="float32"), SAMPLE_RATE, format="WAV")
        return buffer.getvalue()


RUNTIME = Runtime()


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        RUNTIME.load()
    except Exception as exc:
        RUNTIME.error = f"{type(exc).__name__}: {exc}"
        raise
    yield


app = FastAPI(title="RAGENT OmniVoice Worker", version="0.1", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "ready": RUNTIME.model is not None,
        "model": CHECKPOINT,
        "voice": str(REF_AUDIO),
        "voice_id": VOICE_ID,
        "device": DEVICE,
        "num_steps": NUM_STEPS,
        "sample_rate": SAMPLE_RATE,
        "error": RUNTIME.error,
    }


@app.post("/synthesize")
def synthesize(request: SynthesisRequest):
    try:
        audio = RUNTIME.synthesize(request.text.strip(), speed=request.speed)
    except Exception as exc:
        raise HTTPException(503, f"{type(exc).__name__}: {exc}") from exc
    return Response(
        content=audio,
        media_type="audio/wav",
        headers={
            "X-TTS-Provider": "omnivoice",
            "X-TTS-Voice": REF_AUDIO.name,
            "X-TTS-Sample-Rate": str(SAMPLE_RATE),
        },
    )
