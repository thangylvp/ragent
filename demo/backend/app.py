"""FastAPI webtest for live VAD followed by the car CommandASR model.

This application intentionally stops at the model boundary. It does not
execute car calls and does not implement the future robot harness.
"""

from __future__ import annotations

import json
import re
import time
import uuid
import wave
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from vad import VadEventKind, create_live_vad, live_vad_catalog
from vad.live import pcm_dbfs

from .model import build_model, load_tools, summarize_tools
from .settings import get_settings

SETTINGS = get_settings()
TOOLS = load_tools(SETTINGS.model_dir)
MODEL = build_model(SETTINGS)
CAPTURE_DIR = Path(SETTINGS.capture_dir)
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="VAD → Car STC Webtest", version="0.1")
app.mount("/static", StaticFiles(directory=SETTINGS.frontend_dir), name="static")


def _catalog() -> list[dict]:
    return live_vad_catalog(SETTINGS.firered_model_dir)


def _default_vad() -> str:
    available = [item["name"] for item in _catalog() if item["available"]]
    return SETTINGS.default_vad if SETTINGS.default_vad in available else available[0]


@app.get("/")
def index():
    path = Path(SETTINGS.frontend_dir) / "index.html"
    if not path.is_file():
        raise HTTPException(500, "frontend/index.html not found")
    return FileResponse(path)


@app.get("/api/config")
def config():
    return {
        "model": MODEL.info,
        "vad_backends": _catalog(),
        "default_vad": _default_vad(),
        "audio": {"sample_rate": 16_000, "channels": 1, "encoding": "pcm16le"},
        "tools": summarize_tools(TOOLS),
        "scope": "VAD and model inference only; tool calls are never executed",
    }


@app.get("/api/health")
def health():
    if hasattr(MODEL, "probe"):
        MODEL.probe()
    return {"ok": True, "model": MODEL.info, "vad_backends": _catalog()}


@app.get("/api/captures/{capture_id}.wav")
def capture(capture_id: str):
    if not re.fullmatch(r"[0-9a-f]{32}", capture_id):
        raise HTTPException(404, "capture not found")
    path = CAPTURE_DIR / f"{capture_id}.wav"
    if not path.is_file():
        raise HTTPException(404, "capture not found")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


def _write_capture(pcm16le: bytes) -> tuple[str, Path]:
    capture_id = uuid.uuid4().hex
    path = CAPTURE_DIR / f"{capture_id}.wav"
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(pcm16le)
    return capture_id, path


async def _send(websocket: WebSocket, event: str, **payload) -> None:
    await websocket.send_json({"event": event, "timestamp_ms": round(time.time() * 1000), **payload})


@app.websocket("/api/audio/stream")
async def audio_stream(websocket: WebSocket):
    await websocket.accept()
    session = None
    try:
        first = await websocket.receive_text()
        request = json.loads(first)
        if request.get("event") != "start_stream":
            raise ValueError("first message must be start_stream")
        if int(request.get("sample_rate", 0)) != 16_000:
            raise ValueError("webtest accepts only 16 kHz PCM")
        if int(request.get("channels", 1)) != 1:
            raise ValueError("webtest accepts only mono PCM")
        if request.get("encoding", "pcm16le") != "pcm16le":
            raise ValueError("webtest accepts only PCM16LE")
        backend = str(request.get("backend") or _default_vad())
        session = await run_in_threadpool(
            create_live_vad,
            backend,
            firered_model_dir=SETTINGS.firered_model_dir,
        )
        frame_bytes = session.frame_samples * 2
        pending = bytearray()
        input_samples = 0
        update_counter = 0
        vad_process_total_ms = 0.0
        vad_process_max_ms = 0.0
        stream_started = time.perf_counter()
        await _send(
            websocket,
            "stream_started",
            backend=session.name,
            sample_rate=session.sample_rate,
            frame_samples=session.frame_samples,
            frame_ms=round(session.frame_samples * 1000 / session.sample_rate, 2),
        )

        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("text") is not None:
                command = json.loads(message["text"])
                if command.get("event") == "stop_stream":
                    await _send(
                        websocket,
                        "stream_stopped",
                        discarded=True,
                        reason="manual_stop_does_not_finalize_partial_speech",
                    )
                    break
                continue
            chunk = message.get("bytes")
            if not chunk:
                continue
            if len(chunk) % 2:
                raise ValueError("PCM16LE packet has an odd byte count")
            pending.extend(chunk)

            while len(pending) >= frame_bytes:
                frame = bytes(pending[:frame_bytes])
                del pending[:frame_bytes]
                frame_started = time.perf_counter()
                update = await run_in_threadpool(session.process_frame, frame)
                process_ms = (time.perf_counter() - frame_started) * 1000
                vad_process_total_ms += process_ms
                vad_process_max_ms = max(vad_process_max_ms, process_ms)
                input_samples += session.frame_samples
                update_counter += 1
                if update_counter % max(1, round(0.1 * 16_000 / session.frame_samples)) == 0:
                    await _send(
                        websocket,
                        "vad_frame",
                        state=update.state.value,
                        confidence=update.confidence,
                        is_speech=update.is_speech,
                        level_dbfs=round(pcm_dbfs(frame), 2),
                        audio_ms=round(input_samples * 1000 / 16_000),
                        process_ms=round(process_ms, 3),
                    )

                for vad_event in update.events:
                    if vad_event.kind is VadEventKind.SPEECH_STARTED:
                        await _send(
                            websocket,
                            "utterance_started",
                            start_ms=round(
                                (vad_event.utterance_start_sample or 0) * 1000 / 16_000
                            ),
                        )
                        continue
                    if vad_event.kind is VadEventKind.SEGMENT_REJECTED:
                        await _send(
                            websocket,
                            "utterance_rejected",
                            reason=vad_event.reason,
                        )
                        continue
                    if vad_event.kind not in {
                        VadEventKind.SPEECH_ENDED,
                        VadEventKind.MAX_DURATION_REACHED,
                    }:
                        continue
                    audio = vad_event.audio_pcm16le or b""
                    if not audio:
                        await _send(websocket, "error", message="VAD finalized empty audio")
                        return
                    vad_finalized_timestamp_ms = time.time() * 1000
                    finalized_started = time.perf_counter()
                    capture_started = time.perf_counter()
                    capture_id, capture_path = _write_capture(audio)
                    capture_write_ms = (time.perf_counter() - capture_started) * 1000
                    duration_ms = round(len(audio) / 2 * 1000 / 16_000)
                    await _send(
                        websocket,
                        "utterance_finalized",
                        capture_id=capture_id,
                        capture_url=f"/api/captures/{capture_id}.wav",
                        duration_ms=duration_ms,
                        reason=vad_event.reason,
                        endpoint_audio_ms=round(input_samples * 1000 / 16_000, 3),
                        vad_finalized_timestamp_ms=round(vad_finalized_timestamp_ms, 3),
                        vad_elapsed_ms=round((time.perf_counter() - stream_started) * 1000, 1),
                        capture_write_ms=round(capture_write_ms, 3),
                        vad_process_total_ms=round(vad_process_total_ms, 3),
                        vad_process_mean_ms=round(
                            vad_process_total_ms / update_counter,
                            3,
                        ),
                        vad_process_max_ms=round(vad_process_max_ms, 3),
                        vad_frames_processed=update_counter,
                    )
                    await _send(websocket, "model_started", model=MODEL.info)
                    model_started = time.perf_counter()
                    result = await run_in_threadpool(MODEL.infer, capture_path, TOOLS)
                    model_adapter_ms = (time.perf_counter() - model_started) * 1000
                    result_ready_from_vad_ms = (
                        time.perf_counter() - finalized_started
                    ) * 1000
                    model_timings = result.get("timings") or {}
                    adapter_total_ms = float(
                        model_timings.get("adapter_total_ms", model_adapter_ms)
                    )
                    to_last_token_ms = float(
                        model_timings.get("to_last_token_ms", adapter_total_ms)
                    )
                    post_token_ms = max(0.0, adapter_total_ms - to_last_token_ms)
                    last_token_from_vad_ms = max(
                        0.0,
                        result_ready_from_vad_ms - post_token_ms,
                    )
                    await _send(
                        websocket,
                        "model_result",
                        result=result,
                        model=MODEL.info,
                        end_to_end_from_vad_ms=round(
                            result_ready_from_vad_ms,
                            1,
                        ),
                        last_token_from_vad_ms=round(last_token_from_vad_ms, 3),
                        component_timings={
                            "capture_write_ms": round(capture_write_ms, 3),
                            "model_thread_ms": round(model_adapter_ms, 3),
                            "model_dispatch_ms": round(
                                max(0.0, model_adapter_ms - adapter_total_ms),
                                3,
                            ),
                            "post_last_token_ms": round(post_token_ms, 3),
                        },
                    )
                    return
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await _send(websocket, "error", message=f"{type(exc).__name__}: {exc}")
        except Exception:
            pass
    finally:
        if session is not None:
            session.reset()
        try:
            await websocket.close()
        except Exception:
            pass
