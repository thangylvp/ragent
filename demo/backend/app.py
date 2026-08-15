"""FastAPI application for the end-to-end RAGENT voice demo."""

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

from enhancement import create_live_enhancer, live_enhancer_catalog
from harness import build_demo_harness
from vad import VadEventKind, create_live_vad, live_vad_catalog
from vad.live import pcm_dbfs

from .model import build_model, load_tools, summarize_tools
from .settings import get_settings

SETTINGS = get_settings()
TOOLS = load_tools(SETTINGS.model_dir)
MODEL = build_model(SETTINGS)
HARNESS = build_demo_harness(SETTINGS, TOOLS)
CAPTURE_DIR = Path(SETTINGS.capture_dir)
VOICE_DIR = Path(SETTINGS.voice_cache_dir)
STATIC_VOICE_DIR = Path(SETTINGS.static_audio_manifest).parent
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
VOICE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="RAGENT Voice Demo", version="0.2")
app.mount("/static", StaticFiles(directory=SETTINGS.frontend_dir), name="static")


def _catalog() -> list[dict]:
    return live_vad_catalog(SETTINGS.firered_model_dir)


def _default_vad() -> str:
    available = [item["name"] for item in _catalog() if item["available"]]
    if not available:
        raise RuntimeError("no VAD backend is available")
    return SETTINGS.default_vad if SETTINGS.default_vad in available else available[0]


def _enhancer_catalog() -> list[dict]:
    return live_enhancer_catalog(SETTINGS.fastenhancer_s_model)


def _default_enhancer() -> str:
    available = [item["name"] for item in _enhancer_catalog() if item["available"]]
    if not available:
        raise RuntimeError("no speech enhancer is available")
    return SETTINGS.default_enhancer if SETTINGS.default_enhancer in available else available[0]


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
        "harness": HARNESS.info,
        "vad_backends": _catalog(),
        "default_vad": _default_vad(),
        "enhancers": _enhancer_catalog(),
        "default_enhancer": _default_enhancer(),
        "audio": {"sample_rate": 16_000, "channels": 1, "encoding": "pcm16le"},
        "tools": summarize_tools(TOOLS),
        "scope": "end-to-end demo with simulated hardware",
        "voice_disclosure": "Giọng nói phản hồi được tạo bởi AI.",
    }


@app.get("/api/health")
def health():
    if hasattr(MODEL, "probe"):
        MODEL.probe()
    return {
        "ok": True,
        "model": MODEL.info,
        "harness": HARNESS.info,
        "vad_backends": _catalog(),
        "enhancers": _enhancer_catalog(),
    }


@app.get("/api/hardware")
def hardware():
    return HARNESS.executor.snapshot()


@app.post("/api/hardware/busy")
def hardware_busy(busy: bool):
    return HARNESS.executor.set_busy(busy)


@app.post("/api/hardware/vehicle")
def hardware_vehicle(running: bool, speed_kph: float = 0):
    return HARNESS.executor.sync_vehicle_status(
        running=running,
        speed_kph=speed_kph,
    )


@app.post("/api/hardware/reset")
def hardware_reset():
    return HARNESS.reset()


def _safe_id(value: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{32}", value):
        raise HTTPException(404, "audio not found")


@app.get("/api/voice/{clip_id}.wav")
def voice(clip_id: str):
    _safe_id(clip_id)
    for root in (VOICE_DIR, STATIC_VOICE_DIR):
        path = root / f"{clip_id}.wav"
        if path.is_file():
            return FileResponse(path, media_type="audio/wav", filename=path.name)
    raise HTTPException(404, "voice response not found")


@app.get("/api/captures/{capture_id}.wav")
def capture(capture_id: str):
    _safe_id(capture_id)
    path = CAPTURE_DIR / f"{capture_id}.wav"
    if not path.is_file():
        raise HTTPException(404, "capture not found")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


@app.get("/api/captures/{capture_id}/before-enhancement.wav")
def capture_before_enhancement(capture_id: str):
    _safe_id(capture_id)
    path = CAPTURE_DIR / f"{capture_id}.before-enhancement.wav"
    if not path.is_file():
        raise HTTPException(404, "capture not found")
    return FileResponse(path, media_type="audio/wav", filename=path.name)


def _write_wav(path: Path, pcm16le: bytes) -> None:
    if len(pcm16le) % 2:
        raise ValueError("PCM16LE capture has an odd byte count")
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(pcm16le)


def _write_capture_pair(
    model_pcm16le: bytes,
    before_enhancement_pcm16le: bytes,
) -> tuple[str, Path, Path]:
    capture_id = uuid.uuid4().hex
    model_path = CAPTURE_DIR / f"{capture_id}.wav"
    before_path = CAPTURE_DIR / f"{capture_id}.before-enhancement.wav"
    _write_wav(model_path, model_pcm16le)
    _write_wav(before_path, before_enhancement_pcm16le)
    return capture_id, model_path, before_path


def _aligned_pcm_window(
    pcm16le: bytes | bytearray,
    *,
    start_sample: int,
    sample_count: int,
) -> bytes:
    """Return a fixed-size PCM window, zero-padding outside the source."""

    if len(pcm16le) % 2:
        raise ValueError("PCM16LE source has an odd byte count")
    if sample_count < 0:
        raise ValueError("sample_count cannot be negative")
    total_samples = len(pcm16le) // 2
    end_sample = start_sample + sample_count
    source_start = max(0, start_sample)
    source_end = min(total_samples, end_sample)
    output = bytearray(sample_count * 2)
    if source_end > source_start:
        target_start = source_start - start_sample
        selected = pcm16le[source_start * 2 : source_end * 2]
        output[target_start * 2 : target_start * 2 + len(selected)] = selected
    return bytes(output)


async def _send(websocket: WebSocket, event: str, **payload) -> None:
    await websocket.send_json(
        {"event": event, "timestamp_ms": round(time.time() * 1000), **payload}
    )


async def _wait_for_playback(websocket: WebSocket, turn_id: str) -> bool:
    """Drain queued microphone packets until the browser finishes speaking."""

    while True:
        message = await websocket.receive()
        if message.get("type") == "websocket.disconnect":
            return False
        text = message.get("text")
        if text is None:
            continue
        command = json.loads(text)
        event = command.get("event")
        if event == "stop_stream":
            return False
        if event == "audio_playback_started" and command.get("turn_id") == turn_id:
            await _send(
                websocket,
                "playback_acknowledged",
                turn_id=turn_id,
                client_audio_from_vad_ms=command.get("audio_from_vad_ms"),
            )
        if event == "playback_finished" and command.get("turn_id") == turn_id:
            return True


@app.websocket("/api/audio/stream")
async def audio_stream(websocket: WebSocket):  # noqa: C901
    await websocket.accept()
    session = None
    enhancer = None
    try:
        first = await websocket.receive_text()
        request = json.loads(first)
        if request.get("event") != "start_stream":
            raise ValueError("first message must be start_stream")
        if int(request.get("sample_rate", 0)) != 16_000:
            raise ValueError("demo accepts only 16 kHz PCM")
        if int(request.get("channels", 1)) != 1:
            raise ValueError("demo accepts only mono PCM")
        if request.get("encoding", "pcm16le") != "pcm16le":
            raise ValueError("demo accepts only PCM16LE")
        backend = str(request.get("backend") or _default_vad())
        enhancer_name = str(request.get("enhancer") or _default_enhancer())
        session = await run_in_threadpool(
            create_live_vad,
            backend,
            firered_model_dir=SETTINGS.firered_model_dir,
        )
        enhancer = await run_in_threadpool(
            create_live_enhancer,
            enhancer_name,
            fastenhancer_s_model=SETTINGS.fastenhancer_s_model,
        )
        frame_bytes = session.frame_samples * 2
        pending = bytearray()
        before_stream = bytearray()
        input_samples = 0
        frame_count = 0
        vad_total_ms = 0.0
        vad_max_ms = 0.0
        gate_open = True
        await _send(
            websocket,
            "stream_started",
            backend=session.name,
            enhancer=enhancer.name,
            enhancement_algorithmic_delay_ms=enhancer.algorithmic_delay_ms,
            sample_rate=session.sample_rate,
            frame_samples=session.frame_samples,
            frame_ms=round(session.frame_samples * 1000 / session.sample_rate, 2),
            hardware=HARNESS.executor.snapshot(),
        )

        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("text") is not None:
                command = json.loads(message["text"])
                if command.get("event") == "stop_stream":
                    await _send(websocket, "stream_stopped", discarded=True)
                    break
                continue
            chunk = message.get("bytes")
            if not chunk or not gate_open:
                continue
            if len(chunk) % 2:
                raise ValueError("PCM16LE packet has an odd byte count")
            before_stream.extend(chunk)
            enhancement = await run_in_threadpool(enhancer.process, chunk)
            pending.extend(enhancement.pcm16le)

            finalized_event = None
            while len(pending) >= frame_bytes and finalized_event is None:
                frame = bytes(pending[:frame_bytes])
                del pending[:frame_bytes]
                frame_started = time.perf_counter()
                update = await run_in_threadpool(session.process_frame, frame)
                process_ms = (time.perf_counter() - frame_started) * 1000
                vad_total_ms += process_ms
                vad_max_ms = max(vad_max_ms, process_ms)
                input_samples += session.frame_samples
                frame_count += 1
                cadence = max(1, round(0.1 * 16_000 / session.frame_samples))
                if frame_count % cadence == 0:
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
                for event in update.events:
                    if event.kind is VadEventKind.SPEECH_STARTED:
                        await _send(
                            websocket,
                            "utterance_started",
                            start_ms=round((event.utterance_start_sample or 0) * 1000 / 16_000),
                        )
                    elif event.kind is VadEventKind.SEGMENT_REJECTED:
                        await _send(websocket, "utterance_rejected", reason=event.reason)
                    elif event.kind in {
                        VadEventKind.SPEECH_ENDED,
                        VadEventKind.MAX_DURATION_REACHED,
                    }:
                        finalized_event = event
                        break

            if finalized_event is None:
                continue
            audio = finalized_event.audio_pcm16le or b""
            if not audio:
                await _send(websocket, "error", message="VAD finalized empty audio")
                continue

            gate_open = False
            turn_id = uuid.uuid4().hex
            vad_ready_clock = time.perf_counter()
            vad_ready_epoch_ms = time.time() * 1000
            await _send(websocket, "input_gate", state="closed", turn_id=turn_id)
            capture_started = time.perf_counter()
            delay_samples = round(enhancer.algorithmic_delay_ms * 16_000 / 1000)
            enhanced_start = finalized_event.utterance_start_sample or 0
            before_audio = _aligned_pcm_window(
                before_stream,
                start_sample=enhanced_start - delay_samples,
                sample_count=len(audio) // 2,
            )
            capture_id, capture_path, _ = _write_capture_pair(audio, before_audio)
            capture_ms = (time.perf_counter() - capture_started) * 1000
            duration_ms = round(len(audio) / 2 * 1000 / 16_000)
            utterance_start_audio_ms = round(
                (finalized_event.utterance_start_sample or 0) * 1000 / 16_000,
                3,
            )
            utterance_end_audio_ms = round(
                (
                    finalized_event.utterance_end_sample
                    if finalized_event.utterance_end_sample is not None
                    else (finalized_event.utterance_start_sample or 0) + len(audio) // 2
                )
                * 1000
                / 16_000,
                3,
            )
            await _send(
                websocket,
                "utterance_finalized",
                turn_id=turn_id,
                capture_id=capture_id,
                capture_url=f"/api/captures/{capture_id}.wav",
                before_enhancement_capture_url=(
                    f"/api/captures/{capture_id}/before-enhancement.wav"
                ),
                duration_ms=duration_ms,
                reason=finalized_event.reason,
                utterance_start_audio_ms=utterance_start_audio_ms,
                utterance_end_audio_ms=utterance_end_audio_ms,
                endpoint_audio_ms=round(input_samples * 1000 / 16_000, 3),
                vad_finalized_timestamp_ms=round(vad_ready_epoch_ms, 3),
                capture_write_ms=round(capture_ms, 3),
                vad_process_total_ms=round(vad_total_ms, 3),
                vad_process_mean_ms=round(vad_total_ms / max(1, frame_count), 3),
                vad_process_max_ms=round(vad_max_ms, 3),
                vad_frames_processed=frame_count,
                enhancement_frames_processed=enhancer.frames_processed,
                enhancement_compute_total_ms=round(enhancer.compute_total_ms, 3),
                enhancement_compute_mean_ms=round(
                    enhancer.compute_total_ms / max(1, enhancer.frames_processed), 3
                ),
                enhancement_compute_max_ms=round(enhancer.compute_max_ms, 3),
                enhancement_algorithmic_delay_ms=enhancer.algorithmic_delay_ms,
            )

            await _send(websocket, "model_started", turn_id=turn_id, model=MODEL.info)
            model_dispatch_clock = time.perf_counter()
            slm_result = await run_in_threadpool(MODEL.infer, capture_path, TOOLS)
            model_done_clock = time.perf_counter()
            model_timings = slm_result.get("timings") or {}
            dispatch_from_vad_ms = (model_dispatch_clock - vad_ready_clock) * 1000
            first_from_request = float(
                model_timings.get("request_to_first_token_ms", 0.0)
            )
            last_from_request = float(
                model_timings.get(
                    "request_to_last_token_ms",
                    model_timings.get("to_last_token_ms", slm_result["latency_ms"]),
                )
            )
            audio_to_first_ms = dispatch_from_vad_ms + first_from_request
            audio_to_last_ms = dispatch_from_vad_ms + last_from_request
            await _send(
                websocket,
                "model_result",
                turn_id=turn_id,
                result=slm_result,
                model=MODEL.info,
                timings={
                    "audio_to_first_llm_token_ms": round(audio_to_first_ms, 3),
                    "audio_to_last_llm_token_ms": round(audio_to_last_ms, 3),
                    "audio_to_model_result_ms": round(
                        (model_done_clock - vad_ready_clock) * 1000,
                        3,
                    ),
                    "capture_write_ms": round(capture_ms, 3),
                    "model_dispatch_ms": round(dispatch_from_vad_ms - capture_ms, 3),
                    "model_thread_ms": round((model_done_clock - model_dispatch_clock) * 1000, 3),
                },
            )

            harness_result = await run_in_threadpool(HARNESS.process, slm_result)
            response_ready_ms = (time.perf_counter() - vad_ready_clock) * 1000
            response = harness_result.as_dict()
            response["timings"].update(
                {
                    "audio_to_first_llm_token_ms": round(audio_to_first_ms, 3),
                    "audio_to_last_llm_token_ms": round(audio_to_last_ms, 3),
                    "audio_to_response_ready_ms": round(response_ready_ms, 3),
                }
            )
            if response["route"] == "cloud":
                # Cloud latency is measured inside the harness. Anchor it after
                # the complete SLM response so this remains an end-of-audio
                # cumulative metric, consistent with the other UI milestones.
                model_result_ms = (model_done_clock - vad_ready_clock) * 1000
                response["timings"]["audio_to_cloud_llm_ms"] = round(
                    model_result_ms + float(response["timings"]["cloud_ms"]),
                    3,
                )
            await _send(
                websocket,
                "assistant_response",
                turn_id=turn_id,
                response=response,
                audio_baseline_timestamp_ms=round(vad_ready_epoch_ms, 3),
            )

            should_continue = await _wait_for_playback(websocket, turn_id)
            if not should_continue:
                break
            session.reset()
            enhancer.reset()
            pending.clear()
            before_stream.clear()
            input_samples = 0
            frame_count = 0
            vad_total_ms = 0.0
            vad_max_ms = 0.0
            gate_open = True
            await _send(
                websocket,
                "input_gate",
                state="open",
                turn_id=turn_id,
                hardware=HARNESS.executor.snapshot(),
            )
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
        if enhancer is not None:
            enhancer.reset()
        try:
            await websocket.close()
        except Exception:
            pass
