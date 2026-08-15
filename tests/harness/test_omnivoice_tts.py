from __future__ import annotations

from types import SimpleNamespace

from harness.tts import OmniVoiceTTS


class _Response:
    content = b"RIFF" + bytes(64)

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"ready": True, "error": None}


class _Client:
    def __init__(self):
        self.requests: list[tuple[str, dict]] = []

    def post(self, url: str, *, json: dict) -> _Response:
        self.requests.append((url, json))
        return _Response()

    def get(self, url: str, **_: object) -> _Response:
        self.requests.append((url, {}))
        return _Response()


def test_omnivoice_generates_once_then_uses_persistent_cache(tmp_path):
    voice = tmp_path / "voice.mp3"
    transcript = tmp_path / "voice.txt"
    voice.write_bytes(b"reference-voice")
    transcript.write_text("giọng nói tham chiếu", encoding="utf-8")
    settings = SimpleNamespace(
        omnivoice_checkpoint="test/omnivoice-vietnamese",
        omnivoice_ref_audio=str(voice),
        omnivoice_ref_text=str(transcript),
        omnivoice_num_steps=32,
        omnivoice_speed=0.8,
        omnivoice_voice_id="female_north_1",
        omnivoice_base_url="http://127.0.0.1:8120",
        omnivoice_timeout_s=10,
        voice_cache_dir=str(tmp_path / "cache"),
    )
    tts = OmniVoiceTTS(settings)
    client = _Client()
    tts._client = client

    assert tts.info["ready"] is True
    first = tts.synthesize("Đã thực hiện yêu cầu.")
    second = tts.synthesize("Đã thực hiện yêu cầu.")

    assert first.id == second.id
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert len(client.requests) == 2
    assert client.requests[0][0] == "http://127.0.0.1:8120/health"
    assert client.requests[1][0] == "http://127.0.0.1:8120/synthesize"
    assert client.requests[1][1] == {
        "text": "Đã thực hiện yêu cầu.",
        "speed": 0.8,
    }


def test_omnivoice_can_force_real_synthesis_for_benchmarks(tmp_path):
    settings = SimpleNamespace(
        omnivoice_checkpoint="test/omnivoice-vietnamese",
        omnivoice_num_steps=32,
        omnivoice_speed=0.8,
        omnivoice_voice_id="female_north_1",
        omnivoice_base_url="http://127.0.0.1:8120",
        omnivoice_timeout_s=10,
        omnivoice_force_synthesis=True,
        voice_cache_dir=str(tmp_path / "cache"),
    )
    tts = OmniVoiceTTS(settings)
    client = _Client()
    tts._client = client

    first = tts.synthesize("Câu trả lời động.")
    second = tts.synthesize("Câu trả lời động.")

    assert first.cache_hit is False
    assert second.cache_hit is False
    assert [request[0] for request in client.requests] == [
        "http://127.0.0.1:8120/synthesize",
        "http://127.0.0.1:8120/synthesize",
    ]
