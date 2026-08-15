from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from harness import ManifestAudioStore, ResponseTemplate, ToneTTS


def _manifest(tmp_path: Path, *, text: str = "Đã thực hiện yêu cầu.") -> Path:
    clip = ToneTTS(tmp_path).synthesize(text)
    wav = Path(clip.path)
    payload = {
        "schema_version": 1,
        "generator": {"provider": "test"},
        "clips": {
            "success.test": {
                "id": clip.id,
                "text": text,
                "file": wav.name,
                "sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
            }
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_manifest_audio_store_resolves_without_synthesis(tmp_path):
    store = ManifestAudioStore(_manifest(tmp_path))
    clip = store.resolve(ResponseTemplate("success.test", "Đã thực hiện yêu cầu."))
    assert clip.cache_hit is True
    assert clip.synthesis_ms == 0
    assert clip.as_dict()["url"].endswith(f"/{clip.id}.wav")
    assert store.info["mode"] == "read-only-manifest"


def test_manifest_audio_store_detects_stale_response_text(tmp_path):
    store = ManifestAudioStore(_manifest(tmp_path))
    with pytest.raises(ValueError, match="stale"):
        store.resolve(ResponseTemplate("success.test", "Nội dung mới"))
