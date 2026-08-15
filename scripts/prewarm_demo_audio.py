#!/usr/bin/env python3
"""Generate every static Vietnamese demo response into the persistent cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from demo.backend.model import load_tools
from demo.backend.settings import get_settings
from harness.responses import ResponseLibrary, ResponseTemplate
from harness.tts import build_tts


def response_templates(tools: list[dict]) -> list[ResponseTemplate]:
    library = ResponseLibrary()
    names = [
        (item.get("function") or {}).get("name")
        for item in tools
        if (item.get("function") or {}).get("name")
    ]
    return library.all_templates(names)


def _manifest_entry(clip, cache_dir: Path) -> dict:
    path = Path(clip.path).resolve()
    if path.parent != cache_dir.resolve():
        raise ValueError(f"generated clip is outside voice cache: {path}")
    data = path.read_bytes()
    with wave.open(str(path), "rb") as reader:
        audio = {
            "sample_rate": reader.getframerate(),
            "channels": reader.getnchannels(),
            "frames": reader.getnframes(),
        }
    return {
        "id": clip.id,
        "text": clip.text,
        "file": path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "audio": audio,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel requests; keep 1 for the serial OmniVoice GPU worker",
    )
    args = parser.parse_args()
    settings = get_settings()
    tts = build_tts(settings)
    tools = load_tools(settings.model_dir)
    templates = response_templates(tools)
    generator_info = tts.info
    print(json.dumps({"tts": generator_info, "clips": len(templates)}, ensure_ascii=False))
    failures = 0
    generated = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(tts.synthesize, template.text): template
            for template in templates
        }
        for index, future in enumerate(as_completed(futures), 1):
            template = futures[future]
            try:
                clip = future.result()
                generated[template.key] = clip
                print(
                    f"[{index:03d}/{len(templates)}] "
                    f"{'cached' if clip.cache_hit else 'generated'} "
                    f"{template.key} {clip.id} {template.text}"
                )
            except Exception as exc:
                failures += 1
                print(
                    f"[{index:03d}/{len(templates)}] FAILED {template.key}: "
                    f"{type(exc).__name__}: {exc}"
                )
    cache_dir = Path(settings.voice_cache_dir)
    if not failures:
        manifest = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator": generator_info,
            "clips": {
                template.key: _manifest_entry(generated[template.key], cache_dir)
                for template in templates
            },
        }
        manifest_path = Path(settings.static_audio_manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        staging = manifest_path.with_suffix(".json.part")
        staging.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, manifest_path)
        print(f"manifest={manifest_path}")
    print(f"cache={cache_dir} failures={failures}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
