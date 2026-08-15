#!/usr/bin/env python3
"""Send one WAV/MP3 utterance to the standalone STCC vLLM endpoint."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    model_dir = os.environ.get("RAGENT_MODEL_DIR", "")
    default_tools = str(Path(model_dir) / "tools_openai.json") if model_dir else None
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Input WAV, MP3, FLAC or OGG file")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("RAGENT_VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        help="OpenAI-compatible vLLM base URL",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("RAGENT_VLLM_MODEL_NAME", "stcc"),
        help="Served model name",
    )
    parser.add_argument(
        "--tools",
        type=Path,
        default=Path(default_tools) if default_tools else None,
        help="tools_openai.json (defaults to RAGENT_MODEL_DIR/tools_openai.json)",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    if not mime.startswith("audio/"):
        raise ValueError(f"unsupported audio extension: {path.suffix or '<none>'}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def load_tools(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        raise ValueError("pass --tools or set RAGENT_MODEL_DIR")
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "tools" in data:
        data = data["tools"]
    if not isinstance(data, list) or len(data) != 33:
        raise ValueError(f"expected 33 OpenAI tool definitions in {path}")
    if not all(item.get("type") == "function" and item.get("function") for item in data):
        raise ValueError(f"invalid OpenAI tool definition in {path}")
    return data


def main() -> int:
    args = parse_args()
    if not args.audio.is_file():
        print(f"audio not found: {args.audio}", file=sys.stderr)
        return 2

    try:
        audio = data_url(args.audio)
        tools = load_tools(args.tools)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": audio},
                    },
                ],
            }
        ],
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('RAGENT_VLLM_API_KEY', 'EMPTY')}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code}: {body}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"request failed: {exc}", file=sys.stderr)
        return 1

    choice = result.get("choices", [{}])[0]
    message = choice.get("message", {})
    print(json.dumps(message, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
