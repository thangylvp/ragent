"""Small Qwen3/Hermes tool-call codec used by the demo's vLLM adapter."""

from __future__ import annotations

import json
import re


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def render_tool_calls(tool_calls) -> str:
    """Render structured calls as Qwen3/Hermes tool-call blocks."""

    blocks = []
    for call in tool_calls or []:
        name = call["name"] if isinstance(call, dict) else call.name
        arguments = (
            call.get("arguments") if isinstance(call, dict) else call.arguments
        ) or {}
        value = json.dumps(
            {"name": name, "arguments": arguments},
            ensure_ascii=False,
        )
        blocks.append(f"<tool_call>\n{value}\n</tool_call>")
    return "\n".join(blocks)


def parse_tool_calls(text: str) -> list[dict]:
    """Parse complete calls plus a final complete JSON block missing its close tag."""

    if "</think>" in text:
        text = text.split("</think>")[-1]
    calls = []
    for candidate in _TOOL_CALL_RE.findall(text):
        try:
            value = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and "name" in value:
            calls.append(
                {
                    "name": value["name"],
                    "arguments": value.get("arguments") or {},
                }
            )

    last_open = text.rfind("<tool_call>")
    last_close = text.rfind("</tool_call>")
    if last_open > last_close:
        payload = text[last_open + len("<tool_call>") :].lstrip()
        try:
            value, end = json.JSONDecoder().raw_decode(payload)
        except (TypeError, json.JSONDecodeError):
            pass
        else:
            if not payload[end:].strip() and isinstance(value, dict) and "name" in value:
                calls.append(
                    {
                        "name": value["name"],
                        "arguments": value.get("arguments") or {},
                    }
                )
    return calls
