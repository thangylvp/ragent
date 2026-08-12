"""Qwen3 tool-call wire format — the codec for the CommandASR model's function calls.

CommandASR's decoder is Qwen3, so the model speaks Qwen3's `<tool_call>\n{json}\n</tool_call>`
format. This is the ONE place that **renders** that format (training's gold target) and **parses**
it back (the evaluator decodes model output) — kept side by side so training and eval can never
drift to different formats.

This is deliberately **model-specific** (a Qwen3.5-based model would emit XML and get its own
codec). The robot contract is model-agnostic and speaks structured calls
(`{name, arguments}`) only; turning model text into a call is the SLM's job.

Pure stdlib (json + re) so it imports cheaply; heavy model code is not pulled in by using it.
"""
from __future__ import annotations
import json
import re

# Qwen3/hermes block: <tool_call> {json} </tool_call>. Non-greedy so multiple calls parse apart.
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
ASSISTANT_EOS_TOKEN = "<|im_end|>"


def assistant_eos_token_id(tokenizer) -> int:
    """Return Qwen's assistant-turn terminator, rejecting tokenizer drift.

    Qwen ChatML ends every assistant tool-call turn with ``<|im_end|>``.  It is
    also the tokenizer EOS token, which lets both Transformers and vLLM stop
    immediately after the supervised turn terminator.
    """
    token_id = tokenizer.convert_tokens_to_ids(ASSISTANT_EOS_TOKEN)
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else None
    if token_id is None or (vocab is not None and ASSISTANT_EOS_TOKEN not in vocab):
        raise ValueError(f"tokenizer is missing required {ASSISTANT_EOS_TOKEN!r}")
    tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
    if tokenizer_eos is not None and int(tokenizer_eos) != int(token_id):
        raise ValueError(
            f"Qwen assistant terminator id={token_id} differs from "
            f"tokenizer.eos_token_id={tokenizer_eos}"
        )
    return int(token_id)


def render_tool_calls(tool_calls) -> str:
    """Render calls to Qwen3 `<tool_call>\n{json}\n</tool_call>` blocks (newline-joined).

    Accepts dicts or objects with `.name`/`.arguments`. Returns "" for an empty/None list (the
    caller decides what an abstention turn looks like)."""
    blocks = []
    for tc in tool_calls or []:
        name = tc["name"] if isinstance(tc, dict) else tc.name
        args = (tc.get("arguments") if isinstance(tc, dict) else tc.arguments) or {}
        blocks.append("<tool_call>\n"
                      + json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
                      + "\n</tool_call>")
    return "\n".join(blocks)


def parse(text: str) -> list[dict]:
    """Decode ALL tool calls from a Qwen3/hermes generation -> [{name, arguments}]; [] = abstention.

    Drops a leading `<think>…</think>` trace, tolerates minor whitespace, and silently skips
    malformed blocks (a half-emitted call never crashes scoring). A model may occasionally stop
    immediately after a complete JSON object without emitting the cosmetic closing XML tag; that
    narrow case is accepted, but incomplete or trailing-garbage JSON is not."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    out = []
    for c in _TOOL_CALL_RE.findall(text):
        try:
            o = json.loads(c)
            if isinstance(o, dict) and "name" in o:
                out.append({"name": o["name"], "arguments": o.get("arguments") or {}})
        except Exception:
            continue

    # Recover only a final unclosed block whose JSON object is itself complete.
    # json.JSONDecoder.raw_decode gives us an exact boundary, preventing a
    # truncated object or arbitrary trailing prose from becoming executable.
    last_open = text.rfind("<tool_call>")
    last_close = text.rfind("</tool_call>")
    if last_open > last_close:
        payload = text[last_open + len("<tool_call>"):].lstrip()
        try:
            obj, end = json.JSONDecoder().raw_decode(payload)
            if (
                not payload[end:].strip()
                and isinstance(obj, dict)
                and "name" in obj
            ):
                out.append(
                    {
                        "name": obj["name"],
                        "arguments": obj.get("arguments") or {},
                    }
                )
        except Exception:
            pass
    return out
