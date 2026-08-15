"""Tests for the small serving-format codec used by the demo."""

from harness.tool_calls import parse_tool_calls, render_tool_calls


def test_round_trip_closed_tool_call():
    calls = [{"name": "lead_way", "arguments": {"destination": "lobby"}}]
    assert parse_tool_calls(render_tool_calls(calls)) == calls


def test_accepts_complete_json_when_only_closing_tag_is_missing():
    raw = '<tool_call>\n{"name":"non_tool","arguments":{"text":"xin chào bạn"}}'
    assert parse_tool_calls(raw) == [
        {"name": "non_tool", "arguments": {"text": "xin chào bạn"}}
    ]


def test_rejects_truncated_or_trailing_garbage_unclosed_call():
    assert parse_tool_calls('<tool_call>{"name":"non_tool","arguments":{"text":"xin') == []
    assert (
        parse_tool_calls(
            '<tool_call>{"name":"non_tool","arguments":{"text":"xin chào"}} trailing'
        )
        == []
    )
