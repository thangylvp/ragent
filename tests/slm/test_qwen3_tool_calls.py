"""Contract tests migrated with the reusable Qwen3 tool-call codec."""

import unittest

from slm.modeling.qwen3_tool_calls import parse, render_tool_calls


class Qwen3ToolCallCodecTest(unittest.TestCase):
    def test_round_trip_closed_tool_call(self):
        calls = [{"name": "lead_way", "arguments": {"destination": "lobby"}}]
        self.assertEqual(parse(render_tool_calls(calls)), calls)

    def test_accepts_complete_json_when_only_closing_tag_is_missing(self):
        raw = '<tool_call>\n{"name":"non_tool","arguments":{"text":"xin chào bạn"}}'
        self.assertEqual(
            parse(raw),
            [{"name": "non_tool", "arguments": {"text": "xin chào bạn"}}],
        )

    def test_rejects_truncated_or_trailing_garbage_unclosed_call(self):
        self.assertEqual(
            parse('<tool_call>{"name":"non_tool","arguments":{"text":"xin'),
            [],
        )
        self.assertEqual(
            parse(
                '<tool_call>{"name":"non_tool","arguments":{"text":"xin chào"}} trailing'
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
