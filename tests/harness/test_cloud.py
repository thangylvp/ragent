from __future__ import annotations

from types import SimpleNamespace

from harness.cloud import GeminiCloudAgent, build_cloud_agent


class StubResponse:
    status_code = 200

    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


class StubClient:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def post(self, url, json):
        self.requests.append((url, json))
        return self.response


def _settings(**overrides):
    values = {
        "cloud_model": "gemini-3.6-flash",
        "gemini_thinking_level": "low",
        "gemini_base_url": "https://generativelanguage.googleapis.com/v1beta",
        "gemini_api_key": "test-key",
        "cloud_timeout_s": 10,
        "cloud_history_turns": 2,
        "cloud_enabled": True,
        "cloud_provider": "gemini",
        "openai_api_key": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_gemini_reply_uses_vietnamese_system_prompt_and_bounded_history():
    agent = GeminiCloudAgent(_settings())
    client = StubClient(
        StubResponse(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"thought": True, "text": "hidden"},
                                {"text": "Xin chào, tôi có thể giúp gì cho bạn?"},
                            ]
                        },
                        "finishReason": "STOP",
                    }
                ]
            }
        )
    )
    agent._client = client

    reply = agent.reply("Xin chào")

    assert reply.model == "gemini-3.6-flash"
    assert reply.text == "Xin chào, tôi có thể giúp gì cho bạn?"
    url, payload = client.requests[0]
    assert url.endswith("/models/gemini-3.6-flash:generateContent")
    assert "tiếng Việt" in payload["systemInstruction"]["parts"][0]["text"]
    assert payload["generationConfig"]["maxOutputTokens"] == 1024
    assert payload["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "low"
    assert payload["contents"][-1] == {
        "role": "user",
        "parts": [{"text": "Xin chào"}],
    }
    assert agent.info["provider"] == "gemini"
    assert agent.info["ready"] is True


def test_build_cloud_agent_disables_gemini_without_key():
    agent = build_cloud_agent(_settings(gemini_api_key=None))
    assert agent.enabled is False
