# Demo integration tests

`tests/vad/test_live_vad.py` checks retroactive neural-VAD onset buffering.
`demo/tests/test_webtest.py` streams deterministic PCM through the real
WebSocket and energy VAD up to a disabled model boundary. Browser automation
remains follow-up work. Real car-model inference is a manual hardware test and
the webtest never connects to robot execution.
