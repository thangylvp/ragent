# Test scope

The automated suite covers the code shipped for the demo and benchmark:

- enhancement and streaming VAD behavior;
- tool-call parsing and schema handoff;
- execute-layer validation and state transitions;
- harness routing, cloud fallbacks and static-audio integrity;
- web-loop API and per-turn timing behavior.

GPU/vLLM latency is exercised by the Jetson benchmark drivers rather than by
the default CPU test suite.
