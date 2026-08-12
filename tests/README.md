# Test plan

The automated suite will cover:

- contract parsing and version compatibility;
- partial-call acceptance and required-field detection;
- route, tool-name and argument scoring;
- manifest validation and split leakage checks;
- prompt/template parity between training and vLLM serving;
- standalone checkpoint export audits;
- sample audio inference on supported vLLM versions.

Unit tests must use small fixtures and fake model clients. GPU integration and
latency tests will be marked separately.
