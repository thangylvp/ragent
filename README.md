# RAGENT voice-agent demo and Jetson benchmark

This repository reproduces the Vietnamese speech-to-action demo and its
performance measurements. It uses the standalone car STCC checkpoint as the
temporary parser while the future robot tool contract is designed.

This is deliberately **not** a training or data-generation repository. It does
not include a trainer, generated dataset, raw recordings or model weights. The
gated standalone checkpoint is downloaded separately from Hugging Face and is
served through vLLM.

## Demo pipeline

```text
browser microphone -> FastEnhancer-S -> OmniVAD -> STCC SLM
  |-> tool_call -> execute simulator -> cached Vietnamese response
  `-> non_tool  -> Gemini -> laptop OmniVoice -> dynamic Vietnamese response
```

| Component | Responsibility |
| --- | --- |
| `src/enhancement` | Causal microphone noise suppression before VAD |
| `src/vad` | Streaming speech detection, buffering and endpointing |
| `demo/backend` | Calls the standalone SLM through vLLM and exposes the web loop |
| `src/harness` | Routes tool/non-tool turns and selects cached or dynamic audio |
| `src/execute` | Validates calls and simulates hardware success/reject/busy state |
| `demo/frontend` | Conversation, raw SLM/cloud output, audio replay and per-turn timing |

The SLM receives only the current audio turn. The harness owns conversation
history for cloud turns. The execute layer owns tool validation and simulated
hardware state. Fixed tool outcomes use a checksum-verified audio cache;
runtime TTS is reserved for dynamic cloud replies.

## Repository layout

```text
robot-speech-to-action/
├── configs/
│   ├── benchmarks/             # Checked-in tool/non-tool corpus metadata
│   └── vad/                    # VAD comparison notes and settings
├── demo/                       # Browser UI, API/WebSocket backend and laptop TTS worker
├── docker/jetson-vllm-audio/  # Pinned audio-capable vLLM image
├── docs/                       # Architecture, measured results and deployment notes
├── scripts/
│   ├── jetson/                 # vLLM, MPS, backend and benchmark launchers
│   └── *.py                    # Smoke, benchmark, analysis and report tools
├── src/
│   ├── enhancement/
│   ├── execute/
│   ├── harness/
│   └── vad/
└── tests/                      # Deterministic component and web-loop tests
```

## Reproduce on Jetson

Follow [docs/jetson-reproduction.md](docs/jetson-reproduction.md) for the exact
clean-device procedure: gated model access, container build, portable path
settings, health/smoke tests, MPS verification, the laptop/Jetson split, and
benchmark commands.

The core serve path is:

```bash
docker build --network=host -t stcc-vllm:0.22.0-audio docker/jetson-vllm-audio
RAGENT_MODEL_DIR="$HOME/models/stcc" scripts/jetson/run_vllm_condition.sh 100
```

Then see [demo/README.md](demo/README.md) for the web demo and tunnel, and
[docs/jetson-full-system-benchmark-2026-08-14.md](docs/jetson-full-system-benchmark-2026-08-14.md)
for definitions, raw-artifact locations and measured results. The presentation
is [docs/jetson-voice-agent-results-by-audio-length.pptx](docs/jetson-voice-agent-results-by-audio-length.pptx).

## Validate the code

```bash
PYTHONPATH=src:. python -m pytest -q
for script in demo/*.sh scripts/jetson/*.sh; do bash -n "$script"; done
```

Raw audio, model checkpoints, cached speech and benchmark runs stay under the
ignored `outputs/` tree or external storage. `.env` and API keys are ignored;
`.env.example` contains only safe placeholders.

## Deployment takeaway

The complete pipeline, including the SLM, can run in the cloud. That is the
simplest deployment and update path. Move the SLM to Jetson only when latency,
connectivity, privacy or local autonomy justify the extra compatibility,
resource-sharing, thermal and fleet-maintenance work.
