# VAD → car-STC component webtest

Version 0.1 is intentionally narrower than the future robot demo:

```text
browser microphone → selected server-side VAD → finalized WAV
                   → existing car CommandASR + its original 33 tools
                   → visible tool_call or non_tool result
```

It exists to test microphone streaming, endpoint behavior, retained audio and
the effect of VAD cuts on an ASR-based semantic parser. Calls are never
executed. It does not define or simulate the robot harness, execution layer or
tool contract.

## Run with the local car checkpoint

Use the existing `mega-asr` environment. Install the two lightweight optional
VAD packages once, without replacing its PyTorch stack:

```bash
cd /home/thangnv94/v/robot-speech-to-action
/home/thangnv94/miniconda3/envs/mega-asr/bin/python -m pip install \
  omnivad==0.2.13 webrtcvad-wheels==2.0.14
./demo/run_local_model.sh
```

Open <http://127.0.0.1:8010>, select a VAD, click **Start listening**, speak
one car request and then stay quiet. The page stops the microphone after the
VAD finalizes one utterance, lets you replay the exact WAV sent to the model,
and shows the parsed call and raw model output.

The local model requires CUDA by default. To allow slow CPU generation for
diagnosis, set `WEBTEST_ALLOW_CPU_MODEL=1`.

## Run against vLLM

Serve the same standalone STC checkpoint with its documented flags, expose it
at local port 8100, then run:

```bash
WEBTEST_VLLM_BASE_URL=http://127.0.0.1:8100/v1 \
  ./demo/run_vllm.sh
```

The model request uses the checkpoint's `tools_openai.json` unchanged, sends
only audio in the user turn, disables thinking, uses greedy decoding and reads
the result from `message.tool_calls`. No additional system prompt is added.

## VAD choices

| Choice | Runtime | Live frame |
| --- | --- | ---: |
| `omnivad` | FireRed Stream-VAD on ncnn CPU | 10 ms |
| `firered` | FireRed Stream-VAD on PyTorch CPU | 10 ms hop / 25 ms window |
| `silero` | Silero VAD v6 on PyTorch CPU | 32 ms |
| `webrtc` | WebRTC GMM VAD mode 2 | 20 ms |
| `energy` | Dependency-free adaptive energy baseline | 20 ms |

Browser echo cancellation, noise suppression and automatic gain control are
disabled so the page exposes the actual microphone signal to the VAD. The
server is authoritative for start/end decisions.

Captured WAV files are stored under the ignored directory
`outputs/demo/captures/`. A manual stop always discards partial speech and does
not call the model.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `WEBTEST_MODEL_MODE` | `local` | `local`, `vllm`, or `disabled` for VAD-only testing |
| `WEBTEST_MODEL_DIR` | local step-1250 STC export | Checkpoint and bundled tool catalog |
| `WEBTEST_VAD` | `omnivad` | Initially selected VAD |
| `WEBTEST_FIRERED_MODEL_DIR` | ignored local FireRed download | Stream-VAD weights |
| `WEBTEST_VLLM_BASE_URL` | `http://127.0.0.1:8100/v1` | vLLM OpenAI API |
| `WEBTEST_PORT` | `8010` | Web server port |
| `WEBTEST_CAPTURE_DIR` | `outputs/demo/captures` | Retained model-input WAV files |

## Implemented layout

```text
demo/
├── README.md
├── requirements.txt
├── run.sh
├── run_local_model.sh
├── run_vllm.sh
├── docs/DESIGN.md
├── backend/
│   ├── app.py                   # WebSocket transport and capture files
│   ├── model.py                 # Local and vLLM car-model adapters
│   └── settings.py
└── frontend/
    ├── index.html
    ├── styles.css
    ├── audio-worklet.js         # Resampling/transport only; no VAD decisions
    └── app.js
```

The detector adapters and PCM buffering live in `src/vad/live.py`. The broader
future robot-demo design remains documented in
[`docs/DESIGN.md`](docs/DESIGN.md), but none of its harness behavior is
implemented by this webtest.

## End-to-end latency benchmark

The reproducible benchmark sends the WAV through the WebSocket at microphone
speed. The speech-end annotation is independent from VAD, and CUDA is
synchronized at the first and last generated tokens:

```bash
/home/thangnv94/miniconda3/envs/mega-asr/bin/python \
  scripts/benchmark_webtest_e2e.py \
  outputs/demo/captures/gpu-smoke-tool.wav \
  --speech-end-ms 1710.125 --backend omnivad --runs 6
```

On 2026-08-12, the 2.0-second `set_fog_lights` sample ran through the complete
path on the RTX 5070 Laptop GPU. Speech end was estimated at 1710.125 ms using
the start of the final silence at -35 dB/30 ms. Five warm runs produced:

| Boundary | Mean | Range |
| --- | ---: | ---: |
| Speech end → OmniVAD endpoint | 294.8 ms | 292.4–296.0 ms |
| Speech end → first SLM token | about 830 ms | derived from endpoint + model TTFT |
| Speech end → last SLM token | 1239.7 ms | 1235.7–1245.0 ms |
| Speech end → parsed result ready | 1239.8 ms | 1235.8–1245.1 ms |

The warm post-VAD mean was 0.49 ms to retain the WAV, 1.51 ms to decode it,
1.42 ms to render the tool prompt, 21.17 ms for processor feature extraction,
0.72 ms for host-to-GPU transfer, and 918.64 ms inside generation. Generation
split into 509.77 ms to the first token and 408.87 ms from first to last token
for 29 output tokens. Decode and tool-call parsing after the last token added
only 0.08 ms. The model returned the expected call on all six runs:

```text
OmniVAD WebSocket stream
  → 1.92 s finalized WAV
  → local STC checkpoint
  → set_fog_lights(state=on, position=front)
```

The first lazy-load run took 2366.0 ms from speech end to last token, including
762.2 ms of model loading with the OS file cache already warm. This is not a
machine-boot cold-start number. Keep the model resident for serving.

OmniVAD used about 2.41 ms of CPU per 10 ms frame in this Python/thread-pool
webtest. Its accumulated work happens online while the user is speaking and
therefore must not be added again to the post-speech critical path.
