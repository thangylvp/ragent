# RAGENT end-to-end voice demo

Version 0.2 composes the current components into one persistent, half-duplex
voice loop:

```text
browser microphone → enhancement → server VAD → standalone STCC through vLLM
  ├─ non_tool transcript → Gemini cloud agent → laptop OmniVoice TTS
  └─ tool call → schema validation → simulated hardware
       ├─ success → cached tool-specific Vietnamese confirmation
       ├─ missing required field → cached Vietnamese follow-up
       ├─ rejected → cached detailed Vietnamese explanation
       └─ busy → cached tool-specific busy response
```

The page is a demo surface, not the authoritative evaluator for any component.
It deliberately uses the existing car checkpoint and its 33-tool catalog until
the robot catalog/model is ready. The harness talks only to the
`src/execute` interface, so that simulator can be replaced without changing
the browser or turn loop.

## Set up the demo runtime

Create or activate the host environment used for enhancement/VAD, then install
the demo dependencies:

```bash
cd /path/to/ragent
python -m pip install -r demo/requirements.txt
mkdir -p outputs/denoise/models
curl -L --fail \
  -o outputs/denoise/models/fastenhancer_s_dns.onnx \
  https://github.com/aask1357/fastenhancer/releases/download/onnx-dns-v1.0.0/fastenhancer_s.onnx
```

Start the resident single-voice OmniVoice service in one laptop terminal. Use
an environment where `omnivoice`, PyTorch, FastAPI, NumPy and SoundFile are
installed. Reference audio is deliberately external to Git; provide one clean
voice clip and its exact transcript:

```bash
export RAGENT_TTS_PYTHON=/path/to/tts-env/bin/python
export DEMO_OMNIVOICE_REF_AUDIO=/path/to/reference.mp3
export DEMO_OMNIVOICE_REF_TEXT=/path/to/reference.txt
./demo/run_omnivoice_worker.sh
```

Run the backend on Jetson. The deployment script binds it to Jetson loopback,
uses the local vLLM service on port 8000, and points dynamic speech at port
8120 supplied by the reverse SSH forward:

```bash
cd "$HOME/ragent"
./scripts/jetson/install_gemini_key.sh  # one-time, reads the key without echo
./scripts/jetson/run_demo_backend.sh
```

The measured demo used one northern Vietnamese female reference voice. The
repository excludes that voice data; any authorized reference clip can be
selected through the two variables above. In a second laptop terminal, create
the bidirectional demo tunnel:

```bash
./demo/connect_jetson.sh
```

Open <http://127.0.0.1:8011> and click **Bắt đầu** once. The page, microphone
and speaker are on the laptop, while the HTML, API, WebSocket, enhancement,
VAD, SLM, harness and execute layer run on Jetson. The tunnel maps laptop
`127.0.0.1:8011` to Jetson `127.0.0.1:8010` and maps Jetson
`127.0.0.1:8120` back to laptop OmniVoice. This works on networks that permit
SSH but isolate arbitrary LAN ports, keeps browser media APIs on a trusted
localhost origin, and requires no CORS configuration.

For a persistent demo, run the tunnel and Jetson backend in separate tmux
sessions. The active sessions can be inspected with:

```bash
tmux attach -t ragent-jetson-tunnel
ssh jetson 'tmux attach -t ragent-demo-backend'
```

Detach with `Ctrl-B`, then `D`. Stopping the laptop tunnel makes the page and
dynamic TTS unavailable but does not stop Jetson vLLM. Stopping the Jetson
backend does not stop vLLM either.

The same WebSocket and microphone stay alive across turns. When VAD finalizes
a turn, the input gate
closes before SLM/cloud/execution/TTS and reopens only after response playback
finishes. This prevents the assistant voice from becoming the next request.
The technical drawer retains time-aligned audio before denoising and the exact
SLM input for diagnosis.

Every AI message retains three timing cards using the VAD-finalized audio
boundary:

- audio ready → first SLM token;
- audio ready → last SLM token;
- audio ready → first response-audio playback in the browser.

The third measurement includes harness, cloud/execution, cache/TTS, HTTP
transfer, browser decode and scheduling. It is the closest demo metric to what
the user feels after endpointing. The same message also keeps the exact raw SLM
output and an expandable component-time breakdown, so earlier turns remain
inspectable instead of being replaced by the latest result.

The conversation separates ownership visibly. Tool turns show **User → SLM →
AI**; non-tool turns show **User → SLM → Cloud Agent → AI**. User contains the
exact WAV and replay control, SLM contains raw generation plus the parsed
route/tool call, Cloud Agent contains its free-form text/model/latency, and AI
contains the final spoken response.
While an input preview is playing, browser microphone transport is paused to
prevent the demo from recognizing its own recording as a new command. The
technical drawer continues to show the latest aligned audio before enhancement
for A/B diagnosis.

FastEnhancer-S is the default enhancement backend; `none` is available for
direct A/B testing. It is a causal 16 kHz model with 16 ms algorithmic delay.
OmniVAD is the default VAD backend. Its FireRed Stream-VAD policy uses threshold
`0.65`, 150 ms minimum speech, 300 ms end silence and 80 ms onset padding. The
browser requests echo cancellation but disables browser noise suppression and
automatic gain control. This avoids cascading two uncontrolled denoisers and
raising the background noise floor. Browser media constraints are requests
rather than guarantees and may vary by browser and input device.

## Standalone SLM serving contract

Serve the same standalone STC checkpoint with its documented flags, expose it
at local port 8100, then run:

```bash
WEBTEST_VLLM_BASE_URL=http://127.0.0.1:8100/v1 \
WEBTEST_VLLM_MODEL=stcc \
  ./demo/run_vllm.sh
```

The model request uses the checkpoint's `tools_openai.json` unchanged, sends
only audio in the user turn, disables thinking, uses greedy decoding and reads
streaming `delta.tool_calls`. No additional SLM system prompt is added.

Replace `OPENAI_API_KEY` before demonstrating non-tool conversation. An invalid
key produces a spoken Vietnamese cloud-unavailable response without breaking
tool execution. OmniVoice is the default TTS provider. Pre-generate every fixed
confirmation and follow-up as a WAV cache hit while the worker is running:

```bash
PYTHONPATH=src:. python scripts/prewarm_demo_audio.py --workers 1
```

The command generates 135 fixed response entries and a checksum-verified
`outputs/demo/voice/manifest.json`: per-tool success, missing-field, busy and
detailed reject clips plus system fallbacks. Runtime synthesis is reserved for
successful cloud answers whose text cannot be known in advance. Copy the WAV
files together with the manifest to Jetson; fixed tool outcomes never contact
the laptop TTS service.

## VAD choices

| Choice | Runtime | Live frame |
| --- | --- | ---: |
| `omnivad` | FireRed Stream-VAD on ncnn CPU | 10 ms |
| `firered` | FireRed Stream-VAD on PyTorch CPU | 10 ms hop / 25 ms window |
| `silero` | Silero VAD v6 on PyTorch CPU | 32 ms |
| `webrtc` | WebRTC GMM VAD mode 2 | 20 ms |
| `energy` | Dependency-free adaptive energy baseline | 20 ms |

Browser echo cancellation is enabled for the interactive demo. Browser noise
suppression and automatic gain control are disabled because the selected
server enhancer owns noise reduction. The server-side VAD remains authoritative
for start/end decisions.

Captured WAV files are stored under the ignored directory
`outputs/demo/captures/`. Each completed turn produces `<id>.wav` (exact model
input) and `<id>.before-enhancement.wav` (time-aligned pre-denoise PCM). A
manual stop always discards partial speech and does not call the model.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `WEBTEST_MODEL_MODE` | `vllm` | `vllm`, or `disabled` for VAD-only testing |
| `WEBTEST_MODEL_DIR` | `outputs/models/stcc` | Standalone checkpoint directory; the backend reads its tool catalog |
| `WEBTEST_VAD` | `omnivad` | Initially selected VAD |
| `WEBTEST_ENHANCER` | `fastenhancer_s` | Initially selected speech enhancer; use `none` for A/B |
| `WEBTEST_FASTENHANCER_S_MODEL` | ignored local ONNX download | FastEnhancer-S DNS checkpoint |
| `WEBTEST_FIRERED_MODEL_DIR` | ignored local FireRed download | Stream-VAD weights |
| `WEBTEST_VLLM_BASE_URL` | `http://127.0.0.1:8100/v1` | vLLM OpenAI API |
| `WEBTEST_PORT` | `8010` | Web server port |
| `WEBTEST_CAPTURE_DIR` | `outputs/demo/captures` | Retained model-input WAV files |
| `DEMO_CLOUD_PROVIDER` | `gemini` | `gemini` or `openai`; affects only `non_tool` conversation |
| `DEMO_CLOUD_MODEL` | `gemini-3.6-flash` | Stable Gemini model for `non_tool` conversation |
| `DEMO_GEMINI_THINKING_LEVEL` | `low` | `minimal`, `low`, `medium`, or `high`; low favors voice latency |
| `GEMINI_API_KEY` | unset | Required only on the Jetson backend for Gemini cloud turns |
| `DEMO_CLOUD_ENABLED` | `1` | Disable cloud routing with `0` |
| `DEMO_TTS_PROVIDER` | `omnivoice` | `omnivoice`, `edge`, `openai`, or `disabled` |
| `DEMO_OMNIVOICE_BASE_URL` | `http://127.0.0.1:8120` | Resident OmniVoice worker |
| `DEMO_OMNIVOICE_CHECKPOINT` | `splendor1811/omnivoice-vietnamese` | Vietnamese OmniVoice checkpoint |
| `DEMO_OMNIVOICE_VOICE_ID` | `female_north_1` | Stable client-side voice identity; reference files remain on the worker |
| `DEMO_OMNIVOICE_REF_AUDIO` | `outputs/demo/reference_voice/reference.mp3` | External single reference voice |
| `DEMO_OMNIVOICE_REF_TEXT` | `outputs/demo/reference_voice/reference.txt` | Exact reference-audio transcript |
| `DEMO_OMNIVOICE_NUM_STEPS` | `32` | Flow-matching synthesis steps |
| `DEMO_OMNIVOICE_SPEED` | `0.80` | Duration safety margin that prevents OmniVoice from dropping initial words or sentences |
| `DEMO_TTS_MODEL` | `gpt-4o-mini-tts` | Used only with OpenAI TTS |
| `DEMO_VOICE_CACHE_DIR` | `outputs/demo/voice` | Persistent response WAV cache |
| `DEMO_STATIC_AUDIO_MANIFEST` | `outputs/demo/voice/manifest.json` | Required read-only fixed-response bundle index |

## Implemented layout

```text
demo/
├── README.md
├── requirements.txt
├── run.sh
├── run_vllm.sh
├── run_omnivoice_worker.sh       # Starts the resident one-voice TTS service
├── connect_jetson.sh             # Browser/API forward + reverse laptop-TTS forward
├── tts_worker.py                 # OmniVoice load, prompt and WAV endpoint
├── docs/DESIGN.md
├── backend/
│   ├── app.py                   # Persistent WebSocket and component composition
│   ├── model.py                 # Streaming vLLM SLM adapter
│   └── settings.py
└── frontend/
    ├── index.html
    ├── styles.css
    ├── audio-worklet.js         # Resampling/transport only; no VAD decisions
    └── app.js
```

The enhancement adapter lives in `src/enhancement/live.py`; detector adapters
and utterance buffering live in `src/vad/live.py`; routing/cloud/TTS live in
`src/harness`; and schema validation plus the digital twin live in
`src/execute`. [`docs/DESIGN.md`](docs/DESIGN.md) records the remaining
migration from car proxy hardware to the future robot contract.

## End-to-end latency benchmark

The reproducible benchmark sends the WAV through the WebSocket at microphone
speed. The speech-end annotation is independent from VAD, and CUDA is
synchronized at the first and last generated tokens:

```bash
python \
  scripts/benchmark_webtest_e2e.py \
  outputs/demo/captures/gpu-smoke-tool.wav \
  --speech-end-ms 1710.125 --enhancer fastenhancer_s --backend omnivad --runs 6
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
