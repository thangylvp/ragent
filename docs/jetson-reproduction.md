# Reproduce the STCC voice-agent demo on Jetson

This is the canonical clean-machine procedure for the tested edge deployment.
The benchmark platform was a Jetson AGX Orin 32 GB running Ubuntu 24.04,
JetPack 7.2 / L4T 39.2, Docker with NVIDIA CDI, and MAXN power mode. The model
server used vLLM 0.22.0. The demo temporarily uses the car STCC checkpoint as
the speech-to-action parser while the robot tool contract is being designed.

## 1. Prerequisites

- Docker must work for the current user without `sudo`.
- `docker run --rm --device nvidia.com/gpu=all ubuntu:24.04 true` must accept
  the NVIDIA CDI device syntax.
- Use the Jetson's unrestricted network interface for GitHub, Hugging Face and
  container downloads. In our setup that is Wi-Fi; the wired network blocks
  some of those services.
- Request access to the gated model repository `thangylvp/stcc` before
  downloading it.

Clone the code and fetch the checkpoint:

```bash
git clone https://github.com/thangylvp/ragent.git "$HOME/ragent"
cd "$HOME/ragent"

python3 -m pip install --user 'huggingface_hub[cli]'
hf auth login
hf download thangylvp/stcc --local-dir "$HOME/models/stcc"
test -f "$HOME/models/stcc/config.json"
test -f "$HOME/models/stcc/tools_openai.json"
```

The Hugging Face repository includes short tool-call and non-tool sample audio
files. Raw/private benchmark audio is intentionally not stored in Git. This
repository contains no training or dataset-generation pipeline.

## 2. Build the Jetson vLLM image

The local image adds the audio decoding packages that the base vLLM server
needs for audio requests:

```bash
docker build --network=host \
  -t stcc-vllm:0.22.0-audio \
  docker/jetson-vllm-audio
```

If the base image tag changes or is mirrored internally, override it with
`--build-arg VLLM_BASE_IMAGE=<image>`. Keep vLLM 0.22.0 for exact benchmark
reproduction; validate model compatibility again before changing versions.

## 3. Configure portable paths

The launchers contain defaults from the benchmark Jetson, but every machine-
specific path can be overridden:

```bash
export RAGENT_JETSON_APP_DIR="$PWD"
export RAGENT_MODEL_DIR="$HOME/models/stcc"
export RAGENT_VLLM_CACHE_DIR="$HOME/.cache/vllm-stcc022"
export RAGENT_VLLM_IMAGE="stcc-vllm:0.22.0-audio"
export RAGENT_VLLM_MODEL_NAME="stcc"
export RAGENT_VLLM_PORT="8000"
```

The host-side VAD/demo uses a separate Python environment. Create one and set
its interpreter path:

```bash
python3 -m venv "$HOME/venvs/ragent-demo"
"$HOME/venvs/ragent-demo/bin/pip" install -r demo/requirements.txt
export RAGENT_JETSON_PYTHON="$HOME/venvs/ragent-demo/bin/python"
```

For the default FastEnhancer path, download the small ONNX model once:

```bash
mkdir -p "$HOME/models/fastenhancer"
curl -fL \
  https://github.com/aask1357/fastenhancer/releases/download/onnx-dns-v1.0.0/fastenhancer_s.onnx \
  -o "$HOME/models/fastenhancer/fastenhancer_s.onnx"
export RAGENT_FASTENHANCER_MODEL="$HOME/models/fastenhancer/fastenhancer_s.onnx"
```

## 4. Start and verify the SLM

Run unconstrained/full GPU scheduling:

```bash
scripts/jetson/run_vllm_condition.sh 100
```

The command stays in the foreground so the model log is visible. Use a
terminal multiplexer or service manager for unattended operation. From another
terminal:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
python3 scripts/smoke_test_vllm_audio.py \
  "$HOME/models/stcc/samples/tool_call_set_fog_lights.wav" \
  --tools "$HOME/models/stcc/tools_openai.json"
```

If the sample filename differs, list `"$HOME/models/stcc/samples"` and pass
one of its WAV/MP3 files. For controlled GPU-sharing measurements, stop the
current container and start a percentage condition such as:

```bash
scripts/jetson/run_vllm_condition.sh 30
```

Values below 100 enable CUDA MPS and set the default active-thread percentage.
This is an SM scheduling quota, not the same experiment as running a synthetic
GPU-busy process. Confirm the applied setting in the startup log:
`MPS_DEFAULT_ACTIVE_THREAD_PERCENTAGE=30`. Use `100` for the normal deployment.

## 5. Run the edge backend and laptop UI/TTS

Install the cloud key on the Jetson without echoing or committing it:

```bash
scripts/jetson/install_gemini_key.sh
```

Alternatively copy `.env.example` to `.env` and edit it locally. The checked-in
example contains no credentials.

`demo/README.md` documents the complete split deployment. In that topology,
VAD, SLM, harness and execution simulation run on Jetson; the browser runs on
the laptop; the Jetson calls laptop-hosted OmniVoice only for dynamic cloud
responses. Deterministic tool success, rejection, missing-field and busy
responses are cached and played on the edge.

The laptop OmniVoice worker requires an authorized external reference voice
and transcript through `DEMO_OMNIVOICE_REF_AUDIO` and
`DEMO_OMNIVOICE_REF_TEXT`. After it is healthy, run
`scripts/prewarm_demo_audio.py` once and copy its ignored voice bundle plus
manifest to the matching Jetson cache path. No reference or generated voice
data is committed to this repository.

Start the Jetson backend after the vLLM health check succeeds:

```bash
scripts/jetson/run_demo_backend.sh
```

Expose only the required ports over SSH from the laptop, following the exact
tunnel commands in `demo/README.md`. Never commit `demo/.env` or API tokens.

## 6. Reproduce measurements

The checked-in benchmark drivers and condition configs are:

- `scripts/benchmark_jetson_full_system.py` and
  `configs/benchmarks/jetson_tool_latency_corpus.json` for the tool-call path;
- `scripts/benchmark_jetson_cloud_path.py` and
  `configs/benchmarks/jetson_cloud_latency_corpus.json` for the non-tool/cloud
  path;
- `scripts/jetson/run_measured_condition.sh` and
  `scripts/jetson/run_measured_cloud_condition.sh` for 100/70/50/30% MPS runs;
- `scripts/analyze_jetson_full_system.py` and
  `scripts/analyze_jetson_cloud_path.py` for reports.

Results go under the ignored `outputs/` tree. Exact number-for-number
reproduction also requires the same WAV corpus and warm-up policy referenced by
the configs; the repository deliberately excludes private/raw audio and runtime
caches. The committed report records the measured environment, definitions,
trial count and aggregate statistics.

## Deployment decision

The entire pipeline, including the SLM, can run in the cloud. That is the
simplest option to deploy, update and monitor. Put the SLM on the edge only when
lower tool-call latency, offline operation, privacy or local autonomy justify
the additional compatibility, memory, thermal and fleet-maintenance work.
