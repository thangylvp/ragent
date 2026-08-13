# Jetson AGX Orin vLLM deployment and benchmark — 2026-08-13

This note records the tested deployment of the standalone car STCC checkpoint
on the Jetson AGX Orin. The car model is used as a proxy for the future robot
SLM because both use the same Qwen3-ASR direct audio-to-structured-call
architecture.

It also compares normal execution with a verified 50% CUDA MPS compute share,
representing a deployment in which STCC must leave GPU execution capacity for
another model.

## Result

The standalone checkpoint serves correctly with vLLM 0.22.0 on this Jetson
after adding vLLM's missing Python audio dependencies to the official ARM64
image. All 27 measured requests produced the expected route, tool name and
arguments at both compute allocations.

Limiting the live vLLM engine to 50% of the GPU's active threads reduced decode
throughput from 78.3 to 69.4 tokens/s and increased mean client-to-full-output
latency from 603.8 to 674.2 ms. The latency penalty was approximately 10–13%,
not 2x, because this small model is not limited only by raw SM compute.

| Audio | TTFT 100% | TTFT 50% | Full output 100% | Full output 50% | RTF 100% | RTF 50% |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 s | 86.1 ms | 88.0 ms | 398.4 ms | 438.7 ms | 0.392 | 0.432 |
| 2 s | 88.7 ms | 88.3 ms | 452.9 ms | 497.9 ms | 0.223 | 0.245 |
| 3 s | 91.3 ms | 92.2 ms | 443.2 ms | 487.7 ms | 0.145 | 0.160 |
| 4 s | 95.8 ms | 104.6 ms | 486.5 ms | 544.0 ms | 0.120 | 0.134 |
| 5 s | 98.1 ms | 107.4 ms | 630.2 ms | 706.0 ms | 0.124 | 0.140 |
| 6 s | 100.8 ms | 109.7 ms | 633.6 ms | 709.5 ms | 0.104 | 0.117 |
| 7 s | 102.9 ms | 114.0 ms | 675.4 ms | 758.5 ms | 0.095 | 0.107 |
| 8 s | 105.2 ms | 114.6 ms | 780.8 ms | 875.0 ms | 0.096 | 0.108 |
| 9 s | 115.6 ms | 131.6 ms | 932.9 ms | 1,050.4 ms | 0.103 | 0.116 |

The nominal 9-second sample is 8.953 seconds long. `TTFT` is vLLM's server
time from request arrival to its first generated token. `Full output` is client
wall time until the complete OpenAI-compatible response is received. `RTF` is
vLLM server end-to-end time divided by audio duration; lower is better and a
value below 1 means processing is faster than real time.

## Tested system

| Item | Value |
| --- | --- |
| Device | NVIDIA Jetson AGX Orin, 32 GB unified memory |
| OS | Ubuntu 24.04, ARM64 |
| JetPack / L4T | JetPack 7.2 / L4T R39.2 |
| Host CUDA | 13.2 |
| Power mode | `MAXN` |
| Storage | 1 TB NVMe |
| vLLM | 0.22.0 |
| Serving image | `stcc-vllm:0.22.0-audio` |
| Base image | `vllm/vllm-openai:v0.22.0-ubuntu2404` |
| PyTorch in image | 2.11.0+cu130 |
| Model dtype | BF16 |
| Model architecture | `Qwen3ASRForConditionalGeneration` |
| Checkpoint on Jetson | `/home/trinq3/models/stcc` |
| Tool catalog | 33 OpenAI-format tools |
| Context limit | 8,192 tokens |
| Client load | One request at a time |

The copied model file was checksum-verified against the source checkpoint. Its
SHA-256 digest was:

```text
af32b8d68c67df7e858f574b3dbcb7a66bfb269753cf08507c3e5e1fea3eaa57
```

## Network routing

The Jetson has wired and wireless interfaces. The wired network restricts
external sites, so model/image downloads should use the wireless default route.
SSH may continue over the wired address while outbound internet traffic uses
Wi-Fi.

Verify the chosen outbound interface before downloading:

```bash
ip route get 1.1.1.1
```

The result should contain the wireless interface, for example:

```text
1.1.1.1 via <wifi-gateway> dev wlP1p1s0 src <wifi-address>
```

Docker on this machine was configured with an unavailable wired-network HTTP
proxy. A regular `docker pull` therefore failed even when the host default
route was Wi-Fi. The tested workaround was a user-level BuildKit builder using
host networking and no proxy. Do not use `sudo` for Docker; the Jetson user is
already in the `docker` group.

## Audio-capable vLLM image

The official vLLM 0.22 ARM64 image can load the Qwen3-ASR architecture, compile
the optimized engine and use FlashAttention. However, it does not contain the
Python packages needed to decode an `audio_url`. The server starts normally,
but an audio request fails with HTTP 400 and an `ImportError` for `av`.

The tested derived image adds these packages:

- `av==18.1.0`
- `scipy==1.18.0`
- `soundfile==0.14.0`
- `soxr==1.1.0`

Because the Jetson's Python wheel download path was unreliable, the ARM64
wheels were downloaded on the workstation, copied to the Jetson and installed
offline:

```dockerfile
FROM vllm/vllm-openai:v0.22.0-ubuntu2404

COPY wheels/ /wheels/
RUN python3 -m pip install --no-cache-dir --no-index \
      --find-links=/wheels \
      av==18.1.0 scipy==1.18.0 soundfile==0.14.0 soxr==1.1.0 \
    && rm -rf /wheels
```

Build it on the Jetson as:

```bash
docker build -t stcc-vllm:0.22.0-audio /path/to/build-context
```

## Normal server configuration

The tested full-compute server command is:

```bash
mkdir -p /home/trinq3/.cache/vllm-stcc022

docker run --rm \
  --name stcc-vllm022 \
  --runtime=nvidia \
  --gpus all \
  --network=host \
  --ipc=host \
  --volume /home/trinq3/models/stcc:/model:ro \
  --volume /home/trinq3/.cache/vllm-stcc022:/root/.cache/vllm \
  stcc-vllm:0.22.0-audio \
  /model \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name stcc \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --generation-config auto \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --limit-mm-per-prompt '{"audio":1}' \
  --mm-processor-cache-gb 0 \
  --gpu-memory-utilization 0.80
```

Disabling the multimodal processor cache avoids retaining copies of past audio
and releases approximately 0.3 GiB in this configuration. vLLM's reusable
text/tool prefix cache remains active. The compile-cache volume is important:
subsequent starts reused the compiled graphs and reduced `torch.compile` time
from approximately 41 seconds to approximately 7 seconds.

Check health and the exact runtime version with:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/version
curl -fsS http://127.0.0.1:8000/v1/models
```

Port 8000 was not reachable directly from the workstation. Use an SSH tunnel:

```bash
ssh -N -L 8000:127.0.0.1:8000 jetson
```

The workstation can then call `http://127.0.0.1:8000`.

## Cold start and warm-up

With a saved compile cache, the HTTP server still takes approximately two
minutes to become ready because it loads the model, profiles memory and captures
CUDA graphs. HTTP readiness is not sufficient for production traffic: the
first real audio request can trigger additional Triton JIT compilation.

Observed first audio request after startup:

| Allocation | Cold TTFT | Cold full output |
| --- | ---: | ---: |
| 100% compute | approximately 1.47 s | approximately 1.82 s |
| 50% MPS compute | approximately 1.93 s | approximately 2.33 s |

Production startup should therefore:

1. wait for `/health`;
2. send one representative audio request with the complete tool list;
3. validate its tool call;
4. only then advertise application readiness.

## 50% CUDA MPS configuration

`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50` limits the CUDA work submitted by this
process to half of the available SM threads. The benchmark also set the MPS
server default before starting vLLM:

```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/stcc-mps/pipe
export CUDA_MPS_LOG_DIRECTORY=/tmp/stcc-mps/log
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50

mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
printf 'set_default_active_thread_percentage 50\n' \
  | nvidia-cuda-mps-control
```

The vLLM process must start with those environment variables and connect to
that MPS daemon. Do not assume the limit is active merely because the variables
exist. Verify the live client:

```bash
printf 'get_server_list\n' | nvidia-cuda-mps-control
printf 'ps\n' | nvidia-cuda-mps-control
printf 'get_active_thread_percentage <SERVER_PID>\n' \
  | nvidia-cuda-mps-control
```

For the measured run, the live `VLLM::EngineCore` client was connected to MPS
server 91 and `get_active_thread_percentage 91` returned `50.0`.

`tegrastats` continued to report approximately 98–99% `GR3D_FREQ`. That is not
evidence that the process escaped the limit: it means the allocated GPU
partition was busy. The MPS controller is the source of truth for the active
thread percentage.

## Benchmark method

The benchmark used nine WAV inputs covering 1 through approximately 9 seconds.
They include both `non_tool` speech and exact structured car commands. Each
duration was tested with three acoustically equivalent but byte-unique audio
files. A single 16-bit PCM sample near the end was changed by only one to three
integer units, preventing vLLM from recognizing the request as an identical
audio replay without audibly changing it.

For each sequential request, the benchmark sampled these vLLM Prometheus
metrics before and after the HTTP call:

- `vllm:time_to_first_token_seconds`
- `vllm:e2e_request_latency_seconds`
- `vllm:prefix_cache_hits_total`
- `vllm:prefix_cache_queries_total`

The client separately measured wall time around the full HTTP request. Output
tool names and arguments were compared with the committed expected outputs.
`tegrastats` sampled the Jetson every 100 ms during the run.

This byte-unique method matters because repeated identical audio benefits from
vLLM caching and makes TTFT unrealistically low for microphone traffic. The
reported table should be treated as a new-utterance, one-client estimate.

## Utilization

| Metric while active | 100% compute | 50% MPS compute |
| --- | ---: | ---: |
| Decode throughput | 78.3 tokens/s | 69.4 tokens/s |
| GPU busy indication | 98.3% mean | 97.8% mean |
| Container memory | 25.1 GiB | 24.8 GiB |
| GPU temperature | 52.5°C mean, 54.1°C max | 51.4°C mean, 52.6°C max |
| GPU/SoC rail power | 25.5 W mean, 27.7 W max | 23.5 W mean, 27.7 W max |
| Aggregate CPU use | 8.8% mean | 8.6% mean |

Thermals were safe in both runs. Compute sharing reduced mean GPU/SoC rail
power by approximately 2 W.

## Multi-model deployment implications

The 50% MPS result isolates SM compute sharing; it is not a full simulation of
a second running model. A real second model also consumes unified memory,
memory bandwidth, KV-cache capacity and CPU time. Those effects may increase
STCC latency more than this test.

Memory is the immediate constraint. At `--gpu-memory-utilization 0.80`, the
STCC container uses approximately 25 GiB on a 32 GB Jetson, while system memory
during inference is approximately 28.5 of 30.6 GiB. Another substantial vLLM
model cannot safely coexist with that reservation.

Before a two-model test:

1. reduce both servers' `--gpu-memory-utilization`, initially to approximately
   `0.35–0.40` each;
2. reduce context length and KV-cache reservation to the real product need;
3. start both models and confirm there is no swap or OOM pressure;
4. warm both services;
5. generate concurrent traffic for the second model while measuring STCC;
6. report p50, p95 and p99 queueing latency, not only isolated request time;
7. verify output accuracy again under sustained thermal and memory load.

For production isolation, keep each model as an MPS client with an explicit
active-thread allocation. MPS controls compute share, while vLLM memory limits
control each engine's KV-cache reservation; both are required.

## Operational commands

The tested service runs in tmux under the name `stcc-vllm022`:

```bash
ssh jetson -t 'tmux attach -t stcc-vllm022'
```

Detach without stopping it using `Ctrl-b`, then `d`. Follow the server log:

```bash
ssh jetson 'tail -f /home/trinq3/logs/stcc_vllm022.log'
```

Stop only the named container:

```bash
ssh jetson 'docker stop --timeout 30 stcc-vllm022'
```

At the end of the benchmark, the temporary MPS-limited container and MPS
daemon were removed. The normal full-compute `stcc-vllm022` service was
restored, health-checked and warmed with a valid `set_fog_lights` request.
