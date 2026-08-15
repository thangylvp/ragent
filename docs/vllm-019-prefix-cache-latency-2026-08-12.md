# vLLM 0.19 STCC latency and prefix-cache notes — 2026-08-12

This note explains the latency terms used for the STCC serving benchmark and
records the local A/B test of vLLM 0.19 automatic prefix caching. The test used
the existing car checkpoint as a proxy for the future robot SLM because both
use the same direct audio-to-structured-call architecture.

## Short conclusion

The model receives approximately 6,400 prompt tokens per request because the
33 tool definitions are included in the chat prompt. Almost all of that text is
identical between requests. Enabling vLLM prefix caching allowed 6,368 prompt
tokens to be reused and reduced steady-state server time to first token from
402.6 ms to 43.6 ms. It did not materially change decode speed after the first
token.

For a single local client on the RTX 5070 Laptop GPU:

| Metric | Prefix cache off | Prefix cache on | Change |
| --- | ---: | ---: | ---: |
| Mean server time to first token | 402.6 ms | 43.6 ms | 89.2% lower, 9.24x faster |
| Mean server time to last token | 623.9 ms | 269.7 ms | 56.8% lower, 2.31x faster |
| Mean client request time | 626.7 ms | 272.6 ms | 56.5% lower, 2.30x faster |
| Mean first-to-last decode time | 221.3 ms | 226.1 ms | effectively unchanged |
| Cached prompt tokens per measured request | 0 | 6,368 | 99% of the prompt |
| Correct route/tool name | 5/5 | 5/5 | unchanged |

Prefix caching should therefore be enabled for this deployment. It attacks the
largest current model-side latency component: repeatedly prefilling the same
tool catalog.

## What the latency names mean

### Time to first token

Time to first token, abbreviated **TTFT**, is the interval between vLLM
accepting a generation request and the model producing its first output token.
It includes work that must happen before generation can begin, principally:

1. request queueing, if another request is already running;
2. multimodal/audio preprocessing and audio-encoder work inside the serving
   path;
3. prompt prefill, including attention over the system prompt and tool schemas;
4. selection of the first generated token.

The value in this benchmark comes from vLLM's
`vllm:time_to_first_token_seconds` metric. In vLLM 0.19 it is calculated from
the engine request arrival timestamp to the first generated-token iteration.
It is a server metric, so it does not include the client's audio base64
encoding or all HTTP/frontend overhead.

TTFT matters because it measures how quickly the model begins answering. For a
tool parser, however, the first token is normally only the start of the tool
syntax; it is not yet safe to execute anything.

### Time to last token

**Time to last token** is the interval from the same request start until the
model has generated its complete output. In these tests it is vLLM's
`vllm:e2e_request_latency_seconds` measurement. It includes TTFT plus generation
of all remaining tokens:

```text
server time to last token = TTFT + first-to-last decode time
```

For the 2-second fog-light sample with prefix caching:

```text
request accepted       first token                       last token
       |-------------------|----------------------------------|
               43.6 ms                 227.5 ms
       |------------------------------------------------------|
                            271.1 ms
```

The complete parsed output was:

```json
{
  "name": "set_fog_lights",
  "arguments": {
    "state": "on",
    "position": "front"
  }
}
```

The execution layer must wait for the complete call and validate it. Therefore,
time to last token is the more important model-side latency for the present
non-incremental tool-execution design.

### Client request time

The client measurement starts immediately before the local HTTP POST and stops
after the complete JSON response has arrived. It adds request serialization,
the local HTTP stack, OpenAI-compatible response formatting and JSON parsing to
the server's time-to-last-token value.

The measured difference was only about 2.9 ms on average because the client and
server were on the same machine. This difference will be larger over a real
network.

### Latency after the user stops speaking

None of the three request metrics above includes the time spent waiting for VAD
to decide that the utterance has ended. The user-visible post-speech latency is
approximately:

```text
speech-end-to-result
  = VAD endpoint delay
  + audio finalization and dispatch
  + client/server request-to-last-token time
  + output parsing and validation
```

The earlier OmniVAD pipeline test measured a mean speech-end-to-VAD delay of
294.8 ms. Combining that with the 272.6 ms prefix-cached local request time gives
a first-order estimate of approximately 567 ms from speech end to a complete
tool call:

```text
294.8 ms + 272.6 ms = 567.4 ms
```

This is an estimate, not a directly measured optimized end-to-end result. The
VAD and vLLM numbers came from separate benchmark runs, and production network,
harness validation and execution-layer work still need to be added. A direct
VAD-to-vLLM benchmark should be run on Jetson before treating 567 ms as a product
latency claim.

## Meaning of “cache off” and “cache on”

The model, weights, audio, tools and generation settings were held constant.
Only automatic prefix caching changed:

- **Last token off**: complete server response latency with
  `--no-enable-prefix-caching`.
- **Last token on**: complete server response latency with
  `--enable-prefix-caching` after a warm-up request populated the cache.

The cache does not store an answer and does not skip audio understanding. It
stores the attention key/value state for full token blocks in an identical
prompt prefix. Each new request still processes its different audio and
generates a new answer.

The prompt layout places the stable system/tool text before the audio tokens:

```text
[system instructions + 33 tool schemas] [audio tokens] [assistant start]
|------------ identical 6,368-token cached prefix -----------|
```

vLLM used a block size of 16 tokens. The exact common prefix was approximately
6,370 tokens, so 6,368 tokens formed complete reusable blocks. The remaining
prompt tail, audio features and assistant output were computed separately for
every request.

## Test environment

| Item | Value |
| --- | --- |
| Date | 2026-08-12 |
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU |
| Available VRAM | 8,151 MiB |
| vLLM | 0.19.0 |
| PyTorch | 2.10.0+cu128 |
| Model dtype | BF16 |
| Checkpoint | `../stc/outputs/models/route_v1_best_step1250_hf` |
| Model architecture | `Qwen3ASRForConditionalGeneration` |
| Tool catalog | 33 OpenAI-format tools |
| Client load | One request at a time; no concurrency |
| Context limit | 8,192 tokens |
| Explicit KV-cache reservation | 2 GiB |
| vLLM-reported KV capacity | 18,720 tokens |
| vLLM-reported maximum 8,192-token concurrency | 2.29x |
| GPU process memory after server startup | approximately 4,164 MiB |
| Total GPU memory after cache test | approximately 4,367 MiB |

The 2 GiB KV reservation is capacity reserved when the server starts; it is not
2 GiB of new memory consumed for every request. Prefix-cache entries occupy
available KV-cache blocks and can be evicted when vLLM needs those blocks for
active requests.

## Method

Two fresh vLLM servers were launched sequentially, one with prefix caching off
and one with it on. Both used the same compile-cache directory. Each server
received one unmeasured warm-up request followed by the same five measured WAV
files in the same order. Requests were sequential, temperature was zero, the
tool list and tool order were identical, and output was limited to 128 tokens.

For each request, the client sampled vLLM Prometheus counters immediately
before and after the request. The difference in histogram sums and counts gave
the individual server TTFT and server end-to-end time. The complete response
was also timed at the client and its parsed tool name was compared with the
expected route.

The five audio inputs covered durations from 1 to 5 seconds and generated
between 25 and 42 tokens. Different audio was intentional: it proves that the
stable tool prefix can be reused without incorrectly caching the audio-specific
result.

## Per-file results

| Audio | Prompt tokens | Output tokens | TTFT off | TTFT on | Last token off | Last token on | Cached tokens | Expected/actual route |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 s non-tool | 6,394 | 25 | 444.4 ms | 39.2 ms | 614.5 ms | 240.1 ms | 6,368 | `non_tool` / `non_tool` |
| 2 s fog lights | 6,407 | 29 | 372.5 ms | 43.6 ms | 588.0 ms | 271.1 ms | 6,368 | `set_fog_lights` / `set_fog_lights` |
| 3 s seat heating | 6,420 | 28 | 417.0 ms | 48.1 ms | 603.4 ms | 228.9 ms | 6,368 | `set_seat_heating` / `set_seat_heating` |
| 4 s rear fan direction | 6,433 | 31 | 395.6 ms | 43.4 ms | 631.5 ms | 246.0 ms | 6,368 | `set_fan_direction` / `set_fan_direction` |
| 5 s ambient light | 6,446 | 42 | 383.4 ms | 43.6 ms | 681.9 ms | 362.3 ms | 6,368 | `set_ambient_light` / `set_ambient_light` |
| **Mean** | **6,420** | **31** | **402.6 ms** | **43.6 ms** | **623.9 ms** | **269.7 ms** | **6,368** | **5/5 correct** |

The prompt grows only slightly as audio duration increases because the large
tool catalog dominates its size. Last-token time does not increase monotonically
with audio duration because output length also varies. The 5-second ambient
light example generated 42 tokens, while the 1-second non-tool example generated
25 tokens.

## Where the improvement comes from

The A/B difference is almost entirely before the first token:

| Component | Cache off | Cache on |
| --- | ---: | ---: |
| Prefill through first token | 402.6 ms | 43.6 ms |
| Remaining output generation | 221.3 ms | 226.1 ms |
| Complete server generation | 623.9 ms | 269.7 ms |

The approximately 359 ms TTFT saving closely matches the approximately 354 ms
last-token saving. Output decoding remained about 226 ms, or roughly 133 output
tokens/second over this small sample after excluding the first token. Prefix
caching cannot improve that phase because the answer is new for every request.

Future work to reduce the remaining 270 ms model-side latency should therefore
focus on shorter serialized tool calls, early validated structured decoding,
model/runtime quantization validated for accuracy, or a smaller task-specific
model. Reducing VAD endpoint delay is a separate optimization and may provide a
larger user-visible benefit than further decode tuning.

## Cold start and warm-up

The first request in the initial server process took 22.7 seconds. vLLM was
compiling a previously unseen multimodal input-shape graph. That request was not
included in the steady-state averages. Once generated, the compilation artifacts
were stored in the configured vLLM/Triton cache and later server startup reused
them.

Production must not send the first real user request directly to a newly
started process. Readiness should have two phases:

1. wait until the HTTP health/model endpoint reports ready;
2. send a representative audio request with the full production tool list and
   wait for it to complete.

The warm-up request also populates the stable prefix cache. Warm-up must be
repeated after restarting the engine, changing the model, changing the system
prompt or changing the tool catalog.

## Recommended vLLM 0.19 command

The following configuration was successfully tested on the local GPU:

```bash
source /home/thangnv94/v/.venvs/stc-vllm019/bin/activate

vllm serve /home/thangnv94/v/stc/outputs/models/route_v1_best_step1250_hf \
  --served-model-name stcc \
  --model-impl vllm \
  --dtype bfloat16 \
  --generation-config auto \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --enable-prefix-caching \
  --prefix-caching-hash-algo sha256 \
  --enable-prompt-tokens-details \
  --performance-mode interactivity \
  --enable-chunked-prefill \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 1 \
  --kv-cache-memory-bytes 2G \
  --mm-processor-cache-gb 0.25 \
  --limit-mm-per-prompt '{"audio":1}'
```

`--performance-mode interactivity` prioritizes latency for this single-user
robot workload. `--max-num-seqs 1` is appropriate only while the robot serves
one active utterance at a time. A multi-user service needs a separate
throughput/concurrency benchmark before using this limit.

The 2 GiB KV-cache value worked on the 8 GiB laptop GPU. It must be retuned on
Jetson according to memory available after the model, CUDA context, audio
encoder and other robot processes are loaded. Do not blindly copy this value if
Jetson is using shared system memory.

## Keeping prefix caching effective

Prefix caching is exact, not semantic. A one-token difference creates a
different cache path. The serving client should therefore:

- keep the production system prompt byte-for-byte stable;
- keep the tool schemas and their order stable;
- put stable instructions and tools before audio and dynamic content;
- avoid inserting timestamps, request IDs, current robot state or user-specific
  text before the reusable prefix;
- represent dynamic robot state in the execution layer where possible;
- use deterministic JSON serialization for tool schemas;
- use the same chat template and `chat_template_kwargs` on every request;
- send a new warm-up request after intentionally changing the prefix.

Dynamic state that is genuinely required for parsing should be appended after
the stable prefix. If it is placed inside the system prompt before the tool
catalog, it can invalidate almost the entire cache for every turn.

Monitor the following vLLM metrics in deployment:

```text
vllm:time_to_first_token_seconds
vllm:e2e_request_latency_seconds
vllm:prefix_cache_queries_total
vllm:prefix_cache_hits_total
vllm:prompt_tokens_cached_total
vllm:request_queue_time_seconds
```

A falling cache-hit ratio normally means the prompt prefix changed, entries
were evicted under memory pressure, or requests use incompatible tool lists.

## Limitations

This is a targeted optimization experiment, not a complete deployment
benchmark:

- only five measured requests were run per condition;
- measurements were single-user and sequential;
- tests ran on an RTX 5070 Laptop GPU, not Jetson AGX Orin;
- the current car model and tool catalog were used as a proxy;
- the client requested a non-streaming JSON response;
- output correctness checked the selected tool name, with spot inspection of
  arguments, rather than a full held-out accuracy suite;
- no network hop, harness logic, execution validation or robot action time was
  included;
- the approximately 567 ms post-speech estimate combines separate VAD and
  serving experiments.

Before deployment, rerun this experiment on Jetson vLLM 0.19 with the final
robot tool catalog and representative audio, then measure p50, p95 and p99 over
hundreds of warm requests. Include cold starts, cache invalidation, sustained
memory pressure, interrupted speech, non-tool queries and concurrent robot
processes.
