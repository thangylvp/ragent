# Jetson voice-agent latency report — 2026-08-14

## Executive result

The primary result is the local `tool_call` path. It is controlled, repeatable,
and is the latency that matters when the SLM directly parses a user request for
the execute layer:

```text
speech -> enhancement -> VAD -> STCC SLM -> execute layer
       -> cached Vietnamese response -> playback
```

All reported latency starts after VAD has finalized the input audio. The VAD
endpoint measurement was removed because the benchmark supplied synthetic
zero-valued trailing silence; it is not representative of a naturally noisy
robot environment. From the VAD-ready audio, p100 needed 202 ms to the first
SLM token, 534 ms to the completed tool call, and 568 ms to the first
cached-audio byte. At p50, those values were 219 ms, 595 ms, and 632 ms. The
first-audio penalty at the 50% CUDA MPS active-thread cap was only 64 ms, so
50% is the recommended scheduling cap when sharing the GPU. At 30%, decode becomes substantially slower and first audio rises
to 1,116 ms in aggregate.

The `non_tool` path is included as a secondary end-to-end observation:

```text
speech -> enhancement -> VAD -> STCC SLM -> Gemini
       -> uncached OmniVoice on laptop -> playback
```

Gemini, Wi-Fi, and laptop TTS latency are not controlled by the Jetson CUDA MPS
cap. Therefore, the non-tool table must not be used to rank the full-GPU
baseline and the 70%, 50%, and 30% MPS configurations. Its useful result is the separation between
the fast local SLM decision and the slower external response path.

## Latency clock used in this report

Every reported milestone starts when VAD has finalized the waveform and the SLM
can be dispatched:

```text
VAD-ready audio -> SLM -> execute or cloud -> response audio
```

This is the same origin as the website's `Audio -> ...` diagnostics. It excludes
the user's speaking duration and VAD endpointing. This report stops at the first
response-audio byte. `Audio -> last audio` is excluded because it was derived
from WAV duration rather than measured with acoustic loopback.

The benchmark streamed each source with synthetic clean silence before and
after it. Raw `speech_end_to_*` diagnostic fields remain in the artifacts, but
they are deliberately excluded from the report. A separate VAD study should use
continuous microphone recordings, natural post-speech room noise, and manually
annotated speech-end timestamps.

## How to read the two tables

Each audio has `P50`, `P90`, and `Max` rows. P50 is the median, P90 uses linear
interpolation over the measured repetitions, and Max is the largest observed
value. Every latency value starts at VAD finalization—the same `audio ->`
origin shown by the web demo.

Tool-call cells use this order:

```text
F / G / T / S / A1
```

Non-tool cells use this order:

```text
F / G / T / S / C / A1
```

- `F`: first meaningful SLM token.
- `G`: SLM generation time from the first to the last meaningful token.
- `T`: SLM decode throughput in tokens/s, calculated as
  `(output tokens - 1) / G`; it excludes TTFT.
- `S`: last meaningful SLM token; for `tool_call`, the structured call is now
  available for parsing and execution.
- `C`: complete Gemini response. Gemini was called with non-streaming
  `generateContent`, so this is response completion rather than a separately
  observed streamed token. It is not applicable to `tool_call`.
- `A1`: first byte from the response-WAV endpoint. Fixed execute-layer replies
  use the Jetson cache. Dynamic non-tool replies are available only after the
  laptop OmniVoice worker has synthesized and returned the complete WAV.

All latency fields are milliseconds; `T` is tokens/s. `Audio -> last audio` is
not reported. For the 30% non-tool condition, P90 is based on only two runs and
is descriptive rather than a reliable tail-latency estimate.

## Tool-call latency — primary benchmark

The 256 measured turns below contain eight warm repetitions for every audio and
MPS active-thread cap. One full corpus cycle per cap was used for warm-up and
excluded. The cell order is `F / G / T / S / A1`; this path does not call
Gemini.

| Audio | Statistic | Full GPU (MPS off) | 70% MPS cap | 50% MPS cap | 30% MPS cap |
| --- | --- | ---: | ---: | ---: | ---: |
| 1.39 s | P50 | 193 / 217 / 124.7 / 410 / 443 | 198 / 229 / 117.9 / 427 / 462 | 206 / 245 / 110.1 / 451 / 487 | 322 / 454 / 59.4 / 777 / 838 |
|  | P90 | 194 / 217 / 124.7 / 410 / 443 | 199 / 229 / 118.0 / 428 / 463 | 207 / 245 / 110.2 / 452 / 488 | 323 / 455 / 59.5 / 778 / 838 |
|  | Max | 194 / 217 / 124.8 / 410 / 444 | 199 / 229 / 118.0 / 428 / 463 | 207 / 245 / 110.2 / 452 / 488 | 323 / 455 / 59.5 / 778 / 839 |
| 2.00 s | P50 | 235 / 191 / 146.6 / 426 / 460 | 242 / 202 / 138.5 / 444 / 480 | 254 / 217 / 129.3 / 470 / 507 | 408 / 401 / 69.9 / 809 / 870 |
|  | P90 | 235 / 191 / 146.6 / 426 / 460 | 243 / 203 / 138.5 / 445 / 480 | 254 / 217 / 129.4 / 471 / 508 | 409 / 401 / 69.9 / 809 / 871 |
|  | Max | 236 / 191 / 146.7 / 427 / 460 | 243 / 203 / 138.5 / 445 / 480 | 254 / 217 / 129.4 / 471 / 508 | 409 / 401 / 69.9 / 809 / 871 |
| 3.00 s | P50 | 193 / 204 / 137.3 / 397 / 431 | 202 / 216 / 129.7 / 418 / 452 | 211 / 231 / 121.2 / 443 / 479 | 348 / 428 / 65.4 / 776 / 837 |
|  | P90 | 194 / 204 / 137.4 / 398 / 431 | 202 / 216 / 129.8 / 418 / 453 | 212 / 231 / 121.4 / 443 / 480 | 348 / 429 / 65.4 / 777 / 838 |
|  | Max | 194 / 205 / 137.4 / 398 / 431 | 203 / 216 / 129.8 / 419 / 453 | 213 / 232 / 121.5 / 443 / 480 | 349 / 429 / 65.4 / 777 / 838 |
| 4.00 s | P50 | 201 / 294 / 119.2 / 495 / 528 | 208 / 310 / 112.7 / 519 / 553 | 218 / 332 / 105.4 / 550 / 587 | 357 / 615 / 56.9 / 973 / 1,034 |
|  | P90 | 202 / 294 / 119.3 / 495 / 529 | 209 / 311 / 112.8 / 519 / 554 | 219 / 332 / 105.4 / 551 / 587 | 358 / 616 / 56.9 / 973 / 1,035 |
|  | Max | 202 / 294 / 119.3 / 495 / 529 | 210 / 311 / 113.0 / 520 / 555 | 219 / 332 / 105.5 / 551 / 587 | 358 / 616 / 56.9 / 974 / 1,035 |
| 5.00 s | P50 | 204 / 371 / 110.6 / 574 / 608 | 210 / 392 / 104.6 / 602 / 637 | 220 / 420 / 97.7 / 640 / 677 | 359 / 777 / 52.8 / 1,136 / 1,198 |
|  | P90 | 204 / 371 / 110.6 / 575 / 609 | 211 / 393 / 104.6 / 603 / 638 | 221 / 420 / 97.7 / 640 / 678 | 360 / 777 / 52.8 / 1,137 / 1,198 |
|  | Max | 204 / 371 / 110.7 / 575 / 610 | 211 / 393 / 104.6 / 603 / 638 | 221 / 420 / 97.7 / 640 / 678 | 360 / 777 / 52.8 / 1,138 / 1,198 |
| 6.00 s | P50 | 206 / 371 / 110.5 / 578 / 612 | 214 / 392 / 104.5 / 606 / 641 | 224 / 421 / 97.5 / 644 / 682 | 363 / 777 / 52.8 / 1,140 / 1,202 |
|  | P90 | 207 / 371 / 110.5 / 578 / 613 | 214 / 392 / 104.5 / 607 / 642 | 224 / 421 / 97.5 / 645 / 682 | 364 / 777 / 52.8 / 1,141 / 1,203 |
|  | Max | 208 / 372 / 110.5 / 579 / 613 | 214 / 393 / 104.6 / 607 / 642 | 224 / 421 / 97.5 / 645 / 683 | 364 / 778 / 52.8 / 1,141 / 1,203 |
| 6.75 s | P50 | 211 / 537 / 96.8 / 748 / 782 | 224 / 569 / 91.4 / 793 / 828 | 234 / 608 / 85.5 / 842 / 879 | 364 / 1,124 / 46.3 / 1,487 / 1,549 |
|  | P90 | 212 / 538 / 96.8 / 749 / 783 | 224 / 569 / 91.5 / 793 / 828 | 234 / 608 / 85.5 / 842 / 879 | 365 / 1,124 / 46.3 / 1,488 / 1,549 |
|  | Max | 212 / 538 / 96.8 / 750 / 784 | 225 / 569 / 91.5 / 794 / 828 | 234 / 608 / 85.5 / 843 / 880 | 365 / 1,124 / 46.3 / 1,489 / 1,550 |
| 7.44 s | P50 | 183 / 564 / 95.8 / 747 / 781 | 189 / 596 / 90.6 / 785 / 820 | 197 / 639 / 84.5 / 836 / 873 | 311 / 1,179 / 45.8 / 1,490 / 1,551 |
|  | P90 | 184 / 564 / 95.9 / 748 / 782 | 189 / 596 / 90.6 / 785 / 820 | 199 / 639 / 84.6 / 837 / 874 | 312 / 1,179 / 45.8 / 1,491 / 1,552 |
|  | Max | 184 / 564 / 95.9 / 748 / 782 | 189 / 596 / 90.6 / 786 / 821 | 200 / 639 / 84.7 / 837 / 875 | 313 / 1,179 / 45.8 / 1,491 / 1,553 |

The key tool-calling observations are:

- From 100% to 50%, aggregate post-VAD first-token latency increased by only
  17 ms and completed-tool-call latency increased by 61 ms. A 50% MPS cap
  remains suitable for an interactive demo.
- Median SLM decode throughput after the first token was 114.9, 108.6, 101.5,
  and 54.8 tokens/s on the full-GPU baseline and at the 70%, 50%, and 30%
  MPS active-thread caps, respectively.
  This uses `(output tokens - 1) / first-to-last-token time`; it excludes TTFT.
- At 30%, long structured calls are the problem: the two longest cases need
  approximately 1.49 seconds after VAD to finish the SLM output and 1.55 seconds
  after VAD before cached audio delivery begins.
- The execute layer takes approximately 0.45 ms, cached-audio lookup takes
  approximately 0.03 ms, and the local response endpoint begins delivery about
  4.5–4.7 ms after the response is ready. The fixed-response cache is therefore
  working as intended.
- The response length, rather than input-audio duration alone, controls the gap
  from first to last SLM token. The 6.75 and 7.44-second samples generate longer
  JSON tool calls.

All 256 rows followed a schema-valid tool path that the simulator accepted, but
that does not mean all calls matched the intended semantics. Across the eight
unique recordings, tool-name accuracy was 7/8 (87.5%) and exact-call accuracy
was 3/8 (37.5%). The wrong predictions were stable across allocations and
byte-unique audio variants, indicating deterministic model errors. Improve the
training data/checkpoint before using this corpus as an accuracy claim.

## Non-tool latency — secondary observation

The real-speech prefixes cover 1.0, 3.0, 5.0, 7.0, and 7.5 seconds. All
included rows routed `non_tool -> Gemini -> dynamic OmniVoice`, had no harness
errors, and forced fresh TTS synthesis rather than a cache hit. p100 and p70
have six runs per audio. p50 has 4/4/3/3/4 runs by increasing length. p30 has
two runs for every audio length. Its per-length P90 values are therefore
descriptive and must not be treated as stable tail estimates. The cell order
is `F / G / T / S / C / A1`.

| Audio | Statistic | Full GPU (MPS off) | 70% MPS cap | 50% MPS cap | 30% MPS cap |
| --- | --- | ---: | ---: | ---: | ---: |
| 1.0 s | P50 | 192 / 178 / 134.9 / 371 / 2,123 / 4,151 | 199 / 189 / 127.2 / 388 / 2,255 / 4,453 | 196 / 201 / 119.6 / 396 / 2,220 / 4,194 | 324 / 374 / 64.1 / 699 / 2,714 / 4,905 |
|  | P90 | 197 / 180 / 135.2 / 376 / 2,174 / 4,422 | 200 / 189 / 127.3 / 388 / 2,751 / 4,890 | 207 / 201 / 119.6 / 408 / 2,285 / 4,431 | 325 / 374 / 64.1 / 699 / 2,749 / 4,922 |
|  | Max | 200 / 181 / 135.2 / 381 / 2,191 / 4,534 | 200 / 189 / 127.4 / 389 / 3,102 / 5,025 | 208 / 201 / 119.6 / 408 / 2,303 / 4,503 | 325 / 375 / 64.1 / 700 / 2,758 / 4,926 |
| 3.0 s | P50 | 198 / 319 / 109.9 / 518 / 2,942 / 5,035 | 205 / 338 / 103.6 / 543 / 3,426 / 5,724 | 214 / 359 / 97.5 / 573 / 3,128 / 5,257 | 333 / 670 / 52.3 / 1,003 / 3,645 / 5,776 |
|  | P90 | 200 / 321 / 109.9 / 518 / 3,326 / 5,479 | 205 / 338 / 103.7 / 543 / 3,834 / 5,941 | 214 / 359 / 97.5 / 573 / 3,528 / 5,732 | 334 / 670 / 52.3 / 1,003 / 3,764 / 5,907 |
|  | Max | 200 / 321 / 110.0 / 518 / 3,464 / 5,663 | 206 / 338 / 103.7 / 543 / 3,874 / 6,074 | 214 / 359 / 97.5 / 574 / 3,678 / 5,889 | 334 / 670 / 52.3 / 1,003 / 3,794 / 5,940 |
| 5.0 s | P50 | 205 / 462 / 99.5 / 667 / 3,285 / 5,473 | 220 / 487 / 94.4 / 707 / 3,733 / 6,055 | 194 / 518 / 88.8 / 711 / 3,560 / 5,960 | 360 / 966 / 47.6 / 1,326 / 3,670 / 6,123 |
|  | P90 | 209 / 467 / 99.9 / 670 / 3,615 / 5,978 | 220 / 487 / 94.4 / 707 / 3,973 / 6,289 | 223 / 518 / 88.8 / 741 / 3,882 / 6,304 | 361 / 966 / 47.6 / 1,326 / 3,777 / 6,145 |
|  | Max | 209 / 473 / 100.1 / 671 / 3,752 / 6,219 | 220 / 488 / 94.4 / 708 / 3,976 / 6,320 | 230 / 518 / 88.8 / 748 / 3,963 / 6,390 | 361 / 966 / 47.7 / 1,326 / 3,804 / 6,150 |
| 7.0 s | P50 | 210 / 577 / 95.2 / 787 / 3,197 / 5,631 | 226 / 610 / 90.1 / 836 / 3,639 / 6,019 | 197 / 649 / 84.8 / 846 / 3,292 / 6,014 | 366 / 1,205 / 45.7 / 1,571 / 4,053 / 6,370 |
|  | P90 | 210 / 579 / 95.5 / 789 / 4,401 / 6,782 | 226 / 610 / 90.1 / 836 / 4,010 / 6,513 | 228 / 649 / 84.8 / 876 / 3,688 / 6,154 | 366 / 1,205 / 45.7 / 1,571 / 4,305 / 6,567 |
|  | Max | 210 / 579 / 95.5 / 789 / 4,733 / 7,096 | 227 / 610 / 90.1 / 837 / 4,059 / 6,545 | 235 / 649 / 84.8 / 884 / 3,787 / 6,189 | 366 / 1,205 / 45.7 / 1,571 / 4,368 / 6,616 |
| 7.5 s | P50 | 212 / 615 / 94.2 / 827 / 2,973 / 5,260 | 227 / 651 / 89.1 / 878 / 3,264 / 5,725 | 215 / 692 / 83.8 / 906 / 3,009 / 5,412 | 370 / 1,288 / 45.0 / 1,659 / 3,772 / 6,131 |
|  | P90 | 214 / 617 / 94.4 / 829 / 3,538 / 5,932 | 228 / 651 / 89.1 / 879 / 3,565 / 6,014 | 236 / 692 / 83.9 / 928 / 3,148 / 5,585 | 370 / 1,288 / 45.0 / 1,659 / 3,980 / 6,338 |
|  | Max | 215 / 617 / 94.4 / 830 / 3,584 / 5,948 | 228 / 651 / 89.1 / 880 / 3,654 / 6,078 | 237 / 692 / 83.9 / 928 / 3,163 / 5,615 | 370 / 1,288 / 45.0 / 1,659 / 4,032 / 6,390 |

The non-tool result supports three conclusions:

- The SLM routing decision is still relatively fast. At 100%, post-VAD
  first-token time is 192–212 ms and SLM completion is 371–827 ms.
- Median non-tool SLM decode throughput was 99.5, 94.4, 88.8, and 47.6
  tokens/s on the full-GPU baseline and at the 70%, 50%, and 30% MPS
  active-thread caps. These values should be
  compared within the non-tool workload because its output lengths differ from
  the controlled tool-call corpus.
- External generation dominates. In the complete p100 set, isolated Gemini
  latency had a 2,135 ms median and uncached OmniVoice synthesis had a 2,258 ms
  median. The complete dynamic path reached its first audio byte at a 5,242 ms
  post-VAD aggregate median.
- Changing the Jetson MPS active-thread cap cannot control Gemini, Wi-Fi, or laptop TTS.
  This explains non-monotonic cells such as p50 occasionally completing before
  p70. Only `F` and `S` should be read as evidence about the Jetson SLM.

The OmniVoice worker produced complete WAV files faster than real time: its p100
median synthesis RTF was 0.358. The remaining high post-VAD completion time is
mostly because the current path is sequential and non-streaming: wait for the SLM, then the
complete Gemini response, then the complete TTS WAV, and finally play it. The
largest future non-tool improvement would come from streaming cloud output into
streaming TTS/audio playback, not from a small Jetson MPS-cap change.

## Method and reliability

For both paths, the real FastAPI WebSocket handler received mono 16 kHz PCM16
audio in 20 ms packets at microphone pace. FastEnhancer-S and OmniVAD ran online
while the user audio arrived. The runner prepended 500 ms and appended 1,000 ms
of zero-valued PCM to each source. Because that artificial suffix is not a
representative robot environment, all headline latency begins only after VAD
finalization.

The tool benchmark used eight recordings from 1.39 to 7.44 seconds. Every MPS
cap received one excluded warm-up cycle and eight measured cycles. Nine
acoustically equivalent variants changed one near-final PCM sample by only 1–9
integer units, preventing identical-audio replay caching. Sample order rotated
across cycles. Temperature was zero and output was capped at 128 tokens.

The non-tool benchmark used byte-unique variants and uncached TTS. p100 and p70
received six measured repetitions per source. Data collection was intentionally
stopped early at p50 after the scope was narrowed to tool-call performance. The
initial compact p30 follow-up covered 1.0, 5.0, and 7.5 seconds twice; a verified
30% MPS supplemental run added two repetitions each for 3.0 and 7.0 seconds.

[CUDA MPS active-thread percentage](https://docs.nvidia.com/deploy/mps/appendix-tools-and-interface-reference.html#cuda-mps-active-thread-percentage)
supplied the 70%, 50%, and 30% scheduling caps. NVIDIA defines it as the
portion of available GPU threads usable by the client contexts. The p100
condition was a direct full-GPU baseline with MPS disabled. Each limited
run captured MPS control output listing the live vLLM engine as a client at the
requested percentage. This places an execution-resource ceiling on vLLM; it is
not the instantaneous utilization reported by `tegrastats`, a GPU-memory
limit, or a clock/power cap. It also does not recreate the memory, CPU,
thermal, or bandwidth contention of a particular second deployed model.

During collection, the Jetson Docker daemon restarted after a Docker/NVIDIA
toolkit update. The interrupted trial was excluded and preserved separately.
The refreshed CDI spec exposed GPU devices but omitted the real Jetson
`libcuda`; the launcher now uses CDI and read-only mounts the host Orin driver
library. CUDA tensor execution and MPS control were validated before final
measurements resumed. A cloud-fallback trial also exposed and fixed a demo bug:
the audio endpoint now serves both the dynamic TTS cache and the read-only
static response manifest.

## Tested configuration

The edge device was an NVIDIA Jetson AGX Orin with 32 GB unified memory, Ubuntu
24.04 ARM64, Linux 6.8.12 Tegra, JetPack 7.2 / L4T R39.2, and MAXN power mode.
The standalone BF16 STCC checkpoint was served from
`/home/trinq3/models/stcc` by vLLM 0.22.0 in
`stcc-vllm:0.22.0-audio`, with an 8,192-token context and
`--gpu-memory-utilization 0.70`.

The edge preprocessing path used OmniVAD 0.2.13 with the FireRed Stream-VAD
ncnn model on CPU and FastEnhancer-S causal DNS ONNX with ONNX Runtime 1.23.2.
Tool responses came from the read-only 135-clip Vietnamese OmniVoice manifest.
Dynamic responses used Gemini 3.6 Flash and the laptop-hosted
`splendor1811/omnivoice-vietnamese` worker with the single configured northern
female reference voice, 32 steps, and speed 0.8. Only one user request was in
flight at a time.

## Scope

- `A1` is first WAV byte, not the browser's first audible sample.
- `Audio -> last audio` is excluded. A physical E2E validation should capture
  browser playback start and acoustic response end.
- VAD endpoint latency is out of scope. The benchmark used synthetic clean
  trailing silence, so its raw endpoint diagnostics must not be presented as
  an in-the-wild VAD result.
- Inputs are clean real speech, not a representative noisy robot microphone
  dataset.
- The tool corpus was selected for executable-path latency and is not an
  unbiased accuracy set.
- Gemini/network values describe this test window only. They should not be used
  as an SLA or attributed to the Jetson MPS cap.

## Raw artifacts and reproduction

Controlled tool-call artifacts are under:

```text
outputs/benchmarks/jetson_full_system_20260814/
```

Non-tool artifacts are under:

```text
outputs/benchmarks/jetson_cloud_path_20260814/
```

The supplemental 30% runs for 3.0 and 7.0 seconds are preserved separately
under `p30_missing_3s_7s/` and are combined with `p30/` during analysis.

Both locations contain raw per-turn JSONL, summaries, benchmark logs,
`tegrastats`, MPS verification, server configuration, and analysis JSON. The
Jetson originals remain under `/home/trinq3/benchmarks/ragent_full/` and
`/home/trinq3/benchmarks/ragent_cloud/`.

Relevant source files are:

- `scripts/benchmark_jetson_full_system.py`: controlled tool-path recorder.
- `scripts/analyze_jetson_full_system.py`: tool-path distributions and resource
  analysis.
- `scripts/benchmark_jetson_cloud_path.py`: non-tool/cloud/TTS recorder.
- `scripts/analyze_jetson_cloud_path.py`: non-tool aggregation.
- `scripts/jetson/run_vllm_condition.sh`: CDI-aware Jetson launcher.
- `scripts/jetson/start_vllm_condition.sh`: vLLM and CUDA MPS configuration.
- `configs/benchmarks/jetson_tool_latency_corpus.json`: tool-call corpus.
- `configs/benchmarks/jetson_cloud_latency_corpus.json`: non-tool corpus.

## Recommendation

Use the direct full-GPU configuration with MPS disabled for the recorded demo
when the Jetson is dedicated. Use a 50% MPS cap when another model needs GPU time: it retained near-p100 tool-calling latency in
this benchmark. Avoid 30% for interaction-focused use unless approximately
1.12 seconds from VAD-ready audio to cached-audio delivery is acceptable as an
aggregate median. The longest calls need approximately 1.55 seconds to begin
cached-audio delivery.

Benchmark endpointing separately with continuous noisy microphone recordings
and manual speech-end annotations. Keep cached Vietnamese audio for
deterministic execute-layer responses. For non-tool responses, prioritize a
streaming Gemini -> TTS -> playback design. Finally, fix the deterministic
parsing errors before presenting the checkpoint as semantically reliable.

## Summary

- The Jetson tool-call path is interactive; a 50% MPS cap kept median first
  cached audio at 632 ms after VAD, versus 568 ms with the full GPU.
- Dynamic non-tool latency is dominated by cloud generation and TTS, so
  streaming those stages matters more than small edge-GPU tuning.
- The whole pipeline, including the SLM, can run in the cloud. Deploy the SLM
  on edge only when latency, connectivity, privacy or autonomy justify the
  extra deployment and maintenance effort.
