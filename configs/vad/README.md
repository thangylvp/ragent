# VAD backends and benchmark settings

The initial component benchmark used activation threshold `0.5`, 80 ms minimum
speech, 200 ms minimum end silence and 50 ms onset padding for FireRed and
OmniVAD. Those historical settings remain part of the recorded benchmark and
must not be confused with the current interactive defaults.

The webtest now selects OmniVAD by default and uses the same tuned FireRed
Stream-VAD policy for both the ncnn and PyTorch runtimes:

| Setting | Frames | Time/value |
| --- | ---: | ---: |
| Activation threshold | - | `0.65` |
| Probability smoothing | 5 | 50 ms |
| Onset padding | 8 | 80 ms |
| Minimum speech | 15 | 150 ms |
| Maximum speech | 2,000 | 20 s |
| End silence | 30 | 300 ms |

The higher threshold and longer minimum speech reject short background noises.
The 300 ms end-silence window tolerates brief pauses while adding 100 ms to the
old endpoint policy. Silero uses its adapter defaults. WebRTC uses mode 2 and
the energy backend retains the repository's streaming baseline settings because
their detector semantics are not directly equivalent.

| Backend | Acoustic model | Runtime | Environment |
| --- | --- | --- | --- |
| `energy` | Adaptive RMS/noise floor baseline | Python stdlib | project/default |
| `firered` | FireRed Stream-VAD | PyTorch CPU | `mega-asr` |
| `omnivad` | Same FireRed Stream-VAD weights | ncnn CPU | `.venvs/vad` |
| `silero` | Silero VAD v6 | PyTorch CPU | `mega-asr` |
| `webrtc` | WebRTC GMM VAD, mode 2 | native extension CPU | `.venvs/vad` |

FireRed and OmniVAD are intentionally both retained: accuracy parity validates
the edge port, while load time, RTF and memory show the deployment trade-off.
They are not treated as independent acoustic architectures.

## Reproduce the local comparison

Install lightweight backends in an isolated environment:

```bash
uv venv .venvs/vad --python 3.12
uv pip install --python .venvs/vad/bin/python \
  omnivad==0.2.13 webrtcvad-wheels==2.0.14
```

Install the PyTorch adapters into an environment that already owns the desired
PyTorch build (this avoids a package manager selecting a different CUDA wheel):

```bash
python -m pip install fireredvad==0.0.2 silero-vad==6.2.1
hf download FireRedTeam/FireRedVAD \
  --local-dir outputs/vad/models/FireRedVAD
```

Run `scripts/evaluate_vad.py` once per backend with that backend's Python, then
combine the resulting JSON files:

```bash
python scripts/compare_vad.py \
  outputs/vad/benchmarks/{energy,webrtc,firered,omnivad,silero}.json \
  --output outputs/vad/benchmarks/README.md
```

Current robot files all contain user speech. The `vmo_2305_16k` folder names
describe recording backgrounds (crowd, park, piano, quiet and washing dishes),
not negative-only audio. Therefore the first report measures speech hit rate,
detected-audio ratio and runtime by acoustic condition. False activations,
endpoint delay and clipped onset require manually annotated continuous positive
and negative sessions and must not be inferred from isolated clip boundaries.
