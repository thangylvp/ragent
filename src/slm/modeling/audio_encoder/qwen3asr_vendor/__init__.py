"""Vendored Qwen3-ASR Transformers backend.

Copied verbatim from `qwen-asr==0.0.6`
(`qwen_asr/core/transformers_backend/{configuration,modeling,processing}_qwen3_asr.py`,
Apache-2.0, Alibaba Qwen). Keeping these three files local makes the native
speech-to-action wrapper and exported Hugging Face checkpoint independent of
the larger `qwen-asr` application dependency tree. The files import only from
`transformers` and each other.

The package exports the audio-tower types used by training; `CommandASR`
imports the full conditional-generation class directly from the modeling file.
"""
from .configuration_qwen3_asr import (
    Qwen3ASRConfig,
    Qwen3ASRAudioEncoderConfig,
)
from .modeling_qwen3_asr import (
    Qwen3ASRAudioEncoder,
    _get_feat_extract_output_lengths,
)

__all__ = [
    "Qwen3ASRConfig",
    "Qwen3ASRAudioEncoderConfig",
    "Qwen3ASRAudioEncoder",
    "_get_feat_extract_output_lengths",
]
