"""Qwen3-ASR direct speech-to-tool model integration.

The lightweight tool-call codec can be imported without model dependencies.
The native model registers itself only when torch/transformers are available.
"""

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
except ImportError:
    # VAD-only and contract-test environments intentionally omit the heavy SLM
    # dependencies. A real model build still fails clearly when requested.
    pass
else:
    # Once the declared heavy dependencies exist, surface internal import
    # errors instead of silently leaving the registry empty.
    from . import command_asr  # noqa: F401

from .build import META_ARCH_REGISTRY, build_model

__all__ = ["META_ARCH_REGISTRY", "build_model"]
