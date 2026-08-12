"""Optional VAD runtime adapters behind one evaluation contract."""

from .base import VadBackend, VadSegment, create_backend

__all__ = ["VadBackend", "VadSegment", "create_backend"]
