"""Run with ``python -m demo.backend`` from the repository root."""

import uvicorn

from .settings import get_settings

settings = get_settings()
uvicorn.run("demo.backend.app:app", host=settings.host, port=settings.port)
