"""Weightpack plugin for vLLM: composition loading, expert pool and routers.

vLLM discovers this package through the ``vllm.general_plugins`` entry point
declared in ``pyproject.toml`` and calls :func:`register` at start-up.
"""

from __future__ import annotations

import logging

__version__ = "0.1.0"

logger = logging.getLogger(__name__)


def register() -> None:
    """Entry point called by vLLM. Hooks are added here as they land."""
    logger.info("weightpack-vllm %s loaded (no hooks registered yet)", __version__)
