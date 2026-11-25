"""Weightpack plugin for vLLM: token routing across a MIX's experts.

vLLM discovers this package through the ``vllm.general_plugins`` entry point
declared in ``pyproject.toml`` and calls :func:`register` in each of its
processes at start-up. It patches vLLM's LoRA path so an adapter that wpkd
writes as a MIX (an adapter directory with a ``weightpack_mix.json``) routes
each token across the MIX's experts; see routing.py and patches.py.
WEIGHTPACK_DISABLE=1 leaves vLLM untouched.
"""

from __future__ import annotations

import logging
import os

__version__ = "0.1.0"

# A child of vLLM's logger, so vLLM's logging prints it.
logger = logging.getLogger("vllm.weightpack")

_registered = False


def register() -> None:
    """Entry point called by vLLM."""
    global _registered
    if _registered or os.environ.get("WEIGHTPACK_DISABLE") == "1":
        return
    try:
        from . import patches

        patches.apply()
    except Exception:  # an unexpected vLLM: serve without mixes, loudly
        logger.exception("weightpack-vllm %s: token routing is off", __version__)
        return
    _registered = True
    logger.info("weightpack-vllm %s: token routing on", __version__)
