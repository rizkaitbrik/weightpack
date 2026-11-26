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

__version__ = "0.2.0"

# A child of vLLM's logger, so vLLM's logging prints it.
logger = logging.getLogger("vllm.weightpack")

_registered = False


def register() -> None:
    """Entry point called by vLLM."""
    global _registered
    if _registered or os.environ.get("WEIGHTPACK_DISABLE") == "1":
        return
    if os.environ.get("WEIGHTPACK_DELTA_SLOTS") == "1":
        # vLLM's compile cache does not know these patches: a graph cached
        # without them would serve every fine-tune as the base, and one
        # cached with them refers to another process's buffers. So an engine
        # with delta slots compiles afresh and caches nothing.
        os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
    try:
        from . import patches

        patches.apply()
    except Exception:  # an unexpected vLLM: serve without the plugin, loudly
        logger.exception(
            "weightpack-vllm %s: token routing and bit1 fine-tunes are off",
            __version__,
        )
        return
    _registered = True
    logger.info("weightpack-vllm %s: token routing on", __version__)
