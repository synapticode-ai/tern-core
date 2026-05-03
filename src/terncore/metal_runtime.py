"""
Process-wide TernaryEngine singleton acquisition.

The Metal kernel engine lives once per process. Lazy-initialised on the
first call to get_engine(). Returns None when Metal is unavailable
(non-macOS host, missing dylib, GPU init failure) so callers can fall
back to the CPU path without explicit feature detection.

Patent 5: Ternary execution path — packed weight computation.

Copyright (c) 2026 Synapticode Co., Ltd. All rights reserved.
"""
from __future__ import annotations

from typing import Optional

# Module-level state. Two-flag pattern so a failed init is sticky:
# record the attempt and don't retry every get_engine() call.
_engine: "Optional[object]" = None
_init_attempted: bool = False


def get_engine() -> "Optional[object]":
    """Return the process-wide TernaryEngine, or None if Metal unavailable.

    Lazy-init on first call. Subsequent calls return the cached engine
    or None if init previously failed.

    Returns:
        TernaryEngine instance, or None on any failure to acquire it
        (non-macOS, missing dylib, Metal device init failure, etc.).
    """
    global _engine, _init_attempted
    if _init_attempted:
        return _engine

    _init_attempted = True
    try:
        from terncore.ternary_metal import TernaryEngine
        _engine = TernaryEngine()
    except Exception:
        # Broad except by design: contract is "graceful fallback when
        # Metal is unavailable for any reason". Captures ImportError
        # (module/dylib chain broken), RuntimeError (Metal device
        # init returned NULL), OSError (ctypes load failure), and any
        # future failure mode without needing to enumerate them.
        _engine = None
    return _engine


def reset_engine() -> None:
    """Reset the singleton (test-only).

    Releases the current engine reference so subsequent get_engine()
    calls reinitialise. Intended for test isolation; not part of the
    production API surface.
    """
    global _engine, _init_attempted
    _engine = None
    _init_attempted = False
