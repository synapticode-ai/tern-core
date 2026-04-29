"""
Tests for the FP16 cast guards in terncore.coreml_export.

Covers _validate_ternary2_alpha and _cast_fp16_retain_with_guards —
the two helpers that gate the FP32→FP16 cast sites against silent
Inf/NaN production. Closes tern-core #1.

All tests use synthetic numpy/scalar inputs — no .tern-model files,
no HuggingFace downloads.

Patents 38, 41: Configurable precision + compiler scheduling — the
guards preserve the FP16-retain compression contract by rejecting
inputs that would silently break it.

Run with: pytest tests/test_coreml_export.py -v
"""

import math

import numpy as np
import pytest

from terncore.coreml_export_helpers import (
    FP16_MAX,
    _cast_fp16_retain_with_guards,
    _validate_ternary2_alpha,
)


# ═══════════════════════════════════════════════════════════════
# _validate_ternary2_alpha
# ═══════════════════════════════════════════════════════════════


def test_alpha_passes_clean():
    """A finite alpha within FP16 range returns None and does not raise."""
    assert _validate_ternary2_alpha(0.05, "model.layers.0.mlp.gate_proj") is None


def test_alpha_raises_on_nonfinite():
    """Non-finite alpha (Inf or NaN) is a quantiser bug — refuse to emit."""
    with pytest.raises(ValueError, match="Non-finite alpha"):
        _validate_ternary2_alpha(float("inf"), "layer.x")
    with pytest.raises(ValueError, match="Non-finite alpha"):
        _validate_ternary2_alpha(float("nan"), "layer.y")


def test_alpha_raises_on_overflow():
    """Finite alpha above FP16 ceiling is a degenerate quantisation case."""
    overflow = FP16_MAX + 1.0
    with pytest.raises(ValueError, match="exceeds FP16 range"):
        _validate_ternary2_alpha(overflow, "layer.overflow")
    with pytest.raises(ValueError, match="exceeds FP16 range"):
        _validate_ternary2_alpha(-overflow, "layer.overflow_neg")


# ═══════════════════════════════════════════════════════════════
# _cast_fp16_retain_with_guards
# ═══════════════════════════════════════════════════════════════


def test_fp16_retain_passes_clean():
    """A finite FP32 array within range casts to FP16 unchanged."""
    src = np.array([-1.5, 0.0, 1.5, 12.0], dtype=np.float32)
    out = _cast_fp16_retain_with_guards(src, "layer.clean")
    assert out.dtype == np.float16
    assert out.shape == src.shape
    assert np.all(np.isfinite(out))
    assert np.allclose(out.astype(np.float32), src, atol=1e-3)


def test_fp16_retain_raises_on_nan():
    """A NaN in the source is source-model corruption — refuse to emit."""
    src = np.array([1.0, float("nan"), 2.0], dtype=np.float32)
    with pytest.raises(ValueError, match="Non-finite values"):
        _cast_fp16_retain_with_guards(src, "layer.nan")


def test_fp16_retain_raises_on_inf():
    """An Inf in the source is source-model corruption — refuse to emit."""
    src = np.array([1.0, float("inf"), 2.0], dtype=np.float32)
    with pytest.raises(ValueError, match="Non-finite values"):
        _cast_fp16_retain_with_guards(src, "layer.inf")


def test_fp16_retain_clamps_high(capsys):
    """Finite-but-out-of-range values clamp to ±FP16_MAX with a stdout WARNING."""
    src = np.array([1.0, 1.0e6, -1.0e6, 5.0], dtype=np.float32)
    out = _cast_fp16_retain_with_guards(src, "layer.clamp")
    assert out.dtype == np.float16
    assert np.all(np.isfinite(out)), "clamped output must be finite"
    out_f32 = out.astype(np.float32)
    assert out_f32[1] == pytest.approx(FP16_MAX, abs=1.0)
    assert out_f32[2] == pytest.approx(-FP16_MAX, abs=1.0)
    assert math.isclose(out_f32[0], 1.0, abs_tol=1e-3)
    assert math.isclose(out_f32[3], 5.0, abs_tol=1e-3)
    captured = capsys.readouterr()
    assert "WARNING:" in captured.out
    assert "layer.clamp" in captured.out
    assert "clamping" in captured.out
