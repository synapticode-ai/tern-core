"""
Test suite for terncore Stage 1B: C extension acceleration.

Tests that TernaryLinearAccel produces results matching TernaryLinear,
verifies fallback behaviour, SIMD detection, and end-to-end model
inference with accelerated layers.

Run with: pytest tests/test_stage1b.py -v
"""

import pytest
import torch
import torch.nn as nn

from terncore.arithmetic.linear import TernaryLinear
from terncore.accel import (
    TernaryLinearAccel,
    is_accelerated,
    get_acceleration_info,
)


# ═══════════════════════════════════════════════════════════════
# Helper model (same structure as test_stage1a.py)
# ═══════════════════════════════════════════════════════════════


class SimpleModel(nn.Module):
    """Small model for testing conversion + acceleration."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32, 16)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


def _replace_with_accel(model: nn.Module) -> None:
    """Replace all TernaryLinear layers with TernaryLinearAccel in-place."""
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, TernaryLinear) and not isinstance(
            module, TernaryLinearAccel
        ):
            replacements.append((name, module))

    for name, module in replacements:
        accel = TernaryLinearAccel.from_ternary_linear(module)
        parts = name.rsplit(".", 1)
        if len(parts) == 1:
            setattr(model, name, accel)
        else:
            parent = model
            for p in parts[0].split("."):
                parent = getattr(parent, p)
            setattr(parent, parts[-1], accel)


# ═══════════════════════════════════════════════════════════════
# TernaryLinearAccel — core tests
# ═══════════════════════════════════════════════════════════════


class TestTernaryLinearAccel:
    """Tests for accelerated ternary linear layer."""

    def test_output_shape(self):
        """Output shape must match TernaryLinear."""
        layer = TernaryLinearAccel(64, 32)
        layer.eval()
        x = torch.randn(8, 64)
        y = layer(x)
        assert y.shape == (8, 32)

    @pytest.mark.skipif(not is_accelerated(), reason="C library not available")
    def test_matches_ternary_linear(self):
        """Accelerated output must match pure PyTorch within FP tolerance."""
        base = TernaryLinear(64, 32)
        accel = TernaryLinearAccel.from_ternary_linear(base)

        base.eval()
        accel.eval()

        x = torch.randn(4, 64)
        y_base = base(x)
        y_accel = accel(x)

        assert torch.allclose(y_base, y_accel, atol=1e-4, rtol=1e-4), (
            f"Max diff: {(y_base - y_accel).abs().max().item()}"
        )

    @pytest.mark.skipif(not is_accelerated(), reason="C library not available")
    def test_matches_no_bias(self):
        """Matches TernaryLinear output without bias."""
        base = TernaryLinear(64, 32, bias=False)
        accel = TernaryLinearAccel.from_ternary_linear(base)

        base.eval()
        accel.eval()

        x = torch.randn(4, 64)
        y_base = base(x)
        y_accel = accel(x)

        assert torch.allclose(y_base, y_accel, atol=1e-4, rtol=1e-4)

    def test_single_sample(self):
        """Works with unbatched 1D input."""
        layer = TernaryLinearAccel(64, 32)
        layer.eval()
        x = torch.randn(64)
        y = layer(x)
        assert y.shape == (32,)

    def test_batch_sizes(self):
        """Works with various batch sizes."""
        layer = TernaryLinearAccel(64, 32)
        layer.eval()
        for B in [1, 4, 16, 64]:
            x = torch.randn(B, 64)
            y = layer(x)
            assert y.shape == (B, 32)

    def test_deterministic_100_runs(self):
        """Same input produces identical output across 100 runs. Patent 36."""
        layer = TernaryLinearAccel(64, 32)
        layer.eval()
        x = torch.randn(4, 64)

        reference = layer(x)
        for _ in range(99):
            y = layer(x)
            assert torch.equal(reference, y), "Must be bit-identical"

    def test_cache_invalidation(self):
        """Invalidating cache clears packed weights."""
        layer = TernaryLinearAccel(64, 32)
        layer.eval()
        _ = layer(torch.randn(1, 64))

        if is_accelerated():
            assert layer._packed_weights_np is not None

        layer.invalidate_cache()
        assert layer._packed_weights_np is None
        assert layer._cached_ternary is None

    def test_from_ternary_linear(self):
        """Conversion from TernaryLinear preserves parameters."""
        base = TernaryLinear(64, 32, threshold=0.8)
        accel = TernaryLinearAccel.from_ternary_linear(base)

        assert accel.in_features == 64
        assert accel.out_features == 32
        assert accel.threshold == 0.8
        assert torch.equal(accel.weight.data, base.weight.data)

    def test_training_mode_unchanged(self):
        """Training forward pass uses STE, same as TernaryLinear."""
        layer = TernaryLinearAccel(64, 32)
        layer.train()
        x = torch.randn(4, 64)
        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert layer.weight.grad is not None
        assert not torch.isnan(layer.weight.grad).any()

    @pytest.mark.skipif(not is_accelerated(), reason="C library not available")
    def test_large_matrix(self):
        """Matches TernaryLinear on larger dimensions."""
        base = TernaryLinear(256, 128)
        accel = TernaryLinearAccel.from_ternary_linear(base)

        base.eval()
        accel.eval()

        x = torch.randn(8, 256)
        y_base = base(x)
        y_accel = accel(x)

        assert torch.allclose(y_base, y_accel, atol=1e-4, rtol=1e-4), (
            f"Max diff: {(y_base - y_accel).abs().max().item()}"
        )

    @pytest.mark.skipif(not is_accelerated(), reason="C library not available")
    def test_extra_repr(self):
        """String representation reports acceleration status."""
        layer = TernaryLinearAccel(64, 32)
        repr_str = layer.extra_repr()
        assert "accel=" in repr_str
        # Accept any C backend label (torch_ext, ctypes, or legacy "C")
        assert any(
            label in repr_str
            for label in ("accel=torch_ext", "accel=ctypes", "accel=C")
        )


# ═══════════════════════════════════════════════════════════════
# Fallback tests
# ═══════════════════════════════════════════════════════════════


class TestAccelFallback:
    """Tests for fallback to pure PyTorch."""

    def test_fallback_unaligned_n(self):
        """Falls back when in_features is not a multiple of 4."""
        base = TernaryLinear(63, 32)
        accel = TernaryLinearAccel.from_ternary_linear(base)

        base.eval()
        accel.eval()

        x = torch.randn(4, 63)
        y_base = base(x)
        y_accel = accel(x)

        # Fallback uses identical code path — output must match exactly
        assert torch.equal(y_base, y_accel)

    def test_fallback_still_correct(self):
        """Fallback mode produces correct output."""
        layer = TernaryLinearAccel(63, 32)
        layer.eval()
        x = torch.randn(4, 63)
        y = layer(x)
        assert y.shape == (4, 32)
        assert not torch.isnan(y).any()

    def test_fallback_extra_repr(self):
        """Fallback repr when library is missing (or N not aligned)."""
        layer = TernaryLinearAccel(63, 32)
        # N=63 is not multiple of 4, so even with library present,
        # this layer falls back — but extra_repr reports library status
        rep = layer.extra_repr()
        assert "accel=" in rep


# ═══════════════════════════════════════════════════════════════
# Acceleration info and SIMD detection
# ═══════════════════════════════════════════════════════════════


class TestAccelInfo:
    """Tests for acceleration info and SIMD detection."""

    def test_is_accelerated_returns_bool(self):
        result = is_accelerated()
        assert isinstance(result, bool)

    def test_info_structure(self):
        """get_acceleration_info returns expected keys."""
        info = get_acceleration_info()
        assert "accelerated" in info
        assert "library_path" in info
        assert "simd_support" in info
        assert "version" in info

    @pytest.mark.skipif(not is_accelerated(), reason="C library not available")
    def test_simd_scalar_always_set(self):
        """Scalar support must always be reported."""
        info = get_acceleration_info()
        assert info["simd_support"]["scalar"] is True

    @pytest.mark.skipif(not is_accelerated(), reason="C library not available")
    def test_simd_keys(self):
        """SIMD support dict has all expected keys."""
        info = get_acceleration_info()
        for key in ["scalar", "avx2", "avx512", "neon"]:
            assert key in info["simd_support"]
            assert isinstance(info["simd_support"][key], bool)

    @pytest.mark.skipif(not is_accelerated(), reason="C library not available")
    def test_version_string(self):
        """Library version must be a non-empty string."""
        info = get_acceleration_info()
        assert isinstance(info["version"], str)
        assert len(info["version"]) > 0

    @pytest.mark.skipif(not is_accelerated(), reason="C library not available")
    def test_version_matches_package(self):
        """Library version should match Python package version."""
        import terncore

        info = get_acceleration_info()
        assert info["version"] == terncore.__version__

    @pytest.mark.skipif(not is_accelerated(), reason="C library not available")
    def test_library_path_exists(self):
        """Library path should point to an existing file."""
        from pathlib import Path

        info = get_acceleration_info()
        assert Path(info["library_path"]).exists()

    def test_info_without_library(self):
        """When library is absent, info reports no acceleration."""
        info = get_acceleration_info()
        if not is_accelerated():
            assert info["accelerated"] is False
            assert info["library_path"] is None
            assert info["version"] is None
            assert info["simd_support"] == {}


# ═══════════════════════════════════════════════════════════════
# Integration tests — model conversion + accelerated inference
# ═══════════════════════════════════════════════════════════════


class TestAccelIntegration:
    """End-to-end integration tests with model conversion."""

    def test_model_with_accel_layers(self):
        """Convert a model's TernaryLinear layers to TernaryLinearAccel."""
        from terncore.engine.inference import TernaryInferenceEngine

        model = SimpleModel()
        engine = TernaryInferenceEngine()
        engine.convert(model, sensitivity_analysis=False)

        _replace_with_accel(model)
        model.eval()

        x = torch.randn(4, 64)
        y = model(x)

        assert y.shape == (4, 16)
        assert not torch.isnan(y).any()

    @pytest.mark.skipif(not is_accelerated(), reason="C library not available")
    def test_accel_matches_pytorch_model(self):
        """Full model with accel layers matches pure PyTorch model."""
        from terncore.engine.inference import TernaryInferenceEngine

        # Create two identical models from same seed
        torch.manual_seed(42)
        model_base = SimpleModel()
        torch.manual_seed(42)
        model_accel = SimpleModel()

        engine = TernaryInferenceEngine()
        engine.convert(model_base, sensitivity_analysis=False)
        engine.convert(model_accel, sensitivity_analysis=False)

        _replace_with_accel(model_accel)

        model_base.eval()
        model_accel.eval()

        x = torch.randn(4, 64)
        y_base = model_base(x)
        y_accel = model_accel(x)

        assert torch.allclose(y_base, y_accel, atol=1e-3, rtol=1e-3), (
            f"Max diff: {(y_base - y_accel).abs().max().item()}"
        )

    def test_deterministic_model_inference(self):
        """Model with accel layers is deterministic. Patent 36."""
        from terncore.engine.inference import TernaryInferenceEngine

        model = SimpleModel()
        engine = TernaryInferenceEngine()
        engine.convert(model, sensitivity_analysis=False)

        _replace_with_accel(model)
        model.eval()

        x = torch.randn(4, 64)

        reference = model(x)
        for _ in range(49):
            y = model(x)
            assert torch.equal(reference, y)

    def test_infer_through_engine(self):
        """Accelerated model works through TernaryInferenceEngine.infer()."""
        from terncore.engine.inference import TernaryInferenceEngine

        model = SimpleModel()
        engine = TernaryInferenceEngine()
        engine.convert(model, sensitivity_analysis=False)

        _replace_with_accel(model)

        x = torch.randn(4, 64)
        result = engine.infer(model, x)

        assert result.output.shape == (4, 16)
        assert result.latency_ms > 0
        assert result.deterministic is True


# ═══════════════════════════════════════════════════════════════
# SIMD acceleration tests (Phase 2)
# ═══════════════════════════════════════════════════════════════


class TestSIMDAcceleration:
    """Tests for SIMD-accelerated dispatch (AVX2 / NEON). Patent 38."""

    @pytest.mark.skipif(not is_accelerated(), reason="C library not available")
    def test_simd_detection_reports_capability(self):
        """SIMD detection reports at least one hardware capability."""
        import platform

        info = get_acceleration_info()
        simd = info["simd_support"]

        # Scalar must always be set
        assert simd["scalar"] is True

        # Platform-aware: expect AVX2 on x86_64, NEON on AArch64
        machine = platform.machine().lower()
        if machine in ("x86_64", "amd64"):
            assert simd["avx2"] is True, "AVX2 expected on x86_64"
        elif machine in ("aarch64", "arm64"):
            assert simd["neon"] is True, "NEON expected on AArch64"

    @pytest.mark.skipif(not is_accelerated(), reason="C library not available")
    def test_simd_matches_scalar_small(self):
        """SIMD dispatch matches TernaryLinear on small matrix (64x32)."""
        torch.manual_seed(100)
        base = TernaryLinear(32, 64)
        accel = TernaryLinearAccel.from_ternary_linear(base)

        base.eval()
        accel.eval()

        x = torch.randn(4, 32)
        y_base = base(x)
        y_accel = accel(x)

        assert torch.allclose(y_base, y_accel, atol=1e-5, rtol=1e-5), (
            f"Max diff: {(y_base - y_accel).abs().max().item()}"
        )

    @pytest.mark.skipif(not is_accelerated(), reason="C library not available")
    def test_simd_matches_scalar_large(self):
        """SIMD dispatch matches TernaryLinear on large matrix (1024x512)."""
        torch.manual_seed(200)
        base = TernaryLinear(512, 1024)
        accel = TernaryLinearAccel.from_ternary_linear(base)

        base.eval()
        accel.eval()

        x = torch.randn(8, 512)
        y_base = base(x)
        y_accel = accel(x)

        assert torch.allclose(y_base, y_accel, atol=1e-4, rtol=1e-4), (
            f"Max diff: {(y_base - y_accel).abs().max().item()}"
        )

    @pytest.mark.skipif(not is_accelerated(), reason="C library not available")
    def test_simd_deterministic(self):
        """SIMD dispatch produces bit-identical output across 100 runs. Patent 36."""
        torch.manual_seed(300)
        layer = TernaryLinearAccel(128, 64)
        layer.eval()

        x = torch.randn(4, 128)
        reference = layer(x)

        for i in range(99):
            y = layer(x)
            assert torch.equal(reference, y), (
                f"Run {i + 2}: not bit-identical, max diff="
                f"{(reference - y).abs().max().item()}"
            )
