"""
Tests for PackedTernaryLinear and packed operations.

All tests use synthetic tensors — no model loading, fast execution.

Run with: pytest tests/test_packed_linear.py -v
"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from terncore.arithmetic.linear import TernaryLinear
from terncore.arithmetic.quantizer import TernaryQuantizer
from terncore.packed_linear import PackedTernaryLinear, convert_model_to_packed
from terncore.packed_ops import packed_ternary_matmul, packed_ternary_matmul_fast
from terncore.sparse import pack_ternary_weights


# ═══════════════════════════════════════════════════════════════
# PackedTernaryLinear tests
# ═══════════════════════════════════════════════════════════════


class TestPackedTernaryLinear:
    """Tests for PackedTernaryLinear module."""

    def test_from_float_basic(self):
        """Convert nn.Linear → PackedTernaryLinear, verify output shape."""
        torch.manual_seed(100)
        linear = nn.Linear(32, 16)

        packed = PackedTernaryLinear.from_float(linear, threshold=0.7)
        packed.eval()

        x = torch.randn(4, 32)
        output = packed(x)
        assert output.shape == (4, 16)

    def test_from_float_matches_ternary(self):
        """PackedTernaryLinear output matches TernaryLinear output."""
        torch.manual_seed(101)
        linear = nn.Linear(64, 32, bias=True)

        # Create both representations
        ternary = TernaryLinear(64, 32, bias=True, threshold=0.7)
        ternary.weight.data.copy_(linear.weight.data)
        ternary.bias.data.copy_(linear.bias.data)
        ternary.eval()

        packed = PackedTernaryLinear.from_float(linear, threshold=0.7)
        packed.eval()

        x = torch.randn(4, 64)
        with torch.no_grad():
            t_out = ternary(x)
            p_out = packed(x)

        assert torch.allclose(t_out, p_out, atol=1e-5), (
            f"Max diff: {(t_out - p_out).abs().max().item()}"
        )

    def test_from_ternary_linear(self):
        """Convert TernaryLinear → PackedTernaryLinear, verify bit-identical."""
        torch.manual_seed(102)
        ternary = TernaryLinear(48, 24, bias=True, threshold=0.7)
        ternary.eval()

        packed = PackedTernaryLinear.from_ternary_linear(ternary)
        packed.eval()

        x = torch.randn(8, 48)
        with torch.no_grad():
            t_out = ternary(x)
            p_out = packed(x)

        assert torch.allclose(t_out, p_out, atol=1e-5), (
            f"Max diff: {(t_out - p_out).abs().max().item()}"
        )

    def test_from_packed_data(self):
        """Create from raw packed bytes + alpha, verify forward."""
        torch.manual_seed(103)
        weights = torch.randn(16, 32)

        q = TernaryQuantizer(threshold=0.7)
        ternary, alpha = q.quantize(weights)
        packed_t, _bitmap = pack_ternary_weights(ternary)

        layer = PackedTernaryLinear.from_packed_data(
            packed_weights=packed_t,
            alpha=alpha.item(),
            in_features=32,
            out_features=16,
        )
        layer.eval()

        x = torch.randn(2, 32)
        output = layer(x)
        assert output.shape == (2, 16)

        # Verify matches manual computation
        expected = x @ (ternary.float() * alpha).T
        assert torch.allclose(output, expected, atol=1e-5)

    def test_memory_footprint(self):
        """Verify 16x compression for weight storage."""
        layer = PackedTernaryLinear(1024, 1024, bias=False)
        fp = layer.memory_footprint()

        assert fp["float32_bytes"] == 1024 * 1024 * 4  # 4 MB
        assert fp["packed_bytes"] == 1024 * 1024 // 4   # 256 KB
        assert fp["compression_ratio"] == 16.0
        assert fp["alpha_bytes"] == 4
        assert fp["bias_bytes"] == 0

    def test_memory_footprint_with_bias(self):
        """Memory footprint includes bias."""
        layer = PackedTernaryLinear(512, 256, bias=True)
        fp = layer.memory_footprint()
        assert fp["bias_bytes"] == 256 * 4

    def test_forward_with_bias(self):
        """Verify bias is added in forward pass."""
        torch.manual_seed(104)
        linear = nn.Linear(16, 8, bias=True)
        linear.bias.data.fill_(5.0)

        packed = PackedTernaryLinear.from_float(linear, threshold=0.7)
        packed.eval()

        x = torch.zeros(1, 16)
        output = packed(x)
        # With zero input, output should be just the bias
        assert torch.allclose(output, packed.bias.unsqueeze(0), atol=1e-6)

    def test_no_bias(self):
        """Forward works without bias."""
        torch.manual_seed(105)
        linear = nn.Linear(32, 16, bias=False)
        packed = PackedTernaryLinear.from_float(linear, threshold=0.7)
        packed.eval()

        x = torch.randn(2, 32)
        output = packed(x)
        assert output.shape == (2, 16)
        assert packed.bias is None

    def test_gradient_not_needed(self):
        """Packed weights should not require grad (inference only)."""
        layer = PackedTernaryLinear(32, 16, bias=False)
        assert not layer.packed_weights.requires_grad
        assert not layer.alpha.requires_grad

    def test_extra_repr(self):
        """Extra repr shows compression info."""
        layer = PackedTernaryLinear(256, 128, bias=True)
        s = layer.extra_repr()
        assert "in_features=256" in s
        assert "out_features=128" in s
        assert "compression=16x" in s

    def test_3d_input(self):
        """Handles 3D input (batch, seq_len, features)."""
        torch.manual_seed(106)
        linear = nn.Linear(64, 32)
        packed = PackedTernaryLinear.from_float(linear, threshold=0.7)
        packed.eval()

        x = torch.randn(2, 10, 64)
        output = packed(x)
        assert output.shape == (2, 10, 32)


# ═══════════════════════════════════════════════════════════════
# Packed operations tests
# ═══════════════════════════════════════════════════════════════


class TestPackedOps:
    """Tests for packed matmul operations."""

    def test_packed_matmul_correctness(self):
        """packed_ternary_matmul matches regular matmul on known values."""
        torch.manual_seed(200)
        weights = torch.randn(16, 32)

        q = TernaryQuantizer(threshold=0.7)
        ternary, alpha = q.quantize(weights)
        packed, _ = pack_ternary_weights(ternary)

        x = torch.randn(4, 32)

        # Packed matmul
        result = packed_ternary_matmul(x, packed, alpha.item(), 16, 32)

        # Reference
        expected = x @ (ternary.float() * alpha).T

        assert torch.allclose(result, expected, atol=1e-5)

    def test_packed_matmul_shapes(self):
        """Test multiple input batch sizes and feature dims."""
        torch.manual_seed(201)
        for out_f, in_f in [(8, 16), (64, 128), (256, 64)]:
            weights = torch.randn(out_f, in_f)
            q = TernaryQuantizer(threshold=0.7)
            ternary, alpha = q.quantize(weights)
            packed, _ = pack_ternary_weights(ternary)

            for batch in [1, 4, 16]:
                x = torch.randn(batch, in_f)
                result = packed_ternary_matmul(x, packed, alpha.item(), out_f, in_f)
                assert result.shape == (batch, out_f)

    def test_packed_matmul_fast_matches_reference(self):
        """Fast path gives same result as reference."""
        torch.manual_seed(202)
        weights = torch.randn(32, 64)

        q = TernaryQuantizer(threshold=0.7)
        ternary, alpha = q.quantize(weights)
        packed, _ = pack_ternary_weights(ternary)

        x = torch.randn(4, 64)

        ref = packed_ternary_matmul(x, packed, alpha.item(), 32, 64)
        fast = packed_ternary_matmul_fast(x, packed, alpha.item(), 32, 64)

        assert torch.allclose(ref, fast, atol=1e-4), (
            f"Max diff: {(ref - fast).abs().max().item()}"
        )


# ═══════════════════════════════════════════════════════════════
# Model conversion tests
# ═══════════════════════════════════════════════════════════════


class TestModelConversion:
    """Tests for convert_model_to_packed utility."""

    def test_convert_simple_model(self):
        """Convert 2-layer model, verify output matches."""
        torch.manual_seed(300)

        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(32, 64)
                self.fc2 = nn.Linear(64, 16)

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        model = SimpleNet()
        model.eval()

        # Reference output
        x = torch.randn(4, 32)
        with torch.no_grad():
            ref = model(x).clone()

        # Convert
        stats = convert_model_to_packed(model, threshold=0.7)
        assert stats["packed_layers"] == 2
        assert stats["protected_layers"] == 0

        # Verify layers are PackedTernaryLinear
        assert isinstance(model.fc1, PackedTernaryLinear)
        assert isinstance(model.fc2, PackedTernaryLinear)

        # Output should be close (quantisation introduces error)
        with torch.no_grad():
            packed_out = model(x)
        assert packed_out.shape == ref.shape

    def test_convert_with_protection(self):
        """Protected layers stay as nn.Linear."""
        torch.manual_seed(301)

        class MixedNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(32, 64)
                self.decoder = nn.Linear(64, 16)

            def forward(self, x):
                return self.decoder(torch.relu(self.encoder(x)))

        model = MixedNet()
        stats = convert_model_to_packed(
            model, threshold=0.7, protection_list=["decoder"]
        )
        assert stats["packed_layers"] == 1
        assert stats["protected_layers"] == 1

        assert isinstance(model.encoder, PackedTernaryLinear)
        assert isinstance(model.decoder, nn.Linear)

    def test_memory_reduction_after_conversion(self):
        """Total model parameter memory decreases after conversion."""
        torch.manual_seed(302)

        class BigNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(1024, 1024, bias=False)

            def forward(self, x):
                return self.fc(x)

        model = BigNet()

        # Before: 1024*1024 float32 params = 4 MB
        before_bytes = sum(
            p.nelement() * p.element_size() for p in model.parameters()
        )

        convert_model_to_packed(model, threshold=0.7)

        # After: packed_weights buffer + alpha buffer
        after_bytes = 0
        for buf in model.buffers():
            after_bytes += buf.nelement() * buf.element_size()
        for p in model.parameters():
            after_bytes += p.nelement() * p.element_size()

        assert after_bytes < before_bytes
        ratio = before_bytes / after_bytes
        # Should be close to 16x (slight overhead from alpha)
        assert ratio > 15.0, f"Compression only {ratio:.1f}x (expected >15x)"


class TestTernModelReaderPacked:
    """Test TernModelReader.load_packed_model integration."""

    def test_load_packed_from_tern_model(self):
        """Load .tern-model as PackedTernaryLinear layers."""
        torch.manual_seed(400)
        from terncore.tern_model import TernModelReader, TernModelWriter

        # Write a mixed model
        writer = TernModelWriter({"source": "packed_test"})
        writer.add_layer("fc1", torch.randn(32, 16), dtype="ternary2")
        writer.add_layer("fc2", torch.randn(8, 32), dtype="float16")

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        try:
            writer.write(path)

            # Build a model with matching architecture
            class TestNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(16, 32)
                    self.fc2 = nn.Linear(32, 8)

                def forward(self, x):
                    return self.fc2(torch.relu(self.fc1(x)))

            model = TestNet()
            reader = TernModelReader(path)
            missing, unexpected = reader.load_packed_model(model)

            # fc1 should be PackedTernaryLinear
            assert isinstance(model.fc1, PackedTernaryLinear)
            # fc2 should stay as nn.Linear (FP16 loaded)
            assert isinstance(model.fc2, nn.Linear)

            # Should be able to run forward
            x = torch.randn(2, 16)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (2, 8)
        finally:
            Path(path).unlink(missing_ok=True)
