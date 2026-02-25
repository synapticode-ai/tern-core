"""
Tests for sparsity bitmap caching, block analysis, and zero-skip.

Patent 7: Sparsity-aware execution — cached bitmap for zero-skip.
Patent 9: Zero-skip via bitmap-driven sparse kernel.

Run with: pytest tests/test_sparsity.py -v
"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from terncore.arithmetic.quantizer import TernaryQuantizer
from terncore.packed_linear import PackedTernaryLinear
from terncore.packed_ops import packed_ternary_matmul, packed_ternary_matmul_fast
from terncore.sparse import (
    analyze_block_sparsity,
    model_sparsity_report,
    pack_ternary_weights,
    unpack_ternary_weights,
)


# ═══════════════════════════════════════════════════════════════
# Bitmap Caching Tests
# ═══════════════════════════════════════════════════════════════


class TestBitmapCaching:
    """Tests for sparsity bitmap caching in PackedTernaryLinear."""

    def test_bitmap_stored_at_construction(self):
        """PackedTernaryLinear has non-zero bitmap after from_float()."""
        torch.manual_seed(900)
        linear = nn.Linear(64, 32)
        packed = PackedTernaryLinear.from_float(linear, threshold=0.7)

        # Bitmap should exist and have some non-zero bytes
        assert packed.sparsity_bitmap is not None
        assert packed.sparsity_bitmap.dtype == torch.uint8
        assert packed.sparsity_bitmap.sum().item() > 0

    def test_bitmap_matches_weights(self):
        """Cached bitmap correctly identifies non-zero weight positions."""
        torch.manual_seed(901)
        linear = nn.Linear(32, 16)
        packed = PackedTernaryLinear.from_float(linear, threshold=0.7)

        # Unpack weights and compare bitmap
        shape = torch.Size([16, 32])
        ternary = unpack_ternary_weights(packed.packed_weights, shape)
        nonzero_mask = (ternary.flatten() != 0)

        # Unpack bitmap back to boolean
        import numpy as np
        bitmap_np = packed.sparsity_bitmap.numpy()
        unpacked_bits = np.unpackbits(bitmap_np, bitorder="little")[
            : 16 * 32
        ]
        bitmap_bool = torch.from_numpy(unpacked_bits).bool()

        assert torch.equal(nonzero_mask, bitmap_bool)

    def test_bitmap_correct_size(self):
        """Bitmap has correct number of bytes for the weight dimensions."""
        layer = PackedTernaryLinear(256, 128, bias=False)
        expected_bytes = (256 * 128 + 7) // 8
        assert layer.sparsity_bitmap.nelement() == expected_bytes

    def test_from_packed_data_with_bitmap(self):
        """from_packed_data() uses provided bitmap without regenerating."""
        torch.manual_seed(902)
        weights = torch.randn(16, 32)
        q = TernaryQuantizer(threshold=0.7)
        ternary, alpha = q.quantize(weights)
        packed_t, _ = pack_ternary_weights(ternary)

        # Create a known bitmap
        import numpy as np
        nonzero = (ternary.flatten() != 0).numpy().astype(np.uint8)
        bitmap = torch.from_numpy(np.packbits(nonzero, bitorder="little").copy())

        layer = PackedTernaryLinear.from_packed_data(
            packed_weights=packed_t,
            alpha=alpha.item(),
            in_features=32,
            out_features=16,
            sparsity_bitmap=bitmap,
        )

        assert torch.equal(layer.sparsity_bitmap, bitmap)

    def test_from_packed_data_generates_bitmap_if_missing(self):
        """from_packed_data() generates bitmap when not provided."""
        torch.manual_seed(903)
        weights = torch.randn(16, 32)
        q = TernaryQuantizer(threshold=0.7)
        ternary, alpha = q.quantize(weights)
        packed_t, _ = pack_ternary_weights(ternary)

        layer = PackedTernaryLinear.from_packed_data(
            packed_weights=packed_t,
            alpha=alpha.item(),
            in_features=32,
            out_features=16,
        )

        # Bitmap should be generated and non-zero
        assert layer.sparsity_bitmap.sum().item() > 0

    def test_forward_with_cached_bitmap_matches_reference(self):
        """Forward with cached bitmap gives same result as reference matmul."""
        torch.manual_seed(904)
        linear = nn.Linear(64, 32, bias=True)
        packed = PackedTernaryLinear.from_float(linear, threshold=0.7)
        packed.eval()

        x = torch.randn(4, 64)
        with torch.no_grad():
            fast_out = packed(x)

        # Reference: unpack → float → F.linear + bias
        ref_out = packed_ternary_matmul(
            x, packed.packed_weights, packed.alpha.item(), 32, 64
        )
        ref_out = ref_out + packed.bias

        assert torch.allclose(fast_out, ref_out, atol=1e-4), (
            f"Max diff: {(fast_out - ref_out).abs().max().item()}"
        )

    def test_from_ternary_linear_has_bitmap(self):
        """from_ternary_linear() caches bitmap."""
        from terncore.arithmetic.linear import TernaryLinear

        torch.manual_seed(905)
        ternary = TernaryLinear(48, 24, bias=False, threshold=0.7)
        packed = PackedTernaryLinear.from_ternary_linear(ternary)

        assert packed.sparsity_bitmap is not None
        assert packed.sparsity_bitmap.sum().item() > 0

    def test_bitmap_from_tern_model(self):
        """load_packed_model() passes .tern-model bitmap to PackedTernaryLinear."""
        torch.manual_seed(906)
        from terncore.tern_model import TernModelReader, TernModelWriter

        writer = TernModelWriter({"source": "bitmap_test"})
        writer.add_layer("fc", torch.randn(16, 32), dtype="ternary2")

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        try:
            writer.write(path)

            class Net(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(32, 16)

                def forward(self, x):
                    return self.fc(x)

            model = Net()
            reader = TernModelReader(path)
            reader.load_packed_model(model)

            # fc should be PackedTernaryLinear with cached bitmap
            assert isinstance(model.fc, PackedTernaryLinear)
            assert model.fc.sparsity_bitmap is not None
            assert model.fc.sparsity_bitmap.sum().item() > 0
        finally:
            Path(path).unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════
# Block Sparsity Analysis Tests
# ═══════════════════════════════════════════════════════════════


class TestBlockSparsity:
    """Tests for block-level sparsity analysis."""

    def test_block_analysis_all_zero(self):
        """100% sparsity → all blocks are zero blocks."""
        # All-zero ternary weights
        ternary = torch.zeros(16, 32)
        packed, _ = pack_ternary_weights(ternary)

        result = analyze_block_sparsity(packed, 16, 32, block_size=64)
        assert result["sparsity"] == 1.0
        assert result["zero_blocks"] == result["total_blocks"]
        assert result["block_skip_ratio"] == 1.0

    def test_block_analysis_no_zero(self):
        """0% sparsity → no zero blocks."""
        # All +1 weights
        ternary = torch.ones(16, 32)
        packed, _ = pack_ternary_weights(ternary)

        result = analyze_block_sparsity(packed, 16, 32, block_size=64)
        assert result["sparsity"] == 0.0
        assert result["zero_blocks"] == 0
        assert result["block_skip_ratio"] == 0.0

    def test_block_analysis_partial(self):
        """Partial sparsity → some fraction of zero blocks."""
        torch.manual_seed(910)
        # Create weights with ~50% zeros
        ternary = torch.zeros(64, 64)
        mask = torch.rand(64, 64) > 0.5
        ternary[mask] = torch.sign(torch.randn(mask.sum()))
        packed, _ = pack_ternary_weights(ternary)

        result = analyze_block_sparsity(packed, 64, 64, block_size=256)
        assert 0.0 < result["sparsity"] < 1.0
        assert result["total_blocks"] > 0
        assert result["block_skip_ratio"] >= 0.0
        assert len(result["block_sparsity_histogram"]) == 10

    def test_block_analysis_returns_expected_keys(self):
        """Result dict contains all expected keys."""
        ternary = torch.zeros(8, 16)
        packed, _ = pack_ternary_weights(ternary)

        result = analyze_block_sparsity(packed, 8, 16, block_size=32)
        expected_keys = {
            "total_weights", "zero_weights", "sparsity",
            "total_blocks", "zero_blocks", "block_skip_ratio",
            "block_size", "block_sparsity_histogram", "mean_block_sparsity",
        }
        assert set(result.keys()) == expected_keys

    def test_model_sparsity_report(self):
        """model_sparsity_report works on model with PackedTernaryLinear layers."""
        torch.manual_seed(911)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(32, 16)
                self.fc2 = nn.Linear(16, 8)

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        model = Net()
        from terncore.packed_linear import convert_model_to_packed
        convert_model_to_packed(model, threshold=0.7)

        report = model_sparsity_report(model)
        assert len(report) == 2
        assert report[0]["name"] == "fc1"
        assert report[1]["name"] == "fc2"
        for entry in report:
            assert "sparsity" in entry
            assert "block_skip_ratio" in entry


# ═══════════════════════════════════════════════════════════════
# Zero-Skip Correctness Tests
# ═══════════════════════════════════════════════════════════════


class TestZeroSkipCorrectness:
    """Tests for zero-skip kernel correctness."""

    def test_zero_skip_same_output(self):
        """Output identical with cached bitmap vs rebuilt bitmap."""
        torch.manual_seed(920)
        weights = torch.randn(32, 64)
        q = TernaryQuantizer(threshold=0.7)
        ternary, alpha = q.quantize(weights)
        packed, _ = pack_ternary_weights(ternary)

        x = torch.randn(4, 64)

        # Reference (no bitmap)
        ref = packed_ternary_matmul(x, packed, alpha.item(), 32, 64)

        # Fast path with cached bitmap
        import numpy as np
        nonzero = (ternary.flatten() != 0).numpy().astype(np.uint8)
        bitmap = torch.from_numpy(np.packbits(nonzero, bitorder="little").copy())
        fast = packed_ternary_matmul_fast(
            x, packed, alpha.item(), 32, 64, sparsity_bitmap=bitmap
        )

        assert torch.allclose(ref, fast, atol=1e-4), (
            f"Max diff: {(ref - fast).abs().max().item()}"
        )

    def test_high_sparsity_correctness(self):
        """Zero-skip correct at 90% sparsity."""
        torch.manual_seed(921)
        # Create 90% sparse weights
        ternary = torch.zeros(32, 64)
        mask = torch.rand(32, 64) > 0.9
        ternary[mask] = torch.sign(torch.randn(mask.sum()))
        packed, _ = pack_ternary_weights(ternary)
        alpha = 0.5

        x = torch.randn(2, 64)

        # Reference
        ref = packed_ternary_matmul(x, packed, alpha, 32, 64)

        # Fast path with bitmap
        import numpy as np
        nonzero = (ternary.flatten() != 0).numpy().astype(np.uint8)
        bitmap = torch.from_numpy(np.packbits(nonzero, bitorder="little").copy())
        fast = packed_ternary_matmul_fast(
            x, packed, alpha, 32, 64, sparsity_bitmap=bitmap
        )

        assert torch.allclose(ref, fast, atol=1e-4), (
            f"Max diff: {(ref - fast).abs().max().item()}"
        )
