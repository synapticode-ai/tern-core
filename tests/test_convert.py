"""
Tests for the tern-convert end-to-end conversion pipeline.

All tests use synthetic models — no HuggingFace downloads.

Patents 10-12: Automated binary-to-ternary conversion pipeline.

Run with: pytest tests/test_convert.py -v
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from terncore.convert import TernaryConverter, DEFAULT_PROTECTION_PATTERNS
from terncore.tern_model import TernModelReader


# ═══════════════════════════════════════════════════════════════
# Synthetic test models
# ═══════════════════════════════════════════════════════════════


class SimpleMLP(nn.Module):
    """3-layer MLP with lm_head for testing protection."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(32, 64, bias=False)  # should be protected
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.lm_head = nn.Linear(64, 32, bias=False)  # should be protected

    def forward(self, x):
        x = torch.relu(self.embed(x))
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.lm_head(x)


class TransformerLike(nn.Module):
    """Model with transformer-like layer naming for protection testing."""

    def __init__(self):
        super().__init__()
        self.model = nn.ModuleDict({
            "embed_tokens": nn.Linear(16, 32, bias=False),
            "layers": nn.ModuleList([
                nn.ModuleDict({
                    "self_attn": nn.ModuleDict({
                        "q_proj": nn.Linear(32, 32, bias=False),
                        "k_proj": nn.Linear(32, 32, bias=False),
                        "v_proj": nn.Linear(32, 32, bias=False),
                        "o_proj": nn.Linear(32, 32, bias=False),
                    }),
                    "mlp": nn.ModuleDict({
                        "gate_proj": nn.Linear(32, 64, bias=False),
                        "up_proj": nn.Linear(32, 64, bias=False),
                        "down_proj": nn.Linear(64, 32, bias=False),
                    }),
                    "input_layernorm": nn.Linear(32, 32, bias=False),  # named like norm
                })
                for _ in range(2)
            ]),
            "norm": nn.Linear(32, 32, bias=False),  # should be protected
        })
        self.lm_head = nn.Linear(32, 16, bias=False)  # should be protected

    def forward(self, x):
        return x  # not testing forward pass here


class BiasModel(nn.Module):
    """Model with bias for testing bias preservation."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32, bias=True)
        self.fc2 = nn.Linear(32, 8, bias=True)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# ═══════════════════════════════════════════════════════════════
# Pipeline tests
# ═══════════════════════════════════════════════════════════════


class TestTernaryConverter:
    """Tests for the end-to-end conversion pipeline."""

    def test_convert_synthetic_model(self):
        """Convert a small synthetic model end-to-end."""
        torch.manual_seed(1000)
        model = SimpleMLP()

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        try:
            converter = TernaryConverter(
                model_id="test/synthetic",
                output_path=path,
                threshold=0.7,
            )
            stats = converter.convert(verbose=False, model=model)

            assert Path(path).exists()
            assert stats["total_layers"] > 0
            assert stats["ternary_layers"] > 0
            assert stats["file_size_bytes"] > 0
            # >= 0 not > 0: a synthetic SimpleMLP can convert faster than
            # the timer's smallest representable delta on Apple Silicon,
            # so 0.0 is a legitimate "successful conversion that took
            # essentially no time" reading. The timer is being honest;
            # the assertion was being unrealistic.
            assert stats["conversion_time_seconds"] >= 0
        finally:
            Path(path).unlink(missing_ok=True)

    def test_protection_patterns(self):
        """Verify protection patterns correctly identify layers."""
        torch.manual_seed(1001)
        model = SimpleMLP()

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        try:
            converter = TernaryConverter(
                model_id="test/synthetic",
                output_path=path,
                threshold=0.7,
            )
            stats = converter.convert(verbose=False, model=model)

            # embed and lm_head should be protected, fc1 and fc2 should be ternary
            assert stats["protected_layers"] == 2  # embed, lm_head
            assert stats["ternary_layers"] == 2  # fc1, fc2
        finally:
            Path(path).unlink(missing_ok=True)

    def test_protection_always_protects_critical(self):
        """Embedding, norm, and head are ALWAYS protected even with empty patterns."""
        torch.manual_seed(1002)
        model = SimpleMLP()

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        try:
            # Pass empty protection patterns — critical layers should still be protected
            converter = TernaryConverter(
                model_id="test/synthetic",
                output_path=path,
                threshold=0.7,
                protection_patterns=[],  # empty!
            )
            stats = converter.convert(verbose=False, model=model)

            # embed and lm_head must still be protected (always-protected patterns)
            assert stats["protected_layers"] >= 2
        finally:
            Path(path).unlink(missing_ok=True)

    def test_convert_stats_returned(self):
        """convert() returns comprehensive stats dict."""
        torch.manual_seed(1003)
        model = SimpleMLP()

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        try:
            converter = TernaryConverter(
                model_id="test/synthetic",
                output_path=path,
                threshold=0.7,
            )
            stats = converter.convert(verbose=False, model=model)

            expected_keys = {
                "model_name", "total_layers", "ternary_layers", "protected_layers",
                "total_params", "ternary_params", "protected_params",
                "file_size_bytes", "compression_ratio", "conversion_time_seconds",
                "per_layer_stats",
            }
            assert expected_keys.issubset(set(stats.keys()))
            assert isinstance(stats["per_layer_stats"], list)
            assert len(stats["per_layer_stats"]) == stats["total_layers"]
        finally:
            Path(path).unlink(missing_ok=True)

    def test_output_file_loadable(self):
        """Output .tern-model can be read by TernModelReader."""
        torch.manual_seed(1004)
        model = SimpleMLP()

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        try:
            converter = TernaryConverter(
                model_id="test/synthetic",
                output_path=path,
                threshold=0.7,
            )
            converter.convert(verbose=False, model=model)

            reader = TernModelReader(path)
            assert reader.header["num_layers"] == 4
            assert reader.header["num_ternary"] == 2
            assert reader.header["num_protected"] == 2
            assert reader.verify()
        finally:
            Path(path).unlink(missing_ok=True)

    def test_round_trip_synthetic(self):
        """Convert -> load -> forward pass produces valid output."""
        torch.manual_seed(1005)
        model = SimpleMLP()
        model.eval()

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        try:
            converter = TernaryConverter(
                model_id="test/synthetic",
                output_path=path,
                threshold=0.7,
            )
            converter.convert(verbose=False, model=model)

            # Load weights back into a fresh model
            reader = TernModelReader(path)
            fresh_model = SimpleMLP()
            reader.load_as_model(fresh_model, strict=False)
            fresh_model.eval()

            # Forward pass should produce valid (non-NaN, non-zero) output
            x = torch.randn(2, 32)
            with torch.no_grad():
                out = fresh_model(x)

            assert out.shape == (2, 32)
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()
        finally:
            Path(path).unlink(missing_ok=True)

    def test_all_linear_layers_converted(self):
        """All unprotected nn.Linear layers become ternary in output."""
        torch.manual_seed(1006)
        model = SimpleMLP()

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        try:
            converter = TernaryConverter(
                model_id="test/synthetic",
                output_path=path,
                threshold=0.7,
            )
            converter.convert(verbose=False, model=model)

            reader = TernModelReader(path)
            for entry in reader.manifest["layers"]:
                name = entry["name"]
                if "embed" in name or "lm_head" in name:
                    assert entry["dtype"] == "float16", f"{name} should be float16"
                else:
                    assert entry["dtype"] == "ternary2", f"{name} should be ternary2"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_convert_with_bias(self):
        """Layers with bias are correctly converted."""
        torch.manual_seed(1007)
        model = BiasModel()

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        try:
            converter = TernaryConverter(
                model_id="test/synthetic",
                output_path=path,
                threshold=0.7,
            )
            stats = converter.convert(verbose=False, model=model)

            # Both layers should be ternary (no protection patterns match)
            assert stats["ternary_layers"] == 2

            # Verify we can read the file and layers have bias
            reader = TernModelReader(path)
            for entry in reader.manifest["layers"]:
                assert entry["has_bias"] is True
        finally:
            Path(path).unlink(missing_ok=True)

    def test_transformer_protection(self):
        """Transformer-like model has correct protection for embed/norm/head."""
        torch.manual_seed(1008)
        model = TransformerLike()

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        try:
            converter = TernaryConverter(
                model_id="test/transformer",
                output_path=path,
                threshold=0.7,
            )
            stats = converter.convert(verbose=False, model=model)

            reader = TernModelReader(path)
            protected_names = [
                e["name"] for e in reader.manifest["layers"]
                if e["dtype"] == "float16"
            ]
            ternary_names = [
                e["name"] for e in reader.manifest["layers"]
                if e["dtype"] == "ternary2"
            ]

            # embed_tokens, norm, lm_head, and input_layernorm should be protected
            assert any("embed_tokens" in n for n in protected_names)
            assert any("lm_head" in n for n in protected_names)

            # q_proj, k_proj, v_proj, o_proj, gate/up/down should be ternary
            assert any("q_proj" in n for n in ternary_names)
            assert any("v_proj" in n for n in ternary_names)
            assert any("gate_proj" in n for n in ternary_names)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_verify_output(self):
        """verify() validates the output file."""
        torch.manual_seed(1009)
        model = SimpleMLP()

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        try:
            converter = TernaryConverter(
                model_id="test/synthetic",
                output_path=path,
                threshold=0.7,
            )
            converter.convert(verbose=False, model=model)

            assert converter.verify(verbose=False)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_compression_ratio(self):
        """Compression ratio is reasonable for mixed-precision output."""
        torch.manual_seed(1010)
        model = SimpleMLP()

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        try:
            converter = TernaryConverter(
                model_id="test/synthetic",
                output_path=path,
                threshold=0.7,
            )
            stats = converter.convert(verbose=False, model=model)

            # Should compress significantly vs FP32
            # Mixed precision (some FP16, some ternary2) → expect 2-8x
            assert stats["compression_ratio"] > 1.5
        finally:
            Path(path).unlink(missing_ok=True)

    def test_custom_protection_patterns(self):
        """Custom protection patterns add to always-protected."""
        torch.manual_seed(1011)
        model = SimpleMLP()

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        try:
            # Protect fc1 in addition to embed/lm_head
            converter = TernaryConverter(
                model_id="test/synthetic",
                output_path=path,
                threshold=0.7,
                protection_patterns=["*fc1*"],
            )
            stats = converter.convert(verbose=False, model=model)

            # embed + lm_head + fc1 = 3 protected, fc2 = 1 ternary
            assert stats["protected_layers"] == 3
            assert stats["ternary_layers"] == 1
        finally:
            Path(path).unlink(missing_ok=True)

    def test_per_layer_stats(self):
        """Per-layer stats include correct metadata."""
        torch.manual_seed(1012)
        model = SimpleMLP()

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        try:
            converter = TernaryConverter(
                model_id="test/synthetic",
                output_path=path,
                threshold=0.7,
            )
            stats = converter.convert(verbose=False, model=model)

            for layer in stats["per_layer_stats"]:
                assert "name" in layer
                assert "dtype" in layer
                assert "shape" in layer
                if layer["dtype"] == "ternary2":
                    assert "sparsity" in layer
                    assert "alpha" in layer
                    assert 0 <= layer["sparsity"] <= 1
                    assert layer["alpha"] > 0
        finally:
            Path(path).unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════
# CLI tests
# ═══════════════════════════════════════════════════════════════


class TestCLI:
    """Tests for the CLI entry point."""

    def test_cli_help(self):
        """CLI --help doesn't crash."""
        result = subprocess.run(
            [sys.executable, "-m", "terncore.convert", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "tern-convert" in result.stdout

    def test_cli_missing_output(self):
        """CLI errors when --output is missing."""
        result = subprocess.run(
            [sys.executable, "-m", "terncore.convert", "some_model"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0


# ═══════════════════════════════════════════════════════════════
# Conv1D support tests
# ═══════════════════════════════════════════════════════════════


class TestConv1DSupport:
    """Tests for HuggingFace Conv1D handling (GPT-2 family)."""

    @pytest.fixture
    def conv1d_available(self):
        """Check if transformers Conv1D is available."""
        try:
            from transformers.pytorch_utils import Conv1D
            return Conv1D
        except ImportError:
            pytest.skip("transformers not installed")

    def test_conv1d_detected(self, conv1d_available):
        """Conv1D layers are detected as weight layers."""
        from terncore.convert import _is_weight_layer
        Conv1D = conv1d_available

        linear = nn.Linear(32, 64)
        conv1d = Conv1D(64, 32)  # Conv1D(nf=out, nx=in): weight (in, out)
        relu = nn.ReLU()

        assert _is_weight_layer(linear) is True
        assert _is_weight_layer(conv1d) is True
        assert _is_weight_layer(relu) is False

    def test_conv1d_weight_transposed(self, conv1d_available):
        """Conv1D weights are transposed to (out, in) for packing."""
        from terncore.convert import _get_weight_and_shape
        Conv1D = conv1d_available

        # Conv1D stores weights as (in_features=32, out_features=64)
        conv1d = Conv1D(64, 32)
        weight, shape = _get_weight_and_shape(conv1d)
        # After transpose: (out=64, in=32)
        assert shape == [64, 32]
        assert weight.shape == (64, 32)

    def test_conv1d_model_conversion(self, conv1d_available):
        """Model with Conv1D layers converts successfully."""
        Conv1D = conv1d_available

        class GPT2Like(nn.Module):
            def __init__(self):
                super().__init__()
                self.c_attn = Conv1D(96, 32)   # 3 * 32 heads
                self.c_proj = Conv1D(32, 32)
                self.c_fc = Conv1D(128, 32)    # MLP
                self.c_fc2 = Conv1D(32, 128)   # MLP
                self.lm_head = nn.Linear(32, 16, bias=False)

            def forward(self, x):
                return x

        torch.manual_seed(2001)
        model = GPT2Like()

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        try:
            converter = TernaryConverter(
                model_id="test/gpt2like",
                output_path=path,
                threshold=0.7,
            )
            stats = converter.convert(verbose=False, model=model)

            # 5 total layers: 4 Conv1D + 1 nn.Linear
            assert stats["total_layers"] == 5
            # lm_head protected (matches *lm_head* pattern), 4 Conv1D ternary
            assert stats["ternary_layers"] == 4
            assert stats["protected_layers"] == 1

            # Verify file integrity
            assert converter.verify(verbose=False)

            # Read back and check shapes are (out, in)
            reader = TernModelReader(path)
            for entry in reader.manifest["layers"]:
                if entry["name"] == "c_attn":
                    assert entry["shape"] == [96, 32]  # transposed
                elif entry["name"] == "c_proj":
                    assert entry["shape"] == [32, 32]
        finally:
            Path(path).unlink(missing_ok=True)
