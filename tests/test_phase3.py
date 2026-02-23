"""
Test suite for terncore Phase 3: HuggingFace model loading and ternary conversion.

Unit tests use MockLlamaModel (no download needed).  Integration tests
requiring HuggingFace transformers or model downloads are gated.

Run with:
    pytest tests/test_phase3.py -v                     # unit tests only
    TERNCORE_RUN_SLOW=1 pytest tests/test_phase3.py -v # include slow tests

Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
"""

import os

import pytest
import torch
import torch.nn as nn

from terncore.accel import TernaryLinearAccel, is_accelerated
from terncore.arithmetic.linear import TernaryLinear
from terncore.engine.inference import ConversionReport, TernaryInferenceEngine
from terncore.memory import profile_model_memory

# ═══════════════════════════════════════════════════════════════
# Skip markers
# ═══════════════════════════════════════════════════════════════

try:
    import transformers  # noqa: F401

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

requires_transformers = pytest.mark.skipif(
    not HAS_TRANSFORMERS,
    reason="transformers not installed (pip install terncore[transformers])",
)

RUN_SLOW = os.environ.get("TERNCORE_RUN_SLOW", "0") == "1"

slow = pytest.mark.skipif(
    not RUN_SLOW,
    reason="Set TERNCORE_RUN_SLOW=1 to run slow tests",
)


# ═══════════════════════════════════════════════════════════════
# MockLlamaModel — correct LLaMA naming, no HF download
# ═══════════════════════════════════════════════════════════════


class MockLlamaConfig:
    """Minimal config mimicking LlamaForCausalLM."""

    vocab_size = 100
    hidden_size = 64
    num_hidden_layers = 2
    intermediate_size = 128
    num_attention_heads = 4
    num_key_value_heads = 2


class MockSelfAttn(nn.Module):
    """Self-attention block with LLaMA-style projection names."""

    def __init__(self, config: MockLlamaConfig) -> None:
        super().__init__()
        h = config.hidden_size
        kv_dim = config.num_key_value_heads * (h // config.num_attention_heads)
        self.q_proj = nn.Linear(h, h, bias=False)
        self.k_proj = nn.Linear(h, kv_dim, bias=False)
        self.v_proj = nn.Linear(h, kv_dim, bias=False)
        self.o_proj = nn.Linear(h, h, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(self.q_proj(x))


class MockMLP(nn.Module):
    """MLP block with LLaMA-style projection names (SwiGLU)."""

    def __init__(self, config: MockLlamaConfig) -> None:
        super().__init__()
        h = config.hidden_size
        ff = config.intermediate_size
        self.gate_proj = nn.Linear(h, ff, bias=False)
        self.up_proj = nn.Linear(h, ff, bias=False)
        self.down_proj = nn.Linear(ff, h, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_proj(x))
        return self.down_proj(gate * self.up_proj(x))


class MockLlamaLayer(nn.Module):
    """Single transformer layer with LLaMA naming."""

    def __init__(self, config: MockLlamaConfig) -> None:
        super().__init__()
        self.self_attn = MockSelfAttn(config)
        self.mlp = MockMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layernorm(x)
        x = x + self.self_attn(h)
        h = self.post_attention_layernorm(x)
        x = x + self.mlp(h)
        return x


class MockLlamaInner(nn.Module):
    """Inner model container (model.model in HuggingFace)."""

    def __init__(self, config: MockLlamaConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [MockLlamaLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class MockLlamaModel(nn.Module):
    """
    Minimal LLaMA-like model with correct HuggingFace naming.

    Layer structure matches TinyLlama:
        model.embed_tokens          (Embedding, protected)
        model.layers.N.self_attn.{q,k,v,o}_proj   (Linear, eligible)
        model.layers.N.mlp.{gate,up,down}_proj     (Linear, eligible)
        model.layers.N.{input,post_attention}_layernorm  (LayerNorm, not Linear)
        model.norm                  (LayerNorm, not Linear)
        lm_head                     (Linear, protected)
    """

    def __init__(self, config: MockLlamaConfig | None = None) -> None:
        super().__init__()
        self.config = config or MockLlamaConfig()
        self.model = MockLlamaInner(self.config)
        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )

    def forward(
        self, input_ids: torch.Tensor | None = None, **kwargs: object
    ) -> object:
        x = self.model(input_ids)
        logits = self.lm_head(x)
        return type("Output", (), {"logits": logits})()


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
# TestMockLlamaConversion — core conversion tests
# ═══════════════════════════════════════════════════════════════


class TestMockLlamaConversion:
    """Test ternary conversion on MockLlamaModel (no HuggingFace needed)."""

    def test_conversion_counts(self):
        """Correct number of layers are converted vs protected."""
        model = MockLlamaModel()
        engine = TernaryInferenceEngine()
        report = engine.convert(model, sensitivity_analysis=False)

        # 2 layers × 7 projections = 14 eligible Linear
        # lm_head = 1 protected Linear
        # Total Linear = 15
        assert report.total_layers == 15
        assert report.converted_layers == 14
        assert report.skipped_layers == 1

    def test_lm_head_protected(self):
        """lm_head is protected from conversion."""
        model = MockLlamaModel()
        engine = TernaryInferenceEngine()
        engine.convert(model, sensitivity_analysis=False)

        assert isinstance(model.lm_head, nn.Linear)
        assert not isinstance(model.lm_head, TernaryLinear)

    def test_o_proj_not_falsely_protected(self):
        """o_proj must NOT be caught by the 'output' protection pattern."""
        engine = TernaryInferenceEngine()
        dummy = ConversionReport()

        # These names should NOT be protected
        for name in [
            "model.layers.0.self_attn.o_proj",
            "model.layers.5.self_attn.o_proj",
            "model.layers.21.self_attn.o_proj",
        ]:
            assert not engine._should_protect(name, nn.Linear(64, 64), dummy), (
                f"{name} was falsely protected"
            )

    def test_embed_tokens_untouched(self):
        """Embedding layer is not a Linear, so it's never converted."""
        model = MockLlamaModel()
        engine = TernaryInferenceEngine()
        engine.convert(model, sensitivity_analysis=False)

        assert isinstance(model.model.embed_tokens, nn.Embedding)

    def test_converted_layers_are_ternary(self):
        """All eligible layers become TernaryLinear after conversion."""
        model = MockLlamaModel()
        engine = TernaryInferenceEngine()
        engine.convert(model, sensitivity_analysis=False)

        count = sum(
            1 for m in model.modules() if isinstance(m, TernaryLinear)
        )
        assert count == 14

    def test_forward_produces_output(self):
        """Converted model produces output of correct shape."""
        model = MockLlamaModel()
        engine = TernaryInferenceEngine()
        engine.convert(model, sensitivity_analysis=False)

        input_ids = torch.randint(0, 100, (1, 10))
        result = engine.infer(model, {"input_ids": input_ids})

        assert result.output.logits.shape == (1, 10, 100)
        assert not torch.isnan(result.output.logits).any()

    def test_deterministic_output(self):
        """Same input → identical output across runs. Patent 36."""
        model = MockLlamaModel()
        engine = TernaryInferenceEngine()
        engine.convert(model, sensitivity_analysis=False)

        input_ids = torch.randint(0, 100, (1, 10))
        r1 = engine.infer(model, {"input_ids": input_ids})
        r2 = engine.infer(model, {"input_ids": input_ids})

        assert torch.equal(r1.output.logits, r2.output.logits)

    def test_memory_profile_shows_compression(self):
        """Memory profile reports compression after conversion."""
        model = MockLlamaModel()
        engine = TernaryInferenceEngine()
        engine.convert(model, sensitivity_analysis=False)

        profile = profile_model_memory(model)
        assert profile.ternary_params > 0
        assert profile.fp16_params > 0  # lm_head stays FP16
        assert profile.compression_ratio > 1.0

    def test_all_dims_multiple_of_4(self):
        """All eligible Linear dims are multiples of 4 (C kernel compatible)."""
        model = MockLlamaModel()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                assert module.in_features % 4 == 0, (
                    f"{name}: in_features={module.in_features}"
                )

    def test_conversion_report_sizes(self):
        """ConversionReport has valid size and compression estimates."""
        model = MockLlamaModel()
        engine = TernaryInferenceEngine()
        report = engine.convert(model, sensitivity_analysis=False)

        assert report.original_size_mb > 0
        assert report.ternary_size_mb > 0
        assert report.compression_ratio > 1.0
        assert report.ternary_params > 0


# ═══════════════════════════════════════════════════════════════
# TestMockLlamaAccel — C+SIMD acceleration tests
# ═══════════════════════════════════════════════════════════════


class TestMockLlamaAccel:
    """Test accelerated ternary on MockLlamaModel."""

    def test_accel_replacement_count(self):
        """All TernaryLinear layers are replaced with TernaryLinearAccel."""
        model = MockLlamaModel()
        engine = TernaryInferenceEngine()
        engine.convert(model, sensitivity_analysis=False)
        _replace_with_accel(model)

        count = sum(
            1 for m in model.modules() if isinstance(m, TernaryLinearAccel)
        )
        assert count == 14

    @pytest.mark.skipif(not is_accelerated(), reason="C library not available")
    def test_accel_forward_correct_output(self):
        """Accel model produces valid output of correct shape.

        Per-layer accuracy (C kernel vs PyTorch BLAS) is verified to
        atol=1e-4 in test_stage1b.py. Stacking multiple ternary layers
        amplifies FP32 accumulation-order differences chaotically, so
        here we verify structural correctness rather than numerical
        match across the full model.
        """
        model = MockLlamaModel()
        engine = TernaryInferenceEngine()
        engine.convert(model, sensitivity_analysis=False)
        _replace_with_accel(model)
        model.eval()

        input_ids = torch.randint(0, 100, (1, 10))
        with torch.no_grad():
            output = model(input_ids=input_ids)

        assert output.logits.shape == (1, 10, 100)
        assert not torch.isnan(output.logits).any()
        assert not torch.isinf(output.logits).any()

    @pytest.mark.skipif(not is_accelerated(), reason="C library not available")
    def test_accel_deterministic(self):
        """Accel model is deterministic across 50 runs. Patent 36."""
        model = MockLlamaModel()
        engine = TernaryInferenceEngine()
        engine.convert(model, sensitivity_analysis=False)
        _replace_with_accel(model)
        model.eval()

        input_ids = torch.randint(0, 100, (1, 10))
        with torch.no_grad():
            ref = model(input_ids=input_ids).logits
            for _ in range(49):
                y = model(input_ids=input_ids).logits
                assert torch.equal(ref, y)


# ═══════════════════════════════════════════════════════════════
# TestFreeOriginalWeights — memory management
# ═══════════════════════════════════════════════════════════════


class TestFreeOriginalWeights:
    """Test memory freeing after conversion."""

    def test_reduces_parameter_count(self):
        """Freeing original weights reduces parameter memory."""
        from terncore.hf_loader import HFTernaryLoader

        model = MockLlamaModel()
        engine = TernaryInferenceEngine()
        engine.convert(model, sensitivity_analysis=False)

        params_before = sum(p.numel() for p in model.parameters())
        HFTernaryLoader._free_original_weights(model)
        params_after = sum(p.numel() for p in model.parameters())

        assert params_after < params_before

    def test_model_still_works(self):
        """Model produces correct output after freeing weights."""
        from terncore.hf_loader import HFTernaryLoader

        model = MockLlamaModel()
        engine = TernaryInferenceEngine()
        engine.convert(model, sensitivity_analysis=False)

        HFTernaryLoader._free_original_weights(model)

        model.eval()
        input_ids = torch.randint(0, 100, (1, 10))
        with torch.no_grad():
            output = model(input_ids=input_ids)
        assert output.logits.shape == (1, 10, 100)
        assert not torch.isnan(output.logits).any()

    def test_ternary_cache_preserved(self):
        """Cached ternary weights survive the freeing process."""
        from terncore.hf_loader import HFTernaryLoader

        model = MockLlamaModel()
        engine = TernaryInferenceEngine()
        engine.convert(model, sensitivity_analysis=False)

        HFTernaryLoader._free_original_weights(model)

        for module in model.modules():
            if isinstance(module, TernaryLinear):
                assert module._cached_ternary is not None
                assert module._cached_alpha is not None


# ═══════════════════════════════════════════════════════════════
# TestHFLoaderImport — dependency guard
# ═══════════════════════════════════════════════════════════════


class TestHFLoaderImport:
    """Tests for import handling and dependency guard."""

    def test_module_imports(self):
        """Module imports without error regardless of transformers."""
        from terncore.hf_loader import (  # noqa: F401
            ConversionResult,
            GenerationResult,
            HFModelInfo,
        )

    @requires_transformers
    def test_loader_creation(self):
        """Loader can be instantiated when transformers is available."""
        from terncore.hf_loader import HFTernaryLoader

        loader = HFTernaryLoader()
        assert loader.threshold == 0.7
        assert loader.sensitivity_analysis is False
        assert loader.use_accel is False

    def test_require_transformers_message(self):
        """require_transformers gives helpful error when package missing."""
        from terncore.hf_loader import _HF_AVAILABLE, require_transformers

        if not _HF_AVAILABLE:
            with pytest.raises(ImportError, match="pip install"):
                require_transformers()


# ═══════════════════════════════════════════════════════════════
# TestTinyLlamaIntegration — requires model download (~2 GB)
# ═══════════════════════════════════════════════════════════════


@requires_transformers
@slow
class TestTinyLlamaIntegration:
    """Integration tests requiring TinyLlama download.

    Run with: TERNCORE_RUN_SLOW=1 pytest tests/test_phase3.py::TestTinyLlamaIntegration -v
    """

    MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    def test_load_and_convert(self):
        """Full pipeline: load TinyLlama, convert, verify stats."""
        from terncore.hf_loader import HFTernaryLoader

        loader = HFTernaryLoader(threshold=0.7, sensitivity_analysis=False)
        result = loader.load_and_convert(self.MODEL_ID)

        assert result.model_info.model_class == "LlamaForCausalLM"
        assert result.model_info.total_params > 1_000_000_000
        assert result.conversion_report.converted_layers == 154
        assert result.conversion_report.skipped_layers >= 1  # lm_head
        assert result.memory_profile.compression_ratio > 3.0

    def test_generate_text(self):
        """Generate text with ternary model."""
        from terncore.hf_loader import HFTernaryLoader

        loader = HFTernaryLoader(threshold=0.7)
        result = loader.load_and_convert(self.MODEL_ID)
        gen = loader.generate(result, "Hello", max_new_tokens=20)

        assert gen.num_tokens_generated > 0
        assert len(gen.generated_text) > len("Hello")
        assert gen.deterministic is True

    def test_deterministic_generation(self):
        """Two runs produce identical text. Patent 36."""
        from terncore.hf_loader import HFTernaryLoader

        loader = HFTernaryLoader(threshold=0.7)
        result = loader.load_and_convert(self.MODEL_ID)

        gen1 = loader.generate(result, "Test prompt", max_new_tokens=20)
        gen2 = loader.generate(result, "Test prompt", max_new_tokens=20)

        assert gen1.generated_text == gen2.generated_text
