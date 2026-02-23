"""
HuggingFace model loader for ternary conversion.

Downloads pre-trained models from HuggingFace Hub, converts eligible
layers to ternary {-1, 0, +1} via TernaryInferenceEngine, and runs
deterministic text generation.

Requires optional dependencies: pip install terncore[transformers]

Patent 12: Auto binary-to-ternary conversion.
Patent 36: Deterministic execution guarantee.
Patent 40: Bandwidth optimisation — streaming ternary weight loader.

Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn

from terncore.arithmetic.linear import TernaryLinear
from terncore.engine.inference import (
    ConversionReport,
    TernaryInferenceEngine,
)
from terncore.memory import MemoryProfile, profile_model_memory

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Dependency guard
# ═══════════════════════════════════════════════════════════════

_HF_AVAILABLE = False
_HF_IMPORT_ERROR: Optional[str] = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _HF_AVAILABLE = True
except ImportError as e:
    _HF_IMPORT_ERROR = str(e)


def require_transformers() -> None:
    """Raise ImportError with install instructions if transformers is missing."""
    if not _HF_AVAILABLE:
        raise ImportError(
            "HuggingFace transformers is required for hf_loader. "
            "Install with: pip install terncore[transformers]\n"
            f"Original error: {_HF_IMPORT_ERROR}"
        )


# ═══════════════════════════════════════════════════════════════
# Result dataclasses
# ═══════════════════════════════════════════════════════════════


@dataclass
class HFModelInfo:
    """Information about a loaded HuggingFace model."""

    model_id: str
    model_class: str
    total_params: int
    num_linear_layers: int
    eligible_linear_layers: int
    protected_layers: list[str] = field(default_factory=list)
    vocab_size: int = 0
    hidden_size: int = 0
    num_layers: int = 0


@dataclass
class GenerationResult:
    """Result from text generation."""

    prompt: str
    generated_text: str
    num_tokens_generated: int
    prefill_ms: float
    decode_total_ms: float
    per_token_ms: float
    total_ms: float
    deterministic: bool = True


@dataclass
class ConversionResult:
    """Full result from loading and converting a HuggingFace model."""

    model_info: HFModelInfo
    conversion_report: ConversionReport
    memory_profile: MemoryProfile
    conversion_time_s: float
    model: nn.Module
    tokenizer: Any


# ═══════════════════════════════════════════════════════════════
# HFTernaryLoader
# ═══════════════════════════════════════════════════════════════


class HFTernaryLoader:
    """
    Load HuggingFace models and convert to ternary.

    Usage::

        loader = HFTernaryLoader()
        result = loader.load_and_convert("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        gen = loader.generate(result, "What is ternary computing?")
        print(gen.generated_text)

    Patent 12: Auto binary-to-ternary conversion.
    Patent 40: Streaming weight loader with bandwidth optimisation.
    """

    def __init__(
        self,
        threshold: float = 0.7,
        sensitivity_analysis: bool = False,
        use_accel: bool = False,
    ) -> None:
        """
        Args:
            threshold:            Quantisation threshold (0.5-0.9).
            sensitivity_analysis: Run per-layer sensitivity analysis (slower).
            use_accel:            Replace TernaryLinear with TernaryLinearAccel
                                  (C+SIMD kernels).
        """
        require_transformers()
        self.threshold = threshold
        self.sensitivity_analysis = sensitivity_analysis
        self.use_accel = use_accel
        self.engine = TernaryInferenceEngine(
            threshold=threshold,
            protect_embeddings=True,
            protect_layernorm=True,
            protect_lm_head=True,
        )

    def load_and_convert(
        self,
        model_id: str,
        torch_dtype: torch.dtype = torch.float32,
        trust_remote_code: bool = False,
    ) -> ConversionResult:
        """
        Load a HuggingFace model and convert to ternary.

        Args:
            model_id:           HuggingFace model ID or local path.
            torch_dtype:        Data type for loading (float32 recommended
                                for accurate quantisation thresholds).
            trust_remote_code:  Allow models with custom code.

        Returns:
            ConversionResult with converted model, tokenizer, and stats.
        """
        logger.info("Loading model: %s", model_id)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch_dtype,
            device_map="cpu",
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
        model.eval()

        model_info = self._gather_model_info(model, model_id)
        logger.info(
            "Loaded: %s params, %d eligible layers",
            f"{model_info.total_params:,}",
            model_info.eligible_linear_layers,
        )

        # Convert to ternary
        t0 = time.perf_counter()
        report = self.engine.convert(
            model, sensitivity_analysis=self.sensitivity_analysis
        )
        conversion_time = time.perf_counter() - t0

        logger.info(
            "Converted %d/%d layers in %.1fs, compression: %.1fx",
            report.converted_layers,
            report.total_layers,
            conversion_time,
            report.compression_ratio,
        )

        # Optionally upgrade to accelerated layers
        if self.use_accel:
            self._replace_with_accel(model)
            logger.info("Upgraded to TernaryLinearAccel (C+SIMD)")

        # Free original FP32 weight copies
        self._free_original_weights(model)

        mem = profile_model_memory(model)

        return ConversionResult(
            model_info=model_info,
            conversion_report=report,
            memory_profile=mem,
            conversion_time_s=conversion_time,
            model=model,
            tokenizer=tokenizer,
        )

    def _gather_model_info(
        self, model: nn.Module, model_id: str
    ) -> HFModelInfo:
        """Extract architecture information from a loaded model."""
        total_params = sum(p.numel() for p in model.parameters())

        linear_names: list[str] = []
        eligible_names: list[str] = []
        protected_names: list[str] = []

        empty_report = ConversionReport()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                linear_names.append(name)
                if self.engine._should_protect(name, module, empty_report):
                    protected_names.append(name)
                else:
                    eligible_names.append(name)

        config = getattr(model, "config", None)

        return HFModelInfo(
            model_id=model_id,
            model_class=type(model).__name__,
            total_params=total_params,
            num_linear_layers=len(linear_names),
            eligible_linear_layers=len(eligible_names),
            protected_layers=protected_names,
            vocab_size=getattr(config, "vocab_size", 0),
            hidden_size=getattr(config, "hidden_size", 0),
            num_layers=getattr(config, "num_hidden_layers", 0),
        )

    # ───────────────────────────────────────────────────────────
    # Text generation
    # ───────────────────────────────────────────────────────────

    @staticmethod
    def generate_text(
        model: nn.Module,
        tokenizer: Any,
        prompt: str,
        max_new_tokens: int = 50,
        seed: int = 0,
    ) -> GenerationResult:
        """
        Generate text with timing breakdown.

        Works with both FP16 baseline and ternary-converted models
        because TernaryLinear is a drop-in for nn.Linear.

        Uses greedy decoding (do_sample=False) for determinism (Patent 36).

        Args:
            model:           The model (FP16 or ternary).
            tokenizer:       HuggingFace tokenizer.
            prompt:          Input text.
            max_new_tokens:  Maximum tokens to generate.
            seed:            Random seed for determinism.

        Returns:
            GenerationResult with text and timing.
        """
        model.eval()
        torch.manual_seed(seed)

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]

        # Prefill: single forward pass on prompt tokens
        torch.manual_seed(seed)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids)
        prefill_ms = (time.perf_counter() - t0) * 1000

        # Full generation (includes its own internal prefill + decode)
        torch.manual_seed(seed)
        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                repetition_penalty=1.0,
            )
        total_ms = (time.perf_counter() - t0) * 1000

        num_generated = output_ids.shape[1] - prompt_len
        # Approximate decode time: total - prefill
        # Note: generate() does its own prefill internally, so this
        # slightly underestimates decode-only time.
        decode_ms = max(total_ms - prefill_ms, 0.0)
        per_token_ms = decode_ms / max(num_generated, 1)

        generated_text = tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        )

        return GenerationResult(
            prompt=prompt,
            generated_text=generated_text,
            num_tokens_generated=num_generated,
            prefill_ms=prefill_ms,
            decode_total_ms=decode_ms,
            per_token_ms=per_token_ms,
            total_ms=total_ms,
            deterministic=True,
        )

    def generate(
        self,
        result: ConversionResult,
        prompt: str,
        max_new_tokens: int = 50,
    ) -> GenerationResult:
        """Generate text using a converted model."""
        return self.generate_text(
            result.model, result.tokenizer, prompt, max_new_tokens
        )

    # ───────────────────────────────────────────────────────────
    # Memory management
    # ───────────────────────────────────────────────────────────

    @staticmethod
    def _free_original_weights(model: nn.Module) -> None:
        """
        Free original FP32 weight data from converted TernaryLinear layers.

        After conversion, TernaryLinear stores both the original weight
        (nn.Parameter, FP32) and the cached ternary values (buffer, FP32).
        For inference-only use, the original weight is unused. Replace it
        with a 1-element placeholder to free ~4 bytes/param.

        For 989M eligible params in TinyLlama, this saves ~3.7 GB.
        """
        for module in model.modules():
            if isinstance(module, TernaryLinear):
                module.eval()
                if module._cached_ternary is None:
                    module._cache_ternary_weights()

                device = module.weight.device
                module.weight = nn.Parameter(
                    torch.zeros(1, device=device), requires_grad=False
                )

        gc.collect()

    @staticmethod
    def _replace_with_accel(model: nn.Module) -> None:
        """Replace all TernaryLinear layers with TernaryLinearAccel."""
        from terncore.accel import TernaryLinearAccel

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


__all__ = [
    "HFTernaryLoader",
    "HFModelInfo",
    "GenerationResult",
    "ConversionResult",
    "require_transformers",
]
