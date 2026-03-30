"""
Mixed-precision ternary converter.

Converts a model to ternary while protecting specified layers in their
original precision (FP32/FP16).  Uses per-layer sensitivity analysis
results to determine which layers benefit most from protection.

When ``auto=True`` (the default) and no ``protection_list`` is given,
a perplexity-gated scan automatically determines which layers can be
safely converted.  Results are cached to ``~/.terncore/model_cache.json``
so repeat runs skip the scan.

Patent 4: Progressive Compression — iterative protection search for
          mixed-precision ternary/FP16 deployment.
Patent 12: Auto binary-to-ternary conversion.

Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
"""

from __future__ import annotations

import torch.nn as nn
from typing import Optional

from terncore.engine.inference import ConversionReport, TernaryInferenceEngine


class MixedPrecisionConverter:
    """
    Convert a model to ternary with per-layer protection.

    Wraps TernaryInferenceEngine's conversion logic but adds explicit
    layer protection based on sensitivity analysis results.  Protected
    layers stay in their original precision; all other eligible Linear
    layers are converted to ternary {-1, 0, +1}.

    Usage — automatic (recommended):
        converter = MixedPrecisionConverter(threshold=0.7)
        report = converter.convert(model, model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    Usage — explicit protection list:
        converter = MixedPrecisionConverter(
            threshold=0.7,
            protection_list=["model.layers.2.mlp.down_proj"],
        )
        report = converter.convert(model)

    Patent 4: Progressive Compression.
    Patent 12: Auto binary-to-ternary conversion.
    """

    def __init__(
        self,
        threshold: float = 0.7,
        protection_list: Optional[list[str]] = None,
        protect_embeddings: bool = True,
        protect_layernorm: bool = True,
        protect_lm_head: bool = True,
        auto: bool = True,
        ppl_headroom: float = 0.20,
    ) -> None:
        """
        Args:
            threshold:         Quantisation threshold for ternary layers.
            protection_list:   Layer names to keep in original precision.
                               When provided, auto-scan is skipped.
            protect_embeddings: Keep embedding layers in FP16 (default True).
            protect_layernorm:  Keep LayerNorm/RMSNorm in FP32 (default True).
            protect_lm_head:    Keep final output projection in FP16 (default True).
            auto:              Run a perplexity-gated scan to find safe layers
                               automatically when no protection_list is given
                               (default True).
            ppl_headroom:      Maximum allowed PPL increase as a fraction when
                               auto-scanning (default 0.20 = 20%).
        """
        self.threshold = threshold
        self.protection_list = set(protection_list) if protection_list is not None else None
        self.protect_embeddings = protect_embeddings
        self.protect_layernorm = protect_layernorm
        self.protect_lm_head = protect_lm_head
        self.auto = auto
        self.ppl_headroom = ppl_headroom
        self._scan_result = None
        # Track whether an explicit list was provided
        self._explicit_protection = protection_list is not None

    @property
    def scan_result(self):
        """The ScanResult from the last auto-scan, or None."""
        return self._scan_result

    def convert(
        self,
        model: nn.Module,
        model_id: Optional[str] = None,
    ) -> ConversionReport:
        """
        Convert model to mixed-precision ternary in-place.

        Protected layers (from protection_list, auto-scan, or default
        patterns) keep their original weights.  All other nn.Linear
        layers are replaced with TernaryLinear.

        Args:
            model:    PyTorch model to convert in-place.
            model_id: HuggingFace model ID (required for auto-scan,
                      ignored when an explicit protection_list was given).

        Returns:
            ConversionReport with layer counts, param stats, compression.
        """
        # Auto-scan if no explicit protection list was provided
        if not self._explicit_protection and self.auto and model_id is not None:
            from terncore.autoscan import auto_scan
            scan = auto_scan(
                model_id,
                threshold=self.threshold,
                ppl_headroom=self.ppl_headroom,
            )
            self._scan_result = scan
            self.protection_list = set(scan.protection_list)

        # Ensure protection_list is a set (may still be None if auto was off
        # and nothing was provided)
        if self.protection_list is None:
            self.protection_list = set()

        report = ConversionReport()
        report.precision_critical_layers = sorted(self.protection_list)

        for name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue

            report.total_layers += 1
            report.total_params += module.weight.numel()

            if self._should_protect(name):
                report.skipped_layers += 1
                continue

            ternary_layer = TernaryInferenceEngine._convert_linear(
                module, self.threshold
            )
            TernaryInferenceEngine._replace_module(model, name, ternary_layer)

            report.converted_layers += 1
            report.ternary_params += module.weight.numel()

        # Size calculations (same formula as TernaryInferenceEngine.convert)
        report.original_size_mb = (report.total_params * 2) / (1024 * 1024)
        ternary_bytes = report.ternary_params * 0.25
        fp16_bytes = (report.total_params - report.ternary_params) * 2
        report.ternary_size_mb = (ternary_bytes + fp16_bytes) / (1024 * 1024)
        if report.ternary_size_mb > 0:
            report.compression_ratio = (
                report.original_size_mb / report.ternary_size_mb
            )

        return report

    def _should_protect(self, name: str) -> bool:
        """Check if a layer should be kept in original precision."""
        if name in self.protection_list:
            return True

        name_lower = name.lower()
        if self.protect_embeddings and "embed" in name_lower:
            return True
        if self.protect_layernorm and any(
            k in name_lower for k in ("layernorm", "layer_norm", "rmsnorm")
        ):
            return True
        if self.protect_lm_head and any(
            k in name_lower for k in ("lm_head", "output", "classifier")
        ):
            return True

        return False
