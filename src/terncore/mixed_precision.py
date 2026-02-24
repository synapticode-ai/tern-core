"""
Mixed-precision ternary converter.

Converts a model to ternary while protecting specified layers in their
original precision (FP32/FP16).  Uses per-layer sensitivity analysis
results to determine which layers benefit most from protection.

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

    Usage:
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
    ) -> None:
        """
        Args:
            threshold:         Quantisation threshold for ternary layers.
            protection_list:   Layer names to keep in original precision.
            protect_embeddings: Keep embedding layers in FP16 (default True).
            protect_layernorm:  Keep LayerNorm/RMSNorm in FP32 (default True).
            protect_lm_head:    Keep final output projection in FP16 (default True).
        """
        self.threshold = threshold
        self.protection_list = set(protection_list or [])
        self.protect_embeddings = protect_embeddings
        self.protect_layernorm = protect_layernorm
        self.protect_lm_head = protect_lm_head

    def convert(self, model: nn.Module) -> ConversionReport:
        """
        Convert model to mixed-precision ternary in-place.

        Protected layers (from protection_list + default patterns) keep
        their original weights.  All other nn.Linear layers are replaced
        with TernaryLinear using the configured threshold.

        Uses TernaryInferenceEngine static methods for the actual
        Linear -> TernaryLinear conversion and module replacement.

        Args:
            model: PyTorch model to convert in-place.

        Returns:
            ConversionReport with layer counts, param stats, compression.
        """
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
