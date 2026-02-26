"""
Ternary weight quantiser.

Patent 1: Ternary weight encoding {-1, 0, +1} on binary hardware.
Patent 4: Progressive compression with sensitivity analysis.
Patent 7: Sparsity optimisation and zero-weight identification.

Every floating-point weight is mapped to one of three states:
    +1  →  excitatory (biological: EPSP, depolarisation)
    -1  →  inhibitory (biological: IPSP, hyperpolarisation)
     0  →  resting    (biological: resting membrane potential)

This is not a metaphor. It is structural homology with biological synapses.
See Patent 36 (CNS biological architecture) for the full correspondence.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class QuantisationStats:
    """Statistics from a single quantisation pass."""

    sparsity: float          # fraction of weights that are zero
    positive_frac: float     # fraction of weights that are +1
    negative_frac: float     # fraction of weights that are -1
    alpha: float             # learned scaling factor
    reconstruction_mse: float  # mean squared error vs original


class TernaryQuantizer:
    """
    Quantise floating-point weights to ternary {-1, 0, +1}.

    Patent 1, Claims 2, 9-10:
        W_ternary = α · T(W)
        where T(W)_ij = +1 if W_ij >  Δ
                         0 if |W_ij| ≤ Δ
                        -1 if W_ij < -Δ
        α = mean(|W|) for non-zero entries (learned scaling factor)
        Δ = threshold * mean(|W|) (adaptive threshold)

    The threshold parameter controls the width of the dead zone around zero.
    Higher threshold → more zeros → higher sparsity → smaller model → less accuracy.
    Lower threshold → fewer zeros → lower sparsity → larger model → more accuracy.

    Typical operating range: threshold ∈ [0.5, 0.9], default 0.7.
    """

    def __init__(self, threshold: float = 0.7) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"Threshold must be in (0, 1), got {threshold}")
        self.threshold = threshold

    def quantize(
        self, weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantise weights to ternary.

        Args:
            weights: Floating-point weight tensor of any shape.

        Returns:
            ternary: Tensor of same shape with values in {-1, 0, +1}.
            alpha:   Scalar scaling factor (mean absolute value of non-zero entries).
        """
        # Adaptive threshold based on weight magnitude distribution
        abs_w = torch.abs(weights)
        delta = self.threshold * torch.mean(abs_w)

        # Ternary assignment: compare-and-assign, no multiplication
        ternary = torch.where(
            weights > delta,
            torch.ones_like(weights),
            torch.where(
                weights < -delta,
                -torch.ones_like(weights),
                torch.zeros_like(weights),
            ),
        )

        # Scaling factor: mean absolute value of original weights at non-zero positions
        non_zero_mask = ternary != 0
        if non_zero_mask.any():
            alpha = torch.mean(abs_w[non_zero_mask])
        else:
            alpha = torch.mean(abs_w)

        return ternary, alpha

    def dequantize(
        self, ternary: torch.Tensor, alpha: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruct approximate floating-point weights from ternary.

        Args:
            ternary: Ternary weight tensor {-1, 0, +1}.
            alpha:   Scaling factor from quantize().

        Returns:
            Approximate floating-point weights: α · T(W).
        """
        return ternary * alpha

    def stats(
        self, weights: torch.Tensor
    ) -> QuantisationStats:
        """
        Quantise and return detailed statistics.

        Useful for sensitivity analysis (Patent 4): determine which layers
        tolerate ternary quantisation and which need protection.

        Args:
            weights: Floating-point weight tensor of any shape.

        Returns:
            ``QuantisationStats`` with sparsity, alpha, and reconstruction MSE.
        """
        ternary, alpha = self.quantize(weights)
        reconstructed = self.dequantize(ternary, alpha)
        mse = torch.mean((weights - reconstructed) ** 2).item()

        total = ternary.numel()
        zeros = (ternary == 0).sum().item()
        positives = (ternary == 1).sum().item()
        negatives = (ternary == -1).sum().item()

        return QuantisationStats(
            sparsity=zeros / total,
            positive_frac=positives / total,
            negative_frac=negatives / total,
            alpha=alpha.item(),
            reconstruction_mse=mse,
        )


class SensitivityAnalyzer:
    """
    Per-layer sensitivity analysis for ternary quantisation.

    Patent 4, Claims 1-3:
        Analyse each layer independently to determine quantisation tolerance.
        Layers with high reconstruction error are flagged as precision-critical
        and should remain in FP16 or use a lower threshold.

    Patent 6: Multi-level quantisation allows different thresholds per layer.
    """

    def __init__(
        self,
        thresholds: Optional[list[float]] = None,
        mse_ceiling: float = 0.01,
    ) -> None:
        """
        Args:
            thresholds: List of thresholds to evaluate. Default: [0.5, 0.6, 0.7, 0.8, 0.9].
            mse_ceiling: Maximum acceptable reconstruction MSE. Layers above
                         this are flagged as precision-critical.
        """
        self.thresholds = thresholds or [0.5, 0.6, 0.7, 0.8, 0.9]
        self.mse_ceiling = mse_ceiling

    def analyze_layer(
        self, name: str, weights: torch.Tensor
    ) -> dict:
        """
        Evaluate a single layer across multiple thresholds.

        Args:
            name: Layer name (for labelling in results).
            weights: FP32 weight tensor to analyse.

        Returns:
            Dict with layer name, best threshold, stats at each threshold,
            and whether the layer is precision-critical.
        """
        results = []
        for t in self.thresholds:
            q = TernaryQuantizer(threshold=t)
            s = q.stats(weights)
            results.append({"threshold": t, **vars(s)})

        # Find the highest threshold (most compression) that stays under MSE ceiling
        viable = [r for r in results if r["reconstruction_mse"] <= self.mse_ceiling]

        if viable:
            best = max(viable, key=lambda r: r["threshold"])
            precision_critical = False
        else:
            # No threshold works within ceiling — layer is precision-critical
            best = min(results, key=lambda r: r["reconstruction_mse"])
            precision_critical = True

        return {
            "name": name,
            "shape": list(weights.shape),
            "num_params": weights.numel(),
            "precision_critical": precision_critical,
            "recommended_threshold": best["threshold"],
            "results": results,
        }

    def analyze_model(self, model: nn.Module) -> list[dict]:
        """
        Analyse all Linear and Conv2d layers in a model.

        Args:
            model: PyTorch model to analyse.

        Returns:
            List of per-layer analysis dicts, sorted by reconstruction MSE
            (worst layers first).
        """
        analyses = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                analysis = self.analyze_layer(name, module.weight.data)
                analyses.append(analysis)

        # Sort: precision-critical first, then by MSE descending
        analyses.sort(
            key=lambda a: (
                not a["precision_critical"],
                -a["results"][-1]["reconstruction_mse"],
            )
        )
        return analyses
