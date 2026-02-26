"""
Straight-Through Estimator (STE) for ternary quantisation-aware training.

Patent 1: Ternary weight encoding {-1, 0, +1}.
Patent 36: Biological neural mapping — STE mimics biological synaptic
           plasticity where discrete synaptic states are trained via
           continuous gradient signals.

The STE trick: in the forward pass, weights are discretised to {-1, 0, +1}
and scaled by alpha. In the backward pass, gradients pass through the
quantisation step as if it were the identity function. This allows
standard optimisers (SGD, Adam) to update the underlying FP32 "latent"
weights, which are re-quantised on the next forward pass.

Key guarantee: STEQuantize.forward produces IDENTICAL ternary weights
and alpha to TernaryQuantizer.quantize(). This ensures that a model
trained with STE will behave identically at inference time when converted
to TernaryLinear.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from terncore.arithmetic.quantizer import TernaryQuantizer
from terncore.arithmetic.linear import TernaryLinear


class STEQuantize(torch.autograd.Function):
    """
    Autograd function for ternary quantisation with straight-through estimator.

    Forward: quantise weights to ternary {-1, 0, +1}, scale by alpha.
             Output = ternary * alpha (same as TernaryQuantizer.dequantize).
    Backward: pass gradient through unchanged (identity).

    This is a custom autograd.Function because torch.where blocks gradients
    through the non-selected branch. The STE requires gradients to flow to
    the original FP32 weights regardless of which ternary bucket they fell into.
    """

    @staticmethod
    def forward(ctx, weights: torch.Tensor, threshold: float) -> torch.Tensor:
        """Quantise weights to ternary and scale by alpha.

        Produces identical output to ``TernaryQuantizer.quantize()`` followed
        by ``dequantize()`` — this is a hard requirement for training/inference
        consistency.

        Args:
            ctx: Autograd context (used to save tensors for backward).
            weights: FP32 latent weight tensor of any shape.
            threshold: Quantisation threshold in (0, 1).

        Returns:
            Dequantised tensor (ternary * alpha), same shape as weights.
        """
        # Exact same computation as TernaryQuantizer.quantize()
        abs_w = torch.abs(weights)
        delta = threshold * torch.mean(abs_w)

        ternary = torch.where(
            weights > delta,
            torch.ones_like(weights),
            torch.where(
                weights < -delta,
                -torch.ones_like(weights),
                torch.zeros_like(weights),
            ),
        )

        non_zero_mask = ternary != 0
        if non_zero_mask.any():
            alpha = torch.mean(abs_w[non_zero_mask])
        else:
            alpha = torch.mean(abs_w)

        # Save for potential use in backward (not needed for basic STE)
        ctx.save_for_backward(weights)

        return ternary * alpha

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Straight-through estimator: pass gradient unchanged.

        Args:
            ctx: Autograd context.
            grad_output: Gradient from downstream layers.

        Returns:
            Tuple of (grad_weights, None) — None for the threshold
            argument which is not a tensor and has no gradient.
        """
        return grad_output, None


class TernaryLinearSTE(nn.Module):
    """
    Linear layer with STE-based ternary quantisation for training.

    Maintains FP32 latent weights that are quantised to ternary on every
    forward pass. The optimizer updates the latent weights; STE ensures
    gradients flow through the discrete quantisation step.

    This is the training counterpart to TernaryLinear (inference).
    After QAT, call to_ternary_linear() to get a frozen inference layer.

    Args:
        in_features:  Size of each input sample.
        out_features: Size of each output sample.
        bias:         If True, adds a learnable bias (kept in FP32).
        threshold:    Quantisation threshold. Default 0.7.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        threshold: float = 0.7,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold

        # FP32 latent weights — the optimizer updates these
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: quantise weights via STE, then linear transform.

        Args:
            x: Input tensor of shape ``(*, in_features)``.

        Returns:
            Output tensor of shape ``(*, out_features)``.
        """
        q_weight = STEQuantize.apply(self.weight, self.threshold)
        return F.linear(x, q_weight, self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear, threshold: float = 0.7) -> TernaryLinearSTE:
        """
        Create from an existing nn.Linear, copying weights and bias.

        Use this to convert a pretrained model's layers for QAT.

        Args:
            linear: Source ``nn.Linear`` layer.
            threshold: Quantisation threshold (default 0.7).

        Returns:
            New ``TernaryLinearSTE`` with copied weights.
        """
        has_bias = linear.bias is not None
        ste = cls(
            linear.in_features,
            linear.out_features,
            bias=has_bias,
            threshold=threshold,
        )
        ste.weight.data.copy_(linear.weight.data)
        if has_bias:
            ste.bias.data.copy_(linear.bias.data)
        return ste

    def to_ternary_linear(self) -> TernaryLinear:
        """
        Convert to inference TernaryLinear with frozen ternary weights.

        Call this after QAT is complete. The resulting TernaryLinear will
        use cached ternary weights for fast inference.

        Returns:
            Frozen ``TernaryLinear`` with pre-cached ternary weights.
        """
        tl = TernaryLinear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            threshold=self.threshold,
        )
        tl.weight.data.copy_(self.weight.data)
        if self.bias is not None:
            tl.bias.data.copy_(self.bias.data)
        tl.eval()
        tl._cache_ternary_weights()
        return tl

    def verify_quantizer_match(self) -> bool:
        """
        Verify that STE forward produces identical output to TernaryQuantizer.

        Returns:
            True if the STE-quantised weights are bit-identical to
            ``TernaryQuantizer.quantize()`` output.
        """
        q = TernaryQuantizer(threshold=self.threshold)
        ternary, alpha = q.quantize(self.weight.data)
        expected = ternary * alpha

        with torch.no_grad():
            actual = STEQuantize.apply(self.weight.data, self.threshold)

        return torch.allclose(expected, actual, atol=0.0, rtol=0.0)

    @property
    def sparsity(self) -> float:
        """Fraction of weights that are currently zero after quantisation.

        Returns:
            Float in [0, 1] representing the zero-weight ratio.
        """
        with torch.no_grad():
            abs_w = torch.abs(self.weight.data)
            delta = self.threshold * torch.mean(abs_w)
            zeros = ((self.weight.data >= -delta) & (self.weight.data <= delta)).sum()
            return zeros.item() / self.weight.numel()

    def extra_repr(self) -> str:
        """Return string representation for ``print(module)``."""
        s = f"in_features={self.in_features}, out_features={self.out_features}"
        s += f", bias={self.bias is not None}"
        s += f", threshold={self.threshold}"
        return s
