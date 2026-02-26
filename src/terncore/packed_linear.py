"""
PackedTernaryLinear — linear layer with 2-bit packed weight storage.

Patent 1: Ternary weight encoding {-1, 0, +1}.
Patent 5: Packed ternary execution path.
Patent 7: Sparsity-aware execution — cached bitmap for zero-skip.
Patent 39: Ternary-native memory — 2-bit packed format (4 weights/byte).

Memory comparison (1M weights):
    float32:  4,000,000 bytes (4 B/weight)
    int8:     1,000,000 bytes (1 B/weight)  — TernaryLinear cached
    2-bit:      250,000 bytes (0.25 B/weight) — PackedTernaryLinear

Forward path: packed weights + cached bitmap → C kernel with zero-skip.
Fallback: unpack 2-bit → ternary float → F.linear.

Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from terncore.arithmetic.quantizer import TernaryQuantizer
from terncore.packed_ops import packed_ternary_matmul, packed_ternary_matmul_fast
from terncore.sparse import pack_ternary_weights, unpack_ternary_weights


def _build_bitmap_from_packed(
    packed_weights: torch.Tensor,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """Build sparsity bitmap (1 bit per weight, LSB-first, uint8-packed).

    Patent 7: Sparsity bitmap for zero-skip execution.
    """
    import numpy as np

    shape = torch.Size([out_features, in_features])
    ternary = unpack_ternary_weights(packed_weights, shape)
    bitmap_bool = (ternary.flatten() != 0).numpy().astype(np.uint8)
    bitmap_np = np.packbits(bitmap_bool, bitorder="little")
    return torch.from_numpy(bitmap_np.copy())


class PackedTernaryLinear(nn.Module):
    """
    Linear layer that stores weights in 2-bit packed format.

    Memory: 2 bits/weight + alpha (float32) + bitmap (1 bit/weight)
    vs 32 bits/weight for float32.  That's ~10.7x effective compression.

    The sparsity bitmap is cached at construction time so the C kernel
    can skip zero-weight blocks without per-call bitmap rebuilds.

    Inference-only: no gradients through packed weights.

    Args:
        in_features:  Input feature dimension.
        out_features: Output feature dimension.
        bias:         Whether to include bias.
        alpha:        Per-layer scaling factor.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Packed weights: 4 weights per byte
        n_packed_bytes = (in_features * out_features + 3) // 4
        self.register_buffer(
            "packed_weights", torch.zeros(n_packed_bytes, dtype=torch.uint8)
        )
        self.register_buffer(
            "alpha", torch.tensor(alpha, dtype=torch.float32)
        )

        # Cached sparsity bitmap: 1 bit per weight, LSB-first (Patent 7)
        n_bitmap_bytes = (in_features * out_features + 7) // 8
        self.register_buffer(
            "sparsity_bitmap", torch.zeros(n_bitmap_bytes, dtype=torch.uint8)
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_float(
        cls,
        linear: nn.Linear,
        threshold: float = 0.7,
    ) -> "PackedTernaryLinear":
        """
        Convert a standard nn.Linear to PackedTernaryLinear.

        1. Quantise weights: float32 → ternary {-1, 0, +1} + alpha
        2. Pack ternary → 2-bit encoding
        3. Store packed weights + alpha

        Args:
            linear:    Source nn.Linear module.
            threshold: Quantisation threshold.

        Returns:
            PackedTernaryLinear with packed weights.
        """
        q = TernaryQuantizer(threshold=threshold)
        ternary, alpha = q.quantize(linear.weight.data)
        packed, _bitmap = pack_ternary_weights(ternary)

        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            alpha=alpha.item(),
        )
        layer.packed_weights.copy_(packed)

        # Build and cache sparsity bitmap (Patent 7)
        bitmap = _build_bitmap_from_packed(
            packed, linear.out_features, linear.in_features
        )
        layer.sparsity_bitmap.copy_(bitmap)

        if linear.bias is not None:
            layer.bias.data.copy_(linear.bias.data)

        return layer

    @classmethod
    def from_ternary_linear(cls, ternary_linear) -> "PackedTernaryLinear":
        """
        Convert an existing TernaryLinear to PackedTernaryLinear.

        TernaryLinear already has quantised weights — just pack them.

        Args:
            ternary_linear: Source TernaryLinear module.

        Returns:
            PackedTernaryLinear with packed weights.
        """
        from terncore.arithmetic.linear import TernaryLinear

        if not isinstance(ternary_linear, TernaryLinear):
            raise TypeError(f"Expected TernaryLinear, got {type(ternary_linear)}")

        # Get the quantised weights (triggers caching if needed)
        q = TernaryQuantizer(threshold=ternary_linear.threshold)
        ternary, alpha = q.quantize(ternary_linear.weight.data)
        packed, _bitmap = pack_ternary_weights(ternary)

        layer = cls(
            in_features=ternary_linear.in_features,
            out_features=ternary_linear.out_features,
            bias=ternary_linear.bias is not None,
            alpha=alpha.item(),
        )
        layer.packed_weights.copy_(packed)

        # Build and cache sparsity bitmap (Patent 7)
        bitmap = _build_bitmap_from_packed(
            packed, ternary_linear.out_features, ternary_linear.in_features
        )
        layer.sparsity_bitmap.copy_(bitmap)

        if ternary_linear.bias is not None:
            layer.bias.data.copy_(ternary_linear.bias.data)

        return layer

    @classmethod
    def from_packed_data(
        cls,
        packed_weights: torch.Tensor,
        alpha: float,
        in_features: int,
        out_features: int,
        bias: Optional[torch.Tensor] = None,
        sparsity_bitmap: Optional[torch.Tensor] = None,
    ) -> "PackedTernaryLinear":
        """
        Create from pre-packed data (e.g. loaded from .tern-model).

        This is the fast path: no re-quantisation, no re-packing.
        Just assign the packed bytes directly.

        If sparsity_bitmap is provided, uses it directly (zero overhead).
        If not provided, generates it from packed_weights (one-time cost).

        Args:
            packed_weights:  Uint8 tensor with 2-bit packed weights.
            alpha:           Per-layer scaling factor.
            in_features:     Input feature dimension.
            out_features:    Output feature dimension.
            bias:            Optional bias tensor.
            sparsity_bitmap: Optional pre-built sparsity bitmap.

        Returns:
            PackedTernaryLinear ready for inference.
        """
        layer = cls(
            in_features=in_features,
            out_features=out_features,
            bias=bias is not None,
            alpha=alpha,
        )
        layer.packed_weights.copy_(packed_weights)

        # Use provided bitmap or generate from packed weights
        if sparsity_bitmap is not None:
            layer.sparsity_bitmap.copy_(sparsity_bitmap)
        else:
            bitmap = _build_bitmap_from_packed(
                packed_weights, out_features, in_features
            )
            layer.sparsity_bitmap.copy_(bitmap)

        if bias is not None:
            layer.bias.data.copy_(bias)

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using packed weights with cached bitmap zero-skip.

        Uses the C kernel fast path with pre-built sparsity bitmap when
        available. Falls back to unpack → float → F.linear.

        Patent 7: Cached bitmap eliminates per-call bitmap rebuild.
        Patent 9: Zero-skip via bitmap-driven sparse kernel.

        Args:
            x: Input tensor (..., in_features).

        Returns:
            Output tensor (..., out_features).
        """
        output = packed_ternary_matmul_fast(
            x,
            self.packed_weights,
            self.alpha.item(),
            self.out_features,
            self.in_features,
            sparsity_bitmap=self.sparsity_bitmap,
        )
        if self.bias is not None:
            output = output + self.bias
        return output

    def memory_footprint(self) -> dict:
        """
        Report memory usage vs equivalent float32 linear.

        Returns:
            Dict with packed_bytes, float32_bytes, compression_ratio, etc.
        """
        packed_bytes = self.packed_weights.nelement()
        bitmap_bytes = self.sparsity_bitmap.nelement()
        float32_bytes = self.in_features * self.out_features * 4
        bias_bytes = self.out_features * 4 if self.bias is not None else 0
        total_packed = packed_bytes + 4 + bitmap_bytes + bias_bytes
        return {
            "packed_bytes": packed_bytes,
            "bitmap_bytes": bitmap_bytes,
            "float32_bytes": float32_bytes,
            "compression_ratio": float32_bytes / packed_bytes if packed_bytes > 0 else 0,
            "alpha_bytes": 4,
            "bias_bytes": bias_bytes,
            "total_packed_bytes": total_packed,
            "total_float32_bytes": float32_bytes + bias_bytes,
        }

    def extra_repr(self) -> str:
        """Return string representation for ``print(module)``."""
        s = f"in_features={self.in_features}, out_features={self.out_features}"
        s += f", bias={self.bias is not None}"
        fp = self.memory_footprint()
        s += f", packed_bytes={fp['packed_bytes']}"
        s += f", compression={fp['compression_ratio']:.0f}x"
        return s


def convert_model_to_packed(
    model: nn.Module,
    threshold: float = 0.7,
    protection_list: Optional[list[str]] = None,
) -> dict:
    """
    Convert a model's Linear/TernaryLinear layers to PackedTernaryLinear.

    Layers in protection_list stay as nn.Linear (FP16/FP32).
    TernaryLinear layers are packed directly.
    Other nn.Linear layers are quantised + packed.

    Args:
        model:           PyTorch model to convert in-place.
        threshold:       Quantisation threshold for nn.Linear layers.
        protection_list: Layer names to keep in original precision.

    Returns:
        Dict with conversion statistics.
    """
    from terncore.arithmetic.linear import TernaryLinear
    from terncore.engine.inference import TernaryInferenceEngine

    protected = set(protection_list or [])
    stats = {
        "packed_layers": 0,
        "protected_layers": 0,
        "total_layers": 0,
    }

    for name, module in list(model.named_modules()):
        if isinstance(module, TernaryLinear):
            packed = PackedTernaryLinear.from_ternary_linear(module)
            TernaryInferenceEngine._replace_module(model, name, packed)
            stats["packed_layers"] += 1
            stats["total_layers"] += 1
        elif isinstance(module, nn.Linear):
            stats["total_layers"] += 1
            if name in protected or _should_protect_default(name):
                stats["protected_layers"] += 1
                continue
            packed = PackedTernaryLinear.from_float(module, threshold=threshold)
            TernaryInferenceEngine._replace_module(model, name, packed)
            stats["packed_layers"] += 1

    return stats


def _should_protect_default(name: str) -> bool:
    """Check if a layer should be protected by default."""
    name_lower = name.lower()
    if "embed" in name_lower:
        return True
    if any(k in name_lower for k in ("layernorm", "layer_norm", "rmsnorm")):
        return True
    if any(k in name_lower for k in ("lm_head", "classifier")):
        return True
    return False
