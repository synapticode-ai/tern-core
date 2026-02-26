"""
Sparsity-aware memory management.

Patent 3: Sparsity-aware memory architecture.
Patent 8: Packed sparse format for NPU memory hierarchy.

CPU reference implementation. NPU memory hierarchy mapping added in Stage 1B.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass

__all__ = [
    "MemoryProfile",
    "profile_model_memory",
]


@dataclass
class MemoryProfile:
    """Memory usage profile for a ternary model."""

    total_params: int
    ternary_params: int
    fp16_params: int
    packed_bytes: int      # actual storage with 2-bit packing
    bitmap_bytes: int      # sparsity bitmap overhead
    fp16_bytes: int        # protected layers in FP16
    total_bytes: int       # total model size
    original_fp16_bytes: int  # original model in FP16
    compression_ratio: float


def profile_model_memory(model: torch.nn.Module) -> MemoryProfile:
    """
    Calculate memory profile for a (possibly converted) model.

    Counts TernaryLinear/TernaryConv2d params as 2-bit packed.
    Counts standard Linear/Conv2d params as FP16.
    """
    from terncore.arithmetic.linear import TernaryLinear, TernaryConv2d

    ternary_params = 0
    fp16_params = 0

    for module in model.modules():
        if isinstance(module, (TernaryLinear, TernaryConv2d)):
            ternary_params += module.weight.numel()
        elif isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            fp16_params += module.weight.numel()

    total_params = ternary_params + fp16_params
    packed_bytes = ternary_params // 4         # 2 bits per weight = 4 per byte
    bitmap_bytes = ternary_params // 8         # 1 bit per weight
    fp16_bytes = fp16_params * 2               # 16 bits per weight
    total_bytes = packed_bytes + bitmap_bytes + fp16_bytes
    original = total_params * 2                # everything in FP16

    return MemoryProfile(
        total_params=total_params,
        ternary_params=ternary_params,
        fp16_params=fp16_params,
        packed_bytes=packed_bytes,
        bitmap_bytes=bitmap_bytes,
        fp16_bytes=fp16_bytes,
        total_bytes=total_bytes,
        original_fp16_bytes=original,
        compression_ratio=original / total_bytes if total_bytes > 0 else 0,
    )
