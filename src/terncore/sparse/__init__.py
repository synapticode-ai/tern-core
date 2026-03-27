"""
Sparsity optimisation: bitmap generation and zero-skip execution.

Patent 7: Sparsity-aware execution — skip computation for zero weights.
Patent 8: Packed sparse format for memory-efficient storage.
Patent 9: Hierarchical sparsity across model layers.

Typical ternary models have 60-70% zero weights. Without zero-skip,
we waste 60-70% of compute on multiplying by zero. The sparsity bitmap
tells the engine exactly which weights to skip.

2-bit encoding with sparsity bitmap:
    Weight storage: 2 bits per weight (01=+1, 10=-1, 00=0, 11=reserved)
    Bitmap storage: 1 bit per weight (1=non-zero, 0=zero/skip)
    Total: 3 bits per weight effective, but skip 60-70% of compute.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Tuple

__all__ = [
    "SparsityInfo",
    "generate_sparsity_bitmap",
    "pack_ternary_weights",
    "unpack_ternary_weights",
    "analyze_block_sparsity",
    "model_sparsity_report",
    "sparsity_info",
    # Channel pruning (structured sparsity for ANE)
    "ChannelPruneStats",
    "score_channel_importance",
    "detect_prunable_channels",
    "prune_linear_output",
    "prune_linear_input",
    "prune_mlp_channels",
    "prune_attention_channels",
]

from terncore.sparse.channel_pruning import (  # noqa: E402
    ChannelPruneStats,
    score_channel_importance,
    detect_prunable_channels,
    prune_linear_output,
    prune_linear_input,
    prune_mlp_channels,
    prune_attention_channels,
)


@dataclass
class SparsityInfo:
    """Sparsity statistics for a weight tensor."""

    total_weights: int
    zero_weights: int
    nonzero_weights: int
    sparsity_ratio: float
    memory_saved_bytes: int  # vs dense FP16 storage


def generate_sparsity_bitmap(ternary_weights: torch.Tensor) -> torch.Tensor:
    """
    Generate a boolean bitmap indicating non-zero weight positions.

    Patent 7, Claim 1: Sparsity bitmap for zero-skip execution.

    Args:
        ternary_weights: Tensor with values in {-1, 0, +1}.

    Returns:
        Boolean tensor of same shape. True = non-zero (compute needed).
        False = zero (skip).
    """
    return ternary_weights != 0


def pack_ternary_weights(
    ternary_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pack ternary weights into 2-bit encoding with sparsity bitmap.

    Patent 2: 2-bit encoding scheme.
    Patent 8: Packed sparse format.

    Encoding:
        +1 → 0b01
        -1 → 0b10
         0 → 0b00

    Packs 4 weights per byte. Combined with bitmap for zero-skip.

    Args:
        ternary_weights: Tensor with values in {-1, 0, +1}.

    Returns:
        packed: Uint8 tensor with 4 weights per byte.
        bitmap: Boolean tensor for zero-skip.
    """
    flat = ternary_weights.flatten().to(torch.int8)
    bitmap = generate_sparsity_bitmap(ternary_weights)

    # Map: +1→1, -1→2, 0→0
    encoded = torch.where(
        flat == 1,
        torch.tensor(1, dtype=torch.uint8),
        torch.where(
            flat == -1,
            torch.tensor(2, dtype=torch.uint8),
            torch.tensor(0, dtype=torch.uint8),
        ),
    )

    # Pad to multiple of 4
    pad_len = (4 - len(encoded) % 4) % 4
    if pad_len > 0:
        encoded = torch.cat([encoded, torch.zeros(pad_len, dtype=torch.uint8)])

    # Pack 4 weights per byte: w0 in bits 0-1, w1 in bits 2-3, etc.
    reshaped = encoded.reshape(-1, 4)
    packed = (
        reshaped[:, 0]
        | (reshaped[:, 1] << 2)
        | (reshaped[:, 2] << 4)
        | (reshaped[:, 3] << 6)
    )

    return packed, bitmap


def unpack_ternary_weights(
    packed: torch.Tensor,
    original_shape: torch.Size,
) -> torch.Tensor:
    """
    Unpack 2-bit encoded weights back to ternary {-1, 0, +1}.

    Args:
        packed:         Uint8 tensor from pack_ternary_weights.
        original_shape: Shape of the original weight tensor.

    Returns:
        Ternary tensor with original shape.
    """
    total = 1
    for s in original_shape:
        total *= s

    # Extract 4 weights per byte
    w0 = packed & 0x03
    w1 = (packed >> 2) & 0x03
    w2 = (packed >> 4) & 0x03
    w3 = (packed >> 6) & 0x03

    interleaved = torch.stack([w0, w1, w2, w3], dim=1).flatten()[:total]

    # Decode: 1→+1, 2→-1, 0→0
    ternary = torch.where(
        interleaved == 1,
        torch.tensor(1, dtype=torch.float32),
        torch.where(
            interleaved == 2,
            torch.tensor(-1, dtype=torch.float32),
            torch.tensor(0, dtype=torch.float32),
        ),
    )

    return ternary.reshape(original_shape)


def analyze_block_sparsity(
    packed_weights: torch.Tensor,
    out_features: int,
    in_features: int,
    block_size: int = 256,
) -> dict:
    """
    Analyze sparsity at the block level for packed ternary weights.

    Determines what fraction of fixed-size weight blocks are entirely
    zero — these blocks can be skipped completely by hardware zero-skip.

    Patent 7: Block-level sparsity analysis for zero-skip optimisation.
    Patent 9: Hierarchical sparsity across weight blocks.

    Args:
        packed_weights: Uint8 tensor with 2-bit packed weights.
        out_features:   Number of output features.
        in_features:    Number of input features.
        block_size:     Number of weights per block (default 256).

    Returns:
        Dict with total_weights, zero_weights, sparsity, total_blocks,
        zero_blocks, block_skip_ratio, block_sparsity_histogram.
    """
    shape = torch.Size([out_features, in_features])
    ternary = unpack_ternary_weights(packed_weights, shape)
    flat = ternary.flatten()

    total_weights = flat.numel()
    zero_weights = (flat == 0).sum().item()
    sparsity = zero_weights / total_weights if total_weights > 0 else 0.0

    # Block-level analysis
    n_blocks = (total_weights + block_size - 1) // block_size
    zero_blocks = 0
    block_sparsities = []

    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, total_weights)
        block = flat[start:end]
        block_zeros = (block == 0).sum().item()
        block_sparsity = block_zeros / block.numel()
        block_sparsities.append(block_sparsity)
        if block_zeros == block.numel():
            zero_blocks += 1

    # Histogram: bucket block sparsities into 10 bins
    histogram = [0] * 10
    for bs in block_sparsities:
        bin_idx = min(int(bs * 10), 9)
        histogram[bin_idx] += 1

    return {
        "total_weights": total_weights,
        "zero_weights": int(zero_weights),
        "sparsity": round(sparsity, 4),
        "total_blocks": n_blocks,
        "zero_blocks": zero_blocks,
        "block_skip_ratio": round(
            zero_blocks / n_blocks if n_blocks > 0 else 0.0, 4
        ),
        "block_size": block_size,
        "block_sparsity_histogram": histogram,
        "mean_block_sparsity": round(
            sum(block_sparsities) / len(block_sparsities)
            if block_sparsities else 0.0,
            4,
        ),
    }


def model_sparsity_report(model) -> list[dict]:
    """
    Generate sparsity report for all PackedTernaryLinear layers in a model.

    Patent 9: Hierarchical sparsity across model layers.

    Args:
        model: PyTorch model with PackedTernaryLinear layers.

    Returns:
        List of per-layer sparsity analysis dicts.
    """
    from terncore.packed_linear import PackedTernaryLinear

    report = []
    for name, module in model.named_modules():
        if isinstance(module, PackedTernaryLinear):
            analysis = analyze_block_sparsity(
                module.packed_weights,
                module.out_features,
                module.in_features,
            )
            analysis["name"] = name
            report.append(analysis)
    return report


def sparsity_info(ternary_weights: torch.Tensor) -> SparsityInfo:
    """Calculate sparsity statistics for a ternary weight tensor.

    Args:
        ternary_weights: Tensor with values in {-1, 0, +1}.

    Returns:
        ``SparsityInfo`` with total/zero/nonzero counts, sparsity ratio,
        and estimated memory savings vs FP16.
    """
    total = ternary_weights.numel()
    zeros = (ternary_weights == 0).sum().item()
    nonzeros = total - zeros

    # FP16 dense: 2 bytes/weight
    # Ternary packed: 0.25 bytes/weight + bitmap overhead
    fp16_bytes = total * 2
    ternary_bytes = total // 4 + total // 8  # packed + bitmap
    saved = fp16_bytes - ternary_bytes

    return SparsityInfo(
        total_weights=total,
        zero_weights=int(zeros),
        nonzero_weights=int(nonzeros),
        sparsity_ratio=zeros / total,
        memory_saved_bytes=int(saved),
    )
