"""
Block-wise INT4 symmetric quantiser — CoreML-native format.

Quantises FP16/FP32 weights to signed 4-bit integers with per-block
FP16 scales.  The format is byte-identical to CoreML's
constexpr_blockwise_shift_scale op (iOS 18+), enabling direct export
without requantisation.

Format:
    - Signed int4, range [-7, 7] (symmetric, excludes -8)
    - Block size 32 (configurable): one FP16 scale per block
    - Packing: LSB-first, two values per byte, first in low nibble
    - Dequantisation: output = scale * data

Part of tern-core v0.6.0: mixed ternary/INT4 quantisation pipeline.

Copyright (c) 2025 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


DEFAULT_BLOCK_SIZE = 32
INT4_MIN = -7   # Symmetric: exclude -8
INT4_MAX = 7


@dataclass
class Int4QuantResult:
    """Result of INT4 block-wise quantisation."""

    packed_weights: bytes       # 2 values per byte, LSB-first
    scales: bytes               # FP16 scales, one per block
    weight_shape: list[int]     # Original weight shape
    scale_shape: list[int]      # Shape of the scale tensor
    block_size: int
    num_params: int
    reconstruction_error: float  # ||W - dequant(Q)||_F / ||W||_F


def quantize_int4_block(
    weights: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> Int4QuantResult:
    """Quantise a weight tensor to block-wise symmetric INT4.

    Args:
        weights:    2-D weight tensor (out_features, in_features).
        block_size: Number of elements per quantisation block along
                    the input dimension.  Default 32 (CoreML convention).

    Returns:
        Int4QuantResult with packed weights and per-block scales.
    """
    assert weights.ndim == 2, f"Expected 2-D tensor, got {weights.ndim}-D"
    w = weights.float()
    out_features, in_features = w.shape

    # Pad in_features to multiple of block_size
    pad = (block_size - in_features % block_size) % block_size
    if pad > 0:
        w = torch.nn.functional.pad(w, (0, pad))

    padded_in = w.shape[1]
    n_blocks = padded_in // block_size

    # Reshape to (out_features, n_blocks, block_size)
    w_blocked = w.reshape(out_features, n_blocks, block_size)

    # Per-block scale: max(|w|) / 7  (symmetric range [-7, 7])
    block_max = w_blocked.abs().amax(dim=-1)  # (out_features, n_blocks)
    scales = block_max / INT4_MAX
    # Avoid division by zero
    scales = scales.clamp(min=1e-10)

    # Quantise: round(w / scale), clamp to [-7, 7]
    scales_expanded = scales.unsqueeze(-1)  # (out, n_blocks, 1)
    q = torch.round(w_blocked / scales_expanded).clamp(INT4_MIN, INT4_MAX).to(torch.int8)

    # Dequantise for reconstruction error
    dequant = q.float() * scales_expanded
    w_orig_blocked = weights.float()
    if pad > 0:
        dequant = dequant.reshape(out_features, padded_in)[:, :in_features]
    else:
        dequant = dequant.reshape(out_features, in_features)
    w_norm = torch.norm(w_orig_blocked).item()
    error = torch.norm(w_orig_blocked - dequant).item() / w_norm if w_norm > 0 else 0.0

    # Pack to bytes — LSB-first, two int4 per byte, first in low nibble
    q_flat = q.reshape(-1).numpy().astype(np.int8)
    # Mask to 4 bits (two's complement)
    q_masked = q_flat & 0x0F

    # Pad to even length
    if len(q_masked) % 2 != 0:
        q_masked = np.append(q_masked, np.int8(0))

    # Pack pairs: byte = (high_nibble << 4) | low_nibble
    low = q_masked[0::2].astype(np.uint8)
    high = q_masked[1::2].astype(np.uint8)
    packed = (high << 4) | low
    packed_bytes = packed.tobytes()

    # Scales as FP16 bytes
    scales_fp16 = scales.to(torch.float16).numpy()
    scales_bytes = scales_fp16.tobytes()

    return Int4QuantResult(
        packed_weights=packed_bytes,
        scales=scales_bytes,
        weight_shape=list(weights.shape),
        scale_shape=list(scales.shape),
        block_size=block_size,
        num_params=weights.numel(),
        reconstruction_error=error,
    )


def dequantize_int4_block(
    packed_weights: bytes,
    scales: bytes,
    weight_shape: list[int],
    scale_shape: list[int],
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    """Dequantise INT4 block-wise weights back to FP32.

    Args:
        packed_weights: Packed int4 bytes (2 per byte, LSB-first).
        scales:         FP16 scale bytes, one per block.
        weight_shape:   Original [out_features, in_features].
        scale_shape:    [out_features, n_blocks].
        block_size:     Block size used during quantisation.

    Returns:
        FP32 tensor of original shape.
    """
    out_features, in_features = weight_shape

    # Unpack int4
    packed = np.frombuffer(packed_weights, dtype=np.uint8)
    low = (packed & 0x0F).astype(np.int8)
    high = ((packed >> 4) & 0x0F).astype(np.int8)

    # Sign-extend from 4-bit two's complement
    low = np.where(low > 7, low - 16, low).astype(np.int8)
    high = np.where(high > 7, high - 16, high).astype(np.int8)

    # Interleave
    q_flat = np.empty(len(packed) * 2, dtype=np.int8)
    q_flat[0::2] = low
    q_flat[1::2] = high

    # Recover scales
    scales_fp16 = np.frombuffer(scales, dtype=np.float16).reshape(scale_shape)
    scales_tensor = torch.from_numpy(scales_fp16.astype(np.float32))

    # Reshape quantised values
    padded_in = scale_shape[1] * block_size
    total_elements = out_features * padded_in
    q = torch.from_numpy(q_flat[:total_elements].copy()).reshape(
        out_features, scale_shape[1], block_size
    ).float()

    # Dequant: scale * data
    dequant = q * scales_tensor.unsqueeze(-1)
    dequant = dequant.reshape(out_features, padded_in)[:, :in_features]

    return dequant
