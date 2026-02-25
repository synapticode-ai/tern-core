"""
Packed ternary matrix operations — compute from 2-bit packed weights.

Patent 1: Ternary weight encoding — compare-and-add arithmetic.
Patent 5: Ternary execution path — packed weight computation.
Patent 39: Ternary-native memory — 2-bit packed storage format.

The weights are stored in 2-bit packed format (4 weights per byte).
At compute time, they are unpacked to ternary {-1, 0, +1} tensors
and used with standard or SIMD matmul. This achieves the memory
savings (16x vs FP32) while reusing proven compute kernels.

Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from terncore.sparse import unpack_ternary_weights


def packed_ternary_matmul(
    input: torch.Tensor,
    packed_weights: torch.Tensor,
    alpha: float,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """
    Matrix multiply with 2-bit packed ternary weights.

    Reference implementation: unpack → float → F.linear.

    Args:
        input:          Input tensor (..., in_features).
        packed_weights: Uint8 tensor with 2-bit packed weights.
        alpha:          Per-layer scaling factor.
        out_features:   Number of output features.
        in_features:    Number of input features.

    Returns:
        Output tensor (..., out_features).
    """
    shape = torch.Size([out_features, in_features])
    ternary = unpack_ternary_weights(packed_weights, shape)
    weights = ternary.float() * alpha
    return F.linear(input, weights)


def packed_ternary_matmul_fast(
    input: torch.Tensor,
    packed_weights: torch.Tensor,
    alpha: float,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """
    Optimised packed matmul — pass packed weights directly to C kernel.

    The C kernel (ternary_matmul_f32) operates on 2-bit packed uint8
    weights with bitmap-driven zero-skip and alpha scaling built in.
    Falls back to reference path if C acceleration is not available.

    Args:
        input:          Input tensor (..., in_features).
        packed_weights: Uint8 tensor with 2-bit packed weights.
        alpha:          Per-layer scaling factor.
        out_features:   Number of output features.
        in_features:    Number of input features.

    Returns:
        Output tensor (..., out_features).
    """
    try:
        from terncore.accel import is_accelerated, _lib

        if is_accelerated() and in_features % 4 == 0 and _lib is not None:
            import ctypes
            import numpy as np

            # Flatten leading dims for the C kernel
            x_shape = input.shape
            x_2d = input.reshape(-1, in_features)
            x_np = np.ascontiguousarray(
                x_2d.detach().cpu().float().numpy(), dtype=np.float32
            )
            batch = x_2d.shape[0]

            # Packed weights as uint8 — passed directly to C kernel
            pw_np = np.ascontiguousarray(
                packed_weights.numpy(), dtype=np.uint8
            )

            # Build sparsity bitmap (1 bit per weight, LSB-first)
            shape = torch.Size([out_features, in_features])
            ternary = unpack_ternary_weights(packed_weights, shape)
            bitmap_bool = (ternary.flatten() != 0).numpy().astype(np.uint8)
            bitmap_np = np.packbits(bitmap_bool, bitorder="little")

            out_np = np.zeros((batch, out_features), dtype=np.float32)

            # C kernel signature: packed_weights, input, output,
            #   bitmap, alpha, bias, M, N, B
            rc = _lib.ternary_matmul_f32(
                pw_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                bitmap_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                ctypes.c_float(alpha),
                None,  # bias handled by caller
                ctypes.c_int(out_features),
                ctypes.c_int(in_features),
                ctypes.c_int(batch),
            )

            if rc == 0:
                result = torch.from_numpy(out_np).reshape(
                    *x_shape[:-1], out_features
                )
                return result
    except (ImportError, AttributeError):
        pass

    # Fallback: unpack → float → F.linear
    shape = torch.Size([out_features, in_features])
    ternary = unpack_ternary_weights(packed_weights, shape)
    weights = ternary.float() * alpha
    return F.linear(input, weights)
