"""
Accelerated ternary linear layer using C kernels via ctypes.

Provides TernaryLinearAccel as a drop-in replacement for TernaryLinear.
When the compiled C library (libterncore) is available, eval-mode
forward passes use optimised C kernels with 2-bit packed weights and
bitmap-driven zero-skip.  Falls back to pure PyTorch otherwise.

Patent 37: Zero-weight clock-gating — sparsity-aware skip logic.
Patent 38: Configurable precision — dual-path dispatch.
Patent 39: Ternary-native memory — packed trit storage format.

Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
"""

from __future__ import annotations

import ctypes
import platform
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from terncore.arithmetic.linear import TernaryLinear
from terncore.sparse import pack_ternary_weights


# ═══════════════════════════════════════════════════════════════
# Library loading
# ═══════════════════════════════════════════════════════════════

_lib: Optional[ctypes.CDLL] = None
_lib_path: Optional[Path] = None


def _setup_signatures(lib: ctypes.CDLL) -> None:
    """Configure ctypes function signatures for the C library."""
    # ternary_matmul_f32 — primary dispatch (Patent 38)
    lib.ternary_matmul_f32.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),  # packed_weights [M*N/4]
        ctypes.POINTER(ctypes.c_float),  # input          [B x N]
        ctypes.POINTER(ctypes.c_float),  # output         [B x M]
        ctypes.POINTER(ctypes.c_uint8),  # bitmap         (or NULL)
        ctypes.c_float,                  # alpha
        ctypes.POINTER(ctypes.c_float),  # bias           (or NULL)
        ctypes.c_int,                    # M
        ctypes.c_int,                    # N
        ctypes.c_int,                    # B
    ]
    lib.ternary_matmul_f32.restype = ctypes.c_int

    # get_simd_support — capability detection (Patent 38)
    lib.get_simd_support.argtypes = []
    lib.get_simd_support.restype = ctypes.c_uint32

    # ternary_matmul_f32_simd — SIMD-accelerated dispatch (Patent 38)
    lib.ternary_matmul_f32_simd.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),  # packed_weights [M*N/4]
        ctypes.POINTER(ctypes.c_float),  # input          [B x N]
        ctypes.POINTER(ctypes.c_float),  # output         [B x M]
        ctypes.POINTER(ctypes.c_uint8),  # bitmap         (or NULL)
        ctypes.c_float,                  # alpha
        ctypes.POINTER(ctypes.c_float),  # bias           (or NULL)
        ctypes.c_int,                    # M
        ctypes.c_int,                    # N
        ctypes.c_int,                    # B
    ]
    lib.ternary_matmul_f32_simd.restype = ctypes.c_int

    # terncore_version — library version string
    lib.terncore_version.argtypes = []
    lib.terncore_version.restype = ctypes.c_char_p


def _load_library() -> None:
    """Attempt to load libterncore from the csrc directory."""
    global _lib, _lib_path

    csrc_dir = Path(__file__).resolve().parent.parent / "csrc"

    if platform.system() == "Darwin":
        candidates = ["libterncore.dylib", "libterncore.so"]
    else:
        candidates = ["libterncore.so", "libterncore.dylib"]

    for name in candidates:
        path = csrc_dir / name
        if path.exists():
            try:
                lib = ctypes.CDLL(str(path))
                _setup_signatures(lib)
                _lib = lib
                _lib_path = path
                return
            except OSError:
                continue


_load_library()


# ═══════════════════════════════════════════════════════════════
# Module-level utilities
# ═══════════════════════════════════════════════════════════════


def is_accelerated() -> bool:
    """Return True if the C kernels are loaded and available."""
    return _lib is not None


def get_acceleration_info() -> dict:
    """
    Return information about the acceleration backend.

    Returns:
        Dict with keys:
            accelerated:  bool — whether C kernels are available
            library_path: str or None — path to loaded library
            simd_support: dict — available SIMD instruction sets
            version:      str or None — library version string
    """
    info: dict = {
        "accelerated": _lib is not None,
        "library_path": str(_lib_path) if _lib_path else None,
        "simd_support": {},
        "version": None,
    }

    if _lib is not None:
        caps = _lib.get_simd_support()
        info["simd_support"] = {
            "scalar": bool(caps & 0x01),
            "avx2": bool(caps & 0x02),
            "avx512": bool(caps & 0x04),
            "neon": bool(caps & 0x08),
        }
        info["version"] = _lib.terncore_version().decode("utf-8")

    return info


# ═══════════════════════════════════════════════════════════════
# TernaryLinearAccel
# ═══════════════════════════════════════════════════════════════


class TernaryLinearAccel(TernaryLinear):
    """
    Accelerated ternary linear layer using C kernels.

    Drop-in replacement for TernaryLinear.  In eval mode, dispatches
    to compiled C kernels that operate on 2-bit packed weights with
    bitmap-driven zero-skip.  Falls back to pure PyTorch when:

        - C library is not available
        - in_features is not a multiple of 4 (packed format requirement)
        - C kernel returns an error

    Training mode uses the same STE forward pass as TernaryLinear.

    Patent 37: Zero-skip via sparsity bitmap.
    Patent 38: Dual-path dispatch (C kernel / PyTorch fallback).
    Patent 39: 2-bit packed trit storage.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        threshold: float = 0.7,
    ) -> None:
        super().__init__(in_features, out_features, bias, threshold)
        # Cached numpy arrays for C kernel
        self._packed_weights_np: Optional[np.ndarray] = None
        self._bitmap_np: Optional[np.ndarray] = None
        self._alpha_val: float = 0.0

    def _forward_eval(self, x: torch.Tensor) -> torch.Tensor:
        """
        Eval forward pass with C kernel acceleration.

        Falls back to PyTorch if the C library is unavailable or
        the input dimension is not aligned for packed format.

        Handles arbitrary leading dimensions (e.g., (batch, seq_len, features)
        from transformer models) by flattening to 2D for the C kernel, then
        restoring the original shape.
        """
        if _lib is None or self.in_features % 4 != 0:
            return super()._forward_eval(x)

        # Ensure ternary weights are cached
        if self._cached_ternary is None:
            self._cache_ternary_weights()

        # Ensure packed weights are cached
        if self._packed_weights_np is None:
            self._cache_accel_weights()

        # Save original shape and flatten leading dims to 2D
        orig_shape = x.shape
        if x.ndim > 2:
            x = x.reshape(-1, x.shape[-1])

        # Prepare input: float32, contiguous, on CPU
        x_np = x.detach().cpu().float().contiguous().numpy()

        # Handle 1D input (single sample)
        squeeze = False
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)
            squeeze = True

        x_np = np.ascontiguousarray(x_np, dtype=np.float32)

        B = x_np.shape[0]
        M = self.out_features
        N = self.in_features

        # Allocate output
        output_np = np.zeros((B, M), dtype=np.float32)

        # Prepare bias pointer (NULL if no bias)
        if self.bias is not None:
            bias_np = np.ascontiguousarray(
                self.bias.detach().cpu().float().numpy(), dtype=np.float32
            )
            bias_ptr = bias_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        else:
            bias_ptr = None

        # Call C kernel with SIMD dispatch (Patent 38: best available path)
        rc = _lib.ternary_matmul_f32_simd(
            self._packed_weights_np.ctypes.data_as(
                ctypes.POINTER(ctypes.c_uint8)
            ),
            x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._bitmap_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_float(self._alpha_val),
            bias_ptr,
            ctypes.c_int(M),
            ctypes.c_int(N),
            ctypes.c_int(B),
        )

        if rc != 0:
            return super()._forward_eval(x)

        result = torch.from_numpy(output_np.copy())
        if squeeze:
            result = result.squeeze(0)

        # Restore leading dimensions
        if len(orig_shape) > 2:
            result = result.reshape(*orig_shape[:-1], M)

        return result

    def _cache_accel_weights(self) -> None:
        """Pack ternary weights and bitmap for C kernel consumption."""
        if self._cached_ternary is None:
            self._cache_ternary_weights()

        # Pack weights to 2-bit format (Patent 39)
        packed_torch, _ = pack_ternary_weights(self._cached_ternary)
        self._packed_weights_np = np.ascontiguousarray(
            packed_torch.numpy(), dtype=np.uint8
        )

        # Pack bitmap: 1 bit per weight, LSB-first (Patent 37)
        bitmap_bool = (
            self._cached_ternary.flatten() != 0
        ).numpy().astype(np.uint8)
        self._bitmap_np = np.packbits(bitmap_bool, bitorder="little")

        # Cache alpha scalar
        self._alpha_val = self._cached_alpha.item()

    def invalidate_cache(self) -> None:
        """Clear all cached weights including packed C kernel data."""
        super().invalidate_cache()
        self._packed_weights_np = None
        self._bitmap_np = None
        self._alpha_val = 0.0

    @classmethod
    def from_ternary_linear(cls, layer: TernaryLinear) -> TernaryLinearAccel:
        """
        Create a TernaryLinearAccel from an existing TernaryLinear.

        Copies weight, alpha, and bias parameters.  The packed weight
        cache is built lazily on first eval forward pass.
        """
        accel = cls(
            layer.in_features,
            layer.out_features,
            bias=layer.bias is not None,
            threshold=layer.threshold,
        )
        # Share parameters
        accel.weight = layer.weight
        accel.alpha = layer.alpha
        if layer.bias is not None:
            accel.bias = layer.bias

        # Copy cached buffers if already computed
        if layer._cached_ternary is not None:
            accel._cached_ternary = layer._cached_ternary
            accel._cached_alpha = layer._cached_alpha
            accel._sparsity_bitmap = layer._sparsity_bitmap

        return accel

    def extra_repr(self) -> str:
        s = super().extra_repr()
        if _lib is not None:
            s += ", accel=C"
        else:
            s += ", accel=fallback"
        return s
