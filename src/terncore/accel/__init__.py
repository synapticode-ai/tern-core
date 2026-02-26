"""
Accelerated ternary linear layer using C kernels.

Provides TernaryLinearAccel as a drop-in replacement for TernaryLinear.
When the compiled C library (libterncore) is available, eval-mode
forward passes use optimised C kernels with 2-bit packed weights and
bitmap-driven zero-skip.

Two acceleration backends are tried in order:
  1. PyTorch C++ extension (torch.utils.cpp_extension) — zero-copy,
     passes tensor data pointers directly to C kernels.  JIT-compiled
     on first import.  (Phase 4: ~100-200us overhead eliminated.)
  2. ctypes fallback — loads the pre-built shared library and marshals
     data through numpy.

Falls back to pure PyTorch if neither backend is available.

Patent 37: Zero-weight clock-gating — sparsity-aware skip logic.
Patent 38: Configurable precision — dual-path dispatch.
Patent 39: Ternary-native memory — packed trit storage format.

Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from terncore.arithmetic.linear import TernaryLinear
from terncore.sparse import pack_ternary_weights

logger = logging.getLogger(__name__)

__all__ = [
    "TernaryLinearAccel",
    "is_accelerated",
    "get_acceleration_info",
]


# ═══════════════════════════════════════════════════════════════
# Library loading
# ═══════════════════════════════════════════════════════════════

_lib: Optional[ctypes.CDLL] = None
_lib_path: Optional[Path] = None
_torch_ext = None  # torch C++ extension module (preferred backend)


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


def _get_omp_flags():
    """Detect OpenMP flags for the torch extension.

    On macOS, prefer PyTorch's own libiomp5 to avoid dual-libomp
    conflicts.  Falls back to brew libomp if torch's copy isn't found.
    On Linux, use the system's -fopenmp.
    """
    extra_cflags = []
    extra_ldflags = []

    if platform.system() == "Darwin":
        # Need omp.h from brew for compilation, but must link against
        # torch's own libiomp5 to avoid dual-libomp segfault.
        omp_include = None
        try:
            omp_prefix = subprocess.check_output(
                ["brew", "--prefix", "libomp"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            omp_inc = os.path.join(omp_prefix, "include")
            if os.path.isfile(os.path.join(omp_inc, "omp.h")):
                omp_include = omp_inc
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        if omp_include is None:
            return extra_cflags, extra_ldflags

        extra_cflags = [
            "-Xpreprocessor",
            "-fopenmp",
            f"-I{omp_include}",
        ]

        # Don't link against any libomp/libiomp5 directly — this avoids
        # dual-initialization crashes.  OpenMP symbols are resolved at
        # runtime from torch's already-loaded libiomp5 via dynamic lookup.
        extra_ldflags = ["-undefined", "dynamic_lookup"]
    else:
        extra_cflags = ["-fopenmp"]
        extra_ldflags = ["-fopenmp"]

    return extra_cflags, extra_ldflags


def _load_torch_extension() -> None:
    """JIT-compile the PyTorch C++ extension for zero-copy kernel calls."""
    global _torch_ext

    try:
        from torch.utils.cpp_extension import load
    except ImportError:
        return

    csrc_dir = Path(__file__).resolve().parent.parent / "csrc"
    cpp_file = csrc_dir / "torch_bindings.cpp"

    if not cpp_file.exists():
        return

    # Collect all C source files needed by the extension
    c_sources = [
        "ternary_matmul.c",
        "ternary_packed.c",
        "sparse_skip.c",
        "bindings.c",
        "ternary_avx2.c",
        "ternary_neon.c",
    ]
    sources = [str(cpp_file)]
    for name in c_sources:
        p = csrc_dir / name
        if p.exists():
            sources.append(str(p))

    # Build flags — include csrc dir for header resolution
    # Note: no -std=c11 since extra_cflags applies to both C and C++ files
    extra_cflags = ["-O2", f"-I{csrc_dir}"]
    extra_ldflags = []

    # Architecture-specific SIMD flags
    import struct
    if struct.calcsize("P") * 8 == 64:
        machine = platform.machine().lower()
        if machine in ("x86_64", "amd64"):
            extra_cflags.append("-mavx2")

    # OpenMP flags
    omp_cflags, omp_ldflags = _get_omp_flags()
    extra_cflags.extend(omp_cflags)
    extra_ldflags.extend(omp_ldflags)

    try:
        _torch_ext = load(
            name="terncore_ext",
            sources=sources,
            extra_cflags=extra_cflags,
            extra_ldflags=extra_ldflags,
            verbose=False,
        )
        logger.info("Loaded terncore PyTorch C++ extension (zero-copy)")
    except Exception as e:
        logger.debug("Failed to load torch C++ extension: %s", e)
        _torch_ext = None


# Load backends: try torch extension first, then ctypes fallback
_load_torch_extension()
_load_library()


# ═══════════════════════════════════════════════════════════════
# Module-level utilities
# ═══════════════════════════════════════════════════════════════


def is_accelerated() -> bool:
    """Return True if C kernels are available (torch ext or ctypes)."""
    return _torch_ext is not None or _lib is not None


def get_acceleration_info() -> dict:
    """
    Return information about the acceleration backend.

    Returns:
        Dict with keys:
            accelerated:  bool — whether C kernels are available
            library_path: str or None — path to loaded library
            simd_support: dict — available SIMD instruction sets
            version:      str or None — library version string
            backend:      str — "torch_ext", "ctypes", or "none"
    """
    info: dict = {
        "accelerated": is_accelerated(),
        "library_path": str(_lib_path) if _lib_path else None,
        "simd_support": {},
        "version": None,
        "backend": "none",
    }

    if _torch_ext is not None:
        info["backend"] = "torch_ext"
        caps = _torch_ext.get_simd_support()
        info["simd_support"] = {
            "scalar": bool(caps & 0x01),
            "avx2": bool(caps & 0x02),
            "avx512": bool(caps & 0x04),
            "neon": bool(caps & 0x08),
        }
        info["version"] = _torch_ext.terncore_version()
    elif _lib is not None:
        info["backend"] = "ctypes"
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

    Two backends are supported (tried in order):
      1. PyTorch C++ extension — zero-copy, passes tensor data_ptr
         directly to C kernels.  Packed weights stored as torch tensors.
      2. ctypes — marshals through numpy.  ~100-200us fixed overhead.

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
        # Cached packed data — stored as torch tensors for zero-copy path,
        # also kept as numpy arrays for ctypes fallback
        self._packed_weights_np: Optional[np.ndarray] = None
        self._bitmap_np: Optional[np.ndarray] = None
        self._packed_weights_t: Optional[torch.Tensor] = None
        self._bitmap_t: Optional[torch.Tensor] = None
        self._bias_t: Optional[torch.Tensor] = None
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
        if not is_accelerated() or self.in_features % 4 != 0:
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

        M = self.out_features
        N = self.in_features

        # ── Backend 1: torch C++ extension (zero-copy) ──────────
        if _torch_ext is not None and self._packed_weights_t is not None:
            # Ensure input is float32 contiguous on CPU
            x_in = x.detach().cpu().float().contiguous()

            # Handle 1D input (single sample)
            squeeze = False
            if x_in.ndim == 1:
                x_in = x_in.unsqueeze(0)
                squeeze = True

            result = _torch_ext.ternary_forward(
                self._packed_weights_t,
                x_in,
                self._bitmap_t,
                self._alpha_val,
                self._bias_t,
                M,
                N,
            )

            if squeeze:
                result = result.squeeze(0)

            # Restore leading dimensions
            if len(orig_shape) > 2:
                result = result.reshape(*orig_shape[:-1], M)

            return result

        # ── Backend 2: ctypes fallback ──────────────────────────
        return self._forward_eval_ctypes(x, orig_shape, M, N)

    def _forward_eval_ctypes(
        self,
        x: torch.Tensor,
        orig_shape: torch.Size,
        M: int,
        N: int,
    ) -> torch.Tensor:
        """Ctypes-based eval forward (legacy path)."""
        if _lib is None:
            return super()._forward_eval(x)

        # Prepare input: float32, contiguous, on CPU
        x_np = x.detach().cpu().float().contiguous().numpy()

        # Handle 1D input (single sample)
        squeeze = False
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)
            squeeze = True

        x_np = np.ascontiguousarray(x_np, dtype=np.float32)

        B = x_np.shape[0]

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

        # Build torch tensor versions for zero-copy path
        self._packed_weights_t = torch.from_numpy(
            self._packed_weights_np.copy()
        ).contiguous()
        self._bitmap_t = torch.from_numpy(
            self._bitmap_np.copy()
        ).contiguous()

        # Cache bias as contiguous float32 tensor (or empty tensor)
        if self.bias is not None:
            self._bias_t = self.bias.detach().cpu().float().contiguous()
        else:
            self._bias_t = torch.empty(0, dtype=torch.float32)

    def invalidate_cache(self) -> None:
        """Clear all cached weights including packed C kernel data."""
        super().invalidate_cache()
        self._packed_weights_np = None
        self._bitmap_np = None
        self._packed_weights_t = None
        self._bitmap_t = None
        self._bias_t = None
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
        if _torch_ext is not None:
            s += ", accel=torch_ext"
        elif _lib is not None:
            s += ", accel=ctypes"
        else:
            s += ", accel=fallback"
        return s
