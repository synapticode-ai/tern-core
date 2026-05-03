"""Metal kernel integration tests for PackedTernaryLinear.

Skips entirely when the Metal engine is unavailable (non-macOS host,
missing dylib, Metal device init failure). When skipped, no assertion
runs — these tests do not gate CI on non-Metal platforms.

Coverage (this commit):
    - Cross-kernel output equivalence: same packed weights through CPU
      C kernel (uint8) and Metal kernel (uint32 via repack), compared
      per-element-tolerance-equivalent at three layer sizes.

Subsequent commits add: forward-path branching test, MPS fallback
regression, end-to-end production-data .tern-model load + inference test.

Tolerance reasoning:
    Metal kernel returns float16; CPU C kernel returns float32. float16
    has ~3 decimal digits of precision (≈ 1e-3 relative). With ternary
    weights composed linearly over K terms and bounded random input,
    accumulated error scales roughly with sqrt(K) for stochastic
    contributions. Empirical calibration (2026-05-04, seed 0):
    K=64 → 0.0021, K=256 → 0.0062, K=2560 → 0.0282 — fits sqrt(K) × ~5e-4.
    Tolerance bound 5e-2 leaves ~1.8× headroom on the largest layer for
    seed/input variance.

Copyright (c) 2026 Synapticode Co., Ltd. All rights reserved.
"""
import numpy as np
import pytest
import torch

from terncore.metal_runtime import get_engine, reset_engine
from terncore.sparse import pack_ternary_weights


def _metal_or_skip():
    engine = get_engine()
    if engine is None:
        pytest.skip("Metal engine unavailable on this host")
    return engine


@pytest.fixture(autouse=True)
def _reset_metal_singleton():
    """Ensure clean engine state between tests."""
    reset_engine()
    yield
    reset_engine()


@pytest.mark.parametrize("M,K", [(64, 64), (256, 256), (2560, 2560)])
def test_repack_cross_kernel_equivalence(M, K):
    """uint8 CPU kernel and uint32 Metal kernel produce equivalent output
    for the same ternary weights and the same input, within a tolerance
    consistent with float16 vs float32 accumulator precision."""
    engine = _metal_or_skip()

    from terncore.ternary_metal import repack_uint8_to_uint32_codes
    from terncore.packed_ops import packed_ternary_matmul_fast

    # Random ternary weights with a balanced distribution
    torch.manual_seed(0)
    ternary = torch.randint(-1, 2, (M, K), dtype=torch.int8).float()
    nonzero = ternary[ternary != 0]
    alpha = float(nonzero.abs().mean()) if nonzero.numel() > 0 else 1.0

    # Pack via the CPU pipeline (returns flat 1D uint8 buffer + bitmap)
    packed_uint8, _bitmap_bool = pack_ternary_weights(ternary)

    # Random input vector (B=1)
    x = torch.randn(1, K, dtype=torch.float32) * 0.5

    # pack_ternary_weights returns a bool bitmap (1 byte/weight, for analysis
    # use); packed_ternary_matmul_fast's cached-bitmap path expects packbits
    # format (1 bit/weight LSB-first). Construct the packbits bitmap explicitly.
    bitmap_packbits_np = np.packbits(
        (ternary != 0).flatten().numpy().astype(np.uint8),
        bitorder="little",
    )
    bitmap_packbits = torch.from_numpy(bitmap_packbits_np)

    # CPU C kernel path
    cpu_out = packed_ternary_matmul_fast(
        x, packed_uint8, alpha, M, K, sparsity_bitmap=bitmap_packbits,
    )

    # Metal kernel path: repack uint8 → uint32, dispatch via numpy matvec
    codes_u32 = repack_uint8_to_uint32_codes(packed_uint8, K, M)
    codes_u32_np = codes_u32.numpy().astype(np.uint32)
    scales_np = np.full(M, alpha, dtype=np.float32)
    x_np = x.numpy().astype(np.float16)
    metal_out_np = engine.matvec(codes_u32_np, scales_np, x_np, fast=True)
    metal_out = torch.from_numpy(metal_out_np.astype(np.float32))

    assert metal_out.shape == cpu_out.shape, (
        f"shape mismatch: metal={metal_out.shape} cpu={cpu_out.shape}"
    )
    max_abs_diff = (metal_out - cpu_out).abs().max().item()
    # Diagnostic: print so calibration is visible in test output
    print(f"\n[M={M} K={K}] max_abs_diff={max_abs_diff:.6f} alpha={alpha:.4f}",
          flush=True)
    # Calibrated tolerance: empirical max ≈ 0.028 at K=2560 (sqrt(K) × ~5e-4
    # from float16 vs float32 accumulator precision); 5e-2 gives ~1.8× headroom.
    assert max_abs_diff < 5e-2, (
        f"cross-kernel divergence {max_abs_diff} exceeds tolerance 0.05 "
        f"for M={M} K={K} (alpha={alpha:.4f})"
    )
