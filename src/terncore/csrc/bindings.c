/*
 * bindings.c — ctypes interface for ternary compute kernels
 *
 * Top-level dispatch layer loaded by the Python accel/ wrapper via
 * ctypes.  Provides:
 *
 *   1. ternary_matmul_f32()      — Primary entry point: selects the
 *                                  best scalar kernel automatically.
 *   2. ternary_matmul_f32_simd() — SIMD-accelerated entry point:
 *                                  dispatches to AVX2/NEON for dense
 *                                  path, scalar for sparse path.
 *   3. get_simd_support()        — Runtime SIMD feature detection
 *                                  via CPUID (x86) or compile-time
 *                                  check (AArch64).
 *   4. terncore_version()        — Library version string.
 *
 * All individual kernel functions (tern_matvec_f32, tern_packed_*,
 * tern_sparse64_*, tern_packed_*_avx2, tern_packed_*_neon) are also
 * available as exported symbols for direct use from Python.
 *
 * Build (shared library):
 *   Use the Makefile:  make          (auto-detects platform)
 *   Or manually:
 *     macOS x86:  cc -std=c11 -O2 -shared -fPIC -mavx2 \
 *                    -o libterncore.dylib *.c
 *     Linux ARM:  cc -std=c11 -O2 -shared -fPIC \
 *                    -o libterncore.so *.c
 *
 * Patent 36: Deterministic execution.
 * Patent 38: Configurable precision → dual-path dispatch.
 *
 * Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
 */

#include "sparse_skip.h"      /* includes ternary_packed.h → ternary_matmul.h */
#include "ternary_simd.h"     /* TERN_SIMD_*, AVX2/NEON kernel declarations   */
#include "terncore_version.h" /* TERNCORE_VERSION_STRING (build-time, from pyproject.toml) */

/* ── Cached SIMD capabilities ────────────────────────────────────── */

static uint32_t cached_caps  = 0;
static int      caps_checked = 0;

/* ══════════════════════════════════════════════════════════════════════
 * ternary_matmul_f32 — Primary scalar dispatch entry point
 *
 *   output[b,i] = alpha * sum_j(W[i,j] * input[b,j]) + bias[i]
 *
 * Selects the best available scalar kernel based on inputs:
 *
 *   bitmap != NULL  →  sparse64 packed kernel (bit-scan skip)
 *   bitmap == NULL  →  dense packed kernel (byte-level skip)
 *
 * Parameters:
 *   packed_weights  [M * N/4] uint8_t, 2-bit packed ternary weights
 *   input           [B x N]   float32 row-major input activations
 *   output          [B x M]   float32 row-major (caller-allocated)
 *   bitmap          packed sparsity bitmap (ceil(M*N/8) bytes), or
 *                   NULL to skip bitmap-based optimization
 *   alpha           per-layer scaling factor
 *   bias            [M] float32 bias, or NULL
 *   M               output dimension (weight rows)
 *   N               input dimension (weight columns, multiple of 4)
 *   B               batch size
 *
 * Returns: TERN_OK on success, TERN_ERR_* on failure.
 *
 * Patent 38: Configurable precision — dispatch to best available
 *            execution path.
 * ═════════════════════════════════════════════════════════════════════*/
int ternary_matmul_f32(
    const uint8_t *packed_weights,
    const float   *input,
    float         *output,
    const uint8_t *bitmap,
    float          alpha,
    const float   *bias,
    int M, int N, int B)
{
    if (bitmap != NULL) {
        return tern_sparse64_packed_matmul_f32(
            packed_weights, input, output, bitmap, M, N, B, alpha, bias);
    }

    return tern_packed_matmul_f32(
        packed_weights, input, output, M, N, B, alpha, bias);
}

/* ══════════════════════════════════════════════════════════════════════
 * ternary_matmul_f32_simd — SIMD-accelerated dispatch
 *
 * For the dense packed path (bitmap == NULL), dispatches to the best
 * available SIMD kernel:
 *
 *   AVX2  (x86_64)  →  tern_packed_matmul_f32_avx2()
 *   NEON  (AArch64)  →  tern_packed_matmul_f32_neon()
 *   fallback         →  tern_packed_matmul_f32()  (scalar)
 *
 * For the sparse path (bitmap != NULL), falls back to the scalar
 * sparse64 kernel (SIMD sparse kernels are a future optimisation).
 *
 * Same parameters and return values as ternary_matmul_f32().
 *
 * Patent 38: Configurable precision — runtime SIMD dispatch.
 * ═════════════════════════════════════════════════════════════════════*/
int ternary_matmul_f32_simd(
    const uint8_t *packed_weights,
    const float   *input,
    float         *output,
    const uint8_t *bitmap,
    float          alpha,
    const float   *bias,
    int M, int N, int B)
{
    /* Sparse path: SIMD sparse kernels not yet implemented */
    if (bitmap != NULL) {
        return tern_sparse64_packed_matmul_f32(
            packed_weights, input, output, bitmap, M, N, B, alpha, bias);
    }

    /* Dense path: dispatch to best available SIMD kernel */
    uint32_t caps = get_simd_support();

#if defined(__x86_64__) || defined(_M_X64)
    if (caps & TERN_SIMD_AVX2) {
        return tern_packed_matmul_f32_avx2(
            packed_weights, input, output, M, N, B, alpha, bias);
    }
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
    if (caps & TERN_SIMD_NEON) {
        return tern_packed_matmul_f32_neon(
            packed_weights, input, output, M, N, B, alpha, bias);
    }
#endif

    /* Scalar fallback */
    return tern_packed_matmul_f32(
        packed_weights, input, output, M, N, B, alpha, bias);
}

/* ══════════════════════════════════════════════════════════════════════
 * get_simd_support — Runtime SIMD feature detection
 *
 * Returns a bitmask of available instruction sets:
 *   TERN_SIMD_SCALAR  (0x01) — always set
 *   TERN_SIMD_AVX2    (0x02) — x86 CPUID leaf 7, EBX bit 5
 *   TERN_SIMD_AVX512  (0x04) — x86 CPUID leaf 7, EBX bit 16
 *   TERN_SIMD_NEON    (0x08) — always set on AArch64
 *
 * Result is cached after the first call (CPUID is only executed once).
 *
 * Patent 38: Configurable precision — runtime capability detection.
 * ═════════════════════════════════════════════════════════════════════*/
uint32_t get_simd_support(void)
{
    if (caps_checked) return cached_caps;

    uint32_t support = TERN_SIMD_SCALAR;

#if defined(__x86_64__) || defined(_M_X64)
    {
        uint32_t eax, ebx, ecx, edx;

#if defined(__GNUC__) || defined(__clang__)
        __asm__ __volatile__(
            "cpuid"
            : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
            : "a"(7), "c"(0)
        );
#elif defined(_MSC_VER)
        int cpuinfo[4];
        __cpuidex(cpuinfo, 7, 0);
        ebx = (uint32_t)cpuinfo[1];
#else
        ebx = 0;
#endif

        if (ebx & (1u << 5))   support |= TERN_SIMD_AVX2;
        if (ebx & (1u << 16))  support |= TERN_SIMD_AVX512;
    }
#elif defined(__aarch64__) || defined(_M_ARM64)
    /* NEON is always available on AArch64 */
    support |= TERN_SIMD_NEON;
#endif

    cached_caps  = support;
    caps_checked = 1;
    return support;
}

/* ══════════════════════════════════════════════════════════════════════
 * terncore_version — Library version string
 *
 * Returns a pointer to a static string.  TERNCORE_VERSION_STRING is
 * generated at build time from pyproject.toml's [project] version
 * field — see terncore_version.h and the Makefile rule.
 * ═════════════════════════════════════════════════════════════════════*/
const char *terncore_version(void)
{
    return TERNCORE_VERSION_STRING;
}
