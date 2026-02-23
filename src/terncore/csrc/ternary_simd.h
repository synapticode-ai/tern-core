/*
 * ternary_simd.h — SIMD kernel declarations and capability detection
 *
 * Shared header for AVX2 (x86_64) and NEON (AArch64) packed ternary
 * matmul kernels.  Included by bindings.c for dispatch and by the
 * individual SIMD kernel files.
 *
 * Patent 37: Zero-weight clock-gating — SIMD branchless skip logic.
 * Patent 38: Configurable precision — SIMD/scalar dual-path dispatch.
 *
 * Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
 */

#ifndef TERNARY_SIMD_H
#define TERNARY_SIMD_H

#include "ternary_packed.h"   /* TERN_OK, TERN_ERR_*, TRIT_*, decode_byte_f32 */

/* ── SIMD capability flags ────────────────────────────────────────── */

#define TERN_SIMD_SCALAR  (1u << 0)   /* 0x01 — Scalar C (always)    */
#define TERN_SIMD_AVX2    (1u << 1)   /* 0x02 — x86 AVX2             */
#define TERN_SIMD_AVX512  (1u << 2)   /* 0x04 — x86 AVX-512          */
#define TERN_SIMD_NEON    (1u << 3)   /* 0x08 — ARM NEON             */

#ifdef __cplusplus
extern "C" {
#endif

/* ── Dispatch and detection (bindings.c) ─────────────────────────── */

uint32_t get_simd_support(void);
const char *terncore_version(void);

int ternary_matmul_f32(
    const uint8_t *packed_weights,
    const float   *input,
    float         *output,
    const uint8_t *bitmap,
    float          alpha,
    const float   *bias,
    int M, int N, int B);

int ternary_matmul_f32_simd(
    const uint8_t *packed_weights,
    const float   *input,
    float         *output,
    const uint8_t *bitmap,
    float          alpha,
    const float   *bias,
    int M, int N, int B);

/* ── AVX2 kernels (ternary_avx2.c) ───────────────────────────────── */

#if defined(__x86_64__) || defined(_M_X64)

int tern_packed_matvec_f32_avx2(
    const uint8_t *packed,
    const float   *input,
    float         *output,
    int M, int N,
    float alpha,
    const float *bias);

int tern_packed_matmul_f32_avx2(
    const uint8_t *packed,
    const float   *input,
    float         *output,
    int M, int N, int B,
    float alpha,
    const float *bias);

#endif /* x86_64 */

/* ── NEON kernels (ternary_neon.c) ────────────────────────────────── */

#if defined(__aarch64__) || defined(_M_ARM64)

int tern_packed_matvec_f32_neon(
    const uint8_t *packed,
    const float   *input,
    float         *output,
    int M, int N,
    float alpha,
    const float *bias);

int tern_packed_matmul_f32_neon(
    const uint8_t *packed,
    const float   *input,
    float         *output,
    int M, int N, int B,
    float alpha,
    const float *bias);

#endif /* aarch64 */

#ifdef __cplusplus
}
#endif

#endif /* TERNARY_SIMD_H */
