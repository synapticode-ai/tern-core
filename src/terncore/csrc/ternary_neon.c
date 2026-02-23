/*
 * ternary_neon.c — ARM NEON intrinsic kernels for packed ternary matmul
 *
 * Replaces the branching decode_byte_f32 inner loop with branchless
 * NEON mask-and-blend operations.  Processes 1 packed byte (4 trits,
 * 4 float32 inputs) per NEON iteration using 128-bit vectors.
 *
 * Algorithm per iteration:
 *   1. Load 1 packed byte, quick zero-skip if 0x00
 *   2. LUT lookup → 4-bit positive/negative masks
 *   3. Expand masks to 4×32-bit lane masks via lookup table
 *   4. Load 4 input floats, blend with vbslq_f32 (branchless)
 *   5. Extract lanes, accumulate left-to-right for bit-identical output
 *
 * Since NEON processes exactly 1 packed byte (4 trits) per iteration,
 * matching the scalar loop granularity, no tail handling is needed.
 *
 * Compile with: cc -std=c11 -O2 -c ternary_neon.c
 * (No special flags needed — NEON is always available on AArch64.)
 *
 * Patent 37: Zero-weight clock-gating — NEON branchless skip logic.
 * Patent 38: Configurable precision — NEON execution path.
 * Patent 39: Ternary-native memory — packed trit storage format.
 *
 * Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
 */

#include "ternary_simd.h"

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ── Trit-to-mask lookup tables ──────────────────────────────────────
 *
 * Same as AVX2 variant: for each packed byte, precompute which of
 * the 4 trits are positive and which are negative.
 *
 * pos_lut[byte]: bit k set if trit k == TRIT_POS (0b01)
 * neg_lut[byte]: bit k set if trit k == TRIT_NEG (0b10)
 * ─────────────────────────────────────────────────────────────────── */

static uint8_t pos_lut[256];
static uint8_t neg_lut[256];
static int luts_initialized = 0;

static void init_trit_luts(void)
{
    if (luts_initialized) return;

    for (int b = 0; b < 256; b++) {
        uint8_t pos = 0, neg = 0;
        for (int k = 0; k < 4; k++) {
            int trit = (b >> (k * 2)) & TRIT_MASK;
            if (trit == TRIT_POS) pos |= (1u << k);
            if (trit == TRIT_NEG) neg |= (1u << k);
        }
        pos_lut[b] = pos;
        neg_lut[b] = neg;
    }

    luts_initialized = 1;
}

/* ── 4-bit mask expansion table ──────────────────────────────────────
 *
 * For a 4-bit mask (bits [3:0]), expand each bit to a full 32-bit
 * lane (0x00000000 or 0xFFFFFFFF).  Only 16 entries needed.
 *
 * mask_expand[m] is a uint32x4_t where lane k is all-ones if
 * bit k of m is set, all-zeros otherwise.
 * ─────────────────────────────────────────────────────────────────── */

static uint32_t mask_expand_data[16][4];
static int mask_expand_initialized = 0;

static void init_mask_expand(void)
{
    if (mask_expand_initialized) return;

    for (int m = 0; m < 16; m++) {
        for (int k = 0; k < 4; k++) {
            mask_expand_data[m][k] = (m & (1u << k)) ? 0xFFFFFFFFu : 0u;
        }
    }

    mask_expand_initialized = 1;
}

/* ══════════════════════════════════════════════════════════════════════
 * tern_packed_matvec_f32_neon — NEON packed ternary matvec
 *
 *   output[i] = alpha * sum_j(W[i,j] * x[j]) + bias[i]
 *
 * Processes 4 weights (1 packed byte) per NEON iteration.  Each
 * iteration maps exactly to one scalar decode_byte_f32 call, so
 * no tail handling is needed.
 *
 * Patent 37: 4-weight block zero-skip.
 * Patent 38: NEON SIMD execution path.
 * ═════════════════════════════════════════════════════════════════════*/
int tern_packed_matvec_f32_neon(
    const uint8_t *packed,
    const float   *input,
    float         *output,
    int M, int N,
    float alpha,
    const float *bias)
{
    if (!packed || !input || !output) return TERN_ERR_NULL;
    if (M <= 0 || N <= 0)            return TERN_ERR_DIM;
    if (N & 3)                       return TERN_ERR_ALIGN;

    init_trit_luts();
    init_mask_expand();

    const int packed_cols = N >> 2;
    const float32x4_t zero_v = vdupq_n_f32(0.0f);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        float acc = 0.0f;
        const uint8_t *row = packed + (size_t)i * (size_t)packed_cols;

        for (int p = 0; p < packed_cols; p++) {
            uint8_t byte = row[p];

            /* 4-weight block zero-skip (Patent 37) */
            if (byte == 0) continue;

            /* LUT → 4-bit masks */
            uint8_t pos4 = pos_lut[byte];
            uint8_t neg4 = neg_lut[byte];

            /* Expand to NEON lane masks (4×uint32) */
            uint32x4_t pos_mask = vld1q_u32(mask_expand_data[pos4]);
            uint32x4_t neg_mask = vld1q_u32(mask_expand_data[neg4]);

            /* Load 4 input floats */
            float32x4_t inp = vld1q_f32(&input[p << 2]);

            /* Branchless blend: select inputs for pos/neg trits */
            float32x4_t pos_v = vbslq_f32(pos_mask, inp, zero_v);
            float32x4_t neg_v = vbslq_f32(neg_mask, inp, zero_v);
            float32x4_t result = vsubq_f32(pos_v, neg_v);

            /*
             * Sequential accumulation for bit-identical output (Patent 36).
             * Extract 4 lanes left-to-right, matching scalar order.
             */
            acc += vgetq_lane_f32(result, 0);
            acc += vgetq_lane_f32(result, 1);
            acc += vgetq_lane_f32(result, 2);
            acc += vgetq_lane_f32(result, 3);
        }

        output[i] = acc * alpha;
        if (bias) {
            output[i] += bias[i];
        }
    }

    return TERN_OK;
}

/* ══════════════════════════════════════════════════════════════════════
 * tern_packed_matmul_f32_neon — NEON packed ternary matmul (batched)
 * ═════════════════════════════════════════════════════════════════════*/
int tern_packed_matmul_f32_neon(
    const uint8_t *packed,
    const float   *input,
    float         *output,
    int M, int N, int B,
    float alpha,
    const float *bias)
{
    if (!packed || !input || !output)  return TERN_ERR_NULL;
    if (M <= 0 || N <= 0 || B <= 0)   return TERN_ERR_DIM;
    if (N & 3)                        return TERN_ERR_ALIGN;

    for (int b = 0; b < B; b++) {
        const float *in_b  = input  + (size_t)b * (size_t)N;
        float       *out_b = output + (size_t)b * (size_t)M;

        int rc = tern_packed_matvec_f32_neon(
            packed, in_b, out_b, M, N, alpha, bias);
        if (rc != TERN_OK) return rc;
    }

    return TERN_OK;
}

#endif /* __aarch64__ || _M_ARM64 */
