/*
 * ternary_packed.c — Packed 2-bit ternary matrix multiplication kernels
 *
 * Operates directly on the 2-bit packed weight format produced by
 * Python sparse/__init__.py::pack_ternary_weights().  Each uint8
 * byte stores 4 ternary weights:
 *
 *   trit 0 = byte        & 0x03      (bits [1:0])
 *   trit 1 = (byte >> 2) & 0x03      (bits [3:2])
 *   trit 2 = (byte >> 4) & 0x03      (bits [5:4])
 *   trit 3 = byte >> 6               (bits [7:6])
 *
 * Encoding per trit:
 *   0b01 (1) → +1  →  accumulator += input   (pass-through)
 *   0b10 (2) → -1  →  accumulator -= input   (negate)
 *   0b00 (0) →  0  →  skip                   (no operation)
 *   0b11 (3) →  reserved (must not appear in valid data)
 *
 * This format stores 4 weights per byte (vs 1 weight per byte for
 * int8), reducing memory bandwidth by 4x.  Combined with inline
 * zero-skip (packed byte == 0x00 → all 4 weights are zero → skip
 * the group), this is the primary compute kernel for inference.
 *
 * All functions require N % 4 == 0 so that rows are byte-aligned
 * in the packed array.  Neural network layer dimensions are nearly
 * always multiples of much larger powers of 2 in practice.
 *
 * Patent 36: Ternary weight encoding, deterministic execution.
 * Patent 37: Zero-weight clock-gating → sparsity-aware skip logic.
 * Patent 39: Ternary-native memory → packed trit storage format.
 *
 * All functions are deterministic: identical inputs always produce
 * bit-identical outputs (IEEE 754 float32, left-to-right order).
 *
 * Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
 */

#include "ternary_packed.h"

#ifdef _OPENMP
#include <omp.h>
#endif

/* decode_byte_f32 is defined in ternary_packed.h for reuse by SIMD kernels */

/* ══════════════════════════════════════════════════════════════════════
 * Packed ternary matrix-vector multiply
 *
 *   output[i] = alpha * sum_j(W[i,j] * x[j]) + bias[i]
 *
 * Processes 4 weights per packed byte.  When a packed byte is 0x00,
 * all 4 trits are zero and the group is skipped without decoding.
 * With 60-70% sparsity, ~18% of packed bytes are all-zero (0.65^4),
 * providing natural 4-weight block skipping from the format itself.
 *
 * Patent 36: ternary compute, no multiply in inner loop.
 * Patent 37: 4-weight block zero-skip via packed byte check.
 * Patent 39: packed trit storage — 4 weights per byte.
 * ═════════════════════════════════════════════════════════════════════*/
int tern_packed_matvec_f32(
    const uint8_t *packed,
    const float   *input,
    float         *output,
    int M, int N,
    float alpha,
    const float *bias)
{
    if (!packed || !input || !output) return TERN_ERR_NULL;
    if (M <= 0 || N <= 0)            return TERN_ERR_DIM;
    if (N & 3)                       return TERN_ERR_ALIGN;  /* N % 4 != 0 */

    const int packed_cols = N >> 2;   /* N / 4 packed bytes per row */

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        float acc = 0.0f;
        const uint8_t *row = packed + (size_t)i * (size_t)packed_cols;

        for (int p = 0; p < packed_cols; p++) {
            uint8_t byte = row[p];

            /*
             * 4-weight block zero-skip (Patent 37):
             * If the packed byte is 0x00, all four trits are zero.
             * Skip the group without decoding — no add, no subtract.
             */
            if (byte == 0) continue;

            /* Decode 4 trits and accumulate (Patent 36, no multiply) */
            decode_byte_f32(byte, &input[p << 2], &acc);
        }

        output[i] = acc * alpha;
        if (bias) {
            output[i] += bias[i];
        }
    }

    return TERN_OK;
}

/* ══════════════════════════════════════════════════════════════════════
 * Packed ternary matrix multiply (batched)
 *
 *   output[b,i] = alpha * sum_j(W[i,j] * input[b,j]) + bias[i]
 *
 * Each batch element is processed independently.  The packed weight
 * array is shared across the batch.
 * ═════════════════════════════════════════════════════════════════════*/
int tern_packed_matmul_f32(
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

        int rc = tern_packed_matvec_f32(packed, in_b, out_b, M, N, alpha, bias);
        if (rc != TERN_OK) return rc;
    }

    return TERN_OK;
}

/* ══════════════════════════════════════════════════════════════════════
 * Packed ternary matrix-vector multiply with bitmap zero-skip
 *
 *   output[i] = alpha * sum_j(W[i,j] * x[j]) + bias[i]
 *
 * Adds a packed sparsity bitmap for coarser block skipping on top
 * of the per-byte zero check.  The bitmap enables 8-weight block
 * skipping (1 bitmap byte = 8 weights = 2 packed bytes): when a
 * bitmap byte is 0x00, two packed bytes are skipped without reading
 * the weight data at all.
 *
 * The function handles N values that are multiples of 4 but not
 * necessarily multiples of 8 by processing in three phases:
 *   Phase 1: unaligned head (0-7 weights to reach bitmap byte
 *            boundary, processed as individual packed bytes)
 *   Phase 2: aligned body (8 weights per bitmap byte = 2 packed
 *            bytes; skip both when bitmap byte is 0x00)
 *   Phase 3: remaining tail (< 8 weights, as individual packed bytes)
 *
 * Patent 37: bitmap-driven 8-weight block zero-skip.
 * Patent 39: packed trit storage format.
 * ═════════════════════════════════════════════════════════════════════*/
int tern_packed_matvec_f32_sparse(
    const uint8_t *packed,
    const float   *input,
    float         *output,
    const uint8_t *bitmap,
    int M, int N,
    float alpha,
    const float *bias)
{
    if (!packed || !input || !output || !bitmap) return TERN_ERR_NULL;
    if (M <= 0 || N <= 0)                       return TERN_ERR_DIM;
    if (N & 3)                                  return TERN_ERR_ALIGN;

    const int packed_cols = N >> 2;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        float acc = 0.0f;
        const size_t row_start = (size_t)i * (size_t)N;
        const uint8_t *packed_row = packed + (size_t)i * (size_t)packed_cols;

        int j = 0;   /* weight index within this row */
        int p = 0;   /* packed byte index within this row */

        /* ── Phase 1: unaligned head ──────────────────────────────
         * If row_start is not aligned to a bitmap byte boundary
         * (i.e., row_start % 8 != 0), process individual packed
         * bytes (4 weights each) until we reach alignment.
         *
         * Because N % 4 == 0, row_start is always a multiple of 4,
         * so the misalignment is either 0 or 4 (i.e., at most one
         * packed byte of head processing).
         */
        if (row_start & 7) {
            /* row_start % 8 == 4 (since row_start is a multiple of 4) */
            uint8_t byte = packed_row[0];
            if (byte != 0) {
                decode_byte_f32(byte, &input[0], &acc);
            }
            j = 4;
            p = 1;
        }

        /* ── Phase 2: aligned body (8 weights = 1 bitmap byte) ────
         * Each bitmap byte covers 8 consecutive weights.  When
         * the bitmap byte is 0x00, skip 2 packed bytes at once
         * without reading the packed weight data.
         * (Patent 37: bitmap-driven 8-weight block zero-skip)
         */
        while (j + 8 <= N) {
            size_t bm_idx = (row_start + (size_t)j) >> 3;
            uint8_t bm = bitmap[bm_idx];

            if (bm == 0) {
                /* All 8 weights zero — skip 2 packed bytes (Patent 37) */
                j += 8;
                p += 2;
                continue;
            }

            /* Bitmap says at least one non-zero in this group.
             * Fall through to per-byte processing with inline
             * 4-weight zero-check.
             */
            uint8_t b0 = packed_row[p];
            if (b0 != 0) {
                decode_byte_f32(b0, &input[j], &acc);
            }
            p++;
            j += 4;

            uint8_t b1 = packed_row[p];
            if (b1 != 0) {
                decode_byte_f32(b1, &input[j], &acc);
            }
            p++;
            j += 4;
        }

        /* ── Phase 3: remaining tail ──────────────────────────────
         * Process any remaining 4-weight group (at most one packed
         * byte when N % 8 == 4).
         */
        if (j + 4 <= N) {
            uint8_t byte = packed_row[p];
            if (byte != 0) {
                decode_byte_f32(byte, &input[j], &acc);
            }
            /* j += 4; p += 1; — not needed, end of row */
        }

        output[i] = acc * alpha;
        if (bias) {
            output[i] += bias[i];
        }
    }

    return TERN_OK;
}

/* ══════════════════════════════════════════════════════════════════════
 * Packed ternary matrix multiply with bitmap (batched)
 *
 * Each batch element is processed independently.  The packed weight
 * array and bitmap are shared across the batch.
 * ═════════════════════════════════════════════════════════════════════*/
int tern_packed_matmul_f32_sparse(
    const uint8_t *packed,
    const float   *input,
    float         *output,
    const uint8_t *bitmap,
    int M, int N, int B,
    float alpha,
    const float *bias)
{
    if (!packed || !input || !output || !bitmap) return TERN_ERR_NULL;
    if (M <= 0 || N <= 0 || B <= 0)             return TERN_ERR_DIM;
    if (N & 3)                                  return TERN_ERR_ALIGN;

    for (int b = 0; b < B; b++) {
        const float *in_b  = input  + (size_t)b * (size_t)N;
        float       *out_b = output + (size_t)b * (size_t)M;

        int rc = tern_packed_matvec_f32_sparse(
            packed, in_b, out_b, bitmap, M, N, alpha, bias);
        if (rc != TERN_OK) return rc;
    }

    return TERN_OK;
}
