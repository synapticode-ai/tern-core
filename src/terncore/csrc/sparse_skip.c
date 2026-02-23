/*
 * sparse_skip.c — 64-bit word bitmap-driven zero-skip engine
 *
 * Processes 64 weights per bitmap word.  When a uint64 word is zero,
 * the entire 64-weight block is skipped without touching the weight
 * or input arrays.  For non-zero words, a CTZ (count-trailing-zeros)
 * bit-scan loop iterates only over the set bits — compute is
 * proportional to non-zero weight count, not total weight count.
 *
 * With 65% sparsity, only ~35% of weights are visited.  At the
 * block level, any 64-weight region that is entirely zero is
 * skipped with a single uint64 comparison.
 *
 * The bitmap uses the same flat LSB-first format as ternary_matmul.h
 * (1 bit per weight, ceil(M*N / 8) bytes).  It is loaded as uint64
 * little-endian words.  This is correct on x86 and ARM-LE (our
 * target platforms).
 *
 * Patent 37: Zero-weight clock-gating → sparsity-aware skip logic.
 * Patent 39: Ternary-native memory → packed trit storage format.
 *
 * All functions are deterministic: bit-scan processes bits from
 * LSB to MSB (ascending column order), matching the left-to-right
 * accumulation of the dense kernels.  (Patent 36)
 *
 * Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
 */

#include "sparse_skip.h"
#include <string.h>   /* memcpy */

#ifdef _OPENMP
#include <omp.h>
#endif

/* ─────────────────────────────────────────────────────────────────────
 * Portable count-trailing-zeros for uint64.
 *
 * Returns the index (0-63) of the lowest set bit.
 * UNDEFINED when x == 0 — caller must check before calling.
 * ─────────────────────────────────────────────────────────────────── */
static inline int ctz64(uint64_t x)
{
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(x);
#else
    /* Portable fallback — de Bruijn sequence */
    static const int debruijn_table[64] = {
         0,  1,  2,  7,  3, 13,  8, 19,  4, 25, 14, 28,  9, 34, 20, 40,
         5, 17, 26, 38, 15, 46, 29, 48, 10, 31, 35, 54, 21, 50, 41, 57,
        63,  6, 12, 18, 24, 27, 33, 39, 16, 37, 45, 47, 30, 53, 49, 56,
        62, 11, 23, 32, 36, 44, 52, 55, 61, 22, 43, 51, 60, 42, 59, 58
    };
    return debruijn_table[((x & (uint64_t)(-(int64_t)x)) * UINT64_C(0x0218A392CD3D5DBF)) >> 58];
#endif
}

/* ─────────────────────────────────────────────────────────────────────
 * Load a uint64 bitmap word from the byte array.
 *
 * Uses memcpy for safe, alignment-independent loading.  On LE
 * platforms (x86, ARM-LE), the compiler optimises this to a single
 * 8-byte load instruction.
 *
 * For the final partial word (when total bitmap bytes < offset + 8),
 * loads only the available bytes and zero-pads the rest.
 * ─────────────────────────────────────────────────────────────────── */
static inline uint64_t load_bitmap_word(
    const uint8_t *bitmap, size_t word_idx, size_t bitmap_bytes)
{
    size_t offset = word_idx * 8;
    uint64_t word = 0;

    if (offset + 8 <= bitmap_bytes) {
        memcpy(&word, bitmap + offset, 8);
    } else if (offset < bitmap_bytes) {
        memcpy(&word, bitmap + offset, bitmap_bytes - offset);
    }

    return word;
}

/* ══════════════════════════════════════════════════════════════════════
 * Sparse matvec — unpacked int8 weights, 64-bit bitmap bit-scan
 *
 *   output[i] = alpha * sum_j(W[i,j] * x[j]) + bias[i]
 *               (only where bitmap bit is set)
 *
 * For each row, the bitmap words overlapping the row are loaded.
 * Bits outside the row are masked off.  A CTZ loop then iterates
 * only over the set bits (non-zero weight positions), performing
 * add or subtract based on the weight value.
 *
 * Patent 37: 64-weight block zero-skip + bit-scan iteration.
 * Patent 36: deterministic left-to-right accumulation via LSB-first
 *            bit-scan.
 * ═════════════════════════════════════════════════════════════════════*/
int tern_sparse64_matvec_f32(
    const int8_t  *weights,
    const float   *input,
    float         *output,
    const uint8_t *bitmap,
    int M, int N,
    float alpha,
    const float *bias)
{
    if (!weights || !input || !output || !bitmap) return TERN_ERR_NULL;
    if (M <= 0 || N <= 0) return TERN_ERR_DIM;

    const size_t total = (size_t)M * (size_t)N;
    const size_t bitmap_bytes = (total + 7) >> 3;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        float acc = 0.0f;
        const size_t row_start = (size_t)i * (size_t)N;
        const size_t row_end   = row_start + (size_t)N;

        size_t first_word = row_start >> 6;           /* / 64 */
        size_t last_word  = (row_end - 1) >> 6;

        for (size_t widx = first_word; widx <= last_word; widx++) {
            uint64_t word = load_bitmap_word(bitmap, widx, bitmap_bytes);

            /* 64-weight block zero-skip (Patent 37) */
            if (word == 0) continue;

            size_t word_bit0 = widx << 6;             /* widx * 64 */

            /* Mask off bits before this row */
            if (word_bit0 < row_start) {
                unsigned int shift = (unsigned int)(row_start - word_bit0);
                word &= ~((UINT64_C(1) << shift) - 1);
            }
            /* Mask off bits after this row */
            if (word_bit0 + 64 > row_end) {
                unsigned int keep = (unsigned int)(row_end - word_bit0);
                word &= (UINT64_C(1) << keep) - 1;
            }

            /* Bit-scan: visit only non-zero positions (Patent 37) */
            while (word != 0) {
                int bit = ctz64(word);
                size_t flat = word_bit0 + (unsigned int)bit;
                size_t col  = flat - row_start;

                int8_t wv = weights[flat];
                if (wv == TERN_POS) acc += input[col];
                else                acc -= input[col];

                word &= word - 1;   /* clear lowest set bit */
            }
        }

        output[i] = acc * alpha;
        if (bias) {
            output[i] += bias[i];
        }
    }

    return TERN_OK;
}

/* ══════════════════════════════════════════════════════════════════════
 * Batched sparse matmul — unpacked int8 weights, 64-bit bitmap
 * ═════════════════════════════════════════════════════════════════════*/
int tern_sparse64_matmul_f32(
    const int8_t  *weights,
    const float   *input,
    float         *output,
    const uint8_t *bitmap,
    int M, int N, int B,
    float alpha,
    const float *bias)
{
    if (!weights || !input || !output || !bitmap) return TERN_ERR_NULL;
    if (M <= 0 || N <= 0 || B <= 0) return TERN_ERR_DIM;

    for (int b = 0; b < B; b++) {
        const float *in_b  = input  + (size_t)b * (size_t)N;
        float       *out_b = output + (size_t)b * (size_t)M;

        int rc = tern_sparse64_matvec_f32(
            weights, in_b, out_b, bitmap, M, N, alpha, bias);
        if (rc != TERN_OK) return rc;
    }

    return TERN_OK;
}

/* ══════════════════════════════════════════════════════════════════════
 * Sparse matvec — packed 2-bit weights, 64-bit bitmap bit-scan
 *
 * Same as the unpacked variant but reads individual trits from the
 * packed array on-the-fly at positions indicated by the bitmap.
 *
 * Trit extraction for weight at flat index k:
 *   packed byte  = packed[k >> 2]
 *   shift amount = (k & 3) << 1       (0, 2, 4, or 6)
 *   trit value   = (byte >> shift) & 0x03
 *
 * N must be a multiple of 4 (TERN_ERR_ALIGN otherwise).
 *
 * Patent 37: 64-weight block skip + bit-scan.
 * Patent 39: packed trit extraction.
 * ═════════════════════════════════════════════════════════════════════*/
int tern_sparse64_packed_matvec_f32(
    const uint8_t *packed,
    const float   *input,
    float         *output,
    const uint8_t *bitmap,
    int M, int N,
    float alpha,
    const float *bias)
{
    if (!packed || !input || !output || !bitmap) return TERN_ERR_NULL;
    if (M <= 0 || N <= 0) return TERN_ERR_DIM;
    if (N & 3)            return TERN_ERR_ALIGN;

    const size_t total = (size_t)M * (size_t)N;
    const size_t bitmap_bytes = (total + 7) >> 3;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        float acc = 0.0f;
        const size_t row_start = (size_t)i * (size_t)N;
        const size_t row_end   = row_start + (size_t)N;

        size_t first_word = row_start >> 6;
        size_t last_word  = (row_end - 1) >> 6;

        for (size_t widx = first_word; widx <= last_word; widx++) {
            uint64_t word = load_bitmap_word(bitmap, widx, bitmap_bytes);

            if (word == 0) continue;

            size_t word_bit0 = widx << 6;

            if (word_bit0 < row_start) {
                unsigned int shift = (unsigned int)(row_start - word_bit0);
                word &= ~((UINT64_C(1) << shift) - 1);
            }
            if (word_bit0 + 64 > row_end) {
                unsigned int keep = (unsigned int)(row_end - word_bit0);
                word &= (UINT64_C(1) << keep) - 1;
            }

            while (word != 0) {
                int bit = ctz64(word);
                size_t flat = word_bit0 + (unsigned int)bit;
                size_t col  = flat - row_start;

                /*
                 * Extract one trit from packed array (Patent 39):
                 *   byte_idx  = flat / 4
                 *   trit_shift = (flat % 4) * 2
                 */
                unsigned int trit_shift = (unsigned int)(flat & 3) << 1;
                int trit = (packed[flat >> 2] >> trit_shift) & TRIT_MASK;

                if (trit == TRIT_POS) acc += input[col];
                else                  acc -= input[col]; /* TRIT_NEG */

                word &= word - 1;
            }
        }

        output[i] = acc * alpha;
        if (bias) {
            output[i] += bias[i];
        }
    }

    return TERN_OK;
}

/* ══════════════════════════════════════════════════════════════════════
 * Batched sparse matmul — packed 2-bit weights, 64-bit bitmap
 * ═════════════════════════════════════════════════════════════════════*/
int tern_sparse64_packed_matmul_f32(
    const uint8_t *packed,
    const float   *input,
    float         *output,
    const uint8_t *bitmap,
    int M, int N, int B,
    float alpha,
    const float *bias)
{
    if (!packed || !input || !output || !bitmap) return TERN_ERR_NULL;
    if (M <= 0 || N <= 0 || B <= 0) return TERN_ERR_DIM;
    if (N & 3)                      return TERN_ERR_ALIGN;

    for (int b = 0; b < B; b++) {
        const float *in_b  = input  + (size_t)b * (size_t)N;
        float       *out_b = output + (size_t)b * (size_t)M;

        int rc = tern_sparse64_packed_matvec_f32(
            packed, in_b, out_b, bitmap, M, N, alpha, bias);
        if (rc != TERN_OK) return rc;
    }

    return TERN_OK;
}
