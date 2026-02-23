/*
 * ternary_avx2.c — AVX2 intrinsic kernels for packed ternary matmul
 *
 * Replaces the branching decode_byte_f32 inner loop with branchless
 * SIMD mask-and-blend operations.  Processes 2 packed bytes (8 trits,
 * 8 float32 inputs) per AVX2 iteration using 256-bit vectors.
 *
 * Algorithm per iteration:
 *   1. Load 2 packed bytes, quick zero-skip if both are 0x00
 *   2. LUT lookup → 8-bit positive/negative masks
 *   3. Expand masks to 8×32-bit lane masks via _mm256_cmpeq_epi32
 *   4. Load 8 input floats, blend with masks (branchless add/sub/skip)
 *   5. Store to temp, accumulate left-to-right for bit-identical output
 *
 * The sequential accumulation (step 5) preserves the exact IEEE 754
 * addition order of the scalar kernel, guaranteeing bit-identical
 * results.  The performance win comes from eliminating 8 branch
 * mispredictions per 2-byte group (steps 1-4).
 *
 * Compile with: cc -std=c11 -O2 -mavx2 -c ternary_avx2.c
 *
 * Patent 37: Zero-weight clock-gating — SIMD branchless skip logic.
 * Patent 38: Configurable precision — AVX2 execution path.
 * Patent 39: Ternary-native memory — packed trit storage format.
 *
 * Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
 */

#include "ternary_simd.h"

#if defined(__x86_64__) || defined(_M_X64)

#include <immintrin.h>   /* AVX2 intrinsics */

#ifdef _OPENMP
#include <omp.h>
#endif

/* ── Trit-to-mask lookup tables ──────────────────────────────────────
 *
 * For each possible packed byte (0-255), precompute which of the
 * 4 trits are positive (+1) and which are negative (-1).
 *
 * pos_lut[byte]: bit k set if trit k == TRIT_POS (0b01)
 * neg_lut[byte]: bit k set if trit k == TRIT_NEG (0b10)
 *
 * Each LUT is 256 bytes — both fit in a single L1 cache line pair.
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

/* ══════════════════════════════════════════════════════════════════════
 * tern_packed_matvec_f32_avx2 — AVX2 packed ternary matvec
 *
 *   output[i] = alpha * sum_j(W[i,j] * x[j]) + bias[i]
 *
 * Processes 8 weights (2 packed bytes) per AVX2 iteration.
 * Tail (when packed_cols is odd) handled by scalar decode_byte_f32.
 *
 * Patent 37: 8-weight block zero-skip.
 * Patent 38: AVX2 SIMD execution path.
 * ═════════════════════════════════════════════════════════════════════*/
int tern_packed_matvec_f32_avx2(
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

    const int packed_cols = N >> 2;        /* N / 4 packed bytes per row */
    const int simd_pairs  = packed_cols >> 1;  /* pairs of 2 bytes       */
    const int has_tail    = packed_cols & 1;    /* 1 if N/4 is odd        */

    /*
     * Bit-expansion selector: each lane selects one bit from the
     * 8-bit mask.  Used to expand an 8-bit pos/neg mask to 8×32-bit
     * all-ones/all-zeros lanes for _mm256_blendv_ps.
     *
     * Lane 0 checks bit 0, lane 1 checks bit 1, ..., lane 7 checks bit 7.
     */
    const __m256i bit_sel = _mm256_set_epi32(
        1 << 7, 1 << 6, 1 << 5, 1 << 4,
        1 << 3, 1 << 2, 1 << 1, 1 << 0);
    const __m256  zero_v  = _mm256_setzero_ps();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        float acc = 0.0f;
        const uint8_t *row = packed + (size_t)i * (size_t)packed_cols;

        int p = 0;
        for (int s = 0; s < simd_pairs; s++, p += 2) {
            uint8_t b0 = row[p];
            uint8_t b1 = row[p + 1];

            /* Prefetch next packed weight chunk (Patent 40) */
            if (s + 1 < simd_pairs)
                _mm_prefetch((const char *)&row[p + 2], _MM_HINT_T0);

            /* 8-weight block zero-skip (Patent 37) */
            if ((b0 | b1) == 0) continue;

            /* LUT → combined 8-bit masks */
            uint8_t pos8 = pos_lut[b0] | (uint8_t)(pos_lut[b1] << 4);
            uint8_t neg8 = neg_lut[b0] | (uint8_t)(neg_lut[b1] << 4);

            /* Expand 8-bit masks to 8×32-bit AVX2 lane masks */
            __m256i pexp = _mm256_set1_epi32((int32_t)pos8);
            __m256i pos_i = _mm256_cmpeq_epi32(
                _mm256_and_si256(pexp, bit_sel), bit_sel);
            __m256  pos_m = _mm256_castsi256_ps(pos_i);

            __m256i nexp = _mm256_set1_epi32((int32_t)neg8);
            __m256i neg_i = _mm256_cmpeq_epi32(
                _mm256_and_si256(nexp, bit_sel), bit_sel);
            __m256  neg_m = _mm256_castsi256_ps(neg_i);

            /* Load 8 input floats */
            __m256 inp = _mm256_loadu_ps(&input[p << 2]);

            /* Branchless blend: select inputs for positive/negative trits */
            __m256 pos_v = _mm256_blendv_ps(zero_v, inp, pos_m);
            __m256 neg_v = _mm256_blendv_ps(zero_v, inp, neg_m);
            __m256 result = _mm256_sub_ps(pos_v, neg_v);

            /*
             * Sequential accumulation for bit-identical output (Patent 36).
             *
             * We store the 8 result floats to a temp array and add them
             * left-to-right, matching the scalar kernel's accumulation
             * order exactly.  The SIMD benefit comes from the branchless
             * mask-and-blend above, not from parallel accumulation.
             */
            float tmp[8];
            _mm256_storeu_ps(tmp, result);
            acc += tmp[0]; acc += tmp[1]; acc += tmp[2]; acc += tmp[3];
            acc += tmp[4]; acc += tmp[5]; acc += tmp[6]; acc += tmp[7];
        }

        /* Handle 4-float tail when packed_cols is odd */
        if (has_tail) {
            uint8_t byte = row[p];
            if (byte != 0) {
                decode_byte_f32(byte, &input[p << 2], &acc);
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
 * tern_packed_matmul_f32_avx2 — AVX2 packed ternary matmul (batched)
 *
 * Each batch element is processed independently via the AVX2 matvec.
 * The packed weight array is shared across the batch.
 * ═════════════════════════════════════════════════════════════════════*/
int tern_packed_matmul_f32_avx2(
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

        int rc = tern_packed_matvec_f32_avx2(
            packed, in_b, out_b, M, N, alpha, bias);
        if (rc != TERN_OK) return rc;
    }

    return TERN_OK;
}

#endif /* __x86_64__ || _M_X64 */
