/*
 * test_sparse_skip.c — Tests for sparse_skip 64-bit engine + bindings
 *
 * Validates:
 *   1. sparse64 unpacked matches dense unpacked (tern_matvec_f32)
 *   2. sparse64 packed matches packed (tern_packed_matvec_f32)
 *   3. Various matrix sizes (aligned and unaligned to 64)
 *   4. High sparsity patterns
 *   5. Batched variants
 *   6. Error handling
 *   7. Determinism
 *   8. bindings dispatch (ternary_matmul_f32)
 *   9. get_simd_support, terncore_version
 */

#include "ternary_matmul.h"
#include "ternary_packed.h"
#include "sparse_skip.h"
#include "terncore_version.h"  /* TERNCORE_VERSION_STRING (build-time, from pyproject.toml) */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── bindings.c forward declarations (no separate header) ────────── */
int ternary_matmul_f32(
    const uint8_t *packed_weights, const float *input, float *output,
    const uint8_t *bitmap, float alpha, const float *bias,
    int M, int N, int B);
int ternary_matmul_f32_simd(
    const uint8_t *packed_weights, const float *input, float *output,
    const uint8_t *bitmap, float alpha, const float *bias,
    int M, int N, int B);
uint32_t get_simd_support(void);
const char *terncore_version(void);

#define ASSERT(cond, msg)                                         \
    do {                                                          \
        if (!(cond)) {                                            \
            fprintf(stderr, "FAIL [%s:%d] %s\n",                  \
                    __FILE__, __LINE__, (msg));                    \
            failures++;                                           \
        }                                                         \
    } while (0)

#define ASSERT_NEAR(a, b, tol, msg)                               \
    do {                                                          \
        if (fabsf((a) - (b)) > (tol)) {                           \
            fprintf(stderr, "FAIL [%s:%d] %s: got %f, want %f\n", \
                    __FILE__, __LINE__, (msg), (a), (b));          \
            failures++;                                           \
        }                                                         \
    } while (0)

static int failures = 0;

/* ── Helpers ─────────────────────────────────────────────────────── */

static uint8_t encode_trit(int8_t w)
{
    if (w == 1)  return 0x01;
    if (w == -1) return 0x02;
    return 0x00;
}

static void pack_weights(const int8_t *w, uint8_t *packed, int count)
{
    for (int i = 0; i < count; i += 4) {
        packed[i >> 2] = (uint8_t)(
              encode_trit(w[i])
            | (encode_trit(w[i + 1]) << 2)
            | (encode_trit(w[i + 2]) << 4)
            | (encode_trit(w[i + 3]) << 6));
    }
}

static void build_bitmap(const int8_t *w, uint8_t *bm, int count)
{
    int nbytes = (count + 7) / 8;
    memset(bm, 0, (size_t)nbytes);
    for (int k = 0; k < count; k++) {
        if (w[k] != 0)
            bm[k >> 3] |= (uint8_t)(1u << (k & 7));
    }
}

/* Fill weight array with pattern (~65% zeros) */
static void fill_sparse(int8_t *w, int count)
{
    for (int k = 0; k < count; k++) {
        int mod = k % 10;
        if (mod < 6)      w[k] = 0;
        else if (mod < 8) w[k] = 1;
        else              w[k] = -1;
    }
}

/* ── Test 1: sparse64 unpacked matches dense — small ─────────────── */
static void test_sparse64_matches_dense_small(void)
{
    int8_t  W[] = {1, -1, 0, 0, 1, 1, -1, 0, 0, 1, -1, 1};
    float   input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float   bias[] = {0.1f, 0.2f, 0.3f};
    uint8_t bitmap[2]; /* ceil(12/8) */
    float   dense_out[3], sparse_out[3];

    build_bitmap(W, bitmap, 12);
    tern_matvec_f32(W, input, dense_out, 3, 4, 0.5f, bias);
    int rc = tern_sparse64_matvec_f32(
        W, input, sparse_out, bitmap, 3, 4, 0.5f, bias);

    ASSERT(rc == TERN_OK, "sparse64 small rc");
    for (int i = 0; i < 3; i++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "sparse64==dense row %d", i);
        ASSERT_NEAR(sparse_out[i], dense_out[i], 1e-6f, msg);
    }
}

/* ── Test 2: sparse64 unpacked — N > 64 ──────────────────────────── */
static void test_sparse64_large_n(void)
{
    const int M = 2, N = 128;
    int8_t  *W     = malloc((size_t)(M * N));
    uint8_t *bm    = malloc((size_t)((M * N + 7) / 8));
    float   *input = malloc((size_t)(N) * sizeof(float));
    float   *d_out = malloc((size_t)(M) * sizeof(float));
    float   *s_out = malloc((size_t)(M) * sizeof(float));

    fill_sparse(W, M * N);
    for (int j = 0; j < N; j++) input[j] = (float)(j % 13) * 0.1f;

    build_bitmap(W, bm, M * N);
    tern_matvec_f32(W, input, d_out, M, N, 1.0f, NULL);
    int rc = tern_sparse64_matvec_f32(W, input, s_out, bm, M, N, 1.0f, NULL);

    ASSERT(rc == TERN_OK, "sparse64 N=128 rc");
    for (int i = 0; i < M; i++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "sparse64 N=128 row %d", i);
        ASSERT_NEAR(s_out[i], d_out[i], 1e-5f, msg);
    }

    free(W); free(bm); free(input); free(d_out); free(s_out);
}

/* ── Test 3: sparse64 unpacked — N not multiple of 64 ────────────── */
static void test_sparse64_unaligned_n(void)
{
    const int M = 3, N = 50;
    int8_t  *W     = malloc((size_t)(M * N));
    uint8_t *bm    = malloc((size_t)((M * N + 7) / 8));
    float   *input = malloc((size_t)(N) * sizeof(float));
    float   *d_out = malloc((size_t)(M) * sizeof(float));
    float   *s_out = malloc((size_t)(M) * sizeof(float));

    fill_sparse(W, M * N);
    for (int j = 0; j < N; j++) input[j] = (float)(j + 1) * 0.05f;

    build_bitmap(W, bm, M * N);
    tern_matvec_f32(W, input, d_out, M, N, 0.8f, NULL);
    int rc = tern_sparse64_matvec_f32(W, input, s_out, bm, M, N, 0.8f, NULL);

    ASSERT(rc == TERN_OK, "sparse64 N=50 rc");
    for (int i = 0; i < M; i++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "sparse64 N=50 row %d", i);
        ASSERT_NEAR(s_out[i], d_out[i], 1e-5f, msg);
    }

    free(W); free(bm); free(input); free(d_out); free(s_out);
}

/* ── Test 4: sparse64 packed matches packed dense ────────────────── */
static void test_sparse64_packed_matches_packed(void)
{
    const int M = 4, N = 16;
    int8_t  W[4 * 16];
    uint8_t packed[4 * 4], bitmap[8]; /* ceil(64/8) */
    float   input[16], bias[4];
    float   packed_out[4], sparse_out[4];

    fill_sparse(W, M * N);
    for (int j = 0; j < N; j++) input[j] = (float)(j + 1) * 0.1f;
    for (int i = 0; i < M; i++) bias[i] = (float)i * 0.05f;

    pack_weights(W, packed, M * N);
    build_bitmap(W, bitmap, M * N);

    tern_packed_matvec_f32(packed, input, packed_out, M, N, 1.5f, bias);
    int rc = tern_sparse64_packed_matvec_f32(
        packed, input, sparse_out, bitmap, M, N, 1.5f, bias);

    ASSERT(rc == TERN_OK, "sparse64 packed rc");
    for (int i = 0; i < M; i++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "sparse64_packed row %d", i);
        ASSERT_NEAR(sparse_out[i], packed_out[i], 1e-5f, msg);
    }
}

/* ── Test 5: sparse64 packed — large matrix ──────────────────────── */
static void test_sparse64_packed_large(void)
{
    const int M = 32, N = 256;
    int8_t  *W      = malloc((size_t)(M * N));
    uint8_t *packed  = malloc((size_t)(M * (N / 4)));
    uint8_t *bm      = malloc((size_t)((M * N + 7) / 8));
    float   *input   = malloc((size_t)(N) * sizeof(float));
    float   *bias    = malloc((size_t)(M) * sizeof(float));
    float   *ref_out = malloc((size_t)(M) * sizeof(float));
    float   *sp_out  = malloc((size_t)(M) * sizeof(float));

    fill_sparse(W, M * N);
    for (int j = 0; j < N; j++) input[j] = (float)(j % 17) * 0.1f;
    for (int i = 0; i < M; i++) bias[i] = (float)(i % 5) * 0.01f;

    pack_weights(W, packed, M * N);
    build_bitmap(W, bm, M * N);

    tern_packed_matvec_f32(packed, input, ref_out, M, N, 0.73f, bias);
    int rc = tern_sparse64_packed_matvec_f32(
        packed, input, sp_out, bm, M, N, 0.73f, bias);

    ASSERT(rc == TERN_OK, "sparse64 packed large rc");
    for (int i = 0; i < M; i++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "sparse64 packed large row %d", i);
        ASSERT_NEAR(sp_out[i], ref_out[i], 1e-4f, msg);
    }

    free(W); free(packed); free(bm); free(input);
    free(bias); free(ref_out); free(sp_out);
}

/* ── Test 6: all-zero weights → output is bias only ──────────────── */
static void test_sparse64_all_zeros(void)
{
    int8_t  W[16]; memset(W, 0, 16);
    float   input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float   bias[] = {0.5f, -0.5f, 1.0f, -1.0f};
    uint8_t bitmap[2]; memset(bitmap, 0, 2);
    float   output[4];

    int rc = tern_sparse64_matvec_f32(
        W, input, output, bitmap, 4, 4, 1.0f, bias);

    ASSERT(rc == TERN_OK, "all-zero rc");
    for (int i = 0; i < 4; i++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "all-zero row %d", i);
        ASSERT_NEAR(output[i], bias[i], 1e-6f, msg);
    }
}

/* ── Test 7: batched sparse64 unpacked ───────────────────────────── */
static void test_sparse64_batched_unpacked(void)
{
    const int M = 3, N = 8, B = 4;
    int8_t  W[3 * 8];
    uint8_t bm[(3 * 8 + 7) / 8];
    float   input[4 * 8], bias[3];
    float   d_out[4 * 3], s_out[4 * 3];

    fill_sparse(W, M * N);
    for (int k = 0; k < B * N; k++) input[k] = (float)(k % 7) * 0.3f;
    for (int i = 0; i < M; i++) bias[i] = (float)i * 0.1f;

    build_bitmap(W, bm, M * N);
    tern_matmul_f32(W, input, d_out, M, N, B, 0.6f, bias);
    int rc = tern_sparse64_matmul_f32(
        W, input, s_out, bm, M, N, B, 0.6f, bias);

    ASSERT(rc == TERN_OK, "batched sparse64 rc");
    for (int k = 0; k < B * M; k++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "batched sparse64 [%d]", k);
        ASSERT_NEAR(s_out[k], d_out[k], 1e-5f, msg);
    }
}

/* ── Test 8: batched sparse64 packed ─────────────────────────────── */
static void test_sparse64_batched_packed(void)
{
    const int M = 2, N = 8, B = 3;
    int8_t  W[2 * 8];
    uint8_t packed[2 * 2], bm[(2 * 8 + 7) / 8];
    float   input[3 * 8], bias[2];
    float   ref_out[3 * 2], sp_out[3 * 2];

    fill_sparse(W, M * N);
    for (int k = 0; k < B * N; k++) input[k] = (float)(k + 1) * 0.1f;
    bias[0] = 0.01f; bias[1] = 0.02f;

    pack_weights(W, packed, M * N);
    build_bitmap(W, bm, M * N);

    tern_packed_matmul_f32(packed, input, ref_out, M, N, B, 0.5f, bias);
    int rc = tern_sparse64_packed_matmul_f32(
        packed, input, sp_out, bm, M, N, B, 0.5f, bias);

    ASSERT(rc == TERN_OK, "batched sparse64 packed rc");
    for (int k = 0; k < B * M; k++) {
        char msg[64];
        snprintf(msg, sizeof(msg), "batched sparse64 packed [%d]", k);
        ASSERT_NEAR(sp_out[k], ref_out[k], 1e-6f, msg);
    }
}

/* ── Test 9: error handling ──────────────────────────────────────── */
static void test_sparse64_errors(void)
{
    int8_t  W[] = {1, 0, 0, 0};
    float   input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float   output[1];
    uint8_t bm[] = {0x01};
    uint8_t packed[] = {0x01};

    /* NULL checks */
    ASSERT(tern_sparse64_matvec_f32(NULL, input, output, bm, 1, 4, 1.0f, NULL)
           == TERN_ERR_NULL, "NULL weights");
    ASSERT(tern_sparse64_matvec_f32(W, input, output, NULL, 1, 4, 1.0f, NULL)
           == TERN_ERR_NULL, "NULL bitmap");

    /* Dim checks */
    ASSERT(tern_sparse64_matvec_f32(W, input, output, bm, 0, 4, 1.0f, NULL)
           == TERN_ERR_DIM, "M=0");
    ASSERT(tern_sparse64_matmul_f32(W, input, output, bm, 1, 4, 0, 1.0f, NULL)
           == TERN_ERR_DIM, "B=0");

    /* Packed alignment */
    ASSERT(tern_sparse64_packed_matvec_f32(packed, input, output, bm, 1, 3, 1.0f, NULL)
           == TERN_ERR_ALIGN, "N=3");
}

/* ── Test 10: determinism — bit-identical across 100 runs ────────── */
static void test_sparse64_determinism(void)
{
    const int M = 4, N = 20;
    int8_t  W[4 * 20];
    uint8_t bm[(4 * 20 + 7) / 8];
    float   input[20];
    float   first[4], current[4];

    fill_sparse(W, M * N);
    for (int j = 0; j < N; j++) input[j] = (float)(j + 1) * 0.123f;
    build_bitmap(W, bm, M * N);

    tern_sparse64_matvec_f32(W, input, first, bm, M, N, 0.42f, NULL);

    for (int run = 0; run < 100; run++) {
        tern_sparse64_matvec_f32(W, input, current, bm, M, N, 0.42f, NULL);
        for (int i = 0; i < M; i++) {
            ASSERT(current[i] == first[i], "determinism: bit-identical");
        }
    }
}

/* ── Test 11: bindings dispatch — with bitmap ────────────────────── */
static void test_bindings_dispatch_sparse(void)
{
    const int M = 2, N = 8, B = 2;
    int8_t  W[2 * 8];
    uint8_t packed[2 * 2], bm[(2 * 8 + 7) / 8];
    float   input[2 * 8], bias[2];
    float   direct_out[2 * 2], dispatch_out[2 * 2];

    fill_sparse(W, M * N);
    for (int k = 0; k < B * N; k++) input[k] = (float)(k + 1) * 0.1f;
    bias[0] = 0.1f; bias[1] = 0.2f;

    pack_weights(W, packed, M * N);
    build_bitmap(W, bm, M * N);

    tern_sparse64_packed_matmul_f32(
        packed, input, direct_out, bm, M, N, B, 0.5f, bias);
    int rc = ternary_matmul_f32(
        packed, input, dispatch_out, bm, 0.5f, bias, M, N, B);

    ASSERT(rc == TERN_OK, "dispatch sparse rc");
    for (int k = 0; k < B * M; k++) {
        ASSERT(dispatch_out[k] == direct_out[k], "dispatch == direct (sparse)");
    }
}

/* ── Test 12: bindings dispatch — without bitmap ─────────────────── */
static void test_bindings_dispatch_dense(void)
{
    const int M = 2, N = 8, B = 2;
    int8_t  W[2 * 8];
    uint8_t packed[2 * 2];
    float   input[2 * 8];
    float   direct_out[2 * 2], dispatch_out[2 * 2];

    fill_sparse(W, M * N);
    for (int k = 0; k < B * N; k++) input[k] = (float)(k + 1) * 0.2f;

    pack_weights(W, packed, M * N);

    tern_packed_matmul_f32(packed, input, direct_out, M, N, B, 1.0f, NULL);
    int rc = ternary_matmul_f32(
        packed, input, dispatch_out, NULL, 1.0f, NULL, M, N, B);

    ASSERT(rc == TERN_OK, "dispatch dense rc");
    for (int k = 0; k < B * M; k++) {
        ASSERT(dispatch_out[k] == direct_out[k], "dispatch == direct (dense)");
    }
}

/* ── Test 13: SIMD stub falls back to scalar ─────────────────────── */
static void test_simd_fallback(void)
{
    const int M = 2, N = 8, B = 1;
    int8_t  W[2 * 8];
    uint8_t packed[2 * 2], bm[2];
    float   input[8];
    float   scalar_out[2], simd_out[2];

    fill_sparse(W, M * N);
    for (int j = 0; j < N; j++) input[j] = (float)(j + 1);

    pack_weights(W, packed, M * N);
    build_bitmap(W, bm, M * N);

    ternary_matmul_f32(packed, input, scalar_out, bm, 1.0f, NULL, M, N, B);
    ternary_matmul_f32_simd(packed, input, simd_out, bm, 1.0f, NULL, M, N, B);

    for (int i = 0; i < M; i++) {
        ASSERT(simd_out[i] == scalar_out[i], "simd stub == scalar");
    }
}

/* ── Test 14: get_simd_support ───────────────────────────────────── */
static void test_simd_support(void)
{
    uint32_t caps = get_simd_support();

    /* Scalar must always be set */
    ASSERT(caps & 0x01, "SCALAR flag set");

    /* Phase 2: platform-specific SIMD detection */
#if defined(__x86_64__) || defined(_M_X64)
    printf("  SIMD caps: 0x%02x (AVX2=%s, AVX512=%s)\n",
           caps,
           (caps & 0x02) ? "yes" : "no",
           (caps & 0x04) ? "yes" : "no");
#elif defined(__aarch64__) || defined(_M_ARM64)
    ASSERT(caps & 0x08, "NEON flag set on AArch64");
#endif
}

/* ── Test 15: terncore_version ───────────────────────────────────── */
static void test_version(void)
{
    const char *ver = terncore_version();
    ASSERT(ver != NULL, "version not NULL");
    ASSERT(strcmp(ver, TERNCORE_VERSION_STRING) == 0,
           "terncore_version() returns build-time TERNCORE_VERSION_STRING");
}

/* ── Main ────────────────────────────────────────────────────────── */
int main(void)
{
    printf("Running sparse_skip + bindings tests...\n\n");

    /* sparse64 engine */
    test_sparse64_matches_dense_small();
    test_sparse64_large_n();
    test_sparse64_unaligned_n();
    test_sparse64_packed_matches_packed();
    test_sparse64_packed_large();
    test_sparse64_all_zeros();
    test_sparse64_batched_unpacked();
    test_sparse64_batched_packed();
    test_sparse64_errors();
    test_sparse64_determinism();

    /* bindings dispatch */
    test_bindings_dispatch_sparse();
    test_bindings_dispatch_dense();
    test_simd_fallback();
    test_simd_support();
    test_version();

    printf("\n%s (%d failure%s)\n",
           failures == 0 ? "ALL TESTS PASSED" : "TESTS FAILED",
           failures, failures == 1 ? "" : "s");

    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
