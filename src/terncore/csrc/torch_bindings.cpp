/*
 * torch_bindings.cpp — PyTorch C++ extension for ternary compute kernels
 *
 * Provides zero-copy access to the C kernels from PyTorch tensors via
 * torch::utils::cpp_extension.  Eliminates the numpy/ctypes overhead
 * (~100-200us per call) by passing tensor.data_ptr<>() directly to
 * the C kernel.
 *
 * JIT-compiled by torch.utils.cpp_extension.load() at import time.
 *
 * Patent 37: Zero-weight clock-gating — sparsity-aware skip logic.
 * Patent 38: Configurable precision — dual-path dispatch.
 * Patent 39: Ternary-native memory — packed trit storage format.
 *
 * Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
 */

#include <torch/extension.h>

/* ── C kernel declarations ─────────────────────────────────────────── */

extern "C" {

int ternary_matmul_f32_simd(
    const uint8_t *packed_weights,
    const float   *input,
    float         *output,
    const uint8_t *bitmap,
    float          alpha,
    const float   *bias,
    int M, int N, int B);

int ternary_matmul_f32(
    const uint8_t *packed_weights,
    const float   *input,
    float         *output,
    const uint8_t *bitmap,
    float          alpha,
    const float   *bias,
    int M, int N, int B);

uint32_t get_simd_support(void);
const char *terncore_version(void);

}  /* extern "C" */

/* ══════════════════════════════════════════════════════════════════════
 * ternary_forward — Zero-copy torch tensor → C kernel → torch tensor
 *
 *   output = W_packed @ input  (ternary matmul with alpha/bias)
 *
 * All inputs are torch tensors on CPU.  No numpy conversion needed.
 *
 * Patent 38: Configurable precision — SIMD dispatch.
 * ═════════════════════════════════════════════════════════════════════*/
torch::Tensor ternary_forward(
    torch::Tensor packed_weights,   /* [M*N/4] uint8                   */
    torch::Tensor input,            /* [B, N]  float32, contiguous     */
    torch::Tensor bitmap,           /* [ceil(M*N/8)] uint8, or empty   */
    double alpha,
    torch::Tensor bias,             /* [M] float32, or empty           */
    int64_t M,
    int64_t N)
{
    TORCH_CHECK(packed_weights.is_contiguous(), "packed_weights must be contiguous");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(packed_weights.dtype() == torch::kUInt8, "packed_weights must be uint8");

    /* Handle batch dimension */
    int64_t B = input.size(0);
    TORCH_CHECK(input.size(1) == N, "input dim 1 must match N");

    /* Allocate output tensor */
    auto output = torch::zeros({B, M}, torch::dtype(torch::kFloat32));

    /* Prepare pointers */
    const uint8_t *pw_ptr = packed_weights.data_ptr<uint8_t>();
    const float   *in_ptr = input.data_ptr<float>();
    float         *out_ptr = output.data_ptr<float>();

    const uint8_t *bm_ptr = nullptr;
    if (bitmap.numel() > 0) {
        TORCH_CHECK(bitmap.is_contiguous(), "bitmap must be contiguous");
        TORCH_CHECK(bitmap.dtype() == torch::kUInt8, "bitmap must be uint8");
        bm_ptr = bitmap.data_ptr<uint8_t>();
    }

    const float *bias_ptr = nullptr;
    if (bias.numel() > 0) {
        TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
        bias_ptr = bias.data_ptr<float>();
    }

    /* Call SIMD-dispatched C kernel (Patent 38) */
    int rc = ternary_matmul_f32_simd(
        pw_ptr, in_ptr, out_ptr, bm_ptr,
        static_cast<float>(alpha), bias_ptr,
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(B));

    TORCH_CHECK(rc == 0, "ternary_matmul_f32_simd failed with rc=", rc);

    return output;
}

/* ══════════════════════════════════════════════════════════════════════
 * Python module registration
 * ═════════════════════════════════════════════════════════════════════*/
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ternary_forward", &ternary_forward,
          "Zero-copy ternary matmul via SIMD-dispatched C kernel (Patent 38)");
    m.def("get_simd_support", []() -> uint32_t { return get_simd_support(); },
          "Runtime SIMD capability detection (Patent 38)");
    m.def("terncore_version", []() -> std::string { return terncore_version(); },
          "Library version string");
}
