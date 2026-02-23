# Stage 1B Benchmark Results

Microbenchmark comparing **TernaryLinearAccel** (C + AVX2 SIMD) against
**TernaryLinear** (pure PyTorch / BLAS) across four matrix sizes at ~65%
ternary weight sparsity.

## System Information

| Item | Value |
|------|-------|
| Machine | iMac (2019), Intel Core i9-9900K @ 3.60 GHz, 8 cores / 16 threads |
| Architecture | x86_64 |
| OS | macOS (Darwin 24.6.0) |
| Compiler | Apple clang 17.0.0 (clang-1700.0.13.3) |
| Python | 3.11.14 |
| PyTorch | 2.2.2 (CPU, linked against Accelerate BLAS) |
| C library | libterncore v0.1.0 |
| SIMD | AVX2 detected via CPUID (AVX-512 not available) |

## Methodology

### Configuration

- **Batch size**: 1 (single-sample matvec — the inference hot path)
- **Warmup**: 100 iterations (populate caches, stabilise branch predictors)
- **Measured iterations**: 1,000 per configuration
- **Random seed**: 42 (fixed for reproducibility, Patent 36)
- **Quantisation threshold**: 0.7
- **Target sparsity**: ~65% (achieved via zero-inflated weight distribution)

### Weight Construction

Standard normal weights with threshold 0.7 yield only ~42% sparsity.
Real ternary models typically have 60-70% zero weights because trained
weight distributions are peaked at zero.  To simulate this realistic
operating point, the benchmark constructs a zero-inflated weight
distribution using binary search to find the mixing fraction that
produces ~65% sparsity after ternary quantisation.

### Timing

Each iteration is timed with `time.perf_counter()` (sub-microsecond
resolution on macOS).  Statistics are computed over the 1,000 measured
iterations after warmup.

### Backends

| Backend | Description |
|---------|-------------|
| **PyTorch** | `TernaryLinear._forward_eval()` — calls `F.linear(x, W * alpha, bias)` which dispatches to Apple Accelerate BLAS (AVX2/FMA-optimised SGEMV). |
| **C+SIMD** | `TernaryLinearAccel._forward_eval()` — packs ternary weights to 2-bit format, calls `ternary_matmul_f32_simd()` via ctypes. AVX2 kernel uses branchless mask-and-blend with 256-entry LUT, sequential accumulation for bit-identical output. |

## Results

### Latency (microseconds)

| Size | Sparsity | PyTorch (mean +/- std) | C+SIMD (mean +/- std) | Speedup |
|------|----------|----------------------|---------------------|---------|
| 256 x 256 | 65.2% | 28.1 +/- 7.9 | 132.3 +/- 39.5 | 0.21x |
| 512 x 512 | 65.3% | 27.8 +/- 9.0 | 447.2 +/- 31.2 | 0.06x |
| 1024 x 1024 | 65.4% | 270.9 +/- 42.3 | 1,667.8 +/- 50.0 | 0.16x |
| 2048 x 2048 | 65.4% | 2,645.3 +/- 370.1 | 6,613.6 +/- 1,570.1 | 0.40x |

### Memory Footprint (bytes, weight storage only)

| Size | Weight Params | FP32 | 2-bit Packed | Bitmap | Ternary Total | Compression |
|------|--------------|------|-------------|--------|---------------|-------------|
| 256 x 256 | 65,536 | 262,144 | 16,384 | 8,192 | 24,580 | **10.7x** |
| 512 x 512 | 262,144 | 1,048,576 | 65,536 | 32,768 | 98,308 | **10.7x** |
| 1024 x 1024 | 1,048,576 | 4,194,304 | 262,144 | 131,072 | 393,220 | **10.7x** |
| 2048 x 2048 | 4,194,304 | 16,777,216 | 1,048,576 | 524,288 | 1,572,868 | **10.7x** |

## Key Findings

### 1. Memory Compression: 10.7x vs FP32

The primary finding.  Ternary 2-bit packing with sparsity bitmap
achieves a consistent **10.7x compression ratio** over FP32 weight
storage across all matrix sizes.

Breakdown per weight:
- **FP32**: 32 bits
- **Ternary packed**: 2 bits (4 weights per byte, encoding: 01=+1, 10=-1, 00=0)
- **Sparsity bitmap**: 1 bit (for zero-skip acceleration)
- **Alpha scalar**: 4 bytes per layer (amortised to ~0 per weight)
- **Effective**: 3 bits per weight = **10.67x** compression

This compression applies at rest (storage), in transit (bandwidth),
and at execution time (cache footprint).  For TinyLlama-1.1B with
~1.1 billion weights, this reduces weight storage from ~4.2 GB (FP32)
to ~393 MB.

### 2. Deterministic Execution: Bit-Identical Output

Both the C scalar and AVX2 SIMD kernels produce **bit-identical** output
for the same input (verified with exact `==` comparison across 100 runs
in both C and Python test suites).

The AVX2 kernel achieves this by:
1. Using SIMD only for branchless trit decoding (mask-and-blend via LUT)
2. Extracting results to a scalar temporary array
3. Accumulating left-to-right in the same order as the scalar kernel

This guarantees IEEE 754 addition order equivalence (Patent 36).

### 3. Latency: PyTorch BLAS is Faster (Expected)

The C+SIMD kernel is currently **2.5-16x slower** than PyTorch for
forward-pass latency.  This is expected and understood:

**Why PyTorch is faster:**

1. **Optimised BLAS**: PyTorch's `F.linear` dispatches to Apple Accelerate
   (or MKL on Linux), which uses AVX2 FMA instructions with:
   - Fully parallel 8-wide accumulation (no sequential constraint)
   - Cache-optimised tiling for large matrices
   - Decades of hand-tuned assembly

2. **ctypes overhead**: Every forward pass through our C kernel:
   - Detaches and copies the PyTorch tensor to numpy (`.detach().cpu().float().numpy()`)
   - Marshals pointers through ctypes
   - Allocates a numpy output buffer
   - Copies the result back to a PyTorch tensor
   - This fixed overhead dominates at small matrix sizes (256x256: ~100 us overhead)

3. **Sequential accumulation**: Our kernel accumulates 8 SIMD lanes
   sequentially for bit-identical output.  BLAS uses tree-reduction
   or parallel accumulation, trading reproducibility for throughput.

**Why the gap narrows at larger sizes (0.06x -> 0.40x):**

The ctypes overhead is fixed (~100-200 us) while compute scales as
O(M*N).  At 2048x2048, the fixed overhead is amortised and the kernel's
zero-skip advantage begins to show (65% of multiply-accumulate
operations are eliminated).

## Optimisation Path (Phase 4)

The latency gap is addressable.  Planned optimisations ranked by
expected impact:

### Tier 1: Eliminate ctypes overhead (expected 5-10x improvement)

1. **PyTorch C extension** (pybind11 or `torch.utils.cpp_extension`):
   Replace ctypes marshalling with direct tensor memory access.
   Zero-copy: read input tensor data pointer, write to output tensor
   directly.  Eliminates ~100-200 us of per-call overhead.

2. **Custom autograd Function**: Register the C kernel as a proper
   `torch.autograd.Function` so it participates in PyTorch's dispatch
   without numpy round-trips.

### Tier 2: Improve kernel throughput (expected 2-4x improvement)

3. **Parallel accumulation with Kahan summation**: Replace sequential
   `acc += tmp[k]` with 4-wide partial sums and compensated final
   reduction.  Maintains near-bit-identical output while enabling
   4x throughput per row.

4. **Cache tiling**: Process weight matrix in L1-sized blocks
   (32-64 KB) to avoid cache thrashing at large matrix sizes.

5. **Sparse SIMD kernel**: Extend AVX2 mask-and-blend to the sparse
   bitmap path.  Current sparse path uses scalar bit-scan; SIMD
   `PDEP`/`PEXT` (BMI2) could process 8 bitmap bits at once.

### Tier 3: Architectural improvements (expected 1.5-2x improvement)

6. **Batched GEMM dispatch**: For B > 1, tile across both batch and
   output dimensions for better utilisation of AVX2 registers.

7. **Prefetch hints**: Insert `_mm_prefetch` for the next weight row
   while processing the current one.

8. **INT8 accumulation**: For the trit add/subtract path, accumulate
   in INT8 before final FP32 conversion.  Doubles effective SIMD
   width (32 INT8 lanes per AVX2 register vs 8 FP32 lanes).

### Projected Performance

With Tier 1 + Tier 2 optimisations, the C kernel should reach
**parity or better** vs PyTorch BLAS at 1024x1024 and above, while
maintaining bit-identical determinism and 10.7x memory compression.

The fundamental advantage of ternary compute — **eliminating 65% of
MAC operations** — is currently masked by overhead.  Once overhead is
removed, the kernel's O(0.35 * M * N) effective FLOP count vs BLAS's
O(M * N) becomes the dominant factor.

## Reproducing These Results

```bash
# Build C library
cd src/terncore/csrc && make clean && make && cd ../../..

# Run C tests (53 tests, zero warnings)
cd src/terncore/csrc && make test && cd ../../..

# Run Python tests (65 tests)
pytest tests/ -v

# Run benchmark
python benchmarks/bench_stage1b.py

# JSON output only
python benchmarks/bench_stage1b.py --json-only

# Custom iteration counts
python benchmarks/bench_stage1b.py --warmup 200 --iters 2000
```

## Patent Coverage

| Patent | Claim | Demonstrated By |
|--------|-------|-----------------|
| Patent 36 | Deterministic execution | 100-run bit-identical tests (C + Python) |
| Patent 37 | Zero-weight clock-gating | Sparsity bitmap zero-skip in sparse kernel |
| Patent 38 | Configurable precision | CPUID detection, AVX2/NEON/scalar dispatch |
| Patent 39 | Ternary-native memory | 2-bit packed format, 10.7x compression |

---

*Results generated 2026-02-23 on Darwin x86_64 (i9-9900K, AVX2).*
*Benchmark script: `benchmarks/bench_stage1b.py`*
