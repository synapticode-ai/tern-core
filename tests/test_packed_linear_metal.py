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


@pytest.mark.parametrize("M,K", [(64, 64), (256, 256), (2560, 2560)])
def test_packed_layer_mps_matches_cpu(M, K):
    """PackedTernaryLinear forward on MPS produces output equivalent to
    forward on CPU. Exercises the new _forward_metal path end-to-end:
    instance construction → .to("mps") → forward dispatch → output read.
    Tolerance carries from the cross-kernel calibration (5e-2)."""
    _metal_or_skip()
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable on this host")

    import torch.nn as nn
    from terncore.packed_linear import PackedTernaryLinear

    torch.manual_seed(700)
    linear = nn.Linear(K, M, bias=True)
    packed_cpu = PackedTernaryLinear.from_float(linear, threshold=0.7)
    packed_cpu.eval()
    packed_mps = PackedTernaryLinear.from_float(linear, threshold=0.7).to("mps")
    packed_mps.eval()

    x_cpu = torch.randn(1, K)
    x_mps = x_cpu.to("mps")

    with torch.no_grad():
        cpu_out = packed_cpu(x_cpu)
        mps_out = packed_mps(x_mps)

    assert mps_out.device.type == "mps"
    mps_out_cpu = mps_out.cpu()
    max_diff = (cpu_out - mps_out_cpu).abs().max().item()
    print(f"\n[M={M} K={K}] mps vs cpu max_abs_diff={max_diff:.6f}", flush=True)
    assert max_diff < 5e-2, (
        f"MPS-via-Metal vs CPU divergence: {max_diff} (atol=5e-2)"
    )


def test_packed_layer_lazy_buffer_allocation():
    """_metal_buffers_initialised is False before first MPS forward, True
    after, and stays True across subsequent forwards. Codes/scales buffers
    are non-None after init and are not re-allocated on subsequent calls."""
    _metal_or_skip()
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable on this host")

    import torch.nn as nn
    from terncore.packed_linear import PackedTernaryLinear

    torch.manual_seed(701)
    linear = nn.Linear(64, 32, bias=False)
    packed = PackedTernaryLinear.from_float(linear, threshold=0.7).to("mps")
    packed.eval()

    assert packed._metal_buffers_initialised is False
    assert packed._metal_codes_buf is None
    assert packed._metal_scales_buf is None

    x = torch.randn(1, 64).to("mps")
    with torch.no_grad():
        _ = packed(x)

    assert packed._metal_buffers_initialised is True
    assert packed._metal_codes_buf is not None
    assert packed._metal_scales_buf is not None

    codes_id = id(packed._metal_codes_buf)
    scales_id = id(packed._metal_scales_buf)
    with torch.no_grad():
        _ = packed(x)

    assert id(packed._metal_codes_buf) == codes_id
    assert id(packed._metal_scales_buf) == scales_id


@pytest.mark.slow
def test_production_data_e2e_gemopus():
    """End-to-end integration on the real gemopus-4-e4b .tern-model artefact.

    Loads via the decoupled path (load_as_model + key_mapping +
    convert_model_to_packed), moves to MPS, runs 5-token generation through
    PackedTernaryLinear.forward's Metal kernel path. Verifies the entire
    Phase 2.5 stack works on real production data without crashing and
    produces structurally coherent output.

    The Metal-path-taken assertion verifies at least one PackedTernaryLinear
    layer has Metal active; a stronger "most layers" assertion is deferred
    to future work since per-layer Metal failures (e.g., K not divisible by
    16 on some shape) are valid edge cases at this verification stage.

    Skips when the artefact, Metal, MPS, or transformers/Gemma 4 are
    unavailable. ~180-300 s runtime dominated by the 8.9 GiB artefact load
    and the random-init 8B model construction."""
    import gc
    from pathlib import Path

    artefact_path = Path(
        "/Volumes/Syn Archive/models/compressed/gemopus-4-e4b/"
        "gemopus_4_e4b_ternary_v0.1.0.tern-model/model.tern-model"
    )
    if not artefact_path.exists():
        pytest.skip(f"artefact not present at {artefact_path}")

    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable on this host")

    _metal_or_skip()

    try:
        from transformers import AutoConfig, AutoTokenizer
        import transformers
    except ImportError:
        pytest.skip("transformers not installed")

    from terncore.tern_model import (
        TernModelReader, GEMMA4_MULTIMODAL_TRANSFORMERS_5_5,
    )
    from terncore.packed_linear import PackedTernaryLinear, convert_model_to_packed

    # Cheap insurance against accumulated state from prior tests in the
    # session affecting this high-memory load.
    gc.collect()

    HF_ID = "Jackrong/Gemopus-4-E4B-it"

    # Architecture from cached HF config; instantiate via the class the
    # config declares (robust to future Gemma 4 class renames).
    config = AutoConfig.from_pretrained(HF_ID)
    arch_name = config.architectures[0]
    if not hasattr(transformers, arch_name):
        pytest.skip(f"transformers lacks {arch_name} (version mismatch)")
    ModelClass = getattr(transformers, arch_name)

    # Random-init in FP16 to fit the M4 Pro memory budget; weights are
    # immediately overwritten by the .tern-model load.
    model = ModelClass._from_config(config, dtype=torch.float16)
    model.eval()

    # Decoupled load: state_dict via load_as_model + key_mapping, then
    # convert eligible Linear layers to PackedTernaryLinear.
    reader = TernModelReader(str(artefact_path))
    missing, unexpected = reader.load_as_model(
        model, strict=False,
        key_mapping=GEMMA4_MULTIMODAL_TRANSFORMERS_5_5,
    )
    # STATUS_PHASE2.md established baseline: 1 missing (lm_head tied),
    # 54 unexpected (sliding-window KV pruning). Test doesn't assert exact
    # counts (transformers version drift could shift them), only that load
    # completed.

    stats = convert_model_to_packed(model, threshold=0.7)
    assert stats["packed_layers"] > 0, (
        f"convert_model_to_packed produced zero packed layers: {stats}"
    )

    packed_layers = [
        m for m in model.modules() if isinstance(m, PackedTernaryLinear)
    ]
    assert len(packed_layers) > 0

    model = model.to("mps")

    tokenizer = AutoTokenizer.from_pretrained(HF_ID)
    messages = [{"role": "user", "content": "Hello"}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("mps")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=5, do_sample=False,
        )

    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print(f"\n[gemopus-4-e4b e2e] generated: {output_text!r}", flush=True)

    assert len(output_text) > 0, "empty generation output"
    assert any(c.isprintable() and not c.isspace() for c in output_text), (
        f"generation output lacks printable content: {output_text!r}"
    )

    metal_path_taken = any(
        layer._metal_buffers_initialised and layer._metal_engine is not None
        for layer in packed_layers
    )
    assert metal_path_taken, (
        "no PackedTernaryLinear layer initialised Metal buffers — "
        "generation ran entirely through the F.linear fallback"
    )

    # Resource cleanup — release ~10 GB of MPS allocation before next test.
    del model, packed_layers, reader, tokenizer
    gc.collect()
    if hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
