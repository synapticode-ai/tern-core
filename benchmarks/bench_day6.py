"""
Day 6 integration: TinyLlama v_proj_late3 → .tern-model v2 format.

Loads TinyLlama-1.1B, applies v_proj_late3 mixed-precision config
(3 ternary v_proj layers at indices 19-21), writes a .tern-model v2
file, and verifies header, manifest, CRC32 integrity, and file size.

Patent 6: Model format specification.
Patent 8: Serialisation and integrity verification.

Usage:
    python benchmarks/bench_day6.py

Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# Ensure imports work from repo root
_BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_BENCH_DIR.parent / "src"))

from terncore.arithmetic.linear import TernaryLinear
from terncore.arithmetic.quantizer import TernaryQuantizer
from terncore.mixed_precision import MixedPrecisionConverter
from terncore.tern_model import TernModelReader, TernModelWriter

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
THRESHOLD = 0.7
OUTPUT_DIR = _BENCH_DIR.parent / "output"
OUTPUT_PATH = OUTPUT_DIR / "tinyllama_v_proj_late3.tern-model"

# v_proj_late3: ternarise v_proj at layers 19, 20, 21
V_PROJ_LATE3_TERNARY = {
    f"model.layers.{i}.self_attn.v_proj" for i in range(19, 22)
}


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════


def _load_model(model_id: str):
    """Load model and tokenizer from HuggingFace."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading {model_id}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()
    dt = time.time() - t0
    print(f"  Loaded in {dt:.1f}s")
    return model, tokenizer


def _build_protection_list(model: nn.Module) -> list[str]:
    """
    Build protection list for v_proj_late3 config.
    Protects everything EXCEPT the 3 v_proj layers we want to ternarise.
    """
    all_linears = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            all_linears.append(name)
    return [n for n in all_linears if n not in V_PROJ_LATE3_TERNARY]


def _estimate_original_size(model: nn.Module) -> int:
    """Estimate original model checkpoint size in bytes (FP32)."""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params * 4  # FP32 = 4 bytes/param


# ═══════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════


def main():
    print("=" * 70)
    print("Day 6: TinyLlama v_proj_late3 → .tern-model v2")
    print("=" * 70)

    # --- Step 1: Load model ---
    print("\n[1/5] Loading TinyLlama...")
    try:
        model, tokenizer = _load_model(MODEL_ID)
    except Exception as e:
        print(f"  ERROR: Could not load model: {e}")
        print("  Requires: pip install transformers sentencepiece accelerate")
        sys.exit(1)

    original_size = _estimate_original_size(model)
    print(f"  Original model: {original_size / 1e6:.1f} MB (FP32 params)")

    # --- Step 2: Apply v_proj_late3 conversion ---
    print("\n[2/5] Converting to v_proj_late3 mixed-precision...")
    protection_list = _build_protection_list(model)
    converter = MixedPrecisionConverter(
        threshold=THRESHOLD,
        protection_list=protection_list,
    )
    t0 = time.time()
    report = converter.convert(model)
    dt = time.time() - t0

    print(f"  Converted {report.converted_layers} layers in {dt:.2f}s")
    print(f"  Total: {report.total_layers} | "
          f"Ternary: {report.converted_layers} | "
          f"Protected: {report.skipped_layers}")
    print(f"  Compression: {report.compression_ratio:.2f}x")

    # --- Step 3: Write .tern-model ---
    print("\n[3/5] Writing .tern-model v2...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    writer = TernModelWriter({
        "source": MODEL_ID,
        "notes": "v_proj_late3: ternary v_proj at layers 19-21, rest FP16/FP32",
        "config": "v_proj_late3",
        "threshold": THRESHOLD,
        "converted_layers": report.converted_layers,
        "protected_layers": report.skipped_layers,
    })

    t0 = time.time()
    for name, module in model.named_modules():
        if isinstance(module, TernaryLinear):
            # Ternary layer — quantise and pack
            q = TernaryQuantizer(threshold=module.threshold)
            ternary, alpha = q.quantize(module.weight.data)
            sparsity = (ternary == 0).sum().item() / ternary.numel()

            packed, a, bitmap, sp = TernModelWriter.pack_ternary(
                module.weight.data, module.threshold
            )
            writer.add_ternary_layer(
                name=name,
                packed_weights=packed,
                alpha=a,
                shape=list(module.weight.shape),
                sparsity_bitmap=bitmap,
                threshold=module.threshold,
                sparsity=sp,
                bias=module.bias,
            )
        elif isinstance(module, nn.Linear):
            # FP16 protected layer
            writer.add_layer(name, module.weight.data, dtype="float16",
                             bias=module.bias)

    stats = writer.write(str(OUTPUT_PATH))
    dt = time.time() - t0
    print(f"  Written in {dt:.1f}s")
    print(f"  File: {OUTPUT_PATH}")
    print(f"  Size: {stats['file_size'] / 1e6:.2f} MB")
    print(f"  Layers: {stats['num_layers']} "
          f"({stats['num_ternary']} ternary, {stats['num_protected']} FP16)")

    # --- Step 4: Verify ---
    print("\n[4/5] Verifying .tern-model integrity...")
    reader = TernModelReader(str(OUTPUT_PATH))

    # Header checks
    assert reader.header["magic"] == b"TERN", "Header magic mismatch"
    assert reader.header["version"] == 2, "Version mismatch"
    assert reader.header["header_size"] == 256, "Header size mismatch"
    assert reader.header["num_layers"] == stats["num_layers"], "Layer count mismatch"
    print(f"  Header: OK (magic=TERN, version=2, {reader.header['num_layers']} layers)")

    # Manifest checks
    manifest = reader.manifest
    assert "model_metadata" in manifest, "Missing model_metadata"
    assert "layers" in manifest, "Missing layers"
    assert len(manifest["layers"]) == stats["num_layers"], "Manifest layer count mismatch"
    ternary_layers = [l for l in manifest["layers"] if l["dtype"] == "ternary2"]
    fp16_layers = [l for l in manifest["layers"] if l["dtype"] == "float16"]
    print(f"  Manifest: OK ({len(ternary_layers)} ternary, {len(fp16_layers)} FP16)")

    # Alignment check
    weights_offset = reader.header["weights_offset"]
    assert weights_offset % 32 == 0, "Weights section not 32-byte aligned"
    for layer in manifest["layers"]:
        abs_offset = weights_offset + layer["offset"]
        assert abs_offset % 32 == 0, f"Layer {layer['name']} not aligned"
    print(f"  Alignment: OK (all layers 32-byte aligned)")

    # CRC32 check
    assert reader.verify(), "CRC32 verification FAILED"
    print(f"  CRC32: OK (footer verified, crc32=0x{stats['crc32']:08X})")

    # File size check
    actual_size = OUTPUT_PATH.stat().st_size
    assert actual_size == stats["file_size"], "File size mismatch"
    print(f"  File size: OK ({actual_size:,} bytes)")

    # --- Step 5: Summary ---
    print("\n[5/5] Summary")
    print("=" * 70)
    compression = original_size / actual_size
    print(f"  Original FP32 size:   {original_size / 1e6:.1f} MB")
    print(f"  .tern-model v2 size:  {actual_size / 1e6:.2f} MB")
    print(f"  Compression ratio:    {compression:.2f}x")
    print(f"  Ternary layers:       {stats['num_ternary']}")
    print(f"  Protected layers:     {stats['num_protected']}")

    # Per-ternary-layer detail
    print("\n  Ternary layer details:")
    for layer in ternary_layers:
        print(f"    {layer['name']}: "
              f"shape={layer['shape']}, "
              f"sparsity={layer.get('sparsity', 0):.1%}, "
              f"alpha={layer.get('alpha', 0):.6f}")

    print("\n  All verification checks PASSED.")
    print("=" * 70)

    # Cleanup: remove the output file to avoid committing model data
    # (CLAUDE.md says never commit model files)
    print(f"\n  Cleaning up {OUTPUT_PATH}...")
    OUTPUT_PATH.unlink(missing_ok=True)
    print("  Done.")


if __name__ == "__main__":
    main()
