"""
End-to-end HuggingFace model -> .tern-model conversion pipeline.

Patents 10-12: Automated binary-to-ternary conversion pipeline.
Patent 6: Model format specification.
Patent 8: Serialisation and integrity verification.

Pipeline stages:
1. Load HuggingFace model (or accept pre-loaded nn.Module)
2. Identify layer types and build protection list
3. Quantise unprotected layers to ternary
4. Pack weights to 2-bit format
5. Generate sparsity bitmaps
6. Write .tern-model with full manifest
7. Validate output (optional quick-probe)

Usage:
    python -m terncore.convert TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output model.tern

    # With architecture adapter and dry-run:
    python -m terncore.convert --model google/gemma-4-E4B-it \\
        --adapter gemma4 --dry-run --output /tmp/dryrun

Copyright (c) 2025 Synapticode Co., Ltd.
Copyright (c) 2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from terncore.arithmetic.quantizer import TernaryQuantizer
from terncore.sparse import pack_ternary_weights
from terncore.tern_model import TernModelWriter, TernModelReader


def _get_conv1d_class():
    """Lazily import HuggingFace Conv1D if available.

    GPT-2 family models use transformers.pytorch_utils.Conv1D instead of
    nn.Linear for attention/MLP layers. Conv1D stores weights as
    (in_features, out_features) — transposed vs nn.Linear's (out_features,
    in_features). Functionally identical: both compute x @ W + b.
    """
    try:
        from transformers.pytorch_utils import Conv1D
        return Conv1D
    except ImportError:
        return None


def _is_weight_layer(module: nn.Module) -> bool:
    """Check if module is nn.Linear or HuggingFace Conv1D."""
    if isinstance(module, nn.Linear):
        return True
    Conv1D = _get_conv1d_class()
    if Conv1D is not None and isinstance(module, Conv1D):
        return True
    return False


def _get_weight_and_shape(module: nn.Module) -> tuple[torch.Tensor, list[int]]:
    """Extract weight tensor in (out_features, in_features) layout.

    Conv1D stores weights as (in_features, out_features), so we transpose
    to match nn.Linear's convention for consistent packing.
    """
    Conv1D = _get_conv1d_class()
    if Conv1D is not None and isinstance(module, Conv1D):
        # Conv1D: weight is (in_features, out_features) — transpose it
        weight = module.weight.data.t().contiguous()
    else:
        weight = module.weight.data
    return weight, list(weight.shape)


# Default protection patterns — proven from Days 2-5.
# These layers are catastrophic to quantise across all transformer architectures.
DEFAULT_PROTECTION_PATTERNS = [
    "*lm_head*",
    "*embed*",
    "*norm*",
    "*head*",
]

# Always-protected patterns (cannot be overridden by user).
# Embedding and norm layers cause model collapse when quantised.
ALWAYS_PROTECTED_PATTERNS = [
    "*embed*",
    "*norm*",
    "*lm_head*",
]


class TernaryConverter:
    """End-to-end HuggingFace model -> .tern-model conversion.

    Patents 10-12: Automated binary-to-ternary conversion pipeline.

    Pipeline stages:
    1. Load HuggingFace model
    2. Identify layer types and build protection list
    3. Quantise unprotected layers to ternary
    4. Pack weights to 2-bit format
    5. Generate sparsity bitmaps
    6. Write .tern-model with full manifest
    7. Validate output (optional quick-probe)
    """

    def __init__(
        self,
        model_id: str,
        output_path: str,
        threshold: float = 0.7,
        protection_patterns: list[str] | None = None,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            model_id: HuggingFace model ID (e.g. "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
                       or local path to model directory.
            output_path: Where to write the .tern-model file.
            threshold: Ternary quantisation threshold (default 0.7).
            protection_patterns: Layer name glob patterns to keep in FP16.
                Default: ["*lm_head*", "*embed*", "*norm*", "*head*"]
            device: Device for model loading.
        """
        self.model_id = model_id
        self.output_path = Path(output_path)
        self.threshold = threshold
        self.device = device

        # Merge user patterns with always-protected patterns
        user_patterns = protection_patterns or DEFAULT_PROTECTION_PATTERNS
        self._protection_patterns = list(set(
            [p.lower() for p in ALWAYS_PROTECTED_PATTERNS]
            + [p.lower() for p in user_patterns]
        ))

    def convert(
        self,
        verbose: bool = True,
        model: nn.Module | None = None,
    ) -> dict:
        """Run the full conversion pipeline.

        Args:
            verbose: Print progress during conversion.
            model: Optional pre-loaded model (skips HuggingFace loading).

        Returns:
            Dict with conversion stats:
            - model_name, total_layers, ternary_layers, protected_layers
            - total_params, ternary_params, protected_params
            - file_size_bytes, compression_ratio
            - conversion_time_seconds
            - per_layer_stats: list of {name, dtype, shape, sparsity, alpha}
        """
        t_start = time.perf_counter()
        _log = _printer(verbose)

        # Stage 1: Load model
        if model is None:
            _log(f"Loading {self.model_id}...")
            t0 = time.perf_counter()
            model = self._load_model()
            _log(f"  Loaded in {time.perf_counter() - t0:.1f}s")
        else:
            _log(f"Using pre-loaded model")

        # Stage 2: Build protection list
        _log("\nAnalyzing layers...")
        protection_list = self._build_protection_list(model)
        all_linear = self._find_linear_layers(model)
        ternary_names = [n for n in all_linear if n not in protection_list]

        _log(f"  Total weight layers: {len(all_linear)}")
        _log(f"  Protected: {len(protection_list)} layers")
        _log(f"  Ternary:   {len(ternary_names)} layers")

        # Stage 3+4+5: Quantise, pack, and generate bitmaps
        _log(f"\nConverting...")
        t0 = time.perf_counter()
        layer_data = self._quantise_and_pack(model, protection_list, _log)
        _log(f"  Conversion done in {time.perf_counter() - t0:.1f}s")

        # Stage 6: Write .tern-model
        _log(f"\nWriting {self.output_path}...")
        t0 = time.perf_counter()
        model_metadata = {
            "source": self.model_id,
            "threshold": self.threshold,
            "notes": f"Converted by tern-convert with threshold={self.threshold}",
        }
        write_stats = self._write_tern_model(layer_data, model_metadata)
        file_size = write_stats["file_size"]
        _log(f"  File size: {file_size / 1024 / 1024:.2f} MB")
        _log(f"  Written in {time.perf_counter() - t0:.1f}s")

        # Compute stats
        total_params = sum(d["num_params"] for d in layer_data.values())
        ternary_params = sum(
            d["num_params"] for d in layer_data.values() if d["dtype"] == "ternary2"
        )
        protected_params = sum(
            d["num_params"] for d in layer_data.values() if d["dtype"] == "float16"
        )

        # Compression: compare to FP32 total
        fp32_size = total_params * 4
        compression_ratio = fp32_size / file_size if file_size > 0 else 0

        conversion_time = time.perf_counter() - t_start

        per_layer_stats = []
        for name, data in layer_data.items():
            entry = {
                "name": name,
                "dtype": data["dtype"],
                "shape": data["shape"],
            }
            if data["dtype"] == "ternary2":
                entry["sparsity"] = data["sparsity"]
                entry["alpha"] = data["alpha"]
            per_layer_stats.append(entry)

        stats = {
            "model_name": self.model_id,
            "total_layers": len(layer_data),
            "ternary_layers": write_stats["num_ternary"],
            "protected_layers": write_stats["num_protected"],
            "total_params": total_params,
            "ternary_params": ternary_params,
            "protected_params": protected_params,
            "file_size_bytes": file_size,
            "compression_ratio": round(compression_ratio, 2),
            "conversion_time_seconds": round(conversion_time, 2),
            "per_layer_stats": per_layer_stats,
        }

        # Summary
        _log(f"\nConversion complete in {conversion_time:.1f}s")
        _log(f"  Ternary layers: {stats['ternary_layers']}/{stats['total_layers']}")
        pct = (ternary_params / total_params * 100) if total_params > 0 else 0
        _log(f"  Ternary params: {ternary_params:,}/{total_params:,} ({pct:.1f}%)")
        _log(f"  Compression: {compression_ratio:.1f}x vs FP32")

        return stats

    def _load_model(self) -> nn.Module:
        """Load model from HuggingFace or local path."""
        try:
            from transformers import AutoModelForCausalLM
        except ImportError:
            raise ImportError(
                "transformers is required for HuggingFace model loading. "
                "Install with: pip install transformers"
            )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.float32,
            device_map=self.device,
            low_cpu_mem_usage=True,
        )
        model.eval()
        return model

    def _find_linear_layers(self, model: nn.Module) -> list[str]:
        """Find all weight layer names (nn.Linear + Conv1D) in the model."""
        return [
            name for name, module in model.named_modules()
            if _is_weight_layer(module)
        ]

    def _build_protection_list(self, model: nn.Module) -> set[str]:
        """Determine which layers to protect based on patterns.

        Default protection (proven from Days 2-5):
        - Embedding layers (model.embed_tokens)
        - LayerNorm / RMSNorm layers
        - LM head (lm_head)

        These layers are always protected regardless of settings.
        User can add additional patterns.
        """
        protected = set()
        for name, module in model.named_modules():
            if not _is_weight_layer(module):
                continue
            name_lower = name.lower()
            for pattern in self._protection_patterns:
                if fnmatch.fnmatch(name_lower, pattern):
                    protected.add(name)
                    break
        return protected

    def _quantise_and_pack(
        self,
        model: nn.Module,
        protection_list: set[str],
        _log,
    ) -> dict:
        """Quantise unprotected layers, pack to 2-bit, generate bitmaps.

        Returns dict mapping layer_name -> {
            'dtype': 'ternary2' or 'float16',
            'packed_weights': bytes (if ternary),
            'alpha': float (if ternary),
            'sparsity_bitmap': bytes (if ternary),
            'weights': tensor (if fp16),
            'shape': list[int],
            'sparsity': float,
            'num_params': int,
            'bias': tensor or None,
        }
        """
        layer_data = {}
        all_linear = [
            (name, module) for name, module in model.named_modules()
            if _is_weight_layer(module)
        ]

        done = 0
        total = len(all_linear)

        for name, module in all_linear:
            weight, shape = _get_weight_and_shape(module)
            bias = module.bias.data if module.bias is not None else None
            num_params = weight.numel()

            if name in protection_list:
                # Protected: store as FP16
                layer_data[name] = {
                    "dtype": "float16",
                    "weights": weight,
                    "shape": shape,
                    "num_params": num_params,
                    "bias": bias,
                }
            else:
                # Ternary: quantise + pack
                packed_bytes, alpha, bitmap_bytes, sparsity = (
                    TernModelWriter.pack_ternary(weight, self.threshold)
                )
                layer_data[name] = {
                    "dtype": "ternary2",
                    "packed_weights": packed_bytes,
                    "alpha": alpha,
                    "sparsity_bitmap": bitmap_bytes,
                    "shape": shape,
                    "sparsity": sparsity,
                    "num_params": num_params,
                    "bias": bias,
                }

            done += 1
            if done % 20 == 0 or done == total:
                _log(f"  [{done}/{total}] layers processed")

        return layer_data

    def _write_tern_model(self, layer_data: dict, model_metadata: dict) -> dict:
        """Write .tern-model file using TernModelWriter.

        Returns write stats dict.
        """
        writer = TernModelWriter(model_metadata)

        for name, data in layer_data.items():
            if data["dtype"] == "ternary2":
                writer.add_ternary_layer(
                    name=name,
                    packed_weights=data["packed_weights"],
                    alpha=data["alpha"],
                    shape=data["shape"],
                    sparsity_bitmap=data["sparsity_bitmap"],
                    threshold=self.threshold,
                    sparsity=data["sparsity"],
                    bias=data.get("bias"),
                )
            else:
                writer.add_layer(
                    name=name,
                    weights=data["weights"],
                    dtype="float16",
                    bias=data.get("bias"),
                )

        return writer.write(self.output_path)

    def verify(self, verbose: bool = True) -> bool:
        """Verify the output .tern-model file integrity.

        Args:
            verbose: Print verification results.

        Returns:
            True if all checks pass.
        """
        _log = _printer(verbose)

        if not self.output_path.exists():
            _log("ERROR: Output file does not exist")
            return False

        _log("\nVerifying...")
        t0 = time.perf_counter()
        reader = TernModelReader(str(self.output_path))
        ok = reader.verify()
        dt = time.perf_counter() - t0

        if ok:
            _log(f"  Integrity check: PASSED ({dt * 1000:.1f}ms)")
        else:
            _log(f"  Integrity check: FAILED ({dt * 1000:.1f}ms)")

        return ok

    @staticmethod
    def info(model_id: str, device: str = "cpu") -> dict:
        """Show model architecture info without converting.

        Args:
            model_id: HuggingFace model ID or local path.
            device: Device for model loading.

        Returns:
            Dict with model architecture info.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoConfig
        except ImportError:
            raise ImportError(
                "transformers is required. Install with: pip install transformers"
            )

        config = AutoConfig.from_pretrained(model_id)

        # Count parameters from config
        info = {
            "model_id": model_id,
            "architecture": config.architectures[0] if config.architectures else "unknown",
            "hidden_size": getattr(config, "hidden_size", None),
            "num_layers": getattr(config, "num_hidden_layers", None),
            "num_heads": getattr(config, "num_attention_heads", None),
            "vocab_size": getattr(config, "vocab_size", None),
            "intermediate_size": getattr(config, "intermediate_size", None),
        }

        return info


def _printer(verbose: bool):
    """Return a print function that respects verbose flag."""
    def _log(msg: str) -> None:
        if verbose:
            print(msg)
    return _log


def _read_hf_arch_from_config(model_dir: Path) -> str:
    """Read the HF architecture from ``config.json`` in ``model_dir``.

    Used to feed :meth:`ArchitectureAdapter.validate_architecture`
    at the entry of the conversion pipeline. Stdlib-only — no
    transformers dependency.

    Raises :class:`ArchitectureMismatch` if ``config.json`` is
    missing, unreadable, or has no usable ``architectures`` field.
    The error message names the path so the operator can
    investigate.
    """
    from terncore.adapters.base import ArchitectureMismatch

    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise ArchitectureMismatch(
            f"config.json not found at {config_path}. Cannot "
            f"validate adapter routing. Verify the model directory "
            f"contains a valid HF config."
        )
    try:
        with open(config_path) as f:
            hf_config = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise ArchitectureMismatch(
            f"Failed to read config.json at {config_path}: {e}. "
            f"Cannot validate adapter routing."
        )
    architectures = hf_config.get("architectures") or []
    if not architectures:
        raise ArchitectureMismatch(
            f"config.json at {config_path} has no 'architectures' "
            f"field or it is empty. Cannot validate adapter routing."
        )
    return architectures[0]


# ── Full adapter-aware conversion (mixed ternary/INT4/FP16) ────────


def full_convert(
    model_id: str,
    adapter_name: str,
    output_dir: str,
    threshold: float = 0.7,
    name: str = "model",
    verbose: bool = True,
) -> dict:
    """Full adapter-aware conversion: safetensors → .tern-model.

    Mixed quantisation:
    - Ternary {-1, 0, +1} for adapter-eligible layers
    - INT4 block-wise for remaining 2-D language weights
    - FP16 for vision/audio encoders, norms, embeddings, scalars

    Args:
        model_id:     HuggingFace model ID or local path.
        adapter_name: Architecture adapter name (e.g. "gemma4").
        output_dir:   Directory for the .tern-model output.
        threshold:    Ternary quantisation threshold.
        name:         Output model name (without .tern-model suffix).
        verbose:      Print progress.

    Returns:
        Dict with conversion stats.
    """
    import json as _json
    import gc
    import time as _time
    from pathlib import Path
    from datetime import datetime, timezone

    from safetensors import safe_open
    from terncore.adapters import get_adapter
    from terncore.tern_model import TernModelWriter
    from terncore.int4_quantizer import quantize_int4_block

    _log = _printer(verbose)
    adapter = get_adapter(adapter_name)
    info = adapter.info()

    _log("=" * 68)
    _log(f"  Full Conversion — {info.name} adapter")
    _log("=" * 68)
    _log(f"  Model:     {model_id}")
    _log(f"  Adapter:   {info.name} ({', '.join(info.architectures)})")
    _log(f"  Threshold: {threshold}")
    _log(f"  Output:    {output_dir}")

    t_start = _time.perf_counter()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Resolve model path ──
    model_path = Path(model_id)
    resolved_dir = None

    if model_path.is_dir():
        resolved_dir = model_path
    else:
        _log(f"\n  Resolving model from HuggingFace Hub cache...")
        try:
            from huggingface_hub import snapshot_download
            local_dir = snapshot_download(
                model_id,
                allow_patterns=["*.safetensors", "*.safetensors.index.json",
                                "config.json"],
            )
            resolved_dir = Path(local_dir)
            _log(f"  Resolved: {resolved_dir}")
        except Exception as e:
            raise RuntimeError(f"Cannot resolve model: {e}")

    # ── Validate adapter routing against HF config ──
    hf_arch = _read_hf_arch_from_config(resolved_dir)
    adapter.validate_architecture(hf_arch)
    _log(f"  Validated: HF arch '{hf_arch}' matches adapter '{info.name}'")

    # ── Discover safetensors files ──
    index_path = resolved_dir / "model.safetensors.index.json"
    single_shard = resolved_dir / "model.safetensors"
    weight_to_file: dict[str, Path] = {}

    if index_path.exists():
        with open(index_path) as f:
            wmap = _json.load(f)["weight_map"]
        for wname, shard_name in wmap.items():
            weight_to_file[wname] = resolved_dir / shard_name
        _log(f"  Sharded model: {len(set(wmap.values()))} shards, "
             f"{len(weight_to_file)} weights")
    elif single_shard.exists():
        with safe_open(str(single_shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                weight_to_file[key] = single_shard
        _log(f"  Single shard: {len(weight_to_file)} weights")
    else:
        st_files = sorted(resolved_dir.glob("*.safetensors"))
        for st_file in st_files:
            with safe_open(str(st_file), framework="pt", device="cpu") as f:
                for key in f.keys():
                    weight_to_file[key] = st_file
        _log(f"  Found {len(weight_to_file)} weights in {len(st_files)} files")

    if not weight_to_file:
        raise FileNotFoundError(f"No safetensors files found in {resolved_dir}")

    # ── Get shapes and classify ──
    _log(f"\n  Reading weight shapes...")
    weight_shapes: dict[str, list[int]] = {}
    # Group by file for efficient reading
    file_to_keys: dict[Path, list[str]] = {}
    for wname, fpath in weight_to_file.items():
        file_to_keys.setdefault(fpath, []).append(wname)

    for fpath, keys in file_to_keys.items():
        with safe_open(str(fpath), framework="pt", device="cpu") as f:
            for key in keys:
                weight_shapes[key] = list(f.get_slice(key).get_shape())

    _log(f"  Classifying {len(weight_shapes)} weights...")
    classifications = adapter.classify_all(weight_shapes)

    eligible = [n for n, c in classifications.items()
                if c.category == "ternary_eligible"]
    retained = [n for n, c in classifications.items()
                if c.category == "fp16_retain"]
    eligible_set = set(eligible)
    retained_set = set(retained)

    # Split eligible into ternary vs INT4 using tolerance scan.
    # Layers with high reconstruction error (> 0.54) fall back to INT4.
    # This targets the sensitive band (layers 19-23, v_proj/o_proj).
    INT4_ERROR_THRESHOLD = 0.54
    int4_candidates = []

    # Try to load dry-run tolerance data for informed split
    dry_run_path = Path(__file__).parent.parent.parent / "benchmarks" / "gemma4_e4b_dryrun.json"
    sensitivity_map: dict[str, float] = {}
    if dry_run_path.exists():
        import json as _jmod
        with open(dry_run_path) as _f:
            dr = _jmod.load(_f)
        for entry in dr.get("tolerance_scan", []):
            sensitivity_map[entry["name"]] = entry["relative_error"]
        _log(f"  Loaded tolerance data for {len(sensitivity_map)} layers")

    if sensitivity_map:
        # Split eligible: high-error → INT4, low-error → ternary
        new_eligible = []
        for n in eligible:
            error = sensitivity_map.get(n, 0)
            if error >= INT4_ERROR_THRESHOLD:
                int4_candidates.append(n)
            else:
                new_eligible.append(n)
        eligible = new_eligible
        eligible_set = set(eligible)
    int4_set = set(int4_candidates)

    _log(f"  Ternary:   {len(eligible)} layers")
    _log(f"  INT4:      {len(int4_candidates)} layers")
    _log(f"  FP16:      {len(retained)} layers")

    # ── Quantise and write ──
    _log(f"\n  Quantising...")
    writer = TernModelWriter({
        "source": model_id,
        "adapter": adapter_name,
        "threshold": threshold,
        "pipeline": f"adapter-convert-v0.6.0",
    })

    total_weights = len(weight_shapes)
    done = 0
    stats = {
        "ternary_layers": 0, "ternary_params": 0,
        "int4_layers": 0, "int4_params": 0,
        "fp16_layers": 0, "fp16_params": 0,
    }

    # Process file by file to minimise open/close overhead
    for fpath, keys in file_to_keys.items():
        with safe_open(str(fpath), framework="pt", device="cpu") as f:
            for wname in sorted(keys):
                tensor = f.get_tensor(wname)
                shape = list(tensor.shape)
                num_params = tensor.numel()
                canonical = adapter.normalize_name(wname)

                if wname in eligible_set:
                    # Ternary
                    packed, alpha, bitmap, sparsity = (
                        TernModelWriter.pack_ternary(tensor.float(), threshold)
                    )
                    writer.add_ternary_layer(
                        name=canonical,
                        packed_weights=packed,
                        alpha=alpha,
                        shape=shape,
                        sparsity_bitmap=bitmap,
                        threshold=threshold,
                        sparsity=sparsity,
                    )
                    stats["ternary_layers"] += 1
                    stats["ternary_params"] += num_params

                elif wname in int4_set:
                    # INT4 block-wise
                    result = quantize_int4_block(tensor.float(), block_size=32)
                    writer.add_int4_layer(
                        name=canonical,
                        packed_weights=result.packed_weights,
                        scales=result.scales,
                        shape=result.weight_shape,
                        scale_shape=result.scale_shape,
                        block_size=result.block_size,
                        quant_error=result.reconstruction_error,
                    )
                    stats["int4_layers"] += 1
                    stats["int4_params"] += num_params

                else:
                    # FP16
                    writer.add_layer(
                        name=canonical,
                        weights=tensor.float(),
                        dtype="float16",
                    )
                    stats["fp16_layers"] += 1
                    stats["fp16_params"] += num_params

                del tensor
                done += 1
                if done % 100 == 0 or done == total_weights:
                    _log(f"    [{done}/{total_weights}] processed")

        gc.collect()

    # ── Write .tern-model ──
    output_file = out_path / f"{name}.tern-model"
    _log(f"\n  Writing {output_file}...")
    write_stats = writer.write(output_file)
    file_size = write_stats["file_size"]
    file_size_mb = file_size / (1024 * 1024)

    elapsed = _time.perf_counter() - t_start

    total_params = (stats["ternary_params"] + stats["int4_params"]
                    + stats["fp16_params"])
    ternary_ratio = (stats["ternary_params"] / total_params * 100
                     if total_params > 0 else 0)

    # FP16 baseline comparison
    fp16_bytes = total_params * 2
    compression = fp16_bytes / file_size if file_size > 0 else 0

    _log(f"\n{'─' * 68}")
    _log(f"  Conversion Complete")
    _log(f"{'─' * 68}")
    _log(f"  .tern-model:        {file_size_mb:.1f} MB "
         f"({compression:.1f}x vs FP16)")
    _log(f"  Ternary layers:     {stats['ternary_layers']} "
         f"({stats['ternary_params']:,} params, {ternary_ratio:.1f}%)")
    _log(f"  INT4 layers:        {stats['int4_layers']} "
         f"({stats['int4_params']:,} params)")
    _log(f"  FP16 layers:        {stats['fp16_layers']} "
         f"({stats['fp16_params']:,} params)")
    _log(f"  Elapsed:            {elapsed:.1f}s")
    _log("=" * 68)

    report = {
        "runner": "terncore.convert --adapter",
        "model_id": model_id,
        "adapter": adapter_name,
        "threshold": threshold,
        "output_path": str(output_file),
        "file_size_bytes": file_size,
        "file_size_mb": round(file_size_mb, 1),
        "compression_vs_fp16": round(compression, 2),
        "total_params": total_params,
        "ternary_ratio_pct": round(ternary_ratio, 1),
        "elapsed_seconds": round(elapsed, 2),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **stats,
    }

    # Save report alongside .tern-model
    report_path = out_path / f"{name}_conversion_report.json"
    report_path.write_text(_json.dumps(report, indent=2) + "\n")
    _log(f"  Report: {report_path}")

    return report


# ── Dry-run conversion (adapter-aware) ─────────────────────────────


def dry_run_convert(
    model_id: str,
    adapter_name: str,
    output_dir: str,
    threshold: float = 0.7,
    verbose: bool = True,
) -> dict:
    """Dry-run conversion: load weight metadata, classify, scan tolerance.

    Does NOT write a .tern-model file.  Produces a JSON report with:
    - Per-layer ternary tolerance scores (reconstruction error)
    - Estimated ternary ratio
    - FP16-retain layers and reasons
    - Estimated .tern-model size

    Args:
        model_id:     HuggingFace model ID or local path.
        adapter_name: Architecture adapter name (e.g. "gemma4").
        output_dir:   Directory for the dry-run report JSON.
        threshold:    Ternary quantisation threshold.
        verbose:      Print progress.

    Returns:
        Dict with full dry-run report.
    """
    import json as _json
    import gc
    import time
    from pathlib import Path
    from datetime import datetime, timezone

    from terncore.adapters import get_adapter
    from terncore.autoscan import _compute_layer_sensitivity

    _log = _printer(verbose)
    adapter = get_adapter(adapter_name)
    info = adapter.info()

    _log("=" * 68)
    _log(f"  Dry-Run Conversion — {info.name} adapter")
    _log("=" * 68)
    _log(f"  Model:     {model_id}")
    _log(f"  Adapter:   {info.name} ({', '.join(info.architectures)})")
    _log(f"  Threshold: {threshold}")
    _log(f"  Output:    {output_dir}")

    t_start = time.perf_counter()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Resolve model path and load weight shapes ──
    model_path = Path(model_id)
    weight_shapes: dict[str, list[int]] = {}
    _resolved_dir: Optional[Path] = None  # set if we have local safetensors

    if model_path.is_dir():
        _resolved_dir = model_path
    else:
        # Download safetensors from HuggingFace Hub
        _log(f"\n  Downloading model files from HuggingFace Hub...")
        try:
            from huggingface_hub import snapshot_download
            local_dir = snapshot_download(
                model_id,
                allow_patterns=["*.safetensors", "*.safetensors.index.json",
                                "config.json", "tokenizer*"],
            )
            _resolved_dir = Path(local_dir)
            _log(f"  Downloaded to: {_resolved_dir}")
        except Exception as e:
            _log(f"  Hub download failed: {e}")
            _resolved_dir = None

    if _resolved_dir is not None:
        # ── Validate adapter routing against HF config ──
        hf_arch = _read_hf_arch_from_config(_resolved_dir)
        adapter.validate_architecture(hf_arch)
        _log(f"  Validated: HF arch '{hf_arch}' matches adapter '{info.name}'")

        from safetensors import safe_open

        index_path = _resolved_dir / "model.safetensors.index.json"
        single_shard = _resolved_dir / "model.safetensors"

        if index_path.exists():
            _log("  Loading weight shapes from safetensors index...")
            import json as json_mod
            with open(index_path) as f:
                index = json_mod.load(f)
            shard_files = set(index["weight_map"].values())
            for shard_name in sorted(shard_files):
                shard_path = _resolved_dir / shard_name
                with safe_open(str(shard_path), framework="pt", device="cpu") as f:
                    for key in f.keys():
                        weight_shapes[key] = list(f.get_slice(key).get_shape())
            _log(f"  Found {len(weight_shapes)} weight tensors across "
                 f"{len(shard_files)} shards")
        elif single_shard.exists():
            _log("  Loading weight shapes from single safetensors file...")
            with safe_open(str(single_shard), framework="pt", device="cpu") as f:
                for key in f.keys():
                    weight_shapes[key] = list(f.get_slice(key).get_shape())
            _log(f"  Found {len(weight_shapes)} weight tensors")
        else:
            # Fallback: look for any .safetensors files
            st_files = sorted(_resolved_dir.glob("*.safetensors"))
            if st_files:
                _log(f"  Loading shapes from {len(st_files)} safetensors files...")
                for st_file in st_files:
                    with safe_open(str(st_file), framework="pt", device="cpu") as f:
                        for key in f.keys():
                            weight_shapes[key] = list(f.get_slice(key).get_shape())
                _log(f"  Found {len(weight_shapes)} weight tensors")

    if not weight_shapes:
        _log("\n  No safetensors found — loading via transformers...")
        try:
            from transformers import AutoModelForCausalLM
        except ImportError:
            raise ImportError("transformers required: pip install transformers")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        for name, param in model.named_parameters():
            weight_shapes[name] = list(param.shape)
        del model
        gc.collect()
        _log(f"  Found {len(weight_shapes)} parameters")

    # ── Classify weights using the adapter ──
    _log("\n  Classifying weights...")
    classifications = adapter.classify_all(weight_shapes)

    by_category: dict[str, list] = {
        "ternary_eligible": [],
        "fp16_retain": [],
        "skip": [],
    }
    by_component: dict[str, int] = {}
    fp16_reasons: dict[str, list[str]] = {}

    for name, cls in classifications.items():
        by_category[cls.category].append(name)
        by_component[cls.component] = by_component.get(cls.component, 0) + 1
        if cls.category == "fp16_retain":
            fp16_reasons.setdefault(cls.reason, []).append(name)

    eligible_names = by_category["ternary_eligible"]
    retained_names = by_category["fp16_retain"]

    total_params = sum(
        _product(weight_shapes[n]) for n in weight_shapes
    )
    eligible_params = sum(
        _product(weight_shapes[n]) for n in eligible_names
    )
    retained_params = sum(
        _product(weight_shapes[n]) for n in retained_names
    )

    _log(f"  Total weights:      {len(weight_shapes)}")
    _log(f"  Ternary-eligible:   {len(eligible_names)} "
         f"({eligible_params:,} params)")
    _log(f"  FP16-retain:        {len(retained_names)} "
         f"({retained_params:,} params)")
    for comp, count in sorted(by_component.items()):
        _log(f"    {comp}: {count} weights")

    # ── Ternary tolerance scan on eligible weights ──
    _log(f"\n  Running ternary tolerance scan on {len(eligible_names)} "
         f"eligible weights...")

    sensitivities: list[dict] = []

    if _resolved_dir is not None:
        from safetensors import safe_open as _safe_open

        # Build a map: weight_name → safetensors file path
        index_path = _resolved_dir / "model.safetensors.index.json"
        single_shard = _resolved_dir / "model.safetensors"
        weight_to_file: dict[str, Path] = {}

        if index_path.exists():
            import json as json_mod
            with open(index_path) as f:
                wmap = json_mod.load(f)["weight_map"]
            for wname, shard_name in wmap.items():
                weight_to_file[wname] = _resolved_dir / shard_name
        elif single_shard.exists():
            with _safe_open(str(single_shard), framework="pt", device="cpu") as f:
                for key in f.keys():
                    weight_to_file[key] = single_shard
        else:
            for st_file in sorted(_resolved_dir.glob("*.safetensors")):
                with _safe_open(str(st_file), framework="pt", device="cpu") as f:
                    for key in f.keys():
                        weight_to_file[key] = st_file

        done = 0
        for name in sorted(eligible_names):
            if name not in weight_to_file:
                continue
            shard_file = weight_to_file[name]
            with _safe_open(str(shard_file), framework="pt", device="cpu") as f:
                tensor = f.get_tensor(name)

            sens = _compute_layer_sensitivity(name, tensor, threshold)
            sensitivities.append({
                "name": name,
                "canonical_name": adapter.normalize_name(name),
                "relative_error": round(sens.relative_error, 6),
                "num_params": sens.num_params,
                "sparsity": round(sens.sparsity, 4),
                "alpha": round(sens.alpha, 6),
            })
            del tensor
            done += 1
            if done % 50 == 0 or done == len(eligible_names):
                _log(f"    [{done}/{len(eligible_names)}] scanned")

        gc.collect()
    else:
        _log("    (skipped — no safetensors available for streaming scan)")

    # Sort by tolerance (lowest error = most tolerant)
    sensitivities.sort(key=lambda s: s["relative_error"])

    # ── Compute estimates ──
    ternary_ratio = eligible_params / total_params if total_params > 0 else 0

    # Estimated .tern-model size:
    # Ternary: 2 bits/param = 0.25 bytes/param
    # FP16:    16 bits/param = 2 bytes/param
    est_ternary_bytes = eligible_params * 0.25
    est_fp16_bytes = retained_params * 2
    est_total_bytes = est_ternary_bytes + est_fp16_bytes
    est_size_mb = est_total_bytes / (1024 * 1024)

    # FP16 baseline size
    fp16_baseline_bytes = total_params * 2
    est_compression = fp16_baseline_bytes / est_total_bytes if est_total_bytes > 0 else 0

    elapsed = time.perf_counter() - t_start

    # ── Print summary ──
    _log(f"\n{'─' * 68}")
    _log("  Dry-Run Results")
    _log(f"{'─' * 68}")
    _log(f"  Total parameters:       {total_params:,}")
    _log(f"  Ternary-eligible:       {eligible_params:,} "
         f"({ternary_ratio * 100:.1f}%)")
    _log(f"  FP16-retain:            {retained_params:,}")
    _log(f"  Estimated .tern-model:  {est_size_mb:.1f} MB "
         f"({est_compression:.1f}x compression vs FP16)")

    if sensitivities:
        _log(f"\n  Top-10 most tolerant layers:")
        for i, s in enumerate(sensitivities[:10]):
            _log(f"    {i+1:3d}. {s['canonical_name']}")
            _log(f"         error={s['relative_error']:.6f}  "
                 f"sparsity={s['sparsity']:.4f}  "
                 f"params={s['num_params']:,}")

        _log(f"\n  Bottom-10 least tolerant layers:")
        for s in sensitivities[-10:]:
            _log(f"    {s['canonical_name']}")
            _log(f"         error={s['relative_error']:.6f}  "
                 f"sparsity={s['sparsity']:.4f}  "
                 f"params={s['num_params']:,}")

    _log(f"\n  FP16-retain reasons:")
    for reason, names in sorted(fp16_reasons.items()):
        _log(f"    {reason}: {len(names)} layers")

    _log(f"\n  Elapsed: {elapsed:.1f}s")
    _log("=" * 68)

    # ── Build report ──
    report = {
        "runner": "terncore.convert --dry-run",
        "model_id": model_id,
        "adapter": info.name,
        "architectures": list(info.architectures),
        "threshold": threshold,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "total_weights": len(weight_shapes),
        "total_params": total_params,
        "ternary_eligible_weights": len(eligible_names),
        "ternary_eligible_params": eligible_params,
        "fp16_retain_weights": len(retained_names),
        "fp16_retain_params": retained_params,
        "ternary_ratio": round(ternary_ratio, 4),
        "estimated_tern_model_mb": round(est_size_mb, 1),
        "estimated_compression_vs_fp16": round(est_compression, 2),
        "fp16_baseline_mb": round(fp16_baseline_bytes / (1024 * 1024), 1),
        "components": by_component,
        "fp16_retain_reasons": {
            reason: len(names) for reason, names in fp16_reasons.items()
        },
        "fp16_retain_layers": [
            {
                "name": name,
                "canonical_name": classifications[name].canonical_name,
                "reason": classifications[name].reason,
                "component": classifications[name].component,
                "shape": weight_shapes[name],
            }
            for name in retained_names
        ],
        "tolerance_scan": sensitivities,
    }

    # Save report
    report_path = out_path / "dry_run_report.json"
    report_path.write_text(_json.dumps(report, indent=2, default=str) + "\n")
    _log(f"\n  Report saved: {report_path}")

    return report


def _product(shape: list[int]) -> int:
    """Compute the product of a shape list (number of elements)."""
    result = 1
    for s in shape:
        result *= s
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace model to .tern-model format",
        prog="tern-convert",
    )
    parser.add_argument(
        "model_id", nargs="?", default=None,
        help="HuggingFace model ID or local path (positional)",
    )
    parser.add_argument(
        "--model", default=None,
        help="HuggingFace model ID or local path (named, overrides positional)",
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output .tern-model path"
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.7,
        help="Quantisation threshold (default: 0.7)",
    )
    parser.add_argument(
        "--protect", nargs="*", default=None,
        help="Additional layer name patterns to protect (glob)",
    )
    parser.add_argument(
        "--adapter", default=None,
        help="Architecture adapter name (e.g. gemma4)",
    )
    parser.add_argument(
        "--name", default="model",
        help="Output model name (without .tern-model suffix)",
    )
    parser.add_argument(
        "--mixed-int4", action="store_true",
        help="Use mixed ternary/INT4/FP16 quantisation (requires --adapter)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Scan tolerance and report — do not write .tern-model",
    )
    parser.add_argument(
        "--info", action="store_true",
        help="Show model info without converting",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify output after conversion (CRC32 check)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", default=True,
        help="Show progress during conversion",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()
    verbose = not args.quiet

    # Resolve model ID: --model takes precedence over positional
    resolved_model_id = args.model or args.model_id
    if resolved_model_id is None:
        parser.error("model ID is required (positional or --model)")

    # Dry-run mode (requires --adapter)
    if args.dry_run:
        if not args.adapter:
            parser.error("--dry-run requires --adapter")
        try:
            dry_run_convert(
                model_id=resolved_model_id,
                adapter_name=args.adapter,
                output_dir=args.output,
                threshold=args.threshold,
                verbose=verbose,
            )
        except Exception as e:
            print(f"ERROR: Dry-run failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        return

    # Full adapter-aware conversion (mixed ternary/INT4/FP16)
    if args.mixed_int4 or (args.adapter and not args.dry_run and not args.info):
        if not args.adapter:
            parser.error("--mixed-int4 requires --adapter")
        try:
            full_convert(
                model_id=resolved_model_id,
                adapter_name=args.adapter,
                output_dir=args.output,
                threshold=args.threshold,
                name=args.name,
                verbose=verbose,
            )
        except Exception as e:
            print(f"ERROR: Conversion failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        return

    if args.info:
        # Info mode: show model architecture without converting
        try:
            info = TernaryConverter.info(resolved_model_id)
            print(f"Model: {info['model_id']}")
            print(f"  Architecture:  {info['architecture']}")
            print(f"  Hidden size:   {info['hidden_size']}")
            print(f"  Layers:        {info['num_layers']}")
            print(f"  Heads:         {info['num_heads']}")
            print(f"  Vocab size:    {info['vocab_size']}")
            print(f"  Intermediate:  {info['intermediate_size']}")
        except Exception as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        return

    # Convert mode
    converter = TernaryConverter(
        model_id=resolved_model_id,
        output_path=args.output,
        threshold=args.threshold,
        protection_patterns=args.protect,
    )

    try:
        stats = converter.convert(verbose=verbose)
    except Exception as e:
        print(f"ERROR: Conversion failed: {e}")
        sys.exit(1)

    if args.verify:
        ok = converter.verify(verbose=verbose)
        if not ok:
            sys.exit(1)

    if verbose:
        print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
