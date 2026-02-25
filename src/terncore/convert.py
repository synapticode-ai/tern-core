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

Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
"""

from __future__ import annotations

import argparse
import fnmatch
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

        _log(f"  Total Linear layers: {len(all_linear)}")
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
        """Find all nn.Linear layer names in the model."""
        return [
            name for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
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
            if not isinstance(module, nn.Linear):
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
            if isinstance(module, nn.Linear)
        ]

        done = 0
        total = len(all_linear)

        for name, module in all_linear:
            weight = module.weight.data
            bias = module.bias.data if module.bias is not None else None
            shape = list(weight.shape)
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


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace model to .tern-model format",
        prog="tern-convert",
    )
    parser.add_argument("model_id", help="HuggingFace model ID or local path")
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

    if args.info:
        # Info mode: show model architecture without converting
        try:
            info = TernaryConverter.info(args.model_id)
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
        model_id=args.model_id,
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
