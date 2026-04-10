"""
Streaming shard-by-shard ternary conversion pipeline.

Converts a sharded safetensors model to .tern-model format without
ever loading the full model into memory.  Peak RAM usage is ~1.7 GB
for a 70B model (one transformer block at a time) vs 140 GB for the
full-model pipeline.

Requires a pre-computed protection list (from a cached ScanResult or
pattern-based defaults).  Use ``streaming_scan()`` in autoscan.py to
generate one without loading the full model.

Part of tern-core v0.5.0: streaming shard-by-shard conversion pipeline.

Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

from terncore.sharded_loader import (
    ShardedWeightIterator,
    WeightBlock,
    NonBlockWeights,
)
from terncore.tern_model import TernModelWriter


# Patterns that are always protected (same as autoscan._SKIP_PATTERNS).
_ALWAYS_PROTECTED = ("embed", "layernorm", "layer_norm", "rmsnorm",
                     "lm_head", "output", "classifier")


@dataclass
class StreamingConversionReport:
    """Stats from a streaming conversion run."""

    model_dir: str
    total_weights: int = 0
    ternary_weights: int = 0
    protected_weights: int = 0
    total_params: int = 0
    ternary_params: int = 0
    protected_params: int = 0
    original_size_mb: float = 0.0
    ternary_size_mb: float = 0.0
    compression_ratio: float = 1.0
    output_path: str = ""
    output_size_bytes: int = 0
    elapsed_seconds: float = 0.0
    blocks_processed: int = 0
    per_layer: list[dict] = field(default_factory=list)


def _is_protected(name: str, protection_set: set[str]) -> bool:
    """Check if a weight should stay in FP16."""
    if name in protection_set:
        return True
    name_lower = name.lower()
    return any(p in name_lower for p in _ALWAYS_PROTECTED)


class StreamingConverter:
    """Convert a sharded safetensors model to .tern-model, streaming.

    Processes one transformer block at a time.  Each block's tensors are
    loaded from the shard file, quantised/packed (or kept as FP16 if
    protected), added to a TernModelWriter, and discarded before the
    next block is loaded.

    Usage::

        converter = StreamingConverter(
            model_dir="./llama70b",
            output_path="llama70b.tern-model",
            protection_list=scan_result.protection_list,
        )
        report = converter.convert()
    """

    def __init__(
        self,
        model_dir: str | Path,
        output_path: str | Path,
        protection_list: Optional[list[str]] = None,
        ternary_list: Optional[list[str]] = None,
        threshold: float = 0.7,
        verbose: bool = True,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.output_path = Path(output_path)
        self.protection_list = protection_list or []
        self.ternary_list = ternary_list or []
        self.threshold = threshold
        self.verbose = verbose
        self._ternary_set: set[str] = set(self.ternary_list)

    def convert(self) -> StreamingConversionReport:
        """Run the streaming conversion pipeline.

        Returns:
            StreamingConversionReport with layer counts, compression, timing.
        """
        t_start = time.perf_counter()
        protection_set = set(self.protection_list)

        loader = ShardedWeightIterator(self.model_dir)
        writer = TernModelWriter({
            "source": str(self.model_dir),
            "threshold": self.threshold,
            "pipeline": "streaming-v0.5.0",
        })

        report = StreamingConversionReport(model_dir=str(self.model_dir))

        self._log(f"Streaming conversion: {self.model_dir}")
        self._log(f"  Blocks: {loader.num_blocks}, Weights: {loader.num_weights}")
        self._log(f"  Protection list: {len(protection_set)} layers")

        for item in loader:
            if isinstance(item, WeightBlock):
                report.blocks_processed += 1
                self._process_block(item, protection_set, writer, report)
                if self.verbose and report.blocks_processed % 10 == 0:
                    self._log(
                        f"  [{report.blocks_processed}/{loader.num_blocks}] "
                        f"blocks processed"
                    )
                gc.collect()
            elif isinstance(item, NonBlockWeights):
                self._process_non_block(item, protection_set, writer, report)

        # Write output (streaming two-pass to avoid holding all weights in memory)
        self._log(f"  Writing {self.output_path} (streaming)...")
        write_stats = writer.write_streaming(self.output_path)

        # Finalise report
        report.output_path = str(self.output_path)
        report.output_size_bytes = write_stats["file_size"]
        report.original_size_mb = (report.total_params * 2) / (1024 * 1024)
        ternary_bytes = report.ternary_params * 0.25
        fp16_bytes = report.protected_params * 2
        report.ternary_size_mb = (ternary_bytes + fp16_bytes) / (1024 * 1024)
        if report.ternary_size_mb > 0:
            report.compression_ratio = report.original_size_mb / report.ternary_size_mb
        report.elapsed_seconds = time.perf_counter() - t_start

        self._log(f"\nStreaming conversion complete in {report.elapsed_seconds:.1f}s")
        self._log(f"  Ternary: {report.ternary_weights}/{report.total_weights} layers "
                  f"({report.ternary_params:,} params)")
        self._log(f"  Protected: {report.protected_weights} layers "
                  f"({report.protected_params:,} params)")
        self._log(f"  Compression: {report.compression_ratio:.2f}x")
        self._log(f"  Output: {report.output_size_bytes / 1024 / 1024:.1f} MB")

        return report

    def _process_block(
        self,
        block: WeightBlock,
        protection_set: set[str],
        writer: TernModelWriter,
        report: StreamingConversionReport,
    ) -> None:
        """Process one transformer block's worth of tensors."""
        for name, tensor in sorted(block.weights.items()):
            self._process_weight(name, tensor, protection_set, writer, report)
            del tensor
        block.weights.clear()

    def _process_non_block(
        self,
        nb: NonBlockWeights,
        protection_set: set[str],
        writer: TernModelWriter,
        report: StreamingConversionReport,
    ) -> None:
        """Process non-block weights (embed, lm_head, norm)."""
        for name, tensor in sorted(nb.weights.items()):
            self._process_weight(name, tensor, protection_set, writer, report)
            del tensor
        nb.weights.clear()

    def _process_weight(
        self,
        name: str,
        tensor: torch.Tensor,
        protection_set: set[str],
        writer: TernModelWriter,
        report: StreamingConversionReport,
    ) -> None:
        """Quantise or protect a single weight tensor.

        Three-tier assignment:
        - ternary_list → ternary {-1, 0, +1}
        - int4_list → INT4 block-wise (CoreML-native)
        - everything else → FP16
        """
        num_params = tensor.numel()
        report.total_weights += 1
        report.total_params += num_params

        is_1d = tensor.ndim < 2  # LayerNorm weights, biases
        always_protected = is_1d or _is_protected(name, protection_set)

        if always_protected:
            writer.add_layer(name, tensor.float(), dtype="float16")
            report.protected_weights += 1
            report.protected_params += num_params
            report.per_layer.append({
                "name": name, "dtype": "float16",
                "shape": list(tensor.shape), "params": num_params,
            })
        elif name in self._ternary_set:
            packed, alpha, bitmap, sparsity = TernModelWriter.pack_ternary(
                tensor.float(), self.threshold
            )
            writer.add_ternary_layer(
                name=name,
                packed_weights=packed,
                alpha=alpha,
                shape=list(tensor.shape),
                sparsity_bitmap=bitmap,
                threshold=self.threshold,
                sparsity=sparsity,
            )
            report.ternary_weights += 1
            report.ternary_params += num_params
            report.per_layer.append({
                "name": name, "dtype": "ternary2",
                "shape": list(tensor.shape), "params": num_params,
                "sparsity": round(sparsity, 4), "alpha": round(alpha, 6),
            })
        else:
            # INT4 block-wise quantisation
            from terncore.int4_quantizer import quantize_int4_block
            result = quantize_int4_block(tensor.float(), block_size=32)
            writer.add_int4_layer(
                name=name,
                packed_weights=result.packed_weights,
                scales=result.scales,
                shape=result.weight_shape,
                scale_shape=result.scale_shape,
                block_size=result.block_size,
                quant_error=result.reconstruction_error,
            )
            report.ternary_weights += 1  # counts as "quantised"
            report.ternary_params += num_params
            report.per_layer.append({
                "name": name, "dtype": "int4_block32",
                "shape": list(tensor.shape), "params": num_params,
                "quant_error": round(result.reconstruction_error, 6),
            })

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)
