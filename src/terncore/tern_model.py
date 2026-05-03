"""
Production .tern-model v2 format: offset-based random access, SIMD alignment.

Patent 6: Model format specification.
Patent 8: Serialisation and integrity verification.

File structure:
    [HEADER]    256 bytes fixed — magic, version, section offsets
    [MANIFEST]  Variable JSON — layer entries with byte offsets
    [WEIGHTS]   Packed ternary (2-bit) + FP16 protected, 32-byte aligned
    [FOOTER]    16 bytes — CRC32, file_size, reverse magic "NRET"

See docs/tern-model-spec.md for the full byte-level specification.

Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
"""

from __future__ import annotations

import io
import json
import struct
import time
import zlib
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from terncore.arithmetic.quantizer import TernaryQuantizer
from terncore.sparse import pack_ternary_weights, unpack_ternary_weights

# Format constants
TERN_MAGIC = b"TERN"
TERN_MAGIC_REVERSE = b"NRET"
TERN_VERSION = 2
HEADER_SIZE = 256
ALIGNMENT = 32  # 32-byte SIMD boundary (AVX2)


# ── Known transformers-API drift presets ─────────────────────────────
# Used as `key_mapping=` argument on load_as_model / load_packed_model
# to translate manifest prefixes when the source model class hierarchy
# has been reorganised since the artefact was packed.
#
# Gemma 4 multimodal: artefacts packed against pre-5.5 transformers had
# text-tower components at `model.embed_tokens.*` etc.; transformers 5.5+
# moved these under `model.language_model.*`. Audio/vision paths align
# in both layouts and need no rewrite.
GEMMA4_MULTIMODAL_TRANSFORMERS_5_5: Dict[str, str] = {
    "model.embed_tokens.": "model.language_model.embed_tokens.",
    "model.embed_tokens_per_layer.": "model.language_model.embed_tokens_per_layer.",
    "model.layers.": "model.language_model.layers.",
    "model.norm.": "model.language_model.norm.",
    "model.per_layer_model_projection.": "model.language_model.per_layer_model_projection.",
    "model.per_layer_projection_norm.": "model.language_model.per_layer_projection_norm.",
}


def _align_to(offset: int, alignment: int = ALIGNMENT) -> int:
    """Round offset up to the next alignment boundary."""
    remainder = offset % alignment
    return offset if remainder == 0 else offset + (alignment - remainder)


def _pad_to(buf: io.BytesIO, alignment: int = ALIGNMENT) -> None:
    """Write zero-padding to bring buffer position to alignment boundary."""
    pos = buf.tell()
    aligned = _align_to(pos, alignment)
    if aligned > pos:
        buf.write(b"\x00" * (aligned - pos))


class TernModelWriter:
    """
    Write models to .tern-model v2 format.

    Patent 6: Model format with random-access manifest.
    Patent 8: Integrity-verified serialisation.

    Usage (raw FP32 weights — quantised internally):
        writer = TernModelWriter({"source": "TinyLlama/TinyLlama-1.1B"})
        writer.add_layer("layer.0.v_proj", fp32_weights, threshold=0.7)
        writer.write("model.tern-model")

    Usage (pre-packed ternary):
        writer = TernModelWriter({"source": "custom"})
        packed, alpha, bitmap, sparsity = TernModelWriter.pack_ternary(weights, 0.7)
        writer.add_ternary_layer("layer.0.v_proj", packed, alpha, shape,
                                 sparsity_bitmap=bitmap)
        writer.write("model.tern-model")
    """

    def __init__(self, model_metadata: Optional[dict] = None) -> None:
        self._metadata = model_metadata or {}
        self._layers: list[dict] = []

    def add_layer(
        self,
        name: str,
        weights: torch.Tensor,
        dtype: str = "ternary2",
        threshold: float = 0.7,
        sensitivity_score: float = 0.0,
        quant_error: float = 0.0,
        bias: Optional[torch.Tensor] = None,
        block_size: int = 32,
    ) -> None:
        """
        Add a layer from raw FP32 weights.

        For dtype="ternary2": quantises, packs, and stores 2-bit weights.
        For dtype="float16": stores weights in half precision.
        For dtype="int4_block32": quantises to block-wise INT4 (CoreML-native).

        Args:
            name:              Layer name (e.g. "model.layers.0.self_attn.v_proj").
            weights:           FP32 weight tensor.
            dtype:             "ternary2", "float16", or "int4_block32".
            threshold:         Quantisation threshold (ternary only).
            sensitivity_score: From per-layer sensitivity analysis.
            quant_error:       Reconstruction MSE vs original.
            bias:              Optional bias tensor.
            block_size:        Block size for INT4 quantisation (default 32).
        """
        if dtype == "ternary2":
            packed, alpha, bitmap, sparsity = self.pack_ternary(weights, threshold)
            self.add_ternary_layer(
                name=name,
                packed_weights=packed,
                alpha=alpha,
                shape=list(weights.shape),
                sparsity_bitmap=bitmap,
                threshold=threshold,
                sparsity=sparsity,
                sensitivity_score=sensitivity_score,
                quant_error=quant_error,
                bias=bias,
            )
        elif dtype == "int4_block32":
            self._add_int4_layer(
                name=name,
                weights=weights,
                block_size=block_size,
                sensitivity_score=sensitivity_score,
                quant_error=quant_error,
                bias=bias,
            )
        elif dtype == "float16":
            self._add_fp16_layer(
                name=name,
                weights=weights,
                sensitivity_score=sensitivity_score,
                quant_error=quant_error,
                bias=bias,
            )
        else:
            raise ValueError(
                f"Unsupported dtype: {dtype!r}. "
                f"Use 'ternary2', 'int4_block32', or 'float16'."
            )

    def add_ternary_layer(
        self,
        name: str,
        packed_weights: bytes,
        alpha: float,
        shape: list[int],
        sparsity_bitmap: Optional[bytes] = None,
        **metadata: Any,
    ) -> None:
        """
        Add a pre-packed ternary layer.

        Args:
            name:            Layer name.
            packed_weights:  2-bit packed weight bytes (from pack_ternary).
            alpha:           Per-layer scaling factor.
            shape:           Original weight shape, e.g. [2048, 2048].
            sparsity_bitmap: Optional sparsity bitmap bytes.
            **metadata:      Additional fields (threshold, sparsity, sensitivity_score,
                            quant_error, bias as torch.Tensor, etc.).
        """
        num_params = 1
        for s in shape:
            num_params *= s

        # Serialise bias if provided
        bias_tensor = metadata.pop("bias", None)
        bias_bytes = None
        has_bias = False
        if bias_tensor is not None:
            has_bias = True
            bias_bytes = bias_tensor.detach().float().numpy().tobytes()

        self._layers.append({
            "name": name,
            "dtype": "ternary2",
            "shape": shape,
            "num_params": num_params,
            "threshold": metadata.get("threshold", 0.7),
            "alpha": alpha,
            "sparsity": metadata.get("sparsity", 0.0),
            "sensitivity_score": metadata.get("sensitivity_score", 0.0),
            "quant_error": metadata.get("quant_error", 0.0),
            "has_bias": has_bias,
            "has_bitmap": sparsity_bitmap is not None,
            # Binary data (not in manifest JSON)
            "_packed": packed_weights,
            "_bitmap": sparsity_bitmap,
            "_bias": bias_bytes,
        })

    def add_int4_layer(
        self,
        name: str,
        packed_weights: bytes,
        scales: bytes,
        shape: list[int],
        scale_shape: list[int],
        block_size: int = 32,
        **metadata: Any,
    ) -> None:
        """
        Add a pre-packed INT4 block-wise quantised layer.

        Args:
            name:           Layer name.
            packed_weights: INT4 packed bytes (2 values per byte, LSB-first).
            scales:         FP16 per-block scale bytes.
            shape:          Original weight shape [out_features, in_features].
            scale_shape:    Scale tensor shape [out_features, n_blocks].
            block_size:     Block size used during quantisation.
            **metadata:     Additional fields (sensitivity_score, quant_error, bias).
        """
        num_params = 1
        for s in shape:
            num_params *= s

        bias_tensor = metadata.pop("bias", None)
        bias_bytes = None
        has_bias = False
        if bias_tensor is not None:
            has_bias = True
            bias_bytes = bias_tensor.detach().float().numpy().tobytes()

        self._layers.append({
            "name": name,
            "dtype": "int4_block32",
            "shape": shape,
            "scale_shape": scale_shape,
            "block_size": block_size,
            "num_params": num_params,
            "sensitivity_score": metadata.get("sensitivity_score", 0.0),
            "quant_error": metadata.get("quant_error", 0.0),
            "has_bias": has_bias,
            # Binary data
            "_packed": packed_weights,
            "_scales": scales,
            "_bias": bias_bytes,
        })

    def _add_int4_layer(
        self,
        name: str,
        weights: torch.Tensor,
        block_size: int = 32,
        sensitivity_score: float = 0.0,
        quant_error: float = 0.0,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        """Quantise and add an INT4 block-wise layer from FP32 weights."""
        from terncore.int4_quantizer import quantize_int4_block

        result = quantize_int4_block(weights, block_size=block_size)
        self.add_int4_layer(
            name=name,
            packed_weights=result.packed_weights,
            scales=result.scales,
            shape=result.weight_shape,
            scale_shape=result.scale_shape,
            block_size=result.block_size,
            sensitivity_score=sensitivity_score,
            quant_error=result.reconstruction_error,
            bias=bias,
        )

    def _add_fp16_layer(
        self,
        name: str,
        weights: torch.Tensor,
        sensitivity_score: float = 0.0,
        quant_error: float = 0.0,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        """Add a protected FP16 layer."""
        weight_bytes = weights.detach().half().numpy().tobytes()

        bias_bytes = None
        has_bias = False
        if bias is not None:
            has_bias = True
            bias_bytes = bias.detach().half().numpy().tobytes()

        self._layers.append({
            "name": name,
            "dtype": "float16",
            "shape": list(weights.shape),
            "num_params": weights.numel(),
            "sensitivity_score": sensitivity_score,
            "quant_error": quant_error,
            "has_bias": has_bias,
            # Binary data
            "_weight_data": weight_bytes,
            "_bias": bias_bytes,
        })

    def write(self, path: str | Path) -> dict:
        """
        Serialise all added layers to a .tern-model v2 file.

        Returns:
            Dict with file stats (size, num_layers, crc32).
        """
        path = Path(path)

        # --- Build weight data section ---
        weight_buf = io.BytesIO()
        layer_manifests = []

        for layer in self._layers:
            # Record offset relative to weight section start
            layer_offset = weight_buf.tell()

            if layer["dtype"] == "ternary2":
                self._write_ternary_layer(weight_buf, layer)
            elif layer["dtype"] == "int4_block32":
                self._write_int4_layer(weight_buf, layer)
            else:
                self._write_fp16_layer(weight_buf, layer)

            layer_size = weight_buf.tell() - layer_offset

            # Pad to 32-byte boundary
            _pad_to(weight_buf, ALIGNMENT)

            # Manifest entry (no underscore-prefixed binary data)
            entry = {
                k: v for k, v in layer.items() if not k.startswith("_")
            }
            entry["offset"] = layer_offset
            entry["size"] = layer_size
            layer_manifests.append(entry)

        weight_data = weight_buf.getvalue()
        weights_size = len(weight_data)

        # --- Build manifest ---
        manifest_obj = {
            "model_metadata": {
                "source": self._metadata.get("source", "unknown"),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "created_by": "terncore",
                "terncore_version": "0.2.0",
                "notes": self._metadata.get("notes", ""),
                **{k: v for k, v in self._metadata.items()
                   if k not in ("source", "notes")},
            },
            "layers": layer_manifests,
        }
        manifest_json = json.dumps(manifest_obj, indent=2).encode("utf-8")

        # Calculate section offsets
        manifest_offset = HEADER_SIZE
        manifest_size = len(manifest_json)
        weights_offset = _align_to(manifest_offset + manifest_size, ALIGNMENT)
        manifest_padding = weights_offset - manifest_offset - manifest_size

        # Layer counts
        num_ternary = sum(1 for l in self._layers if l["dtype"] == "ternary2")
        num_protected = sum(1 for l in self._layers if l["dtype"] == "float16")

        # --- Build header (256 bytes) ---
        header = io.BytesIO()
        header.write(TERN_MAGIC)                                    # 0:  magic (4)
        header.write(struct.pack("<H", TERN_VERSION))               # 4:  version (2)
        header.write(struct.pack("<H", HEADER_SIZE))                # 6:  header_size (2)
        header.write(struct.pack("<Q", manifest_offset))            # 8:  manifest_offset (8)
        header.write(struct.pack("<Q", manifest_size))              # 16: manifest_size (8)
        header.write(struct.pack("<Q", weights_offset))             # 24: weights_offset (8)
        header.write(struct.pack("<Q", weights_size))               # 32: weights_size (8)
        header.write(struct.pack("<I", len(self._layers)))          # 40: num_layers (4)
        header.write(struct.pack("<I", num_ternary))                # 44: num_ternary (4)
        header.write(struct.pack("<I", num_protected))              # 48: num_protected (4)
        # Reserved (pad to 256 bytes)
        header.write(b"\x00" * (HEADER_SIZE - header.tell()))

        header_data = header.getvalue()
        assert len(header_data) == HEADER_SIZE

        # --- CRC32 over weight data ---
        crc = zlib.crc32(weight_data) & 0xFFFFFFFF

        # --- Footer (16 bytes) ---
        total_size = HEADER_SIZE + manifest_size + manifest_padding + weights_size + 16
        footer = struct.pack("<I", crc)                             # crc32 (4)
        footer += struct.pack("<Q", total_size)                     # file_size (8)
        footer += TERN_MAGIC_REVERSE                                # "NRET" (4)

        # --- Write file ---
        with open(path, "wb") as f:
            f.write(header_data)
            f.write(manifest_json)
            f.write(b"\x00" * manifest_padding)
            f.write(weight_data)
            f.write(footer)

        return {
            "file_size": total_size,
            "num_layers": len(self._layers),
            "num_ternary": num_ternary,
            "num_protected": num_protected,
            "weights_size": weights_size,
            "crc32": crc,
        }

    def _write_ternary_layer(self, buf: io.BytesIO, layer: dict) -> None:
        """Write a ternary layer's binary data to the buffer."""
        packed = layer["_packed"]
        bitmap = layer.get("_bitmap")
        bias = layer.get("_bias")

        # alpha (float32)
        buf.write(struct.pack("<f", layer["alpha"]))
        # packed weights
        buf.write(struct.pack("<I", len(packed)))
        buf.write(packed)
        # bitmap
        if bitmap is not None:
            buf.write(struct.pack("<I", len(bitmap)))
            buf.write(bitmap)
        else:
            buf.write(struct.pack("<I", 0))
        # bias
        if bias is not None:
            buf.write(struct.pack("<I", len(bias)))
            buf.write(bias)
        else:
            buf.write(struct.pack("<I", 0))

    def _write_int4_layer(self, buf: io.BytesIO, layer: dict) -> None:
        """Write an INT4 block-wise layer's binary data to the buffer."""
        packed = layer["_packed"]
        scales = layer["_scales"]
        bias = layer.get("_bias")

        # block_size (uint32)
        buf.write(struct.pack("<I", layer["block_size"]))
        # packed weights
        buf.write(struct.pack("<I", len(packed)))
        buf.write(packed)
        # scales
        buf.write(struct.pack("<I", len(scales)))
        buf.write(scales)
        # bias
        if bias is not None:
            buf.write(struct.pack("<I", len(bias)))
            buf.write(bias)
        else:
            buf.write(struct.pack("<I", 0))

    def _write_fp16_layer(self, buf: io.BytesIO, layer: dict) -> None:
        """Write an FP16 protected layer's binary data to the buffer.

        Uses 8-byte (<Q) size prefix for weight data to support tensors
        larger than 4 GB (e.g. per-layer embeddings in multimodal models).
        """
        weight_data = layer["_weight_data"]
        bias = layer.get("_bias")

        buf.write(struct.pack("<Q", len(weight_data)))
        buf.write(weight_data)

        if bias is not None:
            buf.write(struct.pack("<I", len(bias)))
            buf.write(bias)
        else:
            buf.write(struct.pack("<I", 0))

    def write_streaming(self, path: str | Path) -> dict:
        """
        Streaming two-pass write for large models.

        Unlike write(), this never holds all weight data in memory.
        Pass 1 writes weights to a temp file, recording offsets and
        computing CRC32 incrementally.  Pass 2 assembles the final
        .tern-model: header + manifest + weights (copied from temp)
        + footer.

        Peak memory: one layer's binary data at a time (~900 MB max
        for the largest FP16 layer in a 70B model).

        Returns:
            Dict with file stats (size, num_layers, crc32).
        """
        import tempfile
        import shutil

        path = Path(path)
        layer_manifests = []
        crc = 0
        weights_size = 0
        num_ternary = 0
        num_protected = 0

        # --- Pass 1: write weights to temp file ---
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tern-weights")
        try:
            with open(tmp_fd, "wb") as tmp_f:
                for layer in self._layers:
                    layer_offset = tmp_f.tell()

                    # Write layer data to a small buffer, then flush to disk
                    layer_buf = io.BytesIO()
                    if layer["dtype"] == "ternary2":
                        self._write_ternary_layer(layer_buf, layer)
                        num_ternary += 1
                    elif layer["dtype"] == "int4_block32":
                        self._write_int4_layer(layer_buf, layer)
                        num_ternary += 1  # count in ternary bucket for header
                    else:
                        self._write_fp16_layer(layer_buf, layer)
                        num_protected += 1

                    layer_bytes = layer_buf.getvalue()
                    layer_size = len(layer_bytes)
                    tmp_f.write(layer_bytes)

                    # Update incremental CRC32
                    crc = zlib.crc32(layer_bytes, crc)

                    # Pad to alignment
                    pad_needed = (_align_to(tmp_f.tell(), ALIGNMENT)
                                  - tmp_f.tell())
                    if pad_needed > 0:
                        pad_bytes = b"\x00" * pad_needed
                        tmp_f.write(pad_bytes)
                        crc = zlib.crc32(pad_bytes, crc)

                    # Manifest entry
                    entry = {
                        k: v for k, v in layer.items()
                        if not k.startswith("_")
                    }
                    entry["offset"] = layer_offset
                    entry["size"] = layer_size
                    layer_manifests.append(entry)

                    # Free the layer's binary data immediately
                    layer.pop("_packed", None)
                    layer.pop("_bitmap", None)
                    layer.pop("_bias", None)
                    layer.pop("_weight_data", None)
                    del layer_bytes, layer_buf

                weights_size = tmp_f.tell()

            crc = crc & 0xFFFFFFFF

            # --- Build manifest ---
            manifest_obj = {
                "model_metadata": {
                    "source": self._metadata.get("source", "unknown"),
                    "created_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                    ),
                    "created_by": "terncore",
                    "terncore_version": "0.5.1",
                    "notes": self._metadata.get("notes", ""),
                    **{k: v for k, v in self._metadata.items()
                       if k not in ("source", "notes")},
                },
                "layers": layer_manifests,
            }
            manifest_json = json.dumps(manifest_obj, indent=2).encode("utf-8")

            # Section offsets
            manifest_offset = HEADER_SIZE
            manifest_size = len(manifest_json)
            weights_offset = _align_to(
                manifest_offset + manifest_size, ALIGNMENT
            )
            manifest_padding = (
                weights_offset - manifest_offset - manifest_size
            )

            # --- Build header ---
            header = io.BytesIO()
            header.write(TERN_MAGIC)
            header.write(struct.pack("<H", TERN_VERSION))
            header.write(struct.pack("<H", HEADER_SIZE))
            header.write(struct.pack("<Q", manifest_offset))
            header.write(struct.pack("<Q", manifest_size))
            header.write(struct.pack("<Q", weights_offset))
            header.write(struct.pack("<Q", weights_size))
            header.write(struct.pack("<I", len(self._layers)))
            header.write(struct.pack("<I", num_ternary))
            header.write(struct.pack("<I", num_protected))
            header.write(b"\x00" * (HEADER_SIZE - header.tell()))
            header_data = header.getvalue()
            assert len(header_data) == HEADER_SIZE

            # --- Footer ---
            total_size = (HEADER_SIZE + manifest_size
                          + manifest_padding + weights_size + 16)
            footer = struct.pack("<I", crc)
            footer += struct.pack("<Q", total_size)
            footer += TERN_MAGIC_REVERSE

            # --- Pass 2: assemble final file ---
            with open(path, "wb") as out_f:
                out_f.write(header_data)
                out_f.write(manifest_json)
                out_f.write(b"\x00" * manifest_padding)

                # Stream weights from temp file in 64 MB chunks
                with open(tmp_path, "rb") as tmp_f:
                    while True:
                        chunk = tmp_f.read(64 * 1024 * 1024)
                        if not chunk:
                            break
                        out_f.write(chunk)

                out_f.write(footer)

        finally:
            # Clean up temp file
            try:
                import os
                os.unlink(tmp_path)
            except OSError:
                pass

        return {
            "file_size": total_size,
            "num_layers": len(self._layers),
            "num_ternary": num_ternary,
            "num_protected": num_protected,
            "weights_size": weights_size,
            "crc32": crc,
        }

    # ── Static helpers ──────────────────────────────────────────

    @staticmethod
    def pack_ternary(
        weights: torch.Tensor, threshold: float = 0.7
    ) -> Tuple[bytes, float, bytes, float]:
        """
        Quantise and pack FP32 weights to 2-bit ternary format.

        Uses TernaryQuantizer for quantisation and sparse.pack_ternary_weights
        for 2-bit packing.

        Args:
            weights:   FP32 weight tensor.
            threshold: Quantisation threshold (0, 1).

        Returns:
            packed_bytes:  2-bit packed weights as bytes.
            alpha:         Per-layer scaling factor (float).
            bitmap_bytes:  Sparsity bitmap as bytes.
            sparsity:      Fraction of zero weights (float).
        """
        q = TernaryQuantizer(threshold=threshold)
        ternary, alpha = q.quantize(weights)

        packed, bitmap = pack_ternary_weights(ternary)

        sparsity = (ternary == 0).sum().item() / ternary.numel()

        packed_bytes = packed.numpy().tobytes()
        bitmap_bytes = bitmap.flatten().numpy().astype(np.uint8)
        # Pack boolean bitmap into bits (8 bools per byte, LSB-first)
        n_bits = len(bitmap_bytes)
        n_bytes = (n_bits + 7) // 8
        bitmap_packed = bytearray(n_bytes)
        for i in range(n_bits):
            if bitmap_bytes[i]:
                bitmap_packed[i // 8] |= 1 << (i % 8)
        bitmap_bytes = bytes(bitmap_packed)

        return packed_bytes, alpha.item(), bitmap_bytes, sparsity

    @staticmethod
    def generate_sparsity_bitmap(packed_weights: bytes, block_size: int = 256) -> bytes:
        """
        Generate a block-level sparsity bitmap from packed 2-bit weights.

        Each bit in the output represents one block of `block_size` weights.
        A bit is 1 if the block contains any non-zero weights, 0 if all zero.

        Args:
            packed_weights: 2-bit packed weight bytes.
            block_size:     Number of weights per block (default 256).

        Returns:
            Block-level bitmap as bytes.
        """
        packed = np.frombuffer(packed_weights, dtype=np.uint8)
        # Each byte holds 4 weights
        total_weights = len(packed) * 4
        num_blocks = (total_weights + block_size - 1) // block_size
        n_bytes = (num_blocks + 7) // 8
        bitmap = bytearray(n_bytes)

        weights_per_byte = 4
        block_bytes = block_size // weights_per_byte

        for block_idx in range(num_blocks):
            start = block_idx * block_bytes
            end = min(start + block_bytes, len(packed))
            block_data = packed[start:end]
            if np.any(block_data != 0):
                bitmap[block_idx // 8] |= 1 << (block_idx % 8)

        return bytes(bitmap)


class TernModelReader:
    """
    Read and validate .tern-model v2 files.

    Patent 6: Model format with random-access manifest.

    Usage:
        reader = TernModelReader("model.tern-model")
        print(reader.header)
        print(reader.manifest)
        assert reader.verify()
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.header = self._read_header()
        self.manifest = self._read_manifest()

    def _read_header(self) -> dict:
        """Read and validate the 256-byte header."""
        with open(self.path, "rb") as f:
            data = f.read(HEADER_SIZE)

        if len(data) < HEADER_SIZE:
            raise ValueError(f"File too small for header ({len(data)} < {HEADER_SIZE})")

        magic = data[0:4]
        if magic != TERN_MAGIC:
            raise ValueError(f"Invalid magic: {magic!r} (expected {TERN_MAGIC!r})")

        version = struct.unpack_from("<H", data, 4)[0]
        if version != TERN_VERSION:
            raise ValueError(f"Unsupported version {version} (expected {TERN_VERSION})")

        return {
            "magic": magic,
            "version": version,
            "header_size": struct.unpack_from("<H", data, 6)[0],
            "manifest_offset": struct.unpack_from("<Q", data, 8)[0],
            "manifest_size": struct.unpack_from("<Q", data, 16)[0],
            "weights_offset": struct.unpack_from("<Q", data, 24)[0],
            "weights_size": struct.unpack_from("<Q", data, 32)[0],
            "num_layers": struct.unpack_from("<I", data, 40)[0],
            "num_ternary": struct.unpack_from("<I", data, 44)[0],
            "num_protected": struct.unpack_from("<I", data, 48)[0],
        }

    def _read_manifest(self) -> dict:
        """Read the manifest JSON."""
        with open(self.path, "rb") as f:
            f.seek(self.header["manifest_offset"])
            data = f.read(self.header["manifest_size"])
        return json.loads(data.decode("utf-8"))

    def verify(self) -> bool:
        """
        Verify file integrity using footer CRC32 and file size.

        Returns:
            True if reverse magic, file size, and CRC32 all match.
        """
        file_size = self.path.stat().st_size

        with open(self.path, "rb") as f:
            # Read footer (last 16 bytes)
            f.seek(file_size - 16)
            footer = f.read(16)

        if len(footer) < 16:
            return False

        stored_crc = struct.unpack_from("<I", footer, 0)[0]
        stored_size = struct.unpack_from("<Q", footer, 4)[0]
        reverse_magic = footer[12:16]

        # Check reverse magic
        if reverse_magic != TERN_MAGIC_REVERSE:
            return False

        # Check file size
        if stored_size != file_size:
            return False

        # Check CRC32 over weight data
        with open(self.path, "rb") as f:
            f.seek(self.header["weights_offset"])
            weight_data = f.read(self.header["weights_size"])

        computed_crc = zlib.crc32(weight_data) & 0xFFFFFFFF
        return computed_crc == stored_crc

    def read_layer_data(self, layer_name: str) -> bytes:
        """
        Random-access read of a single layer's raw binary data.

        Args:
            layer_name: Layer name from manifest.

        Returns:
            Raw bytes for the layer's weight data.
        """
        entry = self._get_manifest_entry(layer_name)
        offset = self.header["weights_offset"] + entry["offset"]
        with open(self.path, "rb") as f:
            f.seek(offset)
            return f.read(entry["size"])

    def _get_manifest_entry(self, layer_name: str) -> dict:
        """Find a manifest entry by layer name."""
        for entry in self.manifest["layers"]:
            if entry["name"] == layer_name:
                return entry
        raise KeyError(f"Layer {layer_name!r} not found in manifest")

    # ── Reconstruction methods ──────────────────────────────────

    def reconstruct_layer(self, name: str) -> dict[str, torch.Tensor]:
        """
        Read a layer and reconstruct as PyTorch tensor(s).

        For ternary layers: unpack 2-bit → ternary {-1,0,+1} → float32 * alpha.
        For float16 layers: reinterpret raw bytes as float16 tensor, cast to float32.

        Args:
            name: Layer name from manifest.

        Returns:
            Dict with "weight" tensor and optionally "bias" tensor.
        """
        entry = self._get_manifest_entry(name)
        raw = self.read_layer_data(name)
        buf = io.BytesIO(raw)

        if entry["dtype"] == "ternary2":
            return self._reconstruct_ternary(buf, entry)
        elif entry["dtype"] == "int4_block32":
            return self._reconstruct_int4(buf, entry)
        elif entry["dtype"] == "float16":
            return self._reconstruct_fp16(buf, entry)
        else:
            raise ValueError(f"Unknown dtype {entry['dtype']!r} for layer {name!r}")

    def _reconstruct_ternary(
        self, buf: io.BytesIO, entry: dict
    ) -> dict[str, torch.Tensor]:
        """Reconstruct a ternary layer from its binary data."""
        shape = entry["shape"]
        num_params = entry["num_params"]

        # Read alpha
        alpha = struct.unpack("<f", buf.read(4))[0]

        # Read packed weights
        packed_size = struct.unpack("<I", buf.read(4))[0]
        packed_bytes = buf.read(packed_size)
        packed_tensor = torch.frombuffer(
            bytearray(packed_bytes), dtype=torch.uint8
        )

        # Unpack 2-bit → ternary {-1, 0, +1}
        ternary = unpack_ternary_weights(packed_tensor, torch.Size(shape))

        # Scale by alpha to get reconstructed weights
        weight = ternary * alpha

        result: dict[str, torch.Tensor] = {"weight": weight}

        # Skip bitmap
        bitmap_size = struct.unpack("<I", buf.read(4))[0]
        if bitmap_size > 0:
            buf.read(bitmap_size)

        # Read bias if present
        bias_size = struct.unpack("<I", buf.read(4))[0]
        if bias_size > 0:
            bias_bytes = buf.read(bias_size)
            result["bias"] = torch.frombuffer(
                bytearray(bias_bytes), dtype=torch.float32
            ).clone()

        return result

    def _reconstruct_int4(
        self, buf: io.BytesIO, entry: dict
    ) -> dict[str, torch.Tensor]:
        """Reconstruct an INT4 block-wise layer from its binary data."""
        from terncore.int4_quantizer import dequantize_int4_block

        shape = entry["shape"]
        scale_shape = entry["scale_shape"]
        block_size = entry.get("block_size", 32)

        # Read block_size
        stored_bs = struct.unpack("<I", buf.read(4))[0]

        # Read packed weights
        packed_size = struct.unpack("<I", buf.read(4))[0]
        packed_bytes = buf.read(packed_size)

        # Read scales
        scales_size = struct.unpack("<I", buf.read(4))[0]
        scales_bytes = buf.read(scales_size)

        # Dequantise
        weight = dequantize_int4_block(
            packed_bytes, scales_bytes, shape, scale_shape, stored_bs
        )

        result: dict[str, torch.Tensor] = {"weight": weight}

        # Read bias
        bias_size = struct.unpack("<I", buf.read(4))[0]
        if bias_size > 0:
            bias_bytes = buf.read(bias_size)
            result["bias"] = torch.frombuffer(
                bytearray(bias_bytes), dtype=torch.float32
            ).clone()

        return result

    def _reconstruct_fp16(
        self, buf: io.BytesIO, entry: dict
    ) -> dict[str, torch.Tensor]:
        """Reconstruct an FP16 layer from its binary data."""
        shape = entry["shape"]

        # Read weight data — 8-byte size prefix (supports >4 GB tensors)
        weight_size = struct.unpack("<Q", buf.read(8))[0]
        weight_bytes = buf.read(weight_size)
        weight = torch.frombuffer(
            bytearray(weight_bytes), dtype=torch.float16
        ).reshape(shape).float().clone()

        result: dict[str, torch.Tensor] = {"weight": weight}

        # Read bias if present
        bias_size = struct.unpack("<I", buf.read(4))[0]
        if bias_size > 0:
            bias_bytes = buf.read(bias_size)
            result["bias"] = torch.frombuffer(
                bytearray(bias_bytes), dtype=torch.float16
            ).float().clone()

        return result

    def reconstruct_all(self) -> dict[str, torch.Tensor]:
        """
        Reconstruct all layers as a flat state_dict-compatible dict.

        Returns:
            {"layer.name.weight": tensor, "layer.name.bias": tensor, ...}
        """
        state_dict: dict[str, torch.Tensor] = {}
        for entry in self.manifest["layers"]:
            name = entry["name"]
            tensors = self.reconstruct_layer(name)
            state_dict[f"{name}.weight"] = tensors["weight"]
            if "bias" in tensors:
                state_dict[f"{name}.bias"] = tensors["bias"]
        return state_dict

    def load_as_model(
        self,
        model: nn.Module,
        strict: bool = False,
        key_mapping: Optional[Dict[str, str]] = None,
    ) -> Tuple[list[str], list[str]]:
        """
        Reconstruct state_dict and load into an existing model.

        Args:
            model:  PyTorch model instance to load weights into.
            strict: If True, raise on missing/unexpected keys.
            key_mapping: Optional prefix-rewrite map applied to state_dict
                keys before load_state_dict. Use named presets like
                GEMMA4_MULTIMODAL_TRANSFORMERS_5_5 to bridge transformers
                API drift between artefact pack-time and load-time.

        Returns:
            (missing_keys, unexpected_keys) from load_state_dict.
        """
        state_dict = self.reconstruct_all()
        if key_mapping:
            rewritten: dict[str, torch.Tensor] = {}
            for k, v in state_dict.items():
                new_k = k
                for src, dst in key_mapping.items():
                    if k.startswith(src):
                        new_k = dst + k[len(src):]
                        break
                rewritten[new_k] = v
            state_dict = rewritten
        result = model.load_state_dict(state_dict, strict=strict)
        return list(result.missing_keys), list(result.unexpected_keys)

    # ── Lazy loading convenience API ────────────────────────────

    def layer(self, name: str) -> torch.Tensor:
        """Load and reconstruct a single layer's weight tensor on demand.

        Args:
            name: Layer name from the manifest.

        Returns:
            Reconstructed FP32 weight tensor.
        """
        return self.reconstruct_layer(name)["weight"]

    def load_all(self) -> dict[str, torch.Tensor]:
        """Load all layers as a state_dict. Alias for reconstruct_all().

        Returns:
            Dict mapping ``"layer.name.weight"`` to reconstructed tensors.
        """
        return self.reconstruct_all()

    def layer_names(self) -> list[str]:
        """List all layer names from the manifest (no weight loading).

        Returns:
            List of layer name strings in manifest order.
        """
        return [entry["name"] for entry in self.manifest["layers"]]

    def layer_info(self, name: str) -> dict:
        """Get manifest metadata for a layer without loading weights.

        Args:
            name: Layer name from the manifest.

        Returns:
            Dict with dtype, shape, num_params, offset, size, and
            dtype-specific fields (threshold, alpha, sparsity for ternary).
        """
        return dict(self._get_manifest_entry(name))

    # ── Packed loading ──────────────────────────────────────────

    def load_packed_model(
        self, model: nn.Module
    ) -> Tuple[list[str], list[str]]:
        """
        Load .tern-model weights as PackedTernaryLinear layers.

        For ternary layers: create PackedTernaryLinear from packed bytes
        (no re-quantisation — directly use the packed data from file).

        For FP16 layers: load as regular nn.Linear weights.

        Args:
            model: PyTorch model to modify in-place.

        Returns:
            (missing_keys, unexpected_keys) analogous to load_state_dict.
        """
        from terncore.packed_linear import PackedTernaryLinear

        loaded_keys: list[str] = []
        model_keys = set(dict(model.named_parameters()).keys())
        model_keys.update(dict(model.named_buffers()).keys())

        for entry in self.manifest["layers"]:
            name = entry["name"]
            raw = self.read_layer_data(name)
            buf = io.BytesIO(raw)

            if entry["dtype"] == "ternary2":
                # Read alpha and packed weights directly
                alpha = struct.unpack("<f", buf.read(4))[0]
                packed_size = struct.unpack("<I", buf.read(4))[0]
                packed_bytes = buf.read(packed_size)
                packed_tensor = torch.frombuffer(
                    bytearray(packed_bytes), dtype=torch.uint8
                ).clone()

                # Read bitmap (pass through to PackedTernaryLinear)
                bitmap_size = struct.unpack("<I", buf.read(4))[0]
                bitmap_tensor = None
                if bitmap_size > 0:
                    bitmap_data = buf.read(bitmap_size)
                    bitmap_tensor = torch.frombuffer(
                        bytearray(bitmap_data), dtype=torch.uint8
                    ).clone()

                # Read bias
                bias_size = struct.unpack("<I", buf.read(4))[0]
                bias = None
                if bias_size > 0:
                    bias_data = buf.read(bias_size)
                    bias = torch.frombuffer(
                        bytearray(bias_data), dtype=torch.float32
                    ).clone()

                packed_layer = PackedTernaryLinear.from_packed_data(
                    packed_weights=packed_tensor,
                    alpha=alpha,
                    in_features=entry["shape"][1],
                    out_features=entry["shape"][0],
                    bias=bias,
                    sparsity_bitmap=bitmap_tensor,
                )

                # Replace module in model
                parts = name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], packed_layer)

                loaded_keys.append(f"{name}.packed_weights")
                loaded_keys.append(f"{name}.alpha")
                if bias is not None:
                    loaded_keys.append(f"{name}.bias")

            elif entry["dtype"] == "float16":
                tensors = self._reconstruct_fp16(buf, entry)
                # Set weight on the existing module
                parts = name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                module = getattr(parent, parts[-1])
                if isinstance(module, nn.Linear):
                    module.weight.data = tensors["weight"]
                    loaded_keys.append(f"{name}.weight")
                    if "bias" in tensors and module.bias is not None:
                        module.bias.data = tensors["bias"]
                        loaded_keys.append(f"{name}.bias")

        missing = [k for k in model_keys if k not in loaded_keys]
        unexpected = [k for k in loaded_keys if k not in model_keys]
        return missing, unexpected
