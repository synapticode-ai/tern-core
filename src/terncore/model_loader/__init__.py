"""
.tern-model format: Synapticode's canonical ternary model format.

Patent 17: Cross-platform model deployment format.
Patent 22: Universal .tern-model specification.

File structure:
    [HEADER]     — Magic bytes, version, metadata length
    [METADATA]   — JSON: model architecture, layer specs, provenance
    [WEIGHTS]    — Packed ternary weights (2-bit encoding) + sparsity bitmaps
    [FP16_DATA]  — Protected layers stored in FP16
    [CHECKSUM]   — SHA-256 for integrity verification

The format is designed for:
    1. Minimal size (2-bit packing + sparsity)
    2. Fast loading (memory-mapped weight access)
    3. Deterministic verification (checksum guarantees bit-identical models)
    4. Provenance chain (who converted, when, from what source, with what settings)
"""

from __future__ import annotations

import io
import json
import hashlib
import struct
import time
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Any

from terncore.arithmetic.linear import TernaryLinear, TernaryConv2d
from terncore.sparse import pack_ternary_weights, unpack_ternary_weights

# Magic bytes: "TERN" in ASCII
TERN_MAGIC = b"TERN"
TERN_VERSION = 1


class TernModelWriter:
    """
    Write a converted model to .tern-model format.

    Usage:
        writer = TernModelWriter()
        writer.save(model, "model.tern-model", source="meta-llama/Llama-3-8B")
    """

    def save(
        self,
        model: nn.Module,
        path: str | Path,
        source: str = "unknown",
        notes: str = "",
    ) -> dict:
        """
        Save a model to .tern-model format.

        Args:
            model:  PyTorch model (should be converted to ternary already).
            path:   Output file path.
            source: Source model identifier (e.g., HuggingFace model ID).
            notes:  Optional notes for provenance.

        Returns:
            Metadata dict (also embedded in the file).
        """
        path = Path(path)

        # Build metadata
        layer_specs = []
        weight_buffers = []
        fp16_buffers = []

        for name, module in model.named_modules():
            if isinstance(module, (TernaryLinear, TernaryConv2d)):
                # Pack ternary weights
                from terncore.arithmetic.quantizer import TernaryQuantizer

                q = TernaryQuantizer(threshold=module.threshold)
                ternary, alpha = q.quantize(module.weight.data)
                packed, bitmap = pack_ternary_weights(ternary)

                weight_data = {
                    "packed": packed.numpy().tobytes(),
                    "bitmap": bitmap.numpy().tobytes(),
                    "alpha": alpha.item(),
                }
                if module.bias is not None:
                    weight_data["bias"] = module.bias.data.numpy().tobytes()

                weight_buffers.append(weight_data)

                spec = {
                    "name": name,
                    "type": "ternary_linear" if isinstance(module, TernaryLinear) else "ternary_conv2d",
                    "shape": list(module.weight.shape),
                    "threshold": module.threshold,
                    "has_bias": module.bias is not None,
                    "weight_index": len(weight_buffers) - 1,
                }
                if isinstance(module, TernaryConv2d):
                    spec["stride"] = module.stride
                    spec["padding"] = module.padding

                layer_specs.append(spec)

            elif isinstance(module, (nn.Linear, nn.Conv2d)):
                # Store protected layers in FP16
                fp16_data = module.weight.data.half().numpy().tobytes()
                bias_data = None
                if module.bias is not None:
                    bias_data = module.bias.data.half().numpy().tobytes()

                fp16_buffers.append({
                    "weight": fp16_data,
                    "bias": bias_data,
                })

                spec = {
                    "name": name,
                    "type": "fp16_linear" if isinstance(module, nn.Linear) else "fp16_conv2d",
                    "shape": list(module.weight.shape),
                    "has_bias": module.bias is not None,
                    "fp16_index": len(fp16_buffers) - 1,
                }
                layer_specs.append(spec)

        metadata = {
            "version": TERN_VERSION,
            "source": source,
            "notes": notes,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "created_by": "terncore",
            "terncore_version": "0.1.0",
            "num_layers": len(layer_specs),
            "num_ternary_layers": sum(
                1 for s in layer_specs if s["type"].startswith("ternary")
            ),
            "num_fp16_layers": sum(
                1 for s in layer_specs if s["type"].startswith("fp16")
            ),
            "layers": layer_specs,
        }

        # Serialise to binary
        metadata_json = json.dumps(metadata, indent=2).encode("utf-8")

        # Build weight section
        weight_section = io.BytesIO()
        for wb in weight_buffers:
            for key in ["packed", "bitmap"]:
                data = wb[key]
                weight_section.write(struct.pack("<I", len(data)))
                weight_section.write(data)
            weight_section.write(struct.pack("<f", wb["alpha"]))
            if wb.get("bias"):
                weight_section.write(struct.pack("<I", len(wb["bias"])))
                weight_section.write(wb["bias"])
            else:
                weight_section.write(struct.pack("<I", 0))
        weight_bytes = weight_section.getvalue()

        # Build FP16 section
        fp16_section = io.BytesIO()
        for fb in fp16_buffers:
            fp16_section.write(struct.pack("<I", len(fb["weight"])))
            fp16_section.write(fb["weight"])
            if fb["bias"]:
                fp16_section.write(struct.pack("<I", len(fb["bias"])))
                fp16_section.write(fb["bias"])
            else:
                fp16_section.write(struct.pack("<I", 0))
        fp16_bytes = fp16_section.getvalue()

        # Assemble file
        with open(path, "wb") as f:
            # Header
            f.write(TERN_MAGIC)
            f.write(struct.pack("<H", TERN_VERSION))
            f.write(struct.pack("<I", len(metadata_json)))
            f.write(struct.pack("<I", len(weight_bytes)))
            f.write(struct.pack("<I", len(fp16_bytes)))

            # Sections
            f.write(metadata_json)
            f.write(weight_bytes)
            f.write(fp16_bytes)

        # Checksum (of everything before checksum)
        with open(path, "rb") as f:
            content = f.read()

        # Append checksum
        checksum = hashlib.sha256(content).digest()
        with open(path, "ab") as f:
            f.write(checksum)

        metadata["file_size_bytes"] = len(content) + 32
        metadata["sha256"] = hashlib.sha256(content).hexdigest()

        return metadata


class TernModelReader:
    """
    Read a .tern-model file and return metadata.

    Full model reconstruction is Stage 2 work (tern-compiler).
    For now this validates the format and returns metadata + checksums.

    Usage:
        reader = TernModelReader()
        meta = reader.read_metadata("model.tern-model")
        valid = reader.verify("model.tern-model")
    """

    def read_metadata(self, path: str | Path) -> dict:
        """Read and return metadata from a .tern-model file."""
        path = Path(path)

        with open(path, "rb") as f:
            # Header
            magic = f.read(4)
            if magic != TERN_MAGIC:
                raise ValueError(f"Not a .tern-model file (magic: {magic!r})")

            version = struct.unpack("<H", f.read(2))[0]
            if version > TERN_VERSION:
                raise ValueError(
                    f"Unsupported version {version} (max supported: {TERN_VERSION})"
                )

            meta_len = struct.unpack("<I", f.read(4))[0]
            _weight_len = struct.unpack("<I", f.read(4))[0]
            _fp16_len = struct.unpack("<I", f.read(4))[0]

            # Metadata
            metadata_json = f.read(meta_len)
            metadata = json.loads(metadata_json.decode("utf-8"))

        return metadata

    def verify(self, path: str | Path) -> bool:
        """
        Verify file integrity using embedded SHA-256 checksum.

        Patent 36, Claim 14: Deterministic reproducibility for verification.

        Returns True if checksum matches, False if corrupted.
        """
        path = Path(path)

        with open(path, "rb") as f:
            content = f.read()

        if len(content) < 32:
            return False

        file_content = content[:-32]
        stored_checksum = content[-32:]
        computed_checksum = hashlib.sha256(file_content).digest()

        return stored_checksum == computed_checksum
