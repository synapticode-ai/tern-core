# .tern-model Binary Format Specification v2

**Version:** 2.0
**Date:** 2026-02-25
**Status:** Production
**Patent alignment:** Patent 6 (Model Format), Patent 8 (Serialisation)

## 1. Overview

The `.tern-model` format is Synapticode's canonical binary format for ternary neural
network models. It stores weights quantised to {-1, 0, +1} in a 2-bit packed encoding,
alongside protected layers in their original FP16/FP32 precision.

### Design goals

1. **Random access** — offset-based manifest enables loading any layer without parsing
   the entire file.
2. **SIMD alignment** — all weight data starts on 32-byte boundaries for direct
   AVX2/NEON DMA without memcpy.
3. **Rich metadata** — per-layer sensitivity scores, quantisation error, sparsity ratio,
   and arbitrary key-value metadata for NPU toolchains.
4. **Integrity** — CRC32 footer over weight data, file size, and reverse magic for
   corruption detection.
5. **Minimal size** — 2-bit packing gives 16x compression vs FP32 with no additional
   compression layer (preserves DMA-readiness).

### File structure

```
+========================+  offset 0
|        HEADER          |  Fixed 256 bytes
+========================+  offset 256
|       MANIFEST         |  Variable-length JSON
|    (32-byte aligned)   |
+========================+  manifest_offset + manifest_size (aligned)
|      WEIGHT DATA       |  Packed ternary + FP16 layers
|   (each layer aligned  |
|    to 32 bytes)        |
+========================+  weights_offset + weights_size
|        FOOTER          |  Fixed 16 bytes
+========================+
```

## 2. Header (256 bytes, fixed)

All multi-byte integers are **little-endian**.

| Offset | Size | Type     | Field             | Description |
|--------|------|----------|-------------------|-------------|
| 0      | 4    | bytes    | `magic`           | `b"TERN"` (0x5445524E) |
| 4      | 2    | uint16   | `version`         | Format version (2 for this spec) |
| 6      | 2    | uint16   | `header_size`     | Always 256 for v2 |
| 8      | 8    | uint64   | `manifest_offset` | Byte offset of manifest section |
| 16     | 8    | uint64   | `manifest_size`   | Size of manifest JSON in bytes |
| 24     | 8    | uint64   | `weights_offset`  | Byte offset of weight data section |
| 32     | 8    | uint64   | `weights_size`    | Total size of weight data section |
| 40     | 4    | uint32   | `num_layers`      | Total number of layers in manifest |
| 44     | 4    | uint32   | `num_ternary`     | Number of ternary-quantised layers |
| 48     | 4    | uint32   | `num_protected`   | Number of FP16/FP32 protected layers |
| 52     | 204  | bytes    | `reserved`        | Zero-filled, reserved for future use |

**Invariants:**
- `magic` must equal `b"TERN"` exactly.
- `version` must be 2; readers must reject unknown versions.
- `header_size` is always 256 for v2.
- `manifest_offset` equals `header_size` (256).
- `weights_offset` is 32-byte aligned.
- `num_layers == num_ternary + num_protected`.

## 3. Manifest (variable-length JSON)

The manifest is a UTF-8 encoded JSON object starting at `manifest_offset`. It is
padded to a 32-byte boundary after the JSON content (padding bytes are `0x00`).

### Top-level fields

```json
{
  "model_metadata": {
    "source": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "created_at": "2026-02-25T12:00:00Z",
    "created_by": "terncore",
    "terncore_version": "0.2.0",
    "notes": "v_proj_late3 mixed-precision config"
  },
  "layers": [ ... ]
}
```

### Layer entry

Each entry in the `layers` array describes one layer:

```json
{
  "name": "model.layers.21.self_attn.v_proj",
  "dtype": "ternary2",
  "shape": [2048, 2048],
  "num_params": 4194304,
  "threshold": 0.7,
  "alpha": 0.00342,
  "sparsity": 0.634,
  "sensitivity_score": 0.02,
  "quant_error": 0.00015,
  "has_bias": false,
  "has_bitmap": true,
  "offset": 8192,
  "size": 1048608
}
```

| Field              | Type   | Required | Description |
|--------------------|--------|----------|-------------|
| `name`             | string | yes      | Layer name (matches HuggingFace state_dict key) |
| `dtype`            | string | yes      | `"ternary2"` (2-bit packed) or `"float16"` |
| `shape`            | array  | yes      | Weight tensor dimensions, e.g. `[2048, 2048]` |
| `num_params`       | int    | yes      | Product of shape dimensions |
| `threshold`        | float  | ternary  | Quantisation threshold used |
| `alpha`            | float  | ternary  | Per-layer scaling factor |
| `sparsity`         | float  | ternary  | Fraction of zero weights |
| `sensitivity_score`| float  | no       | From per-layer sensitivity analysis |
| `quant_error`      | float  | no       | Reconstruction MSE vs original weights |
| `has_bias`         | bool   | yes      | Whether bias vector is stored |
| `has_bitmap`       | bool   | ternary  | Whether sparsity bitmap is included |
| `offset`           | int    | yes      | Byte offset from start of weight data section |
| `size`             | int    | yes      | Total bytes for this layer in weight data |

**dtype values:**
- `"ternary2"` — 2-bit packed ternary weights (4 weights per byte)
- `"float16"` — IEEE 754 half-precision (2 bytes per weight)

## 4. Weight Data Section

Starts at `weights_offset` (32-byte aligned). Contains all layer weight data
concatenated, with each layer starting on a 32-byte boundary.

### 4.1 Ternary layer layout (dtype="ternary2")

Within each ternary layer's data block:

```
+----------------------------+
| alpha (float32, 4 bytes)   |
+----------------------------+
| packed_size (uint32, 4 B)  |
+----------------------------+
| packed_weights (N bytes)   |  2 bits per weight, 4 per byte
+----------------------------+
| bitmap_size (uint32, 4 B)  |  0 if no bitmap
+----------------------------+
| bitmap_data (M bytes)      |  1 bit per weight (optional)
+----------------------------+
| bias_size (uint32, 4 B)    |  0 if no bias
+----------------------------+
| bias_data (K bytes)        |  float32 bias vector (optional)
+----------------------------+
| alignment padding          |  0x00 bytes to next 32-byte boundary
+----------------------------+
```

- `packed_size` = ceil(num_params / 4) bytes
- `bitmap_size` = ceil(num_params / 8) bytes (if present)
- `bias_size` = out_features * 4 bytes (float32)

### 4.2 Protected layer layout (dtype="float16")

```
+----------------------------+
| weight_size (uint32, 4 B)  |
+----------------------------+
| weight_data (N bytes)      |  FP16 weights, row-major
+----------------------------+
| bias_size (uint32, 4 B)    |  0 if no bias
+----------------------------+
| bias_data (K bytes)        |  float16 bias vector (optional)
+----------------------------+
| alignment padding          |  0x00 bytes to next 32-byte boundary
+----------------------------+
```

- `weight_size` = num_params * 2 bytes

### 4.3 Alignment rules

- Each layer's data starts on a 32-byte boundary (relative to file start).
- The `offset` field in the manifest is relative to the start of the weight data
  section.
- Padding bytes between layers are `0x00`.
- This ensures AVX2 (32-byte) and NEON (16-byte) aligned loads without memcpy.

## 5. Footer (16 bytes, fixed)

| Offset    | Size | Type   | Field         | Description |
|-----------|------|--------|---------------|-------------|
| EOF - 16  | 4    | uint32 | `crc32`       | CRC32 of entire weight data section |
| EOF - 12  | 8    | uint64 | `file_size`   | Total file size including footer |
| EOF - 4   | 4    | bytes  | `reverse_magic` | `b"NRET"` (0x4E524554) |

**Verification procedure:**
1. Read last 4 bytes; verify they equal `b"NRET"`.
2. Read `file_size` (bytes EOF-12 to EOF-4); verify it matches `os.path.getsize()`.
3. Read `crc32` (bytes EOF-16 to EOF-12).
4. Seek to `weights_offset`, read `weights_size` bytes, compute CRC32.
5. Compare computed CRC32 with stored `crc32`.

## 6. 2-Bit Encoding Table

| Value | Bits | Meaning |
|-------|------|---------|
| 0     | `00` | Zero weight (skip in computation) |
| +1    | `01` | Excitatory (add input to accumulator) |
| -1    | `10` | Inhibitory (subtract input from accumulator) |
| (reserved) | `11` | Reserved for future use |

**Packing order:** 4 weights per byte, LSB-first.
- Bits [1:0] = weight 0
- Bits [3:2] = weight 1
- Bits [5:4] = weight 2
- Bits [7:6] = weight 3

Example: weights `[+1, -1, 0, +1]` → bits `01_10_00_01` → `0b01001001` → byte `0x49`.

**Padding:** If `num_params` is not divisible by 4, the final byte is zero-padded
in the upper bits.

## 7. Sparsity Bitmap Format

- 1 bit per weight: `1` = non-zero, `0` = zero (skip).
- LSB-first within each byte.
- Padded to byte boundary (upper bits zero-filled).
- 32-byte aligned blocks enable SIMD-width skip decisions.

Example: weights `[+1, 0, -1, 0, 0, +1, 0, 0]` → bitmap `0b00100101` → byte `0x25`.

## 8. Byte Order

All multi-byte fields are **little-endian** throughout. This matches:
- x86/x86-64 native order
- ARM in default (LE) mode
- Most NPU DMA engines

## 9. Version Strategy

| Version | Description |
|---------|-------------|
| 1       | Legacy sequential format (model_loader/) — deprecated |
| 2       | Production format with manifest, alignment, CRC32 (this spec) |

Readers must reject files with `version > 2`. Writers must set `version = 2`.

## 10. Limitations (v2)

- No compression (2-bit is already 16x vs FP32; avoids DMA complexity).
- No encryption (out of scope for v2).
- Maximum file size: 2^64 - 1 bytes (uint64 offsets).
- Manifest is JSON — not suitable for >100K layers (use binary manifest in v3).
