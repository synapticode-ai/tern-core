"""
CoreML exporter for .tern-model → .mlpackage conversion.

Builds a complete MIL (Model Intermediate Language) computation graph
for LlamaForCausalLM and injects pre-quantised weights from the
.tern-model file.  The INT4 weights are byte-identical to CoreML's
constexpr_blockwise_shift_scale (iOS 18+); no re-quantisation occurs.

Part of tern-core v0.6.0 compression stack — Layer 4 (CoreML/ANE export).

Copyright (c) 2025 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

import gc
import io
import math
import struct
import time
from pathlib import Path
from typing import Optional

import numpy as np
import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb, types

from terncore.tern_model import TernModelReader
from terncore.sparse import unpack_ternary_weights
from terncore.coreml_export_helpers import (
    _validate_ternary2_alpha,
    _cast_fp16_retain_with_guards,
)
import torch


# ---------------------------------------------------------------------------
# Architecture constants (from config.json)
# ---------------------------------------------------------------------------

# Default architecture constants — Mistral-7B / Llama-3.1-70B.
# Overridden by --config or --arch-preset on the CLI.
HIDDEN_SIZE = 8192
INTERMEDIATE_SIZE = 28672
NUM_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS  # 128
GQA_GROUPS = NUM_HEADS // NUM_KV_HEADS  # 8
RMS_NORM_EPS = 1e-5
ROPE_THETA = 500000.0
VOCAB_SIZE = 128256
NUM_LAYERS = 80

# Architecture presets for common models.
ARCH_PRESETS = {
    "llama32-1b": {
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 64,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "vocab_size": 128256,
        "num_layers": 16,
        "tie_word_embeddings": True,
    },
    "llama32-3b": {
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_heads": 24,
        "num_kv_heads": 8,
        "head_dim": 128,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "vocab_size": 128256,
        "num_layers": 28,
        "tie_word_embeddings": True,
    },
    "mistral-7b": {
        "hidden_size": 8192,
        "intermediate_size": 28672,
        "num_heads": 64,
        "num_kv_heads": 8,
        "head_dim": 128,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "vocab_size": 128256,
        "num_layers": 80,
        "tie_word_embeddings": False,
    },
    "gemma3-4b": {
        "hidden_size": 2560,
        "intermediate_size": 10240,
        "num_heads": 8,
        "num_kv_heads": 4,
        "head_dim": 256,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "vocab_size": 262208,
        "num_layers": 34,
        "tie_word_embeddings": True,
        "activation": "gelu",
        "has_qk_norm": True,
        "has_pre_ffn_norm": True,
        "has_post_ffn_norm": True,
    },
    "gemma3-12b": {
        "hidden_size": 3840,
        "intermediate_size": 15360,
        "num_heads": 16,
        "num_kv_heads": 8,
        "head_dim": 256,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "vocab_size": 262208,
        "num_layers": 48,
        "tie_word_embeddings": True,
        "activation": "gelu",
        "has_qk_norm": True,
        "has_pre_ffn_norm": True,
        "has_post_ffn_norm": True,
    },
    "phi4-14b": {
        "hidden_size": 5120,
        "intermediate_size": 17920,
        "num_heads": 40,
        "num_kv_heads": 10,
        "head_dim": 128,
        "rms_norm_eps": 1e-5,
        "rope_theta": 250000.0,
        "vocab_size": 100352,
        "num_layers": 40,
        "tie_word_embeddings": False,
        "fused_qkv": True,
        "fused_gate_up": True,
    },
    "qwen25-7b": {
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_heads": 28,
        "num_kv_heads": 4,
        "head_dim": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "vocab_size": 152064,
        "num_layers": 28,
        "tie_word_embeddings": False,
    },
    "dsr1-7b": {
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_heads": 28,
        "num_kv_heads": 4,
        "head_dim": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "vocab_size": 152064,
        "num_layers": 28,
        "tie_word_embeddings": False,
    },
    "dsr1-14b": {
        "hidden_size": 5120,
        "intermediate_size": 13824,
        "num_heads": 40,
        "num_kv_heads": 8,
        "head_dim": 128,
        "rms_norm_eps": 1e-5,
        "rope_theta": 1000000.0,
        "vocab_size": 152064,
        "num_layers": 48,
        "tie_word_embeddings": False,
    },
}


# ---------------------------------------------------------------------------
# Weight loading from .tern-model
# ---------------------------------------------------------------------------

def _load_weight_for_coreml(reader: TernModelReader, name: str):
    """Load a weight from the .tern-model and prepare for CoreML injection.

    Returns:
        For int4_block32: (int4_data tagged as np_int4_dtype, scales_fp16)
        For ternary2: (int4_data tagged as np_int4_dtype, uniform_scales_fp16)
        For float16: fp16_numpy_array
    """
    entry = reader._get_manifest_entry(name)
    dtype = entry["dtype"]
    shape = entry["shape"]

    if dtype == "int4_block32":
        raw = reader.read_layer_data(name)
        buf = io.BytesIO(raw)
        block_size = struct.unpack("<I", buf.read(4))[0]
        packed_size = struct.unpack("<I", buf.read(4))[0]
        packed_bytes = buf.read(packed_size)
        scales_size = struct.unpack("<I", buf.read(4))[0]
        scales_bytes = buf.read(scales_size)

        packed = np.frombuffer(packed_bytes, dtype=np.uint8)
        low = (packed & 0x0F).astype(np.int8)
        high = ((packed >> 4) & 0x0F).astype(np.int8)
        low = np.where(low > 7, low - 16, low).astype(np.int8)
        high = np.where(high > 7, high - 16, high).astype(np.int8)
        q_flat = np.empty(len(packed) * 2, dtype=np.int8)
        q_flat[0::2] = low
        q_flat[1::2] = high

        scale_shape = entry["scale_shape"]
        padded_in = scale_shape[1] * block_size
        out_f = shape[0]
        int4_data = q_flat[:out_f * padded_in].reshape(out_f, padded_in)
        scales = np.frombuffer(scales_bytes, dtype=np.float16).reshape(scale_shape)

        return "int4", int4_data.astype(types.np_int4_dtype), scales, padded_in

    elif dtype == "ternary2":
        raw = reader.read_layer_data(name)
        buf = io.BytesIO(raw)
        alpha = struct.unpack("<f", buf.read(4))[0]
        packed_size = struct.unpack("<I", buf.read(4))[0]
        packed_bytes = buf.read(packed_size)

        packed_tensor = torch.frombuffer(bytearray(packed_bytes), dtype=torch.uint8)
        ternary = unpack_ternary_weights(packed_tensor, torch.Size(shape)).numpy()

        block_size = 32
        in_f = shape[1]
        padded_in = ((in_f + block_size - 1) // block_size) * block_size
        if padded_in > in_f:
            ternary = np.pad(ternary, ((0, 0), (0, padded_in - in_f)))
        n_blocks = padded_in // block_size
        int4_data = ternary.astype(np.int8).astype(types.np_int4_dtype)
        _validate_ternary2_alpha(alpha, name)
        scales = np.full((shape[0], n_blocks), alpha, dtype=np.float16)

        return "int4", int4_data, scales, padded_in

    elif dtype == "float16":
        tensors = reader.reconstruct_layer(name)
        weight_fp32 = tensors["weight"].numpy()
        weight_fp16 = _cast_fp16_retain_with_guards(weight_fp32, name)
        return "fp16", weight_fp16

    else:
        raise ValueError(f"Unknown dtype {dtype} for {name}")


def _inject_weight(reader, name):
    """Load weight and inject as MIL constant or constexpr."""
    result = _load_weight_for_coreml(reader, name)
    if result[0] == "int4":
        _, data, scales, _ = result
        return mb.constexpr_blockwise_shift_scale(data=data, scale=scales)
    else:
        _, fp16_data = result
        return mb.const(val=fp16_data)


def _inject_split_weight(reader, name, split_sizes):
    """Load a fused weight, split along axis 0, and inject each part.

    Used for fused QKV (split into Q, K, V) and fused gate_up (split into
    gate, up). Returns a list of MIL variables, one per split.
    """
    result = _load_weight_for_coreml(reader, name)
    parts = []
    if result[0] == "int4":
        _, data, scales, padded_in = result
        # Split data [out_total, padded_in] and scales [out_total, n_blocks]
        offset = 0
        for sz in split_sizes:
            d = data[offset:offset + sz]
            s = scales[offset:offset + sz]
            parts.append(mb.constexpr_blockwise_shift_scale(data=d, scale=s))
            offset += sz
    else:
        _, fp16_data = result
        offset = 0
        for sz in split_sizes:
            parts.append(mb.const(val=fp16_data[offset:offset + sz]))
            offset += sz
    return parts


# ---------------------------------------------------------------------------
# MIL building blocks
# ---------------------------------------------------------------------------

def _rms_norm(x, weight_var, eps=RMS_NORM_EPS):
    """RMSNorm: x * rsqrt(mean(x^2) + eps) * weight."""
    sq = mb.mul(x=x, y=x)
    mean_sq = mb.reduce_mean(x=sq, axes=[-1], keep_dims=True)
    eps_const = mb.const(val=np.float16(eps))
    normed = mb.mul(x=x, y=mb.rsqrt(x=mb.add(x=mean_sq, y=eps_const)))
    return mb.mul(x=normed, y=weight_var)


def _precompute_rope_freqs(seq_len, head_dim=HEAD_DIM, theta=ROPE_THETA):
    """Precompute cos/sin tables for RoPE."""
    freqs = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    t = np.arange(seq_len, dtype=np.float32)
    freqs_outer = np.outer(t, freqs)  # [seq_len, head_dim/2]
    cos_table = np.cos(freqs_outer).astype(np.float16)
    sin_table = np.sin(freqs_outer).astype(np.float16)
    return cos_table, sin_table  # [seq_len, head_dim/2]


def _apply_rope(q, k, cos_table, sin_table):
    """Apply rotary positional embeddings to Q and K.

    q, k: [batch, heads, seq, head_dim]
    cos_table, sin_table: MIL vars [1, 1, seq, head_dim/2]
    """
    def _rotate(x, cos_t, sin_t):
        # Split x into two halves along last dim
        x1 = mb.slice_by_index(x=x, begin=[0, 0, 0, 0],
                               end=[0, 0, 0, HEAD_DIM // 2],
                               end_mask=[True, True, True, False])
        x2 = mb.slice_by_index(x=x, begin=[0, 0, 0, HEAD_DIM // 2],
                               end=[0, 0, 0, 0],
                               end_mask=[True, True, True, True])
        # x1 * cos - x2 * sin
        r1 = mb.sub(x=mb.mul(x=x1, y=cos_t), y=mb.mul(x=x2, y=sin_t))
        # x2 * cos + x1 * sin
        r2 = mb.add(x=mb.mul(x=x2, y=cos_t), y=mb.mul(x=x1, y=sin_t))
        return mb.concat(values=[r1, r2], axis=-1)

    q_rot = _rotate(q, cos_table, sin_table)
    k_rot = _rotate(k, cos_table, sin_table)
    return q_rot, k_rot


def _gqa_attention(hidden, q_w, k_w, v_w, o_w, cos_table, sin_table):
    """Grouped Query Attention with RoPE.

    hidden: [batch, seq, hidden_size]
    """
    batch_seq_shape = [1, -1, HIDDEN_SIZE]

    # Project Q, K, V
    q = mb.linear(x=hidden, weight=q_w)  # [b, s, 8192]
    k = mb.linear(x=hidden, weight=k_w)  # [b, s, 1024]
    v = mb.linear(x=hidden, weight=v_w)  # [b, s, 1024]

    # Reshape to [batch, seq, heads, head_dim] then transpose to [batch, heads, seq, head_dim]
    q = mb.reshape(x=q, shape=[1, -1, NUM_HEADS, HEAD_DIM])
    q = mb.transpose(x=q, perm=[0, 2, 1, 3])
    k = mb.reshape(x=k, shape=[1, -1, NUM_KV_HEADS, HEAD_DIM])
    k = mb.transpose(x=k, perm=[0, 2, 1, 3])
    v = mb.reshape(x=v, shape=[1, -1, NUM_KV_HEADS, HEAD_DIM])
    v = mb.transpose(x=v, perm=[0, 2, 1, 3])

    # Apply RoPE
    q, k = _apply_rope(q, k, cos_table, sin_table)

    # GQA: expand K, V from 8 heads to 64 heads by repeating
    # k: [1, 8, seq, 128] → [1, 64, seq, 128]
    k = mb.tile(x=k, reps=[1, GQA_GROUPS, 1, 1])
    v = mb.tile(x=v, reps=[1, GQA_GROUPS, 1, 1])

    # Attention: Q @ K^T / sqrt(head_dim)
    scale = mb.const(val=np.float16(1.0 / math.sqrt(HEAD_DIM)))
    k_t = mb.transpose(x=k, perm=[0, 1, 3, 2])
    attn = mb.matmul(x=q, y=k_t)
    attn = mb.mul(x=attn, y=scale)

    # Causal mask would go here for generation; for export validation
    # we use softmax without mask (full attention)
    attn = mb.softmax(x=attn, axis=-1)

    # Attention @ V
    out = mb.matmul(x=attn, y=v)  # [1, 64, seq, 128]

    # Reshape back: [1, 64, seq, 128] → [1, seq, 8192]
    out = mb.transpose(x=out, perm=[0, 2, 1, 3])
    out = mb.reshape(x=out, shape=[1, -1, HIDDEN_SIZE])

    # Output projection
    out = mb.linear(x=out, weight=o_w)
    return out


def _mlp(hidden, gate_w, up_w, down_w):
    """SwiGLU MLP: down_proj(silu(gate_proj(x)) * up_proj(x))."""
    gate = mb.linear(x=hidden, weight=gate_w)
    up = mb.linear(x=hidden, weight=up_w)
    activated = mb.mul(x=mb.silu(x=gate), y=up)
    return mb.linear(x=activated, weight=down_w)


# ---------------------------------------------------------------------------
# Full model builder
# ---------------------------------------------------------------------------

def build_llama_coreml(
    tern_model_path: str,
    output_path: str,
    seq_len: int = 512,
    num_blocks: Optional[int] = None,
    arch_preset: Optional[str] = None,
    verbose: bool = True,
):
    """Build a CoreML .mlpackage from a .tern-model file.

    Args:
        tern_model_path: Path to the .tern-model file.
        output_path:     Path for the .mlpackage output.
        seq_len:         Fixed sequence length for the model.
        num_blocks:      Number of transformer blocks (default: from config).
        arch_preset:     Architecture preset name (e.g. "llama32-1b").
        verbose:         Print progress.
    """
    # Resolve architecture config
    if arch_preset and arch_preset in ARCH_PRESETS:
        cfg = ARCH_PRESETS[arch_preset]
    else:
        cfg = ARCH_PRESETS.get("mistral-7b")  # default

    hidden_size = cfg["hidden_size"]
    num_heads = cfg["num_heads"]
    num_kv_heads = cfg["num_kv_heads"]
    head_dim = cfg["head_dim"]
    gqa_groups = num_heads // num_kv_heads
    rms_norm_eps = cfg["rms_norm_eps"]
    rope_theta = cfg["rope_theta"]
    tie_embeddings = cfg.get("tie_word_embeddings", False)
    use_gelu = cfg.get("activation") == "gelu"
    has_qk_norm = cfg.get("has_qk_norm", False)
    has_pre_ffn_norm = cfg.get("has_pre_ffn_norm", False)
    has_post_ffn_norm = cfg.get("has_post_ffn_norm", False)
    fused_qkv = cfg.get("fused_qkv", False)
    fused_gate_up = cfg.get("fused_gate_up", False)

    t0 = time.time()
    reader = TernModelReader(tern_model_path)
    n_blocks = num_blocks or cfg["num_layers"]

    if verbose:
        print(f"Building CoreML model from {tern_model_path}", flush=True)
        print(f"  Preset: {arch_preset or 'default'}", flush=True)
        print(f"  Blocks: {n_blocks}, Seq len: {seq_len}, "
              f"Hidden: {hidden_size}, Heads: {num_heads}/{num_kv_heads}, "
              f"Head dim: {head_dim}", flush=True)

    # Precompute RoPE tables
    cos_table, sin_table = _precompute_rope_freqs(
        seq_len, head_dim=head_dim, theta=rope_theta
    )
    cos_np = cos_table.reshape(1, 1, seq_len, head_dim // 2)
    sin_np = sin_table.reshape(1, 1, seq_len, head_dim // 2)

    # Capture config in closure for the MIL builder functions
    _hidden = hidden_size
    _nheads = num_heads
    _nkv = num_kv_heads
    _hdim = head_dim
    _gqa = gqa_groups
    _eps = rms_norm_eps

    def _rms_norm_cfg(x, weight_var):
        sq = mb.mul(x=x, y=x)
        mean_sq = mb.reduce_mean(x=sq, axes=[-1], keep_dims=True)
        eps_const = mb.const(val=np.float16(_eps))
        normed = mb.mul(x=x, y=mb.rsqrt(x=mb.add(x=mean_sq, y=eps_const)))
        return mb.mul(x=normed, y=weight_var)

    def _apply_rope_cfg(q, k, cos_t, sin_t):
        def _rotate(x, ct, st):
            x1 = mb.slice_by_index(x=x, begin=[0, 0, 0, 0],
                                   end=[0, 0, 0, _hdim // 2],
                                   end_mask=[True, True, True, False])
            x2 = mb.slice_by_index(x=x, begin=[0, 0, 0, _hdim // 2],
                                   end=[0, 0, 0, 0],
                                   end_mask=[True, True, True, True])
            r1 = mb.sub(x=mb.mul(x=x1, y=ct), y=mb.mul(x=x2, y=st))
            r2 = mb.add(x=mb.mul(x=x2, y=ct), y=mb.mul(x=x1, y=st))
            return mb.concat(values=[r1, r2], axis=-1)
        return _rotate(q, cos_t, sin_t), _rotate(k, cos_t, sin_t)

    def _gqa_cfg(hidden, q_w, k_w, v_w, o_w, cos_t, sin_t,
                 q_norm_w=None, k_norm_w=None):
        q = mb.linear(x=hidden, weight=q_w)
        k = mb.linear(x=hidden, weight=k_w)
        v = mb.linear(x=hidden, weight=v_w)
        q = mb.reshape(x=q, shape=[1, -1, _nheads, _hdim])
        q = mb.transpose(x=q, perm=[0, 2, 1, 3])
        k = mb.reshape(x=k, shape=[1, -1, _nkv, _hdim])
        k = mb.transpose(x=k, perm=[0, 2, 1, 3])
        v = mb.reshape(x=v, shape=[1, -1, _nkv, _hdim])
        v = mb.transpose(x=v, perm=[0, 2, 1, 3])
        # QK norm (Gemma 3): RMSNorm on Q and K per-head before RoPE
        if q_norm_w is not None:
            q = _rms_norm_cfg(q, q_norm_w)
        if k_norm_w is not None:
            k = _rms_norm_cfg(k, k_norm_w)
        q, k = _apply_rope_cfg(q, k, cos_t, sin_t)
        if _gqa > 1:
            k = mb.tile(x=k, reps=[1, _gqa, 1, 1])
            v = mb.tile(x=v, reps=[1, _gqa, 1, 1])
        scale = mb.const(val=np.float16(1.0 / math.sqrt(_hdim)))
        k_t = mb.transpose(x=k, perm=[0, 1, 3, 2])
        attn = mb.matmul(x=q, y=k_t)
        attn = mb.mul(x=attn, y=scale)
        attn = mb.softmax(x=attn, axis=-1)
        out = mb.matmul(x=attn, y=v)
        out = mb.transpose(x=out, perm=[0, 2, 1, 3])
        out = mb.reshape(x=out, shape=[1, -1, _nheads * _hdim])
        return mb.linear(x=out, weight=o_w)

    def _mlp_cfg(hidden, gate_w, up_w, down_w):
        """MLP with configurable activation (SiLU for Llama, GELU for Gemma)."""
        gate = mb.linear(x=hidden, weight=gate_w)
        up = mb.linear(x=hidden, weight=up_w)
        if use_gelu:
            activated = mb.mul(x=mb.gelu(x=gate), y=up)
        else:
            activated = mb.mul(x=mb.silu(x=gate), y=up)
        return mb.linear(x=activated, weight=down_w)

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, seq_len), dtype=types.int32)],
        opset_version=ct.target.iOS18,
    )
    def prog(input_ids):
        if verbose:
            print("  Loading embed_tokens...", flush=True)

        embed_w = _inject_weight(reader, "model.embed_tokens.weight")
        hidden = mb.gather(x=embed_w, indices=input_ids, axis=0)

        cos_var = mb.const(val=cos_np)
        sin_var = mb.const(val=sin_np)

        for i in range(n_blocks):
            if verbose:
                print(f"  Block {i}/{n_blocks}...", flush=True)

            prefix = f"model.layers.{i}"
            ln1_w = _inject_weight(reader, f"{prefix}.input_layernorm.weight")

            if fused_qkv:
                q_size = _nheads * _hdim
                kv_size = _nkv * _hdim
                q_w, k_w, v_w = _inject_split_weight(
                    reader, f"{prefix}.self_attn.qkv_proj.weight",
                    [q_size, kv_size, kv_size])
            else:
                q_w = _inject_weight(reader, f"{prefix}.self_attn.q_proj.weight")
                k_w = _inject_weight(reader, f"{prefix}.self_attn.k_proj.weight")
                v_w = _inject_weight(reader, f"{prefix}.self_attn.v_proj.weight")
            o_w = _inject_weight(reader, f"{prefix}.self_attn.o_proj.weight")

            ln2_w = _inject_weight(reader, f"{prefix}.post_attention_layernorm.weight")

            if fused_gate_up:
                inter = cfg["intermediate_size"]
                gate_w, up_w = _inject_split_weight(
                    reader, f"{prefix}.mlp.gate_up_proj.weight",
                    [inter, inter])
            else:
                gate_w = _inject_weight(reader, f"{prefix}.mlp.gate_proj.weight")
                up_w = _inject_weight(reader, f"{prefix}.mlp.up_proj.weight")
            down_w = _inject_weight(reader, f"{prefix}.mlp.down_proj.weight")

            # Optional QK norm (Gemma 3)
            q_norm_w = None
            k_norm_w = None
            if has_qk_norm:
                q_norm_w = _inject_weight(reader, f"{prefix}.self_attn.q_norm.weight")
                k_norm_w = _inject_weight(reader, f"{prefix}.self_attn.k_norm.weight")

            normed = _rms_norm_cfg(hidden, ln1_w)
            attn_out = _gqa_cfg(normed, q_w, k_w, v_w, o_w,
                                cos_var, sin_var,
                                q_norm_w=q_norm_w, k_norm_w=k_norm_w)
            hidden = mb.add(x=hidden, y=attn_out)

            # MLP with optional extra norms (Gemma 3)
            if has_pre_ffn_norm:
                pre_ffn_w = _inject_weight(reader, f"{prefix}.pre_feedforward_layernorm.weight")
                normed2 = _rms_norm_cfg(hidden, pre_ffn_w)
            else:
                normed2 = _rms_norm_cfg(hidden, ln2_w)
            mlp_out = _mlp_cfg(normed2, gate_w, up_w, down_w)
            if has_post_ffn_norm:
                post_ffn_w = _inject_weight(reader, f"{prefix}.post_feedforward_layernorm.weight")
                mlp_out = _rms_norm_cfg(mlp_out, post_ffn_w)
            hidden = mb.add(x=hidden, y=mlp_out)

            gc.collect()

        if verbose:
            print("  Loading final norm + lm_head...", flush=True)
        final_ln_w = _inject_weight(reader, "model.norm.weight")
        hidden = _rms_norm_cfg(hidden, final_ln_w)

        if tie_embeddings:
            # Reuse embed_tokens.weight transposed as LM head
            lm_head_w = _inject_weight(reader, "model.embed_tokens.weight")
        else:
            lm_head_w = _inject_weight(reader, "lm_head.weight")
        logits = mb.linear(x=hidden, weight=lm_head_w, name="logits")

        return logits

    if verbose:
        print("  Converting MIL → mlprogram...", flush=True)
    mlmodel = ct.convert(
        prog,
        source="milinternal",
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.ALL,
    )

    if verbose:
        print(f"  Saving to {output_path}...", flush=True)
    mlmodel.save(output_path)

    elapsed = time.time() - t0
    if verbose:
        print(f"  Done in {elapsed:.1f}s", flush=True)

    return mlmodel


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export .tern-model to CoreML .mlpackage"
    )
    parser.add_argument("--model", required=True,
                        help="Path to .tern-model file")
    parser.add_argument("--output", required=True,
                        help="Output .mlpackage path")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Sequence length (default: 512)")
    parser.add_argument("--blocks", type=int, default=None,
                        help="Number of blocks (default: from config)")
    parser.add_argument("--arch-preset", default=None,
                        choices=list(ARCH_PRESETS.keys()),
                        help="Architecture preset")
    args = parser.parse_args()

    build_llama_coreml(
        args.model, args.output,
        seq_len=args.seq_len,
        num_blocks=args.blocks,
        arch_preset=args.arch_preset,
    )
