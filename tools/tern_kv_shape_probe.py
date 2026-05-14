"""
PATH-1 + PATH-3 combined probe — Gemma 4 KV cache shape disambiguation
and TurboQuant rotation smoke-test at the empirically observed width.

Loads Gemma 4 E4B in FP16, runs one prefill, walks past_key_values
exhaustively (cache type, per-layer item type, full tensor shapes,
dtype, GQA expansion check), then attempts a single TurboQuant
rotation pass at the *actually observed* per-head width to confirm
whether re-initing with the correct d unblocks the encode path.

Exit code 0 = probe complete (rotation smoke-test may still report failure
in the JSON output). Exit code != 0 = setup failure (model load, etc.).
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


def shape_or_none(obj):
    if isinstance(obj, torch.Tensor):
        return list(obj.shape)
    return None


def dtype_or_none(obj):
    if isinstance(obj, torch.Tensor):
        return str(obj.dtype)
    return None


def main() -> int:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/gemma4_kv_shape.json")

    t0 = time.perf_counter()
    print("[probe] loading transformers + Gemma 4 config...")
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

    model_id = "google/gemma-4-E4B-it"
    cfg = AutoConfig.from_pretrained(model_id)
    text_cfg = cfg.text_config
    text_config_summary = {
        "hidden_size": text_cfg.hidden_size,
        "num_attention_heads": text_cfg.num_attention_heads,
        "num_key_value_heads": text_cfg.num_key_value_heads,
        "head_dim": text_cfg.head_dim,
        "num_hidden_layers": text_cfg.num_hidden_layers,
        "model_type": text_cfg.model_type,
        "sliding_window": getattr(text_cfg, "sliding_window", None),
        "rope_theta": getattr(text_cfg, "rope_theta", None),
        "rope_local_base_freq": getattr(text_cfg, "rope_local_base_freq", None),
        "layer_types": getattr(text_cfg, "layer_types", None),
    }
    print(f"[probe] text_config: {text_config_summary}")

    print(f"[probe] loading model FP16...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16)
    model.eval()
    print(f"[probe] model class: {type(model).__name__}")
    print(f"[probe] load took {time.perf_counter()-t0:.1f}s")

    prompt = "The future of computing"
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    print(f"[probe] prefill input shape: {tuple(ids.shape)}")

    t_prefill = time.perf_counter()
    with torch.no_grad():
        out = model(ids, use_cache=True)
    print(f"[probe] prefill took {time.perf_counter()-t_prefill:.1f}s")

    pkv = out.past_key_values
    pkv_type = type(pkv).__name__
    print(f"[probe] past_key_values type: {pkv_type}")

    # Walk
    layers_info = []
    try:
        n_layers = len(pkv)
    except TypeError:
        n_layers = None
    print(f"[probe] len(pkv) = {n_layers}")

    # Sample first 3 + last 1 + any layer that differs from layer 0
    sample_indices = list(range(min(3, n_layers or 0)))
    if n_layers and n_layers - 1 not in sample_indices:
        sample_indices.append(n_layers - 1)

    for li, item in enumerate(pkv):
        item_type = type(item).__name__
        if isinstance(item, (tuple, list)):
            k, v = item[0], item[1]
            info = {
                "layer": li,
                "item_type": item_type,
                "k_shape": shape_or_none(k),
                "k_dtype": dtype_or_none(k),
                "v_shape": shape_or_none(v),
                "v_dtype": dtype_or_none(v),
            }
        else:
            # HF Cache object — try attribute access
            k = getattr(item, "keys", None)
            v = getattr(item, "values", None)
            info = {
                "layer": li,
                "item_type": item_type,
                "k_shape": shape_or_none(k),
                "k_dtype": dtype_or_none(k),
                "v_shape": shape_or_none(v),
                "v_dtype": dtype_or_none(v),
                "attrs": [a for a in dir(item) if not a.startswith("_")][:30],
            }
        layers_info.append(info)
        if li in sample_indices:
            print(f"[probe] layer {li}: {info}")

    # Detect heterogeneity
    shapes_seen = {tuple(li["k_shape"]) if li["k_shape"] else None for li in layers_info}
    print(f"[probe] distinct K shapes across {len(layers_info)} layers: {len(shapes_seen)}")
    for sh in shapes_seen:
        count = sum(1 for li in layers_info if tuple(li["k_shape"] or ()) == sh)
        print(f"[probe]   shape {sh}: {count} layers")

    # Reference: also inspect via _extract_kv_pairs (the path the
    # IncrementalTQCompressor takes)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tern_infer import _extract_kv_pairs
    kv_pairs = _extract_kv_pairs(pkv)
    print(f"[probe] _extract_kv_pairs returned {len(kv_pairs)} tuples")
    extract_summary = []
    for li, (k, v) in enumerate(kv_pairs[:3] + ([kv_pairs[-1]] if len(kv_pairs) > 3 else [])):
        extract_summary.append({
            "layer": li if li < 3 else len(kv_pairs) - 1,
            "k_shape": shape_or_none(k),
            "v_shape": shape_or_none(v),
        })
    print(f"[probe] _extract_kv_pairs sample shapes: {extract_summary}")

    # Per-head slice (what IncrementalTQCompressor.append does internally)
    k0 = kv_pairs[0][0]
    head_h0_slice = k0[0, 0, :].float()  # (seq_len, head_dim_actual)
    print(f"[probe] layer-0 head-0 slice shape (n_tokens, head_dim_actual): {tuple(head_h0_slice.shape)}")
    head_dim_actual_layer0 = head_h0_slice.shape[-1]

    # Try the same on the largest-shape layer (if heterogeneous)
    largest_layer_idx = max(range(len(kv_pairs)), key=lambda i: kv_pairs[i][0].shape[-1])
    k_largest = kv_pairs[largest_layer_idx][0]
    head_largest = k_largest[0, 0, :].float()
    head_dim_actual_largest = head_largest.shape[-1]
    print(f"[probe] largest-head-dim layer={largest_layer_idx} slice shape: {tuple(head_largest.shape)}")

    # PATH-3 smoke test: build RandomHadamardRotation at the
    # *empirically observed* d, attempt a forward.
    smoke = {}
    try:
        sys.path.insert(0, "/Users/syn/synapticode/venv/src/turboquant")
        from src.cache import RandomHadamardRotation, _next_power_of_two
        d_actual = int(head_dim_actual_largest)
        d_padded = _next_power_of_two(d_actual)
        print(f"[probe] PATH-3 smoke: d_actual={d_actual}, d_padded={d_padded}")
        rotation = RandomHadamardRotation(d=d_padded, seed=42, device=torch.device("cpu"))
        # pad if needed
        x = head_largest[:1].clone()  # (1, d_actual)
        if d_padded != d_actual:
            x = torch.nn.functional.pad(x, (0, d_padded - d_actual))
        y = rotation.forward(x)
        smoke = {
            "ok": True,
            "d_actual": d_actual,
            "d_padded": d_padded,
            "input_shape": list(x.shape),
            "output_shape": list(y.shape),
        }
        print(f"[probe] PATH-3 smoke OK: in {tuple(x.shape)} -> out {tuple(y.shape)}")
    except Exception as e:
        smoke = {
            "ok": False,
            "error_type": type(e).__name__,
            "error": str(e),
        }
        print(f"[probe] PATH-3 smoke FAILED: {type(e).__name__}: {e}")

    iso = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record = {
        "probe": "Gemma 4 E4B KV shape diagnostic + TurboQuant rotation smoke test",
        "date_utc": iso,
        "model_id": model_id,
        "model_class": type(model).__name__,
        "text_config_summary": text_config_summary,
        "past_key_values_type": pkv_type,
        "n_layers": n_layers,
        "prefill_input_shape": list(ids.shape),
        "layers_info": layers_info,
        "shapes_distinct": len(shapes_seen),
        "extract_pairs_sample": extract_summary,
        "head_dim_actual_layer0": head_dim_actual_layer0,
        "head_dim_actual_largest": head_dim_actual_largest,
        "largest_layer_idx": largest_layer_idx,
        "path3_smoke_test": smoke,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    print(f"[probe] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
