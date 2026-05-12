"""
Interactive ternary inference demo.

Loads a HuggingFace model, runs a perplexity-gated auto-scan to find the
maximum safe ternary conversion, applies it, and generates text.  Scan
results are cached to ~/.terncore/model_cache.json so repeat runs are
instant.

Usage:
    python tools/tern_infer.py --prompt "The future of computing lies in"
    python tools/tern_infer.py --interactive
    python tools/tern_infer.py --prompt "Once upon a time" --max-tokens 100

Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import torch

# Ensure tern-core is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from terncore.mixed_precision import MixedPrecisionConverter

DEFAULT_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_MAX_TOKENS = 50


def load_model(model_id: str = DEFAULT_MODEL_ID):
    """Load model and apply perplexity-gated ternary conversion.

    Runs an automatic PPL scan (cached after the first run) to find the
    maximum number of layers that can be safely converted to ternary
    within a +20% PPL budget, then converts only those layers.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_id}...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_time = time.perf_counter() - t0
    print(f"  Loaded in {load_time:.1f}s")

    print("Converting to mixed-precision ternary (auto-scan)...")
    t0 = time.perf_counter()
    converter = MixedPrecisionConverter(threshold=0.7)
    report = converter.convert(model, model_id=model_id)
    conv_time = time.perf_counter() - t0
    print(f"  Converted {report.converted_layers} layers in {conv_time:.1f}s")
    print(f"  Protected: {report.skipped_layers}, "
          f"Compression: {report.compression_ratio:.2f}x")

    model.eval()
    return model, tokenizer


def generate_streaming(
    model, tokenizer, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS,
    use_kv_cache: bool = True,
) -> tuple[str, float, int]:
    """Generate text token by token with streaming output.

    Args:
        use_kv_cache: If True, use HuggingFace past_key_values KV cache
            to avoid full recomputation each token.

    Returns (full_text, tokens_per_second, num_tokens).
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generated_ids = input_ids.clone()

    print(f"\n{prompt}", end="", flush=True)

    t_start = time.perf_counter()
    tokens_generated = 0
    past_key_values = None

    with torch.no_grad():
        for _ in range(max_tokens):
            if use_kv_cache and past_key_values is not None:
                # Only feed the last token; reuse cached KV states
                outputs = model(
                    generated_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            else:
                # First pass (prefill) or no-cache mode
                outputs = model(
                    generated_ids,
                    use_cache=use_kv_cache,
                )

            if use_kv_cache:
                past_key_values = outputs.past_key_values

            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)

            # Stop on EOS
            if next_token_id.item() == tokenizer.eos_token_id:
                break

            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            tokens_generated += 1

            # Decode and print just the new token
            new_token = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            print(new_token, end="", flush=True)

    t_elapsed = time.perf_counter() - t_start
    tps = tokens_generated / t_elapsed if t_elapsed > 0 else 0

    print()  # newline
    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return full_text, tps, tokens_generated


def _get_rss_mb() -> float:
    """Return current process RSS in MiB."""
    import resource
    # maxrss is in bytes on macOS
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def _extract_kv_pairs(past_key_values):
    """Extract (key, value) tensor pairs from DynamicCache or legacy tuple."""
    kv_pairs = []
    for item in past_key_values:
        if isinstance(item, (tuple, list)):
            kv_pairs.append((item[0], item[1]))
        else:
            kv_pairs.append((item.keys, item.values))
    return kv_pairs


class IncrementalTQCompressor:
    """Incrementally compresses KV cache vectors via TurboQuant.

    Per-(layer, head) rotation + QJL state is built lazily on first
    encode, keyed by the layer's observed head_dim. This supports
    heterogeneous attention architectures (e.g. Gemma 4's alternating
    sliding_attention head_dim=256 / full_attention head_dim=512
    layers) where a single uniform TurboQuantConfig would fail at the
    first off-shape layer.
    """

    def __init__(self, n_layers=None, n_heads=None, head_dim=None, device="cpu"):
        # n_layers/n_heads/head_dim retained as legacy positional hints
        # for back-compat with prior callers; ignored — state is built
        # lazily from the observed cache shapes at first encode.
        sys.path.insert(0, "/Users/syn/synapticode/venv/src/turboquant")
        self.device = torch.device(device)

        # Per-head_dim TurboQuantConfig (codebook + d_padded etc.)
        self._configs: dict[int, object] = {}
        # rotations[head_dim][(layer_idx, head_idx)] = rotation
        self._rotations: dict[int, dict[tuple[int, int], object]] = {}
        self._qjl: dict[int, dict[tuple[int, int], torch.Tensor]] = {}
        self._mixed: dict[int, dict[tuple[int, int], object]] = {}

        # compressed[(layer_idx, head_idx)] = list of (k_c, v_c)
        self.compressed: dict[tuple[int, int], list] = {}
        # head_dim_layers[head_dim] = set of layer_idx seen for this dim
        self.head_dim_layers: dict[int, set[int]] = {}
        self.seq_len = 0

    def _ensure_config(self, head_dim: int) -> None:
        if head_dim in self._configs:
            return
        from src.cache import TurboQuantConfig
        self._configs[head_dim] = TurboQuantConfig(
            d=head_dim, b_mse=3,
            device=self.device, mixed_precision=True,
        )
        self._rotations[head_dim] = {}
        self._qjl[head_dim] = {}
        self._mixed[head_dim] = {}
        self.head_dim_layers[head_dim] = set()

    def _ensure_state(self, head_dim: int, l: int, h: int) -> None:
        self._ensure_config(head_dim)
        key = (l, h)
        if key not in self._rotations[head_dim]:
            cfg = self._configs[head_dim]
            self._rotations[head_dim][key] = cfg.make_rotation(l, h)
            self._qjl[head_dim][key] = cfg.make_qjl_matrix(l, h)
            self._mixed[head_dim][key] = cfg.get_mixed_config(l, h)
            self.head_dim_layers[head_dim].add(l)

    def append(self, past_key_values, *, encode_from: int | None = None):
        """Encode positions [encode_from:] from the live KV cache.

        If encode_from is None, encodes from self.seq_len (i.e. only new
        positions since the last call). Per-layer head_dim is read from
        each layer's K tensor; rotation state is materialised on demand
        for any (head_dim, layer, head) combination seen for the first
        time.
        """
        from src.cache import turboquant_encode_internal

        kv_pairs = _extract_kv_pairs(past_key_values)
        total_seq = kv_pairs[0][0].shape[2]
        start = encode_from if encode_from is not None else self.seq_len
        if start >= total_seq:
            return  # nothing new

        for l, (keys, values) in enumerate(kv_pairs):
            # (1, n_kv_heads, seq_len, head_dim) — per-layer head_dim
            head_dim = int(keys.shape[-1])
            n_kv_heads = int(keys.shape[1])
            self._ensure_config(head_dim)
            cfg = self._configs[head_dim]
            for h in range(n_kv_heads):
                self._ensure_state(head_dim, l, h)
                k_new = keys[0, h, start:total_seq].float()
                v_new = values[0, h, start:total_seq].float()

                rotation = self._rotations[head_dim][(l, h)]
                S = self._qjl[head_dim][(l, h)]
                mixed = self._mixed[head_dim][(l, h)]

                k_c = turboquant_encode_internal(
                    k_new, cfg.codebook, rotation, S, mixed=mixed,
                )
                v_c = turboquant_encode_internal(
                    v_new, cfg.codebook, rotation, S, mixed=mixed,
                )
                self.compressed.setdefault((l, h), []).append((k_c, v_c))

        self.seq_len = total_seq

    def summary(self) -> dict:
        """Per-head_dim layer counts; for row-output breakdown."""
        return {
            int(d): {
                "n_layers": len(layers),
                "layer_idxs": sorted(int(x) for x in layers),
            }
            for d, layers in self.head_dim_layers.items()
        }


def generate_streaming_turboquant(
    model, tokenizer, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[str, float, int]:
    """Generate with KV cache + incremental TurboQuant compression.

    On the prefill pass the full prompt's KV vectors are compressed in one
    batch.  Each subsequent decode step encodes only the single new KV pair,
    carrying the previously-compressed cache forward without re-encoding.

    Returns (full_text, tokens_per_second, num_tokens).
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generated_ids = input_ids.clone()

    print(f"\n{prompt}", end="", flush=True)

    t_start = time.perf_counter()
    tokens_generated = 0
    past_key_values = None
    compressor = None

    with torch.no_grad():
        for step in range(max_tokens):
            if past_key_values is not None:
                outputs = model(
                    generated_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            else:
                outputs = model(generated_ids, use_cache=True)

            past_key_values = outputs.past_key_values

            # Lazy-init compressor on first pass (now we know dimensions)
            if compressor is None:
                kv0 = _extract_kv_pairs(past_key_values)
                compressor = IncrementalTQCompressor(
                    n_layers=len(kv0),
                    n_heads=kv0[0][0].shape[1],
                    head_dim=kv0[0][0].shape[3],
                )

            # Encode only the new position(s) since last call
            compressor.append(past_key_values)

            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)

            if next_token_id.item() == tokenizer.eos_token_id:
                break

            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            tokens_generated += 1

            new_token = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            print(new_token, end="", flush=True)

    t_elapsed = time.perf_counter() - t_start
    tps = tokens_generated / t_elapsed if t_elapsed > 0 else 0

    print()
    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return full_text, tps, tokens_generated


def interactive_mode(model, tokenizer, max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
    """Interactive prompt loop."""
    print()
    print("=" * 60)
    print("  Ternary Inference Demo (auto-scan)")
    print("  Type a prompt and press Enter. Type 'quit' to exit.")
    print("=" * 60)
    print()

    while True:
        try:
            prompt = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        _, tps, n_tokens = generate_streaming(model, tokenizer, prompt, max_tokens)
        print(f"  [{n_tokens} tokens, {tps:.1f} tok/s]")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ternary inference demo")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="Model ID")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)

    if args.interactive:
        interactive_mode(model, tokenizer, args.max_tokens)
    elif args.prompt:
        full_text, tps, n_tokens = generate_streaming(
            model, tokenizer, args.prompt, args.max_tokens,
        )
        print(f"\n  [{n_tokens} tokens, {tps:.1f} tok/s]")
    else:
        # Default demo prompt
        full_text, tps, n_tokens = generate_streaming(
            model, tokenizer,
            "The future of computing lies in",
            args.max_tokens,
        )
        print(f"\n  [{n_tokens} tokens, {tps:.1f} tok/s]")


if __name__ == "__main__":
    main()
