"""TN-003 TurboQuant baseline measurement orchestration.

Loads a per-expert-sliced ``.tern-model`` artefact via
``load_packed_model``, runs a TurboQuant-aware generation loop with
the compressor instance captured for footprint measurement, computes
standalone WikiText-2 perplexity (caller-controlled), and reports a
single JSON record per measurement run with field names that
disambiguate "compression overhead + footprint" from "quality impact".

**Measurement scope (open-loop integration per 2026-05-08 probe C
finding)**: This script measures TurboQuant's compression operation
overhead and compressed-state footprint. It does NOT measure quality
impact because the existing TurboQuant integration is open-loop —
compressed KV state is recorded as a side effect (``compressor.append()``)
but the model's next forward pass uses the original uncompressed
``past_key_values`` from the previous output. Perplexity reflects the
post-``load_packed_model`` model's baseline, NOT TurboQuant impact.

True quality-vs-compression Pareto measurement requires closed-loop
integration (banked as backlog item "Close TurboQuant compress→decompress
loop for true quality measurement"). This script's JSON schema uses
field names that surface the open-loop scope honestly:

- ``kv_cache_compressed_bytes_snapshot`` — what TurboQuant WOULD store
  under its compression scheme; not what's used at inference time
- ``compression_operation_wall_clock_seconds`` — TurboQuant
  ``.append()`` overhead component of the generation wall-clock
- ``kv_cache_hypothetical_compression_ratio`` — ratio of uncompressed
  actual vs compressed snapshot; "hypothetical" because closed-loop
  substitution doesn't happen
- ``model_baseline_perplexity`` — post-``load_packed_model`` model's
  baseline PPL; NOT TurboQuant quality impact

This is the first TN-003 measurement infrastructure script. Future
KIVI / KVQuant / kvtc / SpQt baseline scripts follow the same pattern
at ``benchmarks/tn003_<technique>_baseline.py`` with the same JSON
schema for apples-to-apples cross-technique comparison.

Wall-clock estimate per Phi-4 measurement run (empirical 2026-05-08/09):

- HF base load: ~27-29 min (HF cache lives on USB-C external storage —
  read-throughput-bound; cold and "cached" loads have similar wall-clock
  in practice, contrary to the original ~3-5 min cached estimate)
- ``load_packed_model``: ~1 min (faster than originally estimated)
- Generation (50 tokens): ~5 min on M4 Pro CPU
- Perplexity at canonical settings (max_length=2048, stride=512):
  **empirically unmeasurable on M4 Pro CPU**. Smoke 2 v1 (2026-05-08)
  ran 14:18 elapsed without completion before kill.
- Perplexity at scope-reduced settings (max_length=512, stride=512;
  ~10× attention reduction per window): **also empirically
  unmeasurable**. Smoke 2-prime (2026-05-09) ran 4:38 elapsed without
  completion before kill.
- **Phi-4 14B PPL via sliding-window perplexity on M4 Pro CPU is
  empirically unmeasurable in reasonable wall-clock at any tested
  setting.** The measurement gap is hardware capability, not tooling:
  Phi-4 14B FP16 forward passes on M4 Pro CPU dominate the wall-clock
  regardless of context length scaling. Tractable measurement requires
  M4 Max / M5 / Mac Studio class hardware, OR algorithmic shortcuts
  (subset of WikiText-2, smaller perplexity surrogate metric, etc.).
- Total ``--no-perplexity`` smoke: **~35 min** (verified across smoke 1
  v1-v4) — substantially longer than the original ~5-10 min projection
  because HF base load is read-throughput-bound on Syn Archive
- Total full smoke at canonical settings: **prohibitive on M4 Pro**
  per the empirical findings above

Copyright (c) 2025-2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from terncore import tern_model as _tern_model_module
from terncore.perplexity import compute_perplexity
from terncore.tern_model import TernModelReader

# Add tools/ so we can import the TurboQuant integration helpers.
# We replicate the generation loop inline (see _generate_with_compressor_capture)
# rather than calling generate_streaming_turboquant directly because we need to
# capture the compressor instance for footprint introspection and time the
# .append() calls separately for compression-overhead measurement.
_TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(_TOOLS_DIR))
from tern_infer import IncrementalTQCompressor, _extract_kv_pairs  # noqa: E402


SCHEMA_VERSION = 1
DEFAULT_PROMPT = "The capital of France is"
# DEFAULT_MAX_TOKENS=50 is a smoke-testing default. Partner-grade
# measurement runs should use 256-512 for stable token-throughput
# statistics — pass via --max-tokens at the CLI.
DEFAULT_MAX_TOKENS = 50
DEFAULT_PPL_STRIDE = 512
DEFAULT_PPL_MAX_LENGTH = 2048


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TN-003 TurboQuant baseline measurement orchestration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifest", required=True,
                   help=".tern-model artefact path")
    p.add_argument("--hf-model-id", required=True,
                   help="HF model identifier for base load")
    p.add_argument("--key-mapping", default=None,
                   help="key_mapping preset name in terncore.tern_model "
                        "(e.g., GEMMA4_MULTIMODAL_TRANSFORMERS_5_5)")
    p.add_argument("--prompt", default=DEFAULT_PROMPT,
                   help="Generation prompt")
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                   help="Maximum tokens to generate. Default 50 is a "
                        "smoke-testing value; partner-grade measurements "
                        "typically use 256-512 for stable token-throughput "
                        "statistics.")
    p.add_argument("--output-json", default=None,
                   help="Output JSON path (default: stdout)")
    p.add_argument("--no-perplexity", action="store_true",
                   help="Skip perplexity computation (fast mode for smoke testing)")
    p.add_argument("--ppl-stride", type=int, default=DEFAULT_PPL_STRIDE,
                   help="Sliding-window stride for perplexity")
    p.add_argument("--ppl-max-length", type=int, default=DEFAULT_PPL_MAX_LENGTH,
                   help="Sliding-window max length for perplexity")
    return p.parse_args()


def _turboquant_compressed_bytes(tqc, seen_ptrs: set) -> int:
    """Sum byte sizes of torch.Tensor fields on a TurboQuantCompressed.

    Traverses ``pq`` (PolarQuantCompressed) and ``qjl`` (QJLCompressed).
    Counts direct torch.Tensor fields only — does NOT recurse into nested
    objects like ``rotation`` (RandomHadamardRotation) or ``codebook``
    (Codebook), which contain their own tensor state. Those nested
    tensors are typically shared across all positions in a layer×head
    and thus don't represent per-position storage cost.

    ``seen_ptrs`` is a caller-supplied set of tensor storage pointers
    (``tensor.data_ptr()``); each tensor counted exactly once across
    the whole compressor traversal to deduplicate shared state (e.g.
    ``qjl.S`` random projection matrix shared across all positions
    within a layer×head — verified empirically 2026-05-08 when smoke 1
    v3 produced a 121× INVERSE compression ratio because qjl.S was
    counted per-instance across ~320,000 instances). data_ptr() used
    rather than id() so tensors that view the same underlying storage
    (tied embeddings, etc.) deduplicate correctly across PyTorch
    versions.

    API discovered empirically 2026-05-08 after smoke 1 failed with
    ``AttributeError: 'TurboQuantCompressed' object has no attribute 'numel'``
    — the original assumption that compressor.compressed leaves were
    torch.Tensor was inferred from variable naming, not probed by
    running an actual compression. Banked as 11th probe-before-committing
    instance.
    """
    total = 0
    for component in (tqc.pq, tqc.qjl):
        for field_name in component.__dataclass_fields__:
            value = getattr(component, field_name, None)
            if isinstance(value, torch.Tensor):
                ptr = value.data_ptr()
                if ptr in seen_ptrs:
                    continue
                seen_ptrs.add(ptr)
                total += value.numel() * value.element_size()
    return total


def _kv_cache_compressed_bytes_snapshot(
    compressor: IncrementalTQCompressor,
) -> int:
    """Sum byte sizes of all compressed KV state in compressor.compressed.

    Traverses the 3-level nested structure ``[layer][head][position_batch]``,
    where each leaf is a ``(k_c, v_c)`` tuple of ``TurboQuantCompressed``
    instances. Per-instance bytes computed via
    ``_turboquant_compressed_bytes`` with a shared ``seen_ptrs`` set so
    tensors with the same underlying storage are counted exactly once.

    Deduplication catches ``qjl.S`` (random projection matrix, shared
    across all positions in a layer×head) and any other shared state
    regardless of which dataclass field exposes it.

    SNAPSHOT semantics: represents what TurboQuant would store under
    its compression scheme, deduplicated across shared tensors. Under
    the existing open-loop pipeline, the model's inference path uses
    the uncompressed ``past_key_values`` not this snapshot.
    """
    seen_ptrs: set = set()
    total = 0
    for layer in compressor.compressed:
        for head in layer:
            for k_c, v_c in head:
                total += _turboquant_compressed_bytes(k_c, seen_ptrs)
                total += _turboquant_compressed_bytes(v_c, seen_ptrs)
    return total


def _model_param_count_via_state_dict(model) -> int:
    """Total parameter count via state_dict (includes buffers, dedups tied weights).

    Replaces ``sum(p.numel() for p in model.parameters())`` because
    PackedTernaryLinear (post-load_packed_model) stores packed_weights
    and scales as BUFFERS, not parameters; ``.parameters()`` misses
    them — verified empirically 2026-05-08 when smoke 1 v3 reported
    1.03B for Phi-4's 14B params, a 14× undercount.

    ``state_dict()`` captures both parameters AND buffers. Dedup by
    tensor storage pointer (``data_ptr()``) to handle tied embeddings
    (e.g. LM head ↔ input embedding shared storage in HF models with
    tied embeddings).
    """
    seen_ptrs: set = set()
    total = 0
    for t in model.state_dict().values():
        ptr = t.data_ptr()
        if ptr in seen_ptrs:
            continue
        seen_ptrs.add(ptr)
        total += t.numel()
    return total


def _kv_cache_uncompressed_bytes_actual(
    compressor: IncrementalTQCompressor,
    dtype: torch.dtype,
) -> int:
    """Compute the uncompressed KV bytes the model ACTUALLY used at inference.

    Formula: n_layers × n_heads × seq_len × head_dim × 2 (K+V) × dtype_bytes.

    ``dtype`` should be the model's parameter/activation dtype (typically
    obtained via ``model.dtype``). Bytes-per-element derived dynamically
    via ``torch.tensor([], dtype=dtype).element_size()`` to handle FP16 /
    BF16 / FP32 / FP64 / INT8 uniformly.

    head_dim accessed via ``compressor.config.d`` — IncrementalTQCompressor's
    constructor takes ``head_dim`` as a parameter and passes it to
    TurboQuantConfig as ``d``, but does NOT store it as
    ``self.head_dim`` (12th probe-before-committing instance, surfaced
    2026-05-08 by smoke 1 v2 AttributeError after the cf73136 fix).
    """
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    return (
        compressor.n_layers
        * compressor.n_heads
        * compressor.seq_len
        * compressor.config.d
        * 2  # K + V
        * dtype_bytes
    )


def _peak_rss_mb() -> float:
    """Peak RSS in MB. macOS returns ru_maxrss in bytes; Linux in kilobytes."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def _resolve_key_mapping(name: Optional[str]):
    if name is None:
        return None
    return getattr(_tern_model_module, name)


def _generate_with_compressor_capture(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
) -> tuple[str, int, float, float, IncrementalTQCompressor]:
    """Replicates generate_streaming_turboquant inline.

    Replication exists so we can:
    1. Capture the compressor instance for ``.compressed[][]`` footprint
       traversal after the loop completes.
    2. Time the ``compressor.append()`` calls separately for
       compression-overhead measurement (vs total generation wall-clock).

    If a future orchestration script needs the same pattern, refactor
    ``tools/tern_infer.py:generate_streaming_turboquant`` to optionally
    return the compressor instance.

    Returns:
        (generated_text, tokens_generated, generation_wall_clock_seconds,
         compression_op_wall_clock_seconds, compressor)
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generated_ids = input_ids.clone()

    t_start = time.perf_counter()
    tokens_generated = 0
    past_key_values = None
    compressor = None
    compression_op_total = 0.0

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

            # Time the compression operation separately. This is the
            # work TurboQuant actually does; the rest of the loop is
            # standard HF generation overhead.
            t_compress = time.perf_counter()
            compressor.append(past_key_values)
            compression_op_total += time.perf_counter() - t_compress

            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)
            if next_token_id.item() == tokenizer.eos_token_id:
                break
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            tokens_generated += 1

    wall_clock = time.perf_counter() - t_start
    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return full_text, tokens_generated, wall_clock, compression_op_total, compressor


def _measure_manifest_bytes(manifest_path: Path) -> int:
    """Manifest size on disk. Handles both single-file and directory layouts."""
    if manifest_path.is_file():
        return manifest_path.stat().st_size
    return sum(p.stat().st_size for p in manifest_path.rglob("*") if p.is_file())


def main() -> int:
    args = _parse_args()
    t_start = time.perf_counter()

    if not Path(args.manifest).exists():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    print(f"[orchestrate] Loading HF base: {args.hf_model_id}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    )
    model.eval()
    t_hf = time.perf_counter()
    print(f"[orchestrate] HF base loaded ({t_hf - t_start:.1f}s).", flush=True)

    print(f"[orchestrate] Loading {args.manifest} via load_packed_model...",
          flush=True)
    reader = TernModelReader(args.manifest)
    key_mapping = _resolve_key_mapping(args.key_mapping)
    missing, unexpected = reader.load_packed_model(model, key_mapping=key_mapping)
    t_load = time.perf_counter()
    print(f"[orchestrate] load_packed_model: missing={len(missing)}, "
          f"unexpected={len(unexpected)} ({t_load - t_hf:.1f}s).", flush=True)

    print(f"[orchestrate] Running generation with compressor capture "
          f"(prompt={args.prompt!r}, max_tokens={args.max_tokens})...", flush=True)
    (generated_text, n_tokens, gen_wall, comp_op_wall, compressor) = (
        _generate_with_compressor_capture(
            model, tokenizer, args.prompt, args.max_tokens,
        )
    )
    print(f"[orchestrate] Generation: {n_tokens} tokens in {gen_wall:.1f}s "
          f"({n_tokens / gen_wall:.2f} tok/s); compression op: "
          f"{comp_op_wall:.2f}s", flush=True)
    print(f"\n[orchestrate] Generated text:\n{generated_text}\n", flush=True)

    kv_compressed_bytes = _kv_cache_compressed_bytes_snapshot(compressor)
    kv_uncompressed_bytes = _kv_cache_uncompressed_bytes_actual(
        compressor, model.dtype,
    )
    hypothetical_ratio = (
        kv_uncompressed_bytes / kv_compressed_bytes if kv_compressed_bytes > 0 else 0
    )
    print(f"[orchestrate] KV cache compressed snapshot: "
          f"{kv_compressed_bytes:,} bytes "
          f"({kv_compressed_bytes / (1024*1024):.2f} MB); "
          f"hypothetical compression ratio: {hypothetical_ratio:.2f}x", flush=True)

    # Perplexity (default on; --no-perplexity to skip)
    ppl_record = None
    if not args.no_perplexity:
        print(f"[orchestrate] Computing WikiText-2 perplexity "
              f"(stride={args.ppl_stride}, max_length={args.ppl_max_length})...",
              flush=True)
        t_ppl = time.perf_counter()
        ppl_value = compute_perplexity(
            model, tokenizer,
            stride=args.ppl_stride,
            max_length=args.ppl_max_length,
        )
        ppl_wall = time.perf_counter() - t_ppl
        print(f"[orchestrate] PPL = {ppl_value:.4f} ({ppl_wall:.1f}s)", flush=True)
        ppl_record = {
            "dataset": "wikitext-2-raw-v1",
            "split": "validation",
            "stride": args.ppl_stride,
            "max_length": args.ppl_max_length,
            "value": ppl_value,
            "wall_clock_seconds": ppl_wall,
            "scope_note": (
                "Open-loop pipeline — PPL reflects post-load_packed_model "
                "model's baseline perplexity, NOT TurboQuant quality impact. "
                "See backlog item 'Close TurboQuant compress→decompress loop'."
            ),
        }

    manifest_bytes = _measure_manifest_bytes(Path(args.manifest))

    record = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "hf_model_id": args.hf_model_id,
        "model_param_count": {
            "value": _model_param_count_via_state_dict(model),
            "scope_note": (
                "Computed via state_dict() with data_ptr() dedup — captures "
                "both Parameters and Buffers (PackedTernaryLinear stores "
                "internal state as buffers post-load_packed_model). May "
                "still undercount when the model uses bit-packed storage "
                "(e.g. ternary trits packed 4-per-byte: numel() returns "
                "byte count, not trit count). Empirical 2026-05-08 (smoke 1 "
                "v4): Phi-4 14B reports 6.14B under this measurement — "
                "~2.3× undercount factor depends on packing scheme of "
                "compressed Linear layers."
            ),
        },
        "config": {
            "key_mapping": args.key_mapping,
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
        },
        "compression": {
            "manifest_bytes_on_disk": manifest_bytes,
            "kv_cache_compressed_bytes_snapshot": kv_compressed_bytes,
            "kv_cache_uncompressed_bytes_actual": kv_uncompressed_bytes,
            "kv_cache_hypothetical_compression_ratio": hypothetical_ratio,
            "scope_note": (
                "Snapshot represents what TurboQuant would store under its "
                "compression scheme — direct dataclass tensor fields only "
                "(rotation/codebook nested objects excluded as typically "
                "shared across positions in a layer×head). 'Hypothetical' "
                "because the existing pipeline is open-loop — the snapshot "
                "is not substituted back into the model's inference path. "
                "Tensor instances deduplicated by storage pointer "
                "(data_ptr()) — qjl.S random projection matrix is shared "
                "across all positions within a layer×head and counted "
                "exactly once. Empirical finding 2026-05-08 (smoke 1 v4): "
                "open-loop measurement at short context (~56 positions) "
                "produces a ratio <1× because per-position state "
                "(qjl.signs as int64, pq.indices, etc.) plus shared "
                "rotation state (qjl.S) exceed FP16 uncompressed KV bytes "
                "for that context length. TurboQuant's published 6× claim "
                "is CLOSED-LOOP and depends on bit-level packing of signs "
                "at storage time + reconstruction of decompressed state "
                "on-the-fly — neither happens in the existing open-loop "
                "pipeline. The ratio reported here is therefore the "
                "open-loop upper bound, not TurboQuant's true compression "
                "capability."
            ),
        },
        "generation": {
            "tokens_generated": n_tokens,
            "wall_clock_seconds": gen_wall,
            "tokens_per_second": n_tokens / gen_wall if gen_wall > 0 else 0,
            "compression_operation_wall_clock_seconds": comp_op_wall,
            "generated_text_first_200_chars": generated_text[:200],
        },
        "model_baseline_perplexity": ppl_record,
        "memory": {
            "peak_rss_mb": _peak_rss_mb(),
        },
        "wall_clock_total_seconds": time.perf_counter() - t_start,
    }

    output = json.dumps(record, indent=2)
    if args.output_json:
        Path(args.output_json).write_text(output + "\n")
        print(f"[orchestrate] JSON record written to: {args.output_json}",
              flush=True)
    else:
        print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
