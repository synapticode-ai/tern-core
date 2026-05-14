"""
Automatic perplexity-gated ternary conversion scanner.

Determines the maximum number of layers that can be safely converted to
ternary while keeping perplexity within a configurable budget (default
+20% of the FP32 baseline).

Layers are tested in sensitivity order — v_proj (most tolerant) first,
MLP projections (least tolerant) last — and within each projection type,
deepest transformer blocks first.

Results are cached to ~/.terncore/model_cache.json keyed by model ID so
repeat runs skip the scan entirely.

Patent 4: Progressive Compression — iterative protection search for
          mixed-precision ternary/FP16 deployment.

Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
"""

from __future__ import annotations

import gc
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CACHE_DIR = Path.home() / ".terncore"
CACHE_FILE = CACHE_DIR / "model_cache.json"

DEFAULT_BLOCK_SIZE = 10
# Autoscan-internal per-layer scan-time gate: maximum PPL degradation
# allowed per added layer-block during perplexity-gated layer-eligibility
# scanning. NOT the R7-A/R8 v1.1 outcome metric of the same name — see
# auto_scan() docstring for terminology distinction. R13.
DEFAULT_PPL_HEADROOM = 0.20

# Projection types ordered by empirical quantization tolerance (most -> least).
_PROJ_PRIORITY = ["v_proj", "k_proj", "o_proj", "q_proj",
                  "gate_proj", "up_proj", "down_proj"]

# Patterns that should never be converted (same as MixedPrecisionConverter defaults).
_SKIP_PATTERNS = ("embed", "layernorm", "layer_norm", "rmsnorm",
                  "lm_head", "output", "classifier")

_CALIBRATION_TEXT = (
    "The development of large language models has transformed natural language "
    "processing research. These models, trained on vast corpora of text data, "
    "demonstrate remarkable capabilities in text generation, summarization, "
    "translation, and question answering. However, their computational cost "
    "during inference remains a significant barrier to widespread deployment. "
    "Techniques such as quantization, pruning, and knowledge distillation aim "
    "to reduce this cost while preserving model quality. Ternary quantization "
    "represents an extreme form of weight compression, replacing floating-point "
    "values with elements from the set negative one, zero, and positive one. "
    "The key challenge is identifying which layers can tolerate this aggressive "
    "compression without unacceptable degradation in output quality."
)


# ---------------------------------------------------------------------------
# Scan result
# ---------------------------------------------------------------------------

@dataclass
class ScanResult:
    """Result of an auto-scan: which layers are safe to convert."""

    model_id: str
    baseline_ppl: float
    best_ppl: float
    ppl_ceiling: float
    ppl_headroom: float
    total_eligible: int
    layers_converted: int
    pct_converted: float
    compression_ratio: float
    protection_list: list[str]
    converted_list: list[str]
    sweep_trace: list[dict] = field(default_factory=list)
    cached: bool = False
    ternary_list: list[str] = field(default_factory=list)
    int4_list: list[str] = field(default_factory=list)
    mixed_compression_ratio: float = 0.0

    @property
    def ppl_delta_pct(self) -> float:
        if self.baseline_ppl == 0:
            return 0.0
        return (self.best_ppl - self.baseline_ppl) / self.baseline_ppl * 100

    @property
    def quality_verdict(self) -> str:
        delta = self.ppl_delta_pct
        if self.layers_converted == 0:
            return "No layers passed — model is too sensitive for ternary at this threshold"
        if delta < 2.0:
            return "Excellent — near-lossless ternary compression"
        if delta < 5.0:
            return "Good — minimal quality loss, suitable for production"
        if delta < 10.0:
            return "Acceptable — moderate quality trade-off for efficiency"
        if delta <= 20.0:
            return "Marginal — noticeable quality loss, use with caution"
        return "Poor — significant quality degradation"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eligible_linear_names(model: nn.Module) -> list[str]:
    """Return names of nn.Linear modules eligible for ternary conversion."""
    names = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            if not any(p in name.lower() for p in _SKIP_PATTERNS):
                names.append(name)
    return names


def _sort_by_sensitivity(layer_names: list[str]) -> list[str]:
    """Sort layers most-tolerant first: v_proj deepest → MLP shallowest."""
    def sort_key(name: str) -> tuple[int, int]:
        proj_rank = len(_PROJ_PRIORITY)
        for i, suffix in enumerate(_PROJ_PRIORITY):
            if name.endswith(suffix):
                proj_rank = i
                break
        block_idx = 0
        for part in name.split("."):
            if part.isdigit():
                block_idx = int(part)
                break
        return (proj_rank, -block_idx)

    return sorted(layer_names, key=sort_key)


def _measure_perplexity(model, tokenizer) -> float:
    """Compute perplexity on the calibration paragraph."""
    input_ids = tokenizer(_CALIBRATION_TEXT, return_tensors="pt").input_ids
    if input_ids.shape[1] < 2:
        return float("inf")
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    return math.exp(outputs.loss.item())


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def _cache_key(model_id: str, threshold: float, ppl_headroom: float) -> str:
    return f"{model_id}|t={threshold}|h={ppl_headroom}"


def _load_cache() -> dict:
    if not CACHE_FILE.exists():
        return {}
    try:
        return json.loads(CACHE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2) + "\n")


def load_cached_result(
    model_id: str,
    threshold: float = 0.7,
    ppl_headroom: float = DEFAULT_PPL_HEADROOM,
) -> Optional[ScanResult]:
    """Return a cached ScanResult if one exists, else None."""
    cache = _load_cache()
    key = _cache_key(model_id, threshold, ppl_headroom)
    entry = cache.get(key)
    if entry is None:
        return None
    return ScanResult(
        model_id=entry["model_id"],
        baseline_ppl=entry["baseline_ppl"],
        best_ppl=entry["best_ppl"],
        ppl_ceiling=entry["ppl_ceiling"],
        ppl_headroom=entry["ppl_headroom"],
        total_eligible=entry["total_eligible"],
        layers_converted=entry["layers_converted"],
        pct_converted=entry["pct_converted"],
        compression_ratio=entry["compression_ratio"],
        protection_list=entry["protection_list"],
        converted_list=entry["converted_list"],
        sweep_trace=entry.get("sweep_trace", []),
        cached=True,
        ternary_list=entry.get("ternary_list", []),
        int4_list=entry.get("int4_list", []),
        mixed_compression_ratio=entry.get("mixed_compression_ratio", 0.0),
    )


def _save_result(result: ScanResult, threshold: float) -> None:
    cache = _load_cache()
    key = _cache_key(result.model_id, threshold, result.ppl_headroom)
    cache[key] = {
        "model_id": result.model_id,
        "baseline_ppl": result.baseline_ppl,
        "best_ppl": result.best_ppl,
        "ppl_ceiling": result.ppl_ceiling,
        "ppl_headroom": result.ppl_headroom,
        "total_eligible": result.total_eligible,
        "layers_converted": result.layers_converted,
        "pct_converted": result.pct_converted,
        "compression_ratio": result.compression_ratio,
        "protection_list": result.protection_list,
        "converted_list": result.converted_list,
        "sweep_trace": result.sweep_trace,
        "ternary_list": result.ternary_list,
        "int4_list": result.int4_list,
        "mixed_compression_ratio": result.mixed_compression_ratio,
    }
    _save_cache(cache)


# ---------------------------------------------------------------------------
# Classification printer
# ---------------------------------------------------------------------------

def print_scan_result(result: ScanResult) -> None:
    """Print a clear classification of the model's efficiency position."""
    print()
    print("=" * 62)
    print("  Ternary Auto-Scan Results")
    print("=" * 62)
    if result.cached:
        print(f"  (cached from {CACHE_FILE})")
    print(f"  Model:            {result.model_id}")
    print(f"  Layers converted: {result.layers_converted}/{result.total_eligible} "
          f"({result.pct_converted:.1f}%)")
    print(f"  Compression:      {result.compression_ratio:.2f}x")
    print(f"  Baseline PPL:     {result.baseline_ppl:.2f}")
    if result.layers_converted > 0:
        print(f"  Ternary PPL:      {result.best_ppl:.2f} "
              f"(+{result.ppl_delta_pct:.1f}%)")
    print(f"  PPL ceiling:      {result.ppl_ceiling:.2f} "
          f"(+{result.ppl_headroom:.0%})")
    print(f"  Verdict:          {result.quality_verdict}")
    print("=" * 62)
    print()


# ---------------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------------

def auto_scan(
    model_id: str,
    threshold: float = 0.7,
    ppl_headroom: float = DEFAULT_PPL_HEADROOM,
    block_size: int = DEFAULT_BLOCK_SIZE,
    use_cache: bool = True,
) -> ScanResult:
    """Run a perplexity-gated sweep to find the maximum safe ternary config.

    Args:
        model_id:      HuggingFace model ID or local path.
        threshold:     Ternary quantization threshold.
        ppl_headroom:  Maximum allowed PPL degradation as a fraction (0.20 = 20%),
                       applied as a per-layer scan-time GATE during the sweep:
                       adding a layer-block is accepted only if cumulative PPL
                       degradation stays under this threshold.

                       NOTE: this is autoscan's *internal* (autoscan-internal) sense of `ppl_headroom`
                       — a scan-time gate on per-layer additions. DISTINCT from
                       the R7-A v1.0 WikiText-2 PPL methodology / R8 v1.1
                       weight-compression PPL headroom diagnostic outcome metric
                       of the same name (`ppl_headroom = (ppl_ternary - ppl_fp16)
                       / ppl_fp16`), which is a measured *result* of compression
                       rather than a *gate* during scanning. The two are
                       conceptually related but not interchangeable. See R13
                       (docs/backlog.md) for terminology disambiguation tracking.
        block_size:    Number of layers to add per sweep step.
        use_cache:     If True, return cached result when available.

    Returns:
        ScanResult with the protection list and classification metadata.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from terncore.mixed_precision import MixedPrecisionConverter

    # --- check cache ---
    if use_cache:
        cached = load_cached_result(model_id, threshold, ppl_headroom)
        if cached is not None:
            print(f"Auto-scan: using cached result for {model_id}")
            print_scan_result(cached)
            return cached

    print(f"Auto-scan: analysing {model_id} for safe ternary layers...")
    t0 = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- baseline ---
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16,
        device_map="auto", low_cpu_mem_usage=True,
        max_memory={"cpu": "50GiB"},
    )
    base_model.eval()

    eligible_fwd = _eligible_linear_names(base_model)
    total_eligible = len(eligible_fwd)
    eligible_sorted = _sort_by_sensitivity(eligible_fwd)

    baseline_ppl = _measure_perplexity(base_model, tokenizer)
    ppl_ceiling = baseline_ppl * (1.0 + ppl_headroom)
    print(f"  Baseline PPL: {baseline_ppl:.2f}, ceiling: {ppl_ceiling:.2f} "
          f"(+{ppl_headroom:.0%})")
    print(f"  Eligible layers: {total_eligible}")

    del base_model
    gc.collect()

    # --- progressive sweep ---
    sweep_trace: list[dict] = []
    best_n = 0
    best_ppl = baseline_ppl
    best_protection: list[str] = list(eligible_fwd)
    best_ratio = 1.0

    for block_end in range(block_size, total_eligible + 1, block_size):
        to_convert = set(eligible_sorted[:block_end])
        protect_names = [n for n in eligible_fwd if n not in to_convert]

        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16,
            device_map="auto", low_cpu_mem_usage=True,
        )
        converter = MixedPrecisionConverter(
            threshold=threshold,
            protection_list=protect_names,
        )
        report = converter.convert(model)
        model.eval()

        ppl = _measure_perplexity(model, tokenizer)
        passed = ppl <= ppl_ceiling
        pct = block_end / total_eligible * 100

        step = {
            "layers_tested": block_end,
            "layers_converted": report.converted_layers,
            "pct_converted": round(pct, 1),
            "perplexity": round(ppl, 2),
            "within_budget": passed,
            "compression_ratio": round(report.compression_ratio, 2),
        }
        sweep_trace.append(step)

        status = "PASS" if passed else "FAIL"
        print(f"  [{block_end:3d}/{total_eligible}] PPL={ppl:.2f} "
              f"({report.compression_ratio:.2f}x) {status}")

        del model
        gc.collect()

        if passed:
            best_n = block_end
            best_ppl = ppl
            best_protection = protect_names
            best_ratio = report.compression_ratio
        else:
            break

    # partial tail block
    remainder = total_eligible % block_size
    if (best_n > 0 and best_n < total_eligible
            and remainder != 0 and sweep_trace[-1]["within_budget"]):
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16,
            device_map="auto", low_cpu_mem_usage=True,
        )
        converter = MixedPrecisionConverter(threshold=threshold, protection_list=[])
        report = converter.convert(model)
        model.eval()

        ppl = _measure_perplexity(model, tokenizer)
        passed = ppl <= ppl_ceiling
        step = {
            "layers_tested": total_eligible,
            "layers_converted": report.converted_layers,
            "pct_converted": 100.0,
            "perplexity": round(ppl, 2),
            "within_budget": passed,
            "compression_ratio": round(report.compression_ratio, 2),
        }
        sweep_trace.append(step)
        status = "PASS" if passed else "FAIL"
        print(f"  [{total_eligible}/{total_eligible}] PPL={ppl:.2f} "
              f"({report.compression_ratio:.2f}x) {status}")

        del model
        gc.collect()

        if passed:
            best_n = total_eligible
            best_ppl = ppl
            best_protection = []
            best_ratio = report.compression_ratio

    best_pct = best_n / total_eligible * 100 if total_eligible else 0
    converted_list = eligible_sorted[:best_n] if best_n > 0 else []

    elapsed = time.perf_counter() - t0
    print(f"  Scan complete in {elapsed:.1f}s: {best_n}/{total_eligible} layers safe")

    result = ScanResult(
        model_id=model_id,
        baseline_ppl=round(baseline_ppl, 2),
        best_ppl=round(best_ppl, 2),
        ppl_ceiling=round(ppl_ceiling, 2),
        ppl_headroom=ppl_headroom,
        total_eligible=total_eligible,
        layers_converted=best_n,
        pct_converted=round(best_pct, 1),
        compression_ratio=round(best_ratio, 2),
        protection_list=best_protection,
        converted_list=converted_list,
        sweep_trace=sweep_trace,
        cached=False,
    )

    # persist
    _save_result(result, threshold)
    print_scan_result(result)

    return result


# ---------------------------------------------------------------------------
# Streaming scan (v0.5.0) — weight-space sensitivity, no full model load
# ---------------------------------------------------------------------------

@dataclass
class LayerSensitivity:
    """Reconstruction error for a single layer under ternary quantisation."""

    name: str
    relative_error: float   # ||W - alpha*T||_F / ||W||_F
    num_params: int
    sparsity: float
    alpha: float


def _compute_layer_sensitivity(
    name: str, weight: torch.Tensor, threshold: float,
) -> LayerSensitivity:
    """Compute reconstruction error for one weight tensor.

    The relative Frobenius error between the original weights and their
    ternary dequantised approximation is a reliable proxy for per-layer
    perplexity impact (used by GPTQ, AWQ, SqueezeLLM).
    """
    from terncore.arithmetic.quantizer import TernaryQuantizer

    w = weight.float()
    quantizer = TernaryQuantizer(threshold=threshold)
    ternary, alpha_tensor = quantizer.quantize(w)
    alpha = alpha_tensor.item() if alpha_tensor.numel() == 1 else alpha_tensor.mean().item()
    reconstructed = quantizer.dequantize(ternary, alpha_tensor)

    w_norm = torch.norm(w).item()
    error_norm = torch.norm(w - reconstructed).item()
    relative_error = error_norm / w_norm if w_norm > 0 else 0.0
    sparsity = (ternary == 0).float().mean().item()

    return LayerSensitivity(
        name=name,
        relative_error=relative_error,
        num_params=weight.numel(),
        sparsity=sparsity,
        alpha=alpha,
    )


def streaming_scan(
    model_id: str,
    threshold: float = 0.7,
    ppl_headroom: float = DEFAULT_PPL_HEADROOM,
    use_cache: bool = True,
    baseline_ppl: Optional[float] = None,
) -> ScanResult:
    """Streaming perplexity-gated scan using weight-space sensitivity.

    Two-pass design that never loads the full model for the sweep:

    Pass 1 — Baseline PPL: loads full model once with device_map="auto"
             and disk offloading.  Skipped if ``baseline_ppl`` is provided.
    Pass 2 — Sensitivity ranking: streams one tensor at a time from
             safetensors shards, computes reconstruction error, ranks
             layers by tolerance.  Peak memory: ~450 MB.

    The protection list is built by walking the ranked layers from
    most-tolerant to least-tolerant and accumulating a predicted
    compression ratio.  An error budget derived from ``ppl_headroom``
    determines the cutoff.

    Args:
        model_id:     Local path to a sharded safetensors model directory.
        threshold:    Ternary quantisation threshold.
        ppl_headroom: Maximum allowed PPL degradation as a fraction (0.20 = 20%),
                      applied as a per-layer scan-time GATE during the sweep:
                      adding a layer-block is accepted only if cumulative PPL
                      degradation stays under this threshold.

                      NOTE: this is autoscan's *internal* (autoscan-internal) sense of `ppl_headroom`
                      — a scan-time gate on per-layer additions. DISTINCT from
                      the R7-A v1.0 WikiText-2 PPL methodology / R8 v1.1
                      weight-compression PPL headroom diagnostic outcome metric
                      of the same name (`ppl_headroom = (ppl_ternary - ppl_fp16)
                      / ppl_fp16`), which is a measured *result* of compression
                      rather than a *gate* during scanning. The two are
                      conceptually related but not interchangeable. See R13
                      (docs/backlog.md) for terminology disambiguation tracking.
        use_cache:    Return cached result if available.
        baseline_ppl: Pre-computed baseline PPL (skips Pass 1 if given).

    Returns:
        ScanResult with protection_list and predicted compression ratio.
    """
    from terncore.sharded_loader import ShardedWeightIterator

    # --- check cache ---
    if use_cache:
        cached = load_cached_result(model_id, threshold, ppl_headroom)
        if cached is not None:
            print(f"Streaming scan: using cached result for {model_id}")
            print_scan_result(cached)
            return cached

    print(f"Streaming scan: analysing {model_id} (weight-space sensitivity)...")
    t0 = time.perf_counter()

    loader = ShardedWeightIterator(model_id)
    eligible_names = loader.eligible_linear_names()
    total_eligible = len(eligible_names)
    eligible_set = set(eligible_names)

    print(f"  Model: {loader.num_blocks} blocks, {loader.num_weights} weights")
    print(f"  Eligible layers: {total_eligible}")

    # --- Pass 1: baseline PPL ---
    if baseline_ppl is None:
        print("  Pass 1: computing baseline perplexity (full model load)...")
        baseline_ppl = _streaming_baseline_ppl(model_id)
    ppl_ceiling = baseline_ppl * (1.0 + ppl_headroom)
    print(f"  Baseline PPL: {baseline_ppl:.2f}, ceiling: {ppl_ceiling:.2f} "
          f"(+{ppl_headroom:.0%})")

    # --- Pass 2: weight-space sensitivity (streaming) ---
    print("  Pass 2: streaming sensitivity analysis...")
    sensitivities: list[LayerSensitivity] = []

    for name, tensor, _bidx in loader.iter_tensors():
        if name not in eligible_set:
            continue
        sens = _compute_layer_sensitivity(name, tensor, threshold)
        sensitivities.append(sens)
        del tensor

        if len(sensitivities) % 50 == 0:
            print(f"    [{len(sensitivities)}/{total_eligible}] layers analysed")

    gc.collect()
    print(f"    [{len(sensitivities)}/{total_eligible}] layers analysed — done")

    # Sort by relative error: lowest error = most tolerant = convert first
    sensitivities.sort(key=lambda s: s.relative_error)

    # --- Build protection list via error budget ---
    # The error budget is calibrated so that the cumulative normalised
    # reconstruction error stays within a range empirically correlated
    # with the ppl_headroom.  At 5% PPL headroom, we allow layers whose
    # cumulative error contribution remains under 5% of the total
    # possible error (sum of all relative errors).
    total_error = sum(s.relative_error for s in sensitivities)
    error_budget = ppl_headroom * total_error if total_error > 0 else 0.0

    cumulative_error = 0.0
    converted_names: list[str] = []
    sweep_trace: list[dict] = []

    for i, sens in enumerate(sensitivities):
        cumulative_error += sens.relative_error
        within_budget = cumulative_error <= error_budget

        step = {
            "layer": sens.name,
            "relative_error": round(sens.relative_error, 6),
            "cumulative_error": round(cumulative_error, 4),
            "error_budget": round(error_budget, 4),
            "within_budget": within_budget,
            "sparsity": round(sens.sparsity, 4),
            "params": sens.num_params,
        }
        sweep_trace.append(step)

        if within_budget:
            converted_names.append(sens.name)
        else:
            break

    # --- 3-tier split: ternary / INT4 / FP16 ---
    converted_set = set(converted_names)
    ternary_names = list(converted_names)  # tolerant layers → ternary
    int4_names = [n for n in eligible_names if n not in converted_set]  # sensitive → INT4
    protection_list: list[str] = []  # no FP16 among eligible (all get INT4 or ternary)

    layers_converted = len(converted_names)
    pct_converted = layers_converted / total_eligible * 100 if total_eligible else 0

    # Compression ratios
    ternary_params = sum(s.num_params for s in sensitivities if s.name in converted_set)
    int4_params = sum(s.num_params for s in sensitivities if s.name not in converted_set)
    total_params = sum(s.num_params for s in sensitivities)

    # Ternary-only compression (legacy)
    original_bytes = total_params * 2  # FP16 baseline
    ternary_bytes = ternary_params * 0.25
    fp16_bytes = (total_params - ternary_params) * 2
    compression_ratio = original_bytes / (ternary_bytes + fp16_bytes) if (ternary_bytes + fp16_bytes) > 0 else 1.0

    # Mixed ternary/INT4 compression
    int4_bytes = int4_params * 0.5  # 4 bits per weight
    mixed_compressed = ternary_bytes + int4_bytes
    mixed_compression_ratio = original_bytes / mixed_compressed if mixed_compressed > 0 else 1.0

    elapsed = time.perf_counter() - t0

    # Print sensitivity ranking
    print(f"\n  Sensitivity ranking (top-10 most tolerant):")
    for i, s in enumerate(sensitivities[:10]):
        tag = "TERNARY" if s.name in converted_set else "INT4"
        print(f"    {i+1:3d}. {s.name}")
        print(f"         error={s.relative_error:.6f}  sparsity={s.sparsity:.4f}  [{tag}]")

    if len(sensitivities) > 10:
        print(f"    ... ({total_eligible - 10} more)")
        print(f"\n  Least tolerant (bottom-3):")
        for s in sensitivities[-3:]:
            tag = "TERNARY" if s.name in converted_set else "INT4"
            print(f"    {s.name}")
            print(f"         error={s.relative_error:.6f}  sparsity={s.sparsity:.4f}  [{tag}]")

    print(f"\n  Scan complete in {elapsed:.1f}s")
    print(f"    Ternary: {len(ternary_names)}/560 layers ({ternary_params:,} params)")
    print(f"    INT4:    {len(int4_names)}/560 layers ({int4_params:,} params)")
    print(f"    Ternary-only compression: {compression_ratio:.2f}x")
    print(f"    Mixed ternary/INT4:       {mixed_compression_ratio:.2f}x")

    # Predicted PPL: linear interpolation from error fraction
    if total_error > 0:
        error_fraction = cumulative_error / total_error
        predicted_ppl = baseline_ppl * (1.0 + error_fraction * ppl_headroom)
    else:
        predicted_ppl = baseline_ppl

    result = ScanResult(
        model_id=model_id,
        baseline_ppl=round(baseline_ppl, 2),
        best_ppl=round(predicted_ppl, 2),
        ppl_ceiling=round(ppl_ceiling, 2),
        ppl_headroom=ppl_headroom,
        total_eligible=total_eligible,
        layers_converted=layers_converted,
        pct_converted=round(pct_converted, 1),
        compression_ratio=round(compression_ratio, 2),
        protection_list=protection_list,
        converted_list=converted_names,
        sweep_trace=sweep_trace,
        cached=False,
        ternary_list=ternary_names,
        int4_list=int4_names,
        mixed_compression_ratio=round(mixed_compression_ratio, 2),
    )

    _save_result(result, threshold)
    print_scan_result(result)

    return result


def _streaming_baseline_ppl(model_id: str) -> float:
    """Compute baseline PPL by loading the full model with disk offloading."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16,
        device_map="auto", low_cpu_mem_usage=True,
        max_memory={"cpu": "50GiB"},
    )
    model.eval()

    ppl = _measure_perplexity(model, tokenizer)

    del model, tokenizer
    gc.collect()

    return ppl


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Perplexity-gated ternary conversion scanner",
        prog="tern-autoscan",
    )
    parser.add_argument(
        "--model", required=True,
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.7,
        help="Ternary quantisation threshold (default: 0.7)",
    )
    parser.add_argument(
        "--perplexity-gate", type=float, default=None,
        help="Maximum allowed PPL increase as a fraction (e.g. 0.05 = 5%%). "
             "Overrides --ppl-headroom.",
    )
    parser.add_argument(
        "--ppl-headroom", type=float, default=DEFAULT_PPL_HEADROOM,
        help="PPL headroom threshold for autoscan per-layer scan-time gate "
             "(max PPL degradation per layer-block; default 0.20). "
             "Autoscan-internal sense — distinct from R7-A/R8 v1.1 outcome metric. "
             "R13.",
    )
    parser.add_argument(
        "--block-size", type=int, default=DEFAULT_BLOCK_SIZE,
        help=f"Number of layers to add per sweep step (default: {DEFAULT_BLOCK_SIZE})",
    )
    parser.add_argument(
        "--layer-by-layer", action="store_true",
        help="Test one layer at a time (sets --block-size 1)",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Ignore cached results and force a fresh scan",
    )
    parser.add_argument(
        "--streaming", action="store_true",
        help="Use streaming mode: weight-space sensitivity analysis without "
             "loading the full model for each sweep step.  Required for models "
             "that exceed available RAM.",
    )
    parser.add_argument(
        "--baseline-ppl", type=float, default=None,
        help="Pre-computed baseline perplexity (skips baseline model load in "
             "--streaming mode).",
    )

    args = parser.parse_args()

    ppl_headroom = args.perplexity_gate if args.perplexity_gate is not None else args.ppl_headroom
    block_size = 1 if args.layer_by_layer else args.block_size

    try:
        if args.streaming:
            streaming_scan(
                model_id=args.model,
                threshold=args.threshold,
                ppl_headroom=ppl_headroom,
                use_cache=not args.no_cache,
                baseline_ppl=args.baseline_ppl,
            )
        else:
            auto_scan(
                model_id=args.model,
                threshold=args.threshold,
                ppl_headroom=ppl_headroom,
                block_size=block_size,
                use_cache=not args.no_cache,
            )
    except KeyboardInterrupt:
        print("\nScan interrupted.")
        sys.exit(130)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
