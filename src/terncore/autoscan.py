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
        ppl_headroom:  Maximum allowed PPL increase as a fraction (0.20 = 20%).
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
    base_model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
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

        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
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
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
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
