"""R12 v1.1 b_mse sweep wrapper for KV-cache compression PPL headroom diagnostic.

Orchestrates a b_mse sweep against the R7-B v1.2 autoregressive harness with the
β1a (make_b_mse_hook_uniform) KV-cache compression hook on MPS. Produces
per-point JSONs conformant to ppl_headroom_kv_cache_point/1.0 and an aggregate
sweep manifest conformant to ppl_headroom_kv_cache_sweep/1.0
(R12 v1.1 §6.1, §6.2).

Canonical first-execution parameters (R12 v1.1 §8):
    --b-mse-values "6,5,4,3,2,1"
    --num-sequences 16
    --ar-seq-len 2048
    --device mps
    --hook-spec uniform
    continue_past_ceiling = false (sweep halts at first ceiling_crossed point)

Termination conditions per R12 v1.1 §6.2:
    ceiling_crossed:           ppl_kv_compressed > baseline × ceiling_multiplier
    grid_exhausted:            all sweep points completed without termination
    hook_construction_failure: factory raised on TurboQuantConfig.__init__
    ppl_eval_failure:          evaluate_ppl_autoregressive raised mid-loop

Per-point JSONs written immediately after each point completes, preserving
partial progress if the sweep halts mid-run. Aggregate manifest written once
at sweep completion (or termination).

Compression ratio per Q6 disposition: theoretical (16 / b_mse), not measured
from past_key_values byte sizes. The β1a hook is a round-trip
(K/V reshape preserved for PPL measurement correctness), so the pre/post-bytes
mechanism described in R12 v1.1 §6.1 is vacuous; the theoretical ratio
captures the deployment-relevant compression. See per-point notes for the
full rationale.

Copyright (c) 2025-2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

# Ensure tern_ppl_bench is importable as a sibling module under tools/
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tern_ppl_bench  # noqa: E402


# ── Schema constants ───────────────────────────────────────────────────

SCHEMA_VERSION_POINT = "ppl_headroom_kv_cache_point/1.0"
SCHEMA_VERSION_SWEEP = "ppl_headroom_kv_cache_sweep/1.0"
SPEC_VERSION = "kv_cache_compression_ppl_headroom_diagnostic v1.2"
METHODOLOGY_CONSUMED = "wikitext2_ppl_methodology_autoregressive v1.2"


# ── Canonical defaults (R12 v1.1 §8) ───────────────────────────────────

DEFAULT_B_MSE_VALUES = "6,5,4,3,2,1"
DEFAULT_NUM_SEQUENCES = 16
DEFAULT_AR_SEQ_LEN = 2048
DEFAULT_DEVICE = "mps"
DEFAULT_HOOK_SPEC = "uniform"
DEFAULT_OUTPUT_DIR = "results/ppl_headroom_kv_cache_sweep"
DEFAULT_CALIBRATION_GATE_PCT = 1.68    # R7-B v1.2 §1.1 N=16 gate
DEFAULT_CEILING_MULTIPLIER = 2.0       # ceiling_crossed if ppl > baseline × this
DEFAULT_CONTINUE_PAST_CEILING = False  # R12 v1.1 §8 canonical
DEFAULT_SEED = 1337                    # R12 v1.1 §8 canonical

# Smoke overrides
SMOKE_B_MSE_VALUES = "4,2"
SMOKE_NUM_SEQUENCES = 2
SMOKE_AR_SEQ_LEN = 128


# ── Compression-ratio helpers (Q6 disposition) ─────────────────────────


def theoretical_compression_ratio(b_mse: int) -> float:
    """Return 16/b_mse — theoretical codebook ratio for FP16 baseline."""
    return 16.0 / b_mse


def compression_ratio_notes(b_mse: int) -> str:
    """Per-point notes documenting the theoretical-ratio derivation."""
    ratio = theoretical_compression_ratio(b_mse)
    return (
        f"kv_cache_compression_ratio derived theoretically from codebook encoding "
        f"(16 bits FP16 / b_mse={b_mse} bits per coordinate = {ratio:.4g}x compression). "
        f"β1a make_b_mse_hook_uniform is a round-trip hook (round-trip K/V preserves "
        f"shape/dtype for PPL measurement correctness), so this ratio is computed from "
        f"b_mse rather than measured from pre/post past_key_values byte sizes. "
        f"R12 v1.1 §6.1's pre/post-bytes mechanism is vacuous for round-trip hooks; "
        f"a future R12 v1.2 amendment will clarify the field semantics."
    )


# ── Result containers ──────────────────────────────────────────────────


@dataclass
class SweepPointResult:
    point_index: int
    b_mse: int
    ppl_eval_run_id: str
    ppl_kv_compressed: Optional[float]
    ppl_headroom: Optional[float]
    ppl_headroom_band: Optional[str]
    kv_cache_compression_ratio: Optional[float]
    factory_build_seconds: float
    eval_wall_time_seconds: float
    tokens_scored: int
    terminated_at_ceiling: bool
    hook_construction_failed: bool
    error_traceback: Optional[str]
    notes: str
    output_filename: str = ""  # set when JSON written


@dataclass
class SweepResult:
    diagnostic_run_id: str
    point_results: list  # list[SweepPointResult]
    termination_reason: str
    terminated_at_point_index: Optional[int]
    failed_at_b_mse: Optional[int]
    total_eval_wall_time_seconds: float
    output_dir: Path
    manifest_path: Optional[Path]


# ── Termination + recommended-operating-point logic ────────────────────


def detect_termination(
    point_result: SweepPointResult, continue_past_ceiling: bool
) -> Optional[str]:
    """Return termination reason if this point triggers one, else None."""
    if point_result.hook_construction_failed:
        return "hook_construction_failure"
    if point_result.ppl_kv_compressed is None:
        # Numerical failure (NaN/Inf or exception); ppl_eval_failure does NOT
        # halt sweep — caller continues. Returning the reason lets the caller
        # decide based on continue-on-failure policy.
        return "ppl_eval_failure"
    if point_result.terminated_at_ceiling and not continue_past_ceiling:
        return "ceiling_crossed"
    return None


def classify_recommended_operating_point(
    point_results: list, calibration_gate_pct: float
) -> dict:
    """Find the most-aggressive b_mse (lowest int) meeting the calibration gate.

    Returns the §6.2 recommended_operating_point dict. If no point qualifies,
    returns the schema-conformant null-stub with explanatory rationale.
    """
    qualifying = [
        p for p in point_results
        if p.ppl_kv_compressed is not None
        and p.ppl_headroom is not None
        and abs(p.ppl_headroom) * 100 < calibration_gate_pct
    ]
    if not qualifying:
        return {
            "point_index": None,
            "b_mse": None,
            "ppl_headroom": None,
            "kv_cache_compression_ratio": None,
            "rationale": (
                f"no sweep point produced ppl_headroom within calibration gate "
                f"{calibration_gate_pct}% (R7-B v1.2 §1.1 N=16 gate)"
            ),
        }
    best = min(qualifying, key=lambda p: p.b_mse)
    return {
        "point_index": best.point_index,
        "b_mse": best.b_mse,
        "ppl_headroom": round(best.ppl_headroom, 4),
        "kv_cache_compression_ratio": best.kv_cache_compression_ratio,
        "rationale": (
            f"minimum b_mse with ppl_headroom within {calibration_gate_pct}% "
            f"calibration gate (R7-B v1.2 §1.1 N=16)"
        ),
    }


# ── Per-point sweep execution ──────────────────────────────────────────


def run_sweep_point(
    *,
    point_index: int,
    b_mse: int,
    model: Any,
    sequences: list,
    baseline_ppl: float,
    ceiling_multiplier: float,
    device: str,
    hook_spec: str,
    model_id: str,
    num_sequences: int,
    seq_len: int,
) -> SweepPointResult:
    """Execute one b_mse point: build factory, run PPL eval, classify result."""
    run_id = tern_ppl_bench.utc_now_compact()
    ratio = theoretical_compression_ratio(b_mse)
    base_notes = compression_ratio_notes(b_mse)

    # Build the hook factory (per-point; closure over b_mse)
    factory_start = time.perf_counter()
    try:
        hook, _params, factory_seconds = tern_ppl_bench.build_kv_cache_hook(
            hook_spec=hook_spec, b_mse=b_mse, model=model, device=device,
        )
    except Exception as exc:
        return SweepPointResult(
            point_index=point_index,
            b_mse=b_mse,
            ppl_eval_run_id=run_id,
            ppl_kv_compressed=None,
            ppl_headroom=None,
            ppl_headroom_band=None,
            kv_cache_compression_ratio=ratio,
            factory_build_seconds=time.perf_counter() - factory_start,
            eval_wall_time_seconds=0.0,
            tokens_scored=0,
            terminated_at_ceiling=False,
            hook_construction_failed=True,
            error_traceback=traceback.format_exc(),
            notes=f"hook construction failed: {exc!r}. " + base_notes,
        )

    print(
        f"[r12_sweep] point {point_index} (b_mse={b_mse}): factory built in "
        f"{factory_seconds:.2f}s ({hook_spec})",
        flush=True,
    )

    # Run the PPL eval with the hook
    eval_start = time.perf_counter()
    try:
        result = tern_ppl_bench.evaluate_ppl_autoregressive(
            model=model, sequences=sequences, kv_cache_hook=hook, device=device,
        )
    except Exception as exc:
        return SweepPointResult(
            point_index=point_index,
            b_mse=b_mse,
            ppl_eval_run_id=run_id,
            ppl_kv_compressed=None,
            ppl_headroom=None,
            ppl_headroom_band=None,
            kv_cache_compression_ratio=ratio,
            factory_build_seconds=factory_seconds,
            eval_wall_time_seconds=time.perf_counter() - eval_start,
            tokens_scored=0,
            terminated_at_ceiling=False,
            hook_construction_failed=False,
            error_traceback=traceback.format_exc(),
            notes=f"ppl eval raised: {exc!r}. " + base_notes,
        )

    # Classify result
    ppl_kv = result.ppl
    if not math.isfinite(ppl_kv):
        return SweepPointResult(
            point_index=point_index,
            b_mse=b_mse,
            ppl_eval_run_id=run_id,
            ppl_kv_compressed=None,
            ppl_headroom=None,
            ppl_headroom_band=None,
            kv_cache_compression_ratio=ratio,
            factory_build_seconds=factory_seconds,
            eval_wall_time_seconds=result.eval_wall_time_seconds,
            tokens_scored=result.tokens_scored,
            terminated_at_ceiling=False,
            hook_construction_failed=False,
            error_traceback=None,
            notes=f"ppl is non-finite ({ppl_kv!r}). " + base_notes,
        )

    ppl_headroom = (ppl_kv - baseline_ppl) / baseline_ppl
    band = tern_ppl_bench.classify_ppl_headroom_band(ppl_headroom)
    ceiling_crossed = ppl_kv > baseline_ppl * ceiling_multiplier

    print(
        f"[r12_sweep] point {point_index} (b_mse={b_mse}): ppl={ppl_kv:.4f} "
        f"headroom={ppl_headroom * 100:+.3f}% band={band} "
        f"wall={result.eval_wall_time_seconds:.1f}s "
        f"ceiling_crossed={ceiling_crossed}",
        flush=True,
    )

    return SweepPointResult(
        point_index=point_index,
        b_mse=b_mse,
        ppl_eval_run_id=run_id,
        ppl_kv_compressed=ppl_kv,
        ppl_headroom=ppl_headroom,
        ppl_headroom_band=band,
        kv_cache_compression_ratio=ratio,
        factory_build_seconds=factory_seconds,
        eval_wall_time_seconds=result.eval_wall_time_seconds,
        tokens_scored=result.tokens_scored,
        terminated_at_ceiling=ceiling_crossed,
        hook_construction_failed=False,
        error_traceback=None,
        notes=base_notes,
    )


# ── JSON assembly (§6.1, §6.2) ─────────────────────────────────────────


def build_point_json(
    *,
    point_result: SweepPointResult,
    diagnostic_run_id: str,
    baseline_ppl: float,
    model_id: str,
    num_sequences: int,
    seq_len: int,
) -> dict:
    """Build per-point JSON conforming to ppl_headroom_kv_cache_point/1.0."""
    return {
        "schema_version": SCHEMA_VERSION_POINT,
        "diagnostic_run_id": diagnostic_run_id,
        "point_index": point_result.point_index,
        "config": {
            "b_mse": point_result.b_mse,
            "weight_threshold": "FP16",
            "kv_hook_application_scope": "all_layers",
            "num_sequences": num_sequences,
            "seq_len": seq_len,
            "model_id": model_id,
        },
        "ppl_eval_run_id": point_result.ppl_eval_run_id,
        "baseline_ppl_r7b": baseline_ppl,
        "ppl_kv_compressed": (
            round(point_result.ppl_kv_compressed, 4)
            if point_result.ppl_kv_compressed is not None else None
        ),
        "ppl_headroom": (
            round(point_result.ppl_headroom, 4)
            if point_result.ppl_headroom is not None else None
        ),
        "ppl_headroom_band": point_result.ppl_headroom_band,
        "kv_cache_compression_ratio": point_result.kv_cache_compression_ratio,
        "terminated_at_ceiling": point_result.terminated_at_ceiling,
        "hook_construction_failed": point_result.hook_construction_failed,
        "notes": point_result.notes,
    }


def build_sweep_manifest(
    *,
    sweep_result: SweepResult,
    inputs: dict,
    calibration_gate_pct: float,
    device: str,
    notes: str,
) -> dict:
    """Build aggregate sweep manifest conforming to ppl_headroom_kv_cache_sweep/1.0."""
    point_results = sweep_result.point_results

    points = [
        {"point_index": p.point_index, "filename": p.output_filename}
        for p in point_results
    ]

    # Frontier: ordered list of per-point summaries
    frontier = [
        {
            "point_index": p.point_index,
            "b_mse": p.b_mse,
            "ppl_headroom": (
                round(p.ppl_headroom, 4) if p.ppl_headroom is not None else None
            ),
            "ppl_headroom_band": p.ppl_headroom_band,
            "kv_cache_compression_ratio": p.kv_cache_compression_ratio,
        }
        for p in point_results
    ]

    recommended = classify_recommended_operating_point(
        point_results, calibration_gate_pct
    )

    return {
        "schema_version": SCHEMA_VERSION_SWEEP,
        "diagnostic_run_id": sweep_result.diagnostic_run_id,
        "timestamp_utc": tern_ppl_bench.utc_now_iso(),
        "tern_core_version": _tern_core_version(),
        "tern_core_git_commit": tern_ppl_bench.git_commit_short(),
        "spec_version": SPEC_VERSION,
        "methodology_consumed": METHODOLOGY_CONSUMED,
        "inputs": inputs,
        "points": points,
        "frontier": frontier,
        "recommended_operating_point": recommended,
        "termination": {
            "terminated_reason": sweep_result.termination_reason,
            "terminated_at_point_index": sweep_result.terminated_at_point_index,
            "failed_at_b_mse": sweep_result.failed_at_b_mse,
        },
        "hardware": {
            "device": device,
            "ppl_eval_wall_time_seconds": round(
                sweep_result.total_eval_wall_time_seconds, 2
            ),
        },
        "notes": notes,
    }


def _tern_core_version() -> str:
    """Best-effort tern-core version string."""
    try:
        import terncore
        return getattr(terncore, "__version__", "unknown")
    except Exception:
        return "unknown"


# ── Main orchestration ─────────────────────────────────────────────────


def run_sweep(args: argparse.Namespace) -> SweepResult:
    """Main sweep orchestration: load model once, loop over b_mse points."""
    diagnostic_run_id = str(uuid.uuid4())
    sweep_dir_name = f"sweep_{tern_ppl_bench.utc_now_compact()}_{diagnostic_run_id[:8]}"
    output_dir = Path(args.output_dir) / sweep_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[r12_sweep] diagnostic_run_id={diagnostic_run_id}\n"
        f"[r12_sweep] output_dir={output_dir}",
        flush=True,
    )

    b_mse_values = [int(x.strip()) for x in args.b_mse_values.split(",") if x.strip()]
    print(
        f"[r12_sweep] sweep grid: b_mse={b_mse_values}, N={args.num_sequences}, "
        f"L={args.ar_seq_len}, device={args.device}, hook={args.hook_spec}",
        flush=True,
    )

    # Load model ONCE for the entire sweep (Q2 disposition)
    print(f"[r12_sweep] loading model {args.model_id} on {args.device}...", flush=True)
    t_load = time.perf_counter()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    dtype = torch.float16 if args.device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    model = model.to(args.device)
    model.eval()
    print(
        f"[r12_sweep]   model loaded in {time.perf_counter() - t_load:.1f}s, "
        f"dtype={next(model.parameters()).dtype}",
        flush=True,
    )

    # Build sequences ONCE for the entire sweep
    print(f"[r12_sweep] loading WikiText-2 test split...", flush=True)
    test_text, hf_revision = tern_ppl_bench.load_wikitext2_test_text()
    tokens = tern_ppl_bench.prepare_tokens(
        test_text, tokenizer, bos_token_id=tokenizer.bos_token_id
    )
    sequences = tern_ppl_bench.build_sequences_autoregressive(
        tokens=tokens,
        num_sequences=args.num_sequences,
        seq_len=args.ar_seq_len,
        bos_token_id=tokenizer.bos_token_id,
    )
    print(
        f"[r12_sweep]   built {len(sequences)} sequences of L_eff={len(sequences[0])} "
        f"(tokens={tokens.shape[0]:,}, hf_revision={hf_revision})",
        flush=True,
    )

    # Sweep loop
    point_results: list = []
    termination_reason = "grid_exhausted"
    terminated_at_point_index: Optional[int] = None
    failed_at_b_mse: Optional[int] = None
    total_eval_wall = 0.0

    for idx, b_mse in enumerate(b_mse_values):
        print(f"\n[r12_sweep] === point {idx} (b_mse={b_mse}) ===", flush=True)
        point_t0 = time.perf_counter()
        point_result = run_sweep_point(
            point_index=idx,
            b_mse=b_mse,
            model=model,
            sequences=sequences,
            baseline_ppl=args.baseline_ppl,
            ceiling_multiplier=args.ceiling_multiplier,
            device=args.device,
            hook_spec=args.hook_spec,
            model_id=args.model_id,
            num_sequences=args.num_sequences,
            seq_len=args.ar_seq_len,
        )

        # Write per-point JSON immediately (partial-progress preservation)
        point_filename = (
            f"point_{idx:02d}_b_mse_{b_mse}_{point_result.ppl_eval_run_id}.json"
        )
        point_path = output_dir / point_filename
        point_payload = build_point_json(
            point_result=point_result,
            diagnostic_run_id=diagnostic_run_id,
            baseline_ppl=args.baseline_ppl,
            model_id=args.model_id,
            num_sequences=args.num_sequences,
            seq_len=args.ar_seq_len,
        )
        point_path.write_text(json.dumps(point_payload, indent=2) + "\n")
        point_result.output_filename = point_filename
        point_results.append(point_result)
        total_eval_wall += point_result.eval_wall_time_seconds

        wall = time.perf_counter() - point_t0
        if wall > 7200:
            print(
                f"[r12_sweep]   WARNING: point {idx} took {wall / 60:.1f} min "
                f"(>2 hours); proceeding",
                flush=True,
            )

        # Termination check (Q4 disposition)
        reason = detect_termination(point_result, args.continue_past_ceiling)
        if reason == "hook_construction_failure":
            print(
                f"[r12_sweep] HALT: hook_construction_failure at b_mse={b_mse} "
                f"(environmental issue suggests halting sweep)",
                flush=True,
            )
            termination_reason = reason
            terminated_at_point_index = idx
            failed_at_b_mse = b_mse
            break
        if reason == "ceiling_crossed":
            print(
                f"[r12_sweep] HALT: ceiling_crossed at b_mse={b_mse} "
                f"(continue_past_ceiling={args.continue_past_ceiling})",
                flush=True,
            )
            termination_reason = reason
            terminated_at_point_index = idx
            failed_at_b_mse = b_mse
            break
        if reason == "ppl_eval_failure":
            print(
                f"[r12_sweep] continuing past ppl_eval_failure at b_mse={b_mse} "
                f"(less-aggressive b_mse may still succeed)",
                flush=True,
            )
            # Record failure but do not halt
            if failed_at_b_mse is None:
                failed_at_b_mse = b_mse
                terminated_at_point_index = idx
                termination_reason = "ppl_eval_failure"

    return SweepResult(
        diagnostic_run_id=diagnostic_run_id,
        point_results=point_results,
        termination_reason=termination_reason,
        terminated_at_point_index=terminated_at_point_index,
        failed_at_b_mse=failed_at_b_mse,
        total_eval_wall_time_seconds=total_eval_wall,
        output_dir=output_dir,
        manifest_path=None,
    )


# ── CLI ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="R12 v1.1 b_mse sweep wrapper for KV-cache PPL headroom diagnostic"
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--baseline-run-id", required=True,
                        help="R7-B baseline run_id (for ppl_headroom comparison record)")
    parser.add_argument("--baseline-ppl", required=True, type=float,
                        help="R7-B baseline PPL value")
    parser.add_argument("--b-mse-values", default=DEFAULT_B_MSE_VALUES,
                        help="Comma-separated b_mse values, e.g. '6,5,4,3,2,1'")
    parser.add_argument("--num-sequences", default=DEFAULT_NUM_SEQUENCES, type=int)
    parser.add_argument("--ar-seq-len", default=DEFAULT_AR_SEQ_LEN, type=int)
    parser.add_argument("--device", default=DEFAULT_DEVICE,
                        choices=["mps", "cpu"])
    parser.add_argument("--hook-spec", default=DEFAULT_HOOK_SPEC,
                        choices=["uniform", "mixed"],
                        help="β1a (uniform) per R12 v1.1 §8.2.3 recommended")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--calibration-gate-pct", default=DEFAULT_CALIBRATION_GATE_PCT,
                        type=float,
                        help="R7-B v1.2 §1.1 N=16 gate (default 1.68)")
    parser.add_argument("--ceiling-multiplier", default=DEFAULT_CEILING_MULTIPLIER,
                        type=float,
                        help="ceiling_crossed if ppl > baseline × this (default 2.0)")
    parser.add_argument("--continue-past-ceiling", action="store_true",
                        default=DEFAULT_CONTINUE_PAST_CEILING,
                        help="Continue sweep after first ceiling_crossed point")
    parser.add_argument("--seed", default=DEFAULT_SEED, type=int)
    parser.add_argument("--notes", default="")
    parser.add_argument("--smoke", action="store_true",
                        help="Reduced-param smoke mode: b_mse=[4,2], N=2, L=128")

    args = parser.parse_args()

    if args.smoke:
        args.b_mse_values = SMOKE_B_MSE_VALUES
        args.num_sequences = SMOKE_NUM_SEQUENCES
        args.ar_seq_len = SMOKE_AR_SEQ_LEN
        if not args.notes:
            args.notes = "R12 v1.1 sweep wrapper smoke mode"
        print(
            f"[r12_sweep] SMOKE MODE: b_mse={args.b_mse_values}, "
            f"N={args.num_sequences}, L={args.ar_seq_len}",
            flush=True,
        )

    sweep_result = run_sweep(args)

    # Build + write aggregate manifest
    b_mse_list = [int(x.strip()) for x in args.b_mse_values.split(",") if x.strip()]
    inputs = {
        "source_model": args.model_id,
        "baseline_ppl_r7b": args.baseline_ppl,
        "baseline_run_id": args.baseline_run_id,
        "ppl_headroom_ceiling": (args.ceiling_multiplier - 1.0),  # 0.50 if mult=1.50; 1.0 if mult=2.0
        "sweep_grid": {"b_mse": b_mse_list},
        "num_sequences": args.num_sequences,
        "seq_len": args.ar_seq_len,
        "seed": args.seed,
        "continue_past_ceiling": args.continue_past_ceiling,
    }
    manifest = build_sweep_manifest(
        sweep_result=sweep_result,
        inputs=inputs,
        calibration_gate_pct=args.calibration_gate_pct,
        device=args.device,
        notes=args.notes,
    )

    manifest_path = sweep_result.output_dir / "sweep_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    sweep_result.manifest_path = manifest_path

    print(
        f"\n[r12_sweep] === SWEEP COMPLETE ===\n"
        f"[r12_sweep] termination: {sweep_result.termination_reason}\n"
        f"[r12_sweep] points completed: {len(sweep_result.point_results)} / "
        f"{len(b_mse_list)}\n"
        f"[r12_sweep] total eval wall-time: {sweep_result.total_eval_wall_time_seconds:.1f}s "
        f"({sweep_result.total_eval_wall_time_seconds / 60:.1f} min)\n"
        f"[r12_sweep] recommended_operating_point: "
        f"b_mse={manifest['recommended_operating_point']['b_mse']}, "
        f"headroom={manifest['recommended_operating_point']['ppl_headroom']}\n"
        f"[r12_sweep] manifest: {manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
