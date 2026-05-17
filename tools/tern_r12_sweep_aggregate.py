"""R12 v1.2 aggregator: build sweep_manifest.json from per-point JSONs.

Standalone aggregator for the per-point process-split orchestrator
(tern_r12_sweep_orchestrator.py). Scans a sweep directory for per-point
JSONs conforming to ppl_headroom_kv_cache_point/1.0, reconstructs
SweepPointResult shells, and reuses tern_r12_sweep.build_sweep_manifest()
to emit an aggregate manifest conforming to ppl_headroom_kv_cache_sweep/1.0
(R12 v1.2 §6.2).

Two invocation modes:

1. Library mode (orchestrator path) — aggregate_sweep_directory() takes the
   sweep_dir, the inputs dict (baseline_ppl, baseline_run_id, ...), and an
   externally-measured total_eval_wall_time_seconds; returns the manifest
   path. The orchestrator measures wall-time across per-point subprocesses
   and passes the total here.

2. CLI mode (post-hoc recovery) — `python tern_r12_sweep_aggregate.py
   <sweep_dir> [--baseline-ppl ...] [...]` reads per-point JSONs from a
   crashed-or-completed sweep directory and emits a manifest. Total
   wall-time defaults to the sum of point JSONs' eval_wall_time_seconds
   if present; otherwise 0.0 with a notes-field flag.

The aggregator is pure over per-point JSONs — re-running it on a partial
sweep produces a valid manifest with termination_reason inferred from the
point set (incomplete | ceiling_crossed | hook_construction_failure |
ppl_eval_failure | grid_exhausted).

Copyright (c) 2025-2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

# Ensure tern_r12_sweep is importable as a sibling module under tools/
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tern_r12_sweep  # noqa: E402
from tern_r12_sweep import (  # noqa: E402
    SweepPointResult,
    SweepResult,
    build_sweep_manifest,
)


POINT_FILENAME_RE = re.compile(r"^point_(\d+)_b_mse_(\d+)_.*\.json$")


def _point_result_from_json(payload: dict, filename: str) -> SweepPointResult:
    """Reconstruct a SweepPointResult shell from a per-point JSON payload.

    Per-point JSON does NOT carry factory_build_seconds / eval_wall_time_seconds
    / tokens_scored / error_traceback (per §6.1 schema). Shell sets those to
    zero/None — they are not used by build_sweep_manifest() except for the
    aggregate total_eval_wall_time, which the orchestrator measures externally
    and passes through SweepResult.total_eval_wall_time_seconds.
    """
    return SweepPointResult(
        point_index=payload["point_index"],
        b_mse=payload["config"]["b_mse"],
        ppl_eval_run_id=payload["ppl_eval_run_id"],
        ppl_kv_compressed=payload.get("ppl_kv_compressed"),
        ppl_headroom=payload.get("ppl_headroom"),
        ppl_headroom_band=payload.get("ppl_headroom_band"),
        kv_cache_compression_ratio=payload.get("kv_cache_compression_ratio"),
        factory_build_seconds=0.0,
        eval_wall_time_seconds=0.0,
        tokens_scored=0,
        terminated_at_ceiling=bool(payload.get("terminated_at_ceiling", False)),
        hook_construction_failed=bool(payload.get("hook_construction_failed", False)),
        error_traceback=None,
        notes=payload.get("notes", ""),
        output_filename=filename,
    )


def _scan_point_jsons(sweep_dir: Path) -> list[tuple[int, int, Path, dict]]:
    """Return [(point_index, b_mse, path, payload), ...] sorted by point_index."""
    rows: list[tuple[int, int, Path, dict]] = []
    for path in sorted(sweep_dir.glob("point_*.json")):
        m = POINT_FILENAME_RE.match(path.name)
        if not m:
            continue
        payload = json.loads(path.read_text())
        rows.append((payload["point_index"], payload["config"]["b_mse"], path, payload))
    rows.sort(key=lambda r: r[0])
    return rows


def _infer_termination(
    point_results: list[SweepPointResult],
    expected_b_mse_grid: Optional[list[int]],
    continue_past_ceiling: bool,
) -> tuple[str, Optional[int], Optional[int]]:
    """Infer termination_reason / terminated_at_point_index / failed_at_b_mse.

    Mirrors tern_r12_sweep.detect_termination() semantics, applied across the
    point set on disk rather than as a live sweep decision.
    """
    for p in point_results:
        if p.hook_construction_failed:
            return ("hook_construction_failure", p.point_index, p.b_mse)
        if p.terminated_at_ceiling and not continue_past_ceiling:
            return ("ceiling_crossed", p.point_index, p.b_mse)
    # ppl_eval_failure: ppl_kv_compressed is None without hook_construction_failed
    first_failure: Optional[SweepPointResult] = None
    for p in point_results:
        if p.ppl_kv_compressed is None and not p.hook_construction_failed:
            first_failure = p
            break
    if expected_b_mse_grid is not None:
        completed = {p.b_mse for p in point_results}
        missing = [b for b in expected_b_mse_grid if b not in completed]
        if missing:
            # Sweep was halted mid-run by something other than a recorded
            # ceiling_crossed / hook_construction_failure point. Most common
            # cause: subprocess killed by OS (libdispatch, OOM, SIGKILL).
            last = point_results[-1] if point_results else None
            return (
                "incomplete",
                last.point_index if last else None,
                last.b_mse if last else None,
            )
    if first_failure is not None:
        return ("ppl_eval_failure", first_failure.point_index, first_failure.b_mse)
    return ("grid_exhausted", None, None)


def aggregate_sweep_directory(
    *,
    sweep_dir: Path,
    diagnostic_run_id: str,
    inputs: dict,
    calibration_gate_pct: float,
    device: str,
    notes: str,
    total_eval_wall_time_seconds: float,
    expected_b_mse_grid: Optional[list[int]] = None,
    continue_past_ceiling: bool = False,
) -> Path:
    """Aggregate per-point JSONs under sweep_dir into sweep_manifest.json.

    Returns the manifest path. Idempotent: re-running overwrites the manifest.
    """
    rows = _scan_point_jsons(sweep_dir)
    point_results = [_point_result_from_json(payload, path.name) for (_, _, path, payload) in rows]

    termination_reason, terminated_at_point_index, failed_at_b_mse = _infer_termination(
        point_results, expected_b_mse_grid, continue_past_ceiling,
    )

    sweep_result = SweepResult(
        diagnostic_run_id=diagnostic_run_id,
        point_results=point_results,
        termination_reason=termination_reason,
        terminated_at_point_index=terminated_at_point_index,
        failed_at_b_mse=failed_at_b_mse,
        total_eval_wall_time_seconds=total_eval_wall_time_seconds,
        output_dir=sweep_dir,
        manifest_path=None,
    )

    manifest = build_sweep_manifest(
        sweep_result=sweep_result,
        inputs=inputs,
        calibration_gate_pct=calibration_gate_pct,
        device=device,
        notes=notes,
    )

    manifest_path = sweep_dir / "sweep_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def _cli_aggregate(args: argparse.Namespace) -> None:
    sweep_dir = Path(args.sweep_dir).resolve()
    if not sweep_dir.is_dir():
        raise SystemExit(f"sweep_dir not a directory: {sweep_dir}")

    rows = _scan_point_jsons(sweep_dir)
    if not rows:
        raise SystemExit(f"no point_*.json files found under {sweep_dir}")

    # Reuse the first point's diagnostic_run_id if --diagnostic-run-id omitted
    diagnostic_run_id = args.diagnostic_run_id or rows[0][3]["diagnostic_run_id"]

    # Reconstruct grid + inputs from first point's payload when CLI args omitted
    first = rows[0][3]
    b_mse_grid = (
        [int(x.strip()) for x in args.b_mse_grid.split(",") if x.strip()]
        if args.b_mse_grid else [b for (_, b, _, _) in rows]
    )

    inputs = {
        "source_model": args.model_id or first["config"]["model_id"],
        "baseline_ppl_r7b": args.baseline_ppl if args.baseline_ppl is not None
                            else first["baseline_ppl_r7b"],
        "baseline_run_id": args.baseline_run_id or "<unknown; post-hoc aggregate>",
        "ppl_headroom_ceiling": args.ceiling_multiplier - 1.0,
        "sweep_grid": {"b_mse": b_mse_grid},
        "num_sequences": args.num_sequences if args.num_sequences is not None
                         else first["config"]["num_sequences"],
        "seq_len": args.seq_len if args.seq_len is not None
                   else first["config"]["seq_len"],
        "seed": args.seed,
        "continue_past_ceiling": args.continue_past_ceiling,
    }

    notes = args.notes or (
        "post-hoc aggregation via tern_r12_sweep_aggregate.py CLI; "
        "total_eval_wall_time_seconds=0.0 because per-point JSONs do not "
        "carry eval_wall_time_seconds (R12 v1.2 §6.1)"
    )

    manifest_path = aggregate_sweep_directory(
        sweep_dir=sweep_dir,
        diagnostic_run_id=diagnostic_run_id,
        inputs=inputs,
        calibration_gate_pct=args.calibration_gate_pct,
        device=args.device,
        notes=notes,
        total_eval_wall_time_seconds=args.total_eval_wall_time_seconds,
        expected_b_mse_grid=b_mse_grid,
        continue_past_ceiling=args.continue_past_ceiling,
    )
    print(f"[r12_aggregate] manifest written: {manifest_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Aggregate per-point JSONs under a sweep directory into "
                    "sweep_manifest.json (R12 v1.2 §6.2). Post-hoc-safe."
    )
    p.add_argument("sweep_dir", help="Directory containing point_*.json files")
    p.add_argument("--diagnostic-run-id", default=None,
                   help="Override diagnostic_run_id (default: reuse first point's)")
    p.add_argument("--model-id", default=None)
    p.add_argument("--baseline-ppl", type=float, default=None)
    p.add_argument("--baseline-run-id", default=None)
    p.add_argument("--b-mse-grid", default=None,
                   help="Expected b_mse grid for termination inference, e.g. '6,5,4,3,2,1'")
    p.add_argument("--num-sequences", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--device", default="mps", choices=["mps", "cpu"])
    p.add_argument("--calibration-gate-pct", type=float,
                   default=tern_r12_sweep.DEFAULT_CALIBRATION_GATE_PCT)
    p.add_argument("--ceiling-multiplier", type=float,
                   default=tern_r12_sweep.DEFAULT_CEILING_MULTIPLIER)
    p.add_argument("--continue-past-ceiling", action="store_true",
                   default=tern_r12_sweep.DEFAULT_CONTINUE_PAST_CEILING)
    p.add_argument("--seed", type=int, default=tern_r12_sweep.DEFAULT_SEED)
    p.add_argument("--total-eval-wall-time-seconds", type=float, default=0.0,
                   help="Total wall-time across points (orchestrator-measured); "
                        "default 0.0 for post-hoc aggregation")
    p.add_argument("--notes", default="")
    args = p.parse_args()
    _cli_aggregate(args)


if __name__ == "__main__":
    main()
