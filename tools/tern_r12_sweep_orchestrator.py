"""R12 v1.2 per-point process-split orchestrator (libdispatch-trap mitigation).

Runs each b_mse sweep point in a fresh Python subprocess so that MPS allocator
state (and any other per-process global) cannot accumulate across points.
Motivated by lesson 16 (feedback_process_death_evidence_completeness_v1) and
the 2026-05-18 A'' R12 sweep PID 8767 failure: PyTorch MPS libdispatch
reentrant deadlock terminated a single long-running process mid-point-0,
leaving no per-point JSON written.

Architecture:
  - Mint diagnostic_run_id + sweep subdir name ONCE here.
  - For each b_mse: spawn `python tools/tern_r12_sweep.py
      --b-mse-values <one> --diagnostic-run-id <id> --output-subdir <subdir>
      --skip-manifest ...` via subprocess.Popen so PID is captured.
  - Wait for the subprocess; measure wall-time around it.
  - Read the just-written per-point JSON; apply detect_termination() to
    decide whether to launch the next point (halt on ceiling_crossed /
    hook_construction_failure; continue past ppl_eval_failure).
  - After loop: call aggregate_sweep_directory() with the orchestrator-
    measured total_eval_wall_time_seconds.

The model loads fresh per subprocess (~30-60s on Mac for Gemma 4 E4B). Across
6 canonical points that's ~3-6 min added wall-clock vs the single-process
sweep — accepted cost of MPS-allocator-state isolation.

Smoke mode (--smoke): b_mse=[4,2], N=2, L=128. Passes those values
EXPLICITLY to each subprocess via --b-mse-values "<one>" (NOT via --smoke,
which would hardcode b_mse=[4,2] inside each subprocess and silently
re-expand the per-point invocation back to multi-point).

Copyright (c) 2025-2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

# Ensure sibling modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tern_r12_sweep  # noqa: E402
import tern_ppl_bench  # noqa: E402
from tern_r12_sweep_aggregate import aggregate_sweep_directory  # noqa: E402


SWEEP_SCRIPT = Path(__file__).resolve().parent / "tern_r12_sweep.py"
POINT_FILENAME_RE = re.compile(r"^point_(\d+)_b_mse_(\d+)_.*\.json$")

# Smoke defaults — explicit per-point fanout (DO NOT pass --smoke to children)
SMOKE_B_MSE_GRID = [4, 2]
SMOKE_NUM_SEQUENCES = 2
SMOKE_AR_SEQ_LEN = 128


def _build_point_argv(
    *,
    python: str,
    b_mse: int,
    point_index: int,
    args: argparse.Namespace,
    diagnostic_run_id: str,
    output_subdir: str,
) -> list[str]:
    """Build the argv for one per-point subprocess invocation."""
    return [
        python, str(SWEEP_SCRIPT),
        "--model-id", args.model_id,
        "--baseline-run-id", args.baseline_run_id,
        "--baseline-ppl", str(args.baseline_ppl),
        "--b-mse-values", str(b_mse),
        "--num-sequences", str(args.num_sequences),
        "--ar-seq-len", str(args.ar_seq_len),
        "--device", args.device,
        "--hook-spec", args.hook_spec,
        "--output-dir", args.output_dir,
        "--calibration-gate-pct", str(args.calibration_gate_pct),
        "--ceiling-multiplier", str(args.ceiling_multiplier),
        "--seed", str(args.seed),
        "--diagnostic-run-id", diagnostic_run_id,
        "--output-subdir", output_subdir,
        "--point-index-offset", str(point_index),
        "--skip-manifest",
    ]


def _read_point_json(sweep_dir: Path, point_index: int, b_mse: int) -> Optional[dict]:
    """Read the per-point JSON written by the just-finished subprocess.

    Subprocess writes filename `point_<NN>_b_mse_<B>_<runid>.json`. The runid
    suffix is unknown to the orchestrator, so we glob by (point_index, b_mse).
    Returns None if no matching file (subprocess died before write).
    """
    pattern = f"point_{point_index:02d}_b_mse_{b_mse}_*.json"
    matches = list(sweep_dir.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        # Multiple writes for same (idx, b_mse) — last-wins by mtime
        matches.sort(key=lambda p: p.stat().st_mtime)
    return json.loads(matches[-1].read_text())


def _termination_from_point_json(
    payload: dict, continue_past_ceiling: bool,
) -> Optional[str]:
    """Decide halt-or-continue from a per-point JSON payload.

    Mirrors tern_r12_sweep.detect_termination() but reads from JSON dict
    rather than SweepPointResult dataclass.
    """
    if payload.get("hook_construction_failed", False):
        return "hook_construction_failure"
    if payload.get("ppl_kv_compressed") is None:
        return "ppl_eval_failure"
    if payload.get("terminated_at_ceiling", False) and not continue_past_ceiling:
        return "ceiling_crossed"
    return None


def run_orchestrated_sweep(args: argparse.Namespace) -> Path:
    """Orchestrate per-point sweep; return manifest path."""
    diagnostic_run_id = str(uuid.uuid4())
    sweep_dir_name = (
        f"sweep_{tern_ppl_bench.utc_now_compact()}_{diagnostic_run_id[:8]}"
    )
    sweep_dir = Path(args.output_dir) / sweep_dir_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    b_mse_grid = [int(x.strip()) for x in args.b_mse_values.split(",") if x.strip()]
    python = args.python or sys.executable

    print(
        f"[r12_orchestrator] diagnostic_run_id={diagnostic_run_id}\n"
        f"[r12_orchestrator] sweep_dir={sweep_dir}\n"
        f"[r12_orchestrator] python={python}\n"
        f"[r12_orchestrator] grid: b_mse={b_mse_grid}, N={args.num_sequences}, "
        f"L={args.ar_seq_len}, device={args.device}, hook={args.hook_spec}\n"
        f"[r12_orchestrator] per-point process split active "
        f"(lesson 16 / libdispatch mitigation)",
        flush=True,
    )

    total_wall = 0.0
    points_launched = 0
    halted_reason: Optional[str] = None

    for idx, b_mse in enumerate(b_mse_grid):
        argv = _build_point_argv(
            python=python, b_mse=b_mse, point_index=idx, args=args,
            diagnostic_run_id=diagnostic_run_id, output_subdir=sweep_dir_name,
        )
        print(
            f"\n[r12_orchestrator] === launching point {idx} (b_mse={b_mse}) ===",
            flush=True,
        )
        t0 = time.perf_counter()
        proc = subprocess.Popen(argv)
        pid = proc.pid
        print(
            f"[r12_orchestrator]   PID={pid} (fresh process; "
            f"MPS allocator state isolated)",
            flush=True,
        )
        returncode = proc.wait()
        wall = time.perf_counter() - t0
        total_wall += wall
        points_launched += 1

        print(
            f"[r12_orchestrator]   PID={pid} exited rc={returncode} "
            f"wall={wall:.1f}s ({wall / 60:.2f} min)",
            flush=True,
        )

        payload = _read_point_json(sweep_dir, idx, b_mse)
        if payload is None:
            # Subprocess died before writing per-point JSON. Halt — sweep is
            # incomplete; surface for surgeon disposition.
            print(
                f"[r12_orchestrator] HALT: no per-point JSON written for "
                f"point {idx} (b_mse={b_mse}); subprocess (PID={pid}) likely "
                f"died before write. Inspect ~/Library/Logs/DiagnosticReports/ "
                f"per lesson 16. Aggregator will mark termination as "
                f"'incomplete'.",
                flush=True,
            )
            halted_reason = "subprocess_died_pre_write"
            break

        # Surface point summary
        print(
            f"[r12_orchestrator]   ppl_kv={payload.get('ppl_kv_compressed')}, "
            f"headroom={payload.get('ppl_headroom')}, "
            f"band={payload.get('ppl_headroom_band')}, "
            f"ceiling_crossed={payload.get('terminated_at_ceiling')}, "
            f"hook_failed={payload.get('hook_construction_failed')}",
            flush=True,
        )

        reason = _termination_from_point_json(payload, args.continue_past_ceiling)
        if reason == "hook_construction_failure":
            print(
                f"[r12_orchestrator] HALT: hook_construction_failure at "
                f"b_mse={b_mse}",
                flush=True,
            )
            halted_reason = reason
            break
        if reason == "ceiling_crossed":
            print(
                f"[r12_orchestrator] HALT: ceiling_crossed at b_mse={b_mse} "
                f"(continue_past_ceiling={args.continue_past_ceiling})",
                flush=True,
            )
            halted_reason = reason
            break
        if reason == "ppl_eval_failure":
            print(
                f"[r12_orchestrator] continuing past ppl_eval_failure at "
                f"b_mse={b_mse} (less-aggressive b_mse may still succeed)",
                flush=True,
            )

    # Build inputs dict for aggregator (matches build_sweep_manifest contract)
    inputs = {
        "source_model": args.model_id,
        "baseline_ppl_r7b": args.baseline_ppl,
        "baseline_run_id": args.baseline_run_id,
        "ppl_headroom_ceiling": args.ceiling_multiplier - 1.0,
        "sweep_grid": {"b_mse": b_mse_grid},
        "num_sequences": args.num_sequences,
        "seq_len": args.ar_seq_len,
        "seed": args.seed,
        "continue_past_ceiling": args.continue_past_ceiling,
    }

    notes_parts = [args.notes] if args.notes else []
    notes_parts.append(
        "per-point process-split orchestration "
        "(tern_r12_sweep_orchestrator.py, lesson 16 mitigation; "
        "MPS allocator state isolated across points)"
    )
    if halted_reason == "subprocess_died_pre_write":
        notes_parts.append(
            "sweep halted: subprocess died before writing per-point JSON; "
            "termination_reason inferred as 'incomplete'"
        )

    manifest_path = aggregate_sweep_directory(
        sweep_dir=sweep_dir,
        diagnostic_run_id=diagnostic_run_id,
        inputs=inputs,
        calibration_gate_pct=args.calibration_gate_pct,
        device=args.device,
        notes="; ".join(notes_parts),
        total_eval_wall_time_seconds=total_wall,
        expected_b_mse_grid=b_mse_grid,
        continue_past_ceiling=args.continue_past_ceiling,
    )

    # Read back the manifest for the closing summary
    manifest = json.loads(manifest_path.read_text())
    rec = manifest["recommended_operating_point"]
    print(
        f"\n[r12_orchestrator] === ORCHESTRATED SWEEP COMPLETE ===\n"
        f"[r12_orchestrator] termination: {manifest['termination']['terminated_reason']}\n"
        f"[r12_orchestrator] points launched: {points_launched} / {len(b_mse_grid)}\n"
        f"[r12_orchestrator] total wall (sum of subprocesses): "
        f"{total_wall:.1f}s ({total_wall / 60:.1f} min)\n"
        f"[r12_orchestrator] recommended_operating_point: "
        f"b_mse={rec.get('b_mse')}, headroom={rec.get('ppl_headroom')}\n"
        f"[r12_orchestrator] manifest: {manifest_path}",
        flush=True,
    )
    return manifest_path


def main() -> None:
    p = argparse.ArgumentParser(
        description="R12 v1.2 per-point process-split orchestrator. "
                    "Runs each b_mse point in a fresh Python subprocess to "
                    "isolate MPS allocator state (lesson 16 / libdispatch "
                    "mitigation)."
    )
    # Mirror tern_r12_sweep.py's required + defaultable args
    p.add_argument("--model-id", required=True)
    p.add_argument("--baseline-run-id", required=True)
    p.add_argument("--baseline-ppl", required=True, type=float)
    p.add_argument("--b-mse-values", default=tern_r12_sweep.DEFAULT_B_MSE_VALUES)
    p.add_argument("--num-sequences", default=tern_r12_sweep.DEFAULT_NUM_SEQUENCES,
                   type=int)
    p.add_argument("--ar-seq-len", default=tern_r12_sweep.DEFAULT_AR_SEQ_LEN,
                   type=int)
    p.add_argument("--device", default=tern_r12_sweep.DEFAULT_DEVICE,
                   choices=["mps", "cpu"])
    p.add_argument("--hook-spec", default=tern_r12_sweep.DEFAULT_HOOK_SPEC,
                   choices=["uniform", "mixed"])
    p.add_argument("--output-dir", default=tern_r12_sweep.DEFAULT_OUTPUT_DIR)
    p.add_argument("--calibration-gate-pct",
                   default=tern_r12_sweep.DEFAULT_CALIBRATION_GATE_PCT, type=float)
    p.add_argument("--ceiling-multiplier",
                   default=tern_r12_sweep.DEFAULT_CEILING_MULTIPLIER, type=float)
    p.add_argument("--continue-past-ceiling", action="store_true",
                   default=tern_r12_sweep.DEFAULT_CONTINUE_PAST_CEILING)
    p.add_argument("--seed", default=tern_r12_sweep.DEFAULT_SEED, type=int)
    p.add_argument("--notes", default="")
    p.add_argument("--smoke", action="store_true",
                   help="Reduced-param smoke mode: b_mse=[4,2], N=2, L=128")
    p.add_argument("--python", default=None,
                   help="Python interpreter for subprocesses (default: sys.executable)")

    args = p.parse_args()

    if args.smoke:
        args.b_mse_values = ",".join(str(b) for b in SMOKE_B_MSE_GRID)
        args.num_sequences = SMOKE_NUM_SEQUENCES
        args.ar_seq_len = SMOKE_AR_SEQ_LEN
        if not args.notes:
            args.notes = ("R12 v1.2 per-point process-split orchestrator "
                          "smoke mode")
        print(
            f"[r12_orchestrator] SMOKE MODE: b_mse={args.b_mse_values}, "
            f"N={args.num_sequences}, L={args.ar_seq_len}",
            flush=True,
        )

    run_orchestrated_sweep(args)


if __name__ == "__main__":
    main()
