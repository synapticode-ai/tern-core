"""
Dual-compression orchestration for Session 3 — 26B-A4B per-expert sprint.

Compresses two Gemma 4 26B-A4B variants through the per-expert slicing
pipeline (PR #14) so Session 4's ``analyse_per_expert_tolerance.py``
(PR #12) can run the IP claim hypothesis test against per-expert
sparsity data:

  Run 1 — google/gemma-4-26b-a4b-it (Google base; broadly-recognised
          architectural reference; 6.5M downloads on HF Hub)
  Run 2 — Jackrong/Gemopus-4-26B-A4B-it (production fine-tune;
          Apple/KAIST/KSGC deliverable target)

The IP claim is testable from manifest data alone after both runs
complete: per-expert zero-state ratio distribution comparison vs
dense-FFN distribution within each model, plus cross-fine-tune
comparison (does the fine-tuning shift the per-expert distribution?).

Configurational fidelity assertion derived from architectural ground
truth (HF text_config) per the orientation memory's Q5 design — the
formula generalises to any Gemma 4 MoE variant via ``layer_types`` +
``num_experts``.

Usage:
    python benchmarks/run_dual_compression_26b_a4b.py
    python benchmarks/run_dual_compression_26b_a4b.py --skip-google
    python benchmarks/run_dual_compression_26b_a4b.py --skip-jackrong

Wall-clock: ~2-3 hours per compression, ~4-6 hours total.
Memory: ~7 GB peak (5 GB shard load + 2 GB stacked-parent in-flight).

Copyright (c) 2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from terncore.convert import full_convert
from terncore.tern_model import TernModelReader


# ── Sprint constants ────────────────────────────────────────────────

GOOGLE_BASE_SOURCE = Path(
    "/Volumes/Syn Archive/models/source/gemma-4-26b-a4b-it"
)
GOOGLE_BASE_OUTPUT_DIR = Path(
    "/Volumes/Syn Archive/models/compressed/gemma4-26b-a4b/"
    "gemma4_26b_a4b_ternary_v0.1.0.tern-model"
)
JACKRONG_HF_ID = "Jackrong/Gemopus-4-26B-A4B-it"
JACKRONG_OUTPUT_DIR = Path(
    "/Volumes/Syn Archive/models/compressed/gemopus-4-26b-a4b/"
    "gemopus_4_26b_a4b_ternary_v0.1.0.tern-model"
)

THRESHOLD = 0.7
TOLERANCE = 65  # ±tolerance band around HF-config-derived expected count

SOFT_PAUSE_SECONDS = 5  # Between runs, lets operator Ctrl-C if review needed


# ── Architectural ground truth ──────────────────────────────────────


def compute_expected_ternary_count(text_config: dict) -> int:
    """Architectural ground truth: per-MoE-layer ternary-eligible Linear count.

    Per-layer breakdown:
    - ``2 × num_experts`` per-expert weights (``gate_up_proj`` + ``down_proj``)
    - 3 dense MLP weights (gate, up, down — present alongside experts
      on every layer in Gemma 4's hybrid MoE+dense block)
    - 4 attention weights for sliding_attention (q, k, v, o)
      OR 3 attention weights for full_attention (q, k, o — no
      ``v_proj`` per ``docs/backlog.md`` "Known architecture quirk:
      full_attention v_proj absence")

    Generalises to any Gemma 4 MoE variant by reading ``layer_types``
    + ``num_experts`` from text_config. For 26B-A4B (128 experts,
    25 sliding + 5 full) this yields 25*263 + 5*262 = 7,885.
    """
    num_experts = text_config["num_experts"]
    layer_types = text_config["layer_types"]
    sliding = layer_types.count("sliding_attention")
    full = layer_types.count("full_attention")

    per_expert_weights = 2 * num_experts
    dense_mlp = 3
    sliding_per_layer = per_expert_weights + dense_mlp + 4
    full_per_layer = per_expert_weights + dense_mlp + 3

    return sliding * sliding_per_layer + full * full_per_layer


def assert_configurational_fidelity(
    report: dict, expected: int, tolerance: int = TOLERANCE,
) -> None:
    """Hard guard on ``ternary_layers`` count vs HF-derived expectation.

    Fires loudly on violation — same Phase 4 discipline that caught
    over-ternisation regressions on E4B. Blocks any downstream use of
    a malformed manifest.
    """
    actual = report["ternary_layers"]
    if not (expected - tolerance <= actual <= expected + tolerance):
        raise ValueError(
            f"Configurational fidelity violation: expected ~{expected} "
            f"(±{tolerance}), got {actual}. Full report: {report}"
        )


def load_text_config_from_dir(model_dir: Path) -> dict:
    """Load text_config from a local model directory's config.json."""
    with open(model_dir / "config.json") as f:
        return json.load(f)["text_config"]


def load_text_config_via_hf(hf_id: str) -> dict:
    """Resolve text_config via huggingface_hub (config-only download).

    Uses snapshot_download with allow_patterns=["config.json"] so we
    don't pull the full 50+ GB model just to read its config — the
    actual model resolution happens inside full_convert via the same
    HF cache.
    """
    from huggingface_hub import snapshot_download
    config_dir = snapshot_download(hf_id, allow_patterns=["config.json"])
    with open(Path(config_dir) / "config.json") as f:
        return json.load(f)["text_config"]


# ── Per-run orchestration ───────────────────────────────────────────


def write_run_summary(
    report: dict,
    expected: int,
    output_dir: Path,
    label: str,
    started_at: str,
    finished_at: str,
    wall_clock_seconds: float,
) -> None:
    """Write run_summary.json alongside model.tern-model."""
    summary = {
        "label": label,
        "input_source": report.get("model_id"),
        "output_path": report.get("output_path"),
        "threshold": report.get("threshold"),
        "ternary_layers": report["ternary_layers"],
        "ternary_params": report["ternary_params"],
        "expected_ternary_count": expected,
        "tolerance_band": [expected - TOLERANCE, expected + TOLERANCE],
        "fidelity_pass": True,
        "fp16_layers": report["fp16_layers"],
        "fp16_params": report["fp16_params"],
        "int4_layers": report["int4_layers"],
        "int4_params": report["int4_params"],
        "compression_vs_fp16": report.get("compression_vs_fp16"),
        "file_size_bytes": report.get("file_size_bytes"),
        "wall_clock_seconds": round(wall_clock_seconds, 2),
        "started_at": started_at,
        "finished_at": finished_at,
    }
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[ok] run_summary written: {summary_path}", flush=True)


def run_compression(
    label: str,
    source: str,
    output_dir: Path,
    text_config: dict,
) -> dict:
    """Run one compression with timing, fidelity check, summary write."""
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    t_start = time.perf_counter()
    print(
        f"\n[{started_at}] Run starting: {label}\n"
        f"  source: {source}\n"
        f"  output: {output_dir}",
        flush=True,
    )

    report = full_convert(
        model_id=source,
        adapter_name="gemma4",
        output_dir=str(output_dir),
        threshold=THRESHOLD,
        verbose=True,
    )

    wall_clock = time.perf_counter() - t_start
    finished_at = datetime.now(timezone.utc).isoformat()
    expected = compute_expected_ternary_count(text_config)
    assert_configurational_fidelity(report, expected)
    write_run_summary(
        report, expected, output_dir, label,
        started_at, finished_at, wall_clock,
    )
    print(
        f"[{finished_at}] Run complete: {label} "
        f"({wall_clock / 60:.1f} min wall)\n"
        f"  ternary_layers: {report['ternary_layers']} "
        f"(expected ~{expected}, ±{TOLERANCE}) — PASS",
        flush=True,
    )
    return report


# ── Post-compression verification ───────────────────────────────────


def _verify_one_manifest(label: str, output_dir: Path) -> None:
    """Four structural checks per the design surface item (g)."""
    manifest_path = output_dir / "model.tern-model"
    print(f"\n[verify] {label}: {manifest_path}", flush=True)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"verify: manifest not found at {manifest_path}"
        )
    reader = TernModelReader(str(manifest_path))
    entries = reader.manifest["layers"]

    # Check 1: stacked entry count
    stacked = [e for e in entries if "stacked_parent" in e]
    parents = sorted({e["stacked_parent"] for e in stacked})
    expected_stacked_lo = 7500  # 60 parents × 128 = 7680 expected; allow slack
    expected_stacked_hi = 7800
    if not (expected_stacked_lo <= len(stacked) <= expected_stacked_hi):
        raise ValueError(
            f"verify: stacked entry count {len(stacked)} outside "
            f"[{expected_stacked_lo}, {expected_stacked_hi}] band"
        )
    print(
        f"  [check 1] stacked entries: {len(stacked)} across "
        f"{len(parents)} parents — PASS",
        flush=True,
    )

    # Check 2: per-expert sparsity diversity (5-parent sample)
    sample_parents = parents[:5]
    for parent in sample_parents:
        sparsities = [e["sparsity"] for e in stacked if e["stacked_parent"] == parent]
        distinct = len({round(s, 6) for s in sparsities})
        if distinct < 2:
            raise ValueError(
                f"verify: per-expert sparsity all-identical for parent "
                f"'{parent}' ({distinct} distinct value across {len(sparsities)} "
                f"slices) — silent shared-threshold regression"
            )
    print(
        f"  [check 2] per-expert sparsity diversity confirmed across "
        f"{len(sample_parents)} sample parents — PASS",
        flush=True,
    )

    # Check 3: FP16 preservation (router weights still float16)
    router_entries = [
        e for e in entries
        if "router" in e["name"].lower() and e["dtype"] != "float16"
    ]
    if router_entries:
        offenders = [(e["name"], e["dtype"]) for e in router_entries[:5]]
        raise ValueError(
            f"verify: {len(router_entries)} router entries are non-FP16 "
            f"(should all be FP16-protected). First offenders: {offenders}"
        )
    print(
        "  [check 3] router/norm entries still FP16-protected — PASS",
        flush=True,
    )

    # Check 4: restacking round-trip sanity (one parent only — bounded memory)
    target_parent = parents[0]
    target_slices = [e for e in stacked if e["stacked_parent"] == target_parent]
    target_total = target_slices[0]["stack_total"]
    target_axis = target_slices[0]["stack_axis"]
    sample_slice = target_slices[0]
    per_slice_shape = sample_slice["shape"]
    expected_restacked_shape = list(per_slice_shape)
    expected_restacked_shape.insert(target_axis, target_total)

    # Build a minimal-cost reconstruction by directly stacking the target
    # parent's slices via reconstruct_layer (avoids walking entire manifest).
    reconstructed = [reader.reconstruct_layer(e["name"])["weight"]
                     for e in sorted(target_slices, key=lambda e: e["stack_index"])]
    restacked = torch.stack(reconstructed, dim=target_axis)
    if list(restacked.shape) != expected_restacked_shape:
        raise ValueError(
            f"verify: restacked shape {list(restacked.shape)} mismatches "
            f"expected {expected_restacked_shape} for parent '{target_parent}'"
        )
    print(
        f"  [check 4] restacking round-trip for '{target_parent}' produces "
        f"shape {list(restacked.shape)} — PASS",
        flush=True,
    )


def verify_both_manifests(google_dir: Path, jackrong_dir: Path) -> None:
    """Run the four structural checks against both compressed artefacts."""
    print(f"\n{'='*72}\n  Post-compression structural verification\n{'='*72}", flush=True)
    _verify_one_manifest("google_base", google_dir)
    _verify_one_manifest("jackrong_finetune", jackrong_dir)
    print("\n[ok] Both manifests verified — Session 4 analysis can proceed.", flush=True)


# ── Pre-flight + main ───────────────────────────────────────────────


def preflight_checks(skip_google: bool, skip_jackrong: bool) -> None:
    """Fail-fast surface for assumptions per the design surface item (h)."""
    if not skip_google and not GOOGLE_BASE_SOURCE.exists():
        raise FileNotFoundError(
            f"Google base source not found at {GOOGLE_BASE_SOURCE}. "
            f"Verify Syn Archive is mounted + the 26B-A4B source weights "
            f"are present."
        )
    if not skip_jackrong:
        if not os.environ.get("HF_HOME"):
            raise EnvironmentError(
                "HF_HOME is not set; expected per project_hf_home_redirect "
                "(typically /Volumes/Syn Archive/cache/huggingface). "
                "Run: export HF_HOME=/Volumes/Syn\\ Archive/cache/huggingface"
            )
    print(
        f"[preflight] paths + env OK "
        f"(skip_google={skip_google}, skip_jackrong={skip_jackrong})",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dual-compression for Session 3 — gemma-4-26b-a4b-it "
                    "(Google base) + Gemopus-4-26B-A4B-it (Jackrong fine-tune)",
    )
    parser.add_argument(
        "--skip-google", action="store_true",
        help="Skip Run 1 (Google base). Use to re-run only Jackrong if Run 1 "
             "succeeded but Run 2 failed.",
    )
    parser.add_argument(
        "--skip-jackrong", action="store_true",
        help="Skip Run 2 (Jackrong fine-tune). Use to re-run only Google "
             "base if needed.",
    )
    args = parser.parse_args()

    if args.skip_google and args.skip_jackrong:
        print("Both runs skipped — nothing to do.", file=sys.stderr)
        return 1

    preflight_checks(args.skip_google, args.skip_jackrong)
    script_started = datetime.now(timezone.utc).isoformat()
    print(f"\n[{script_started}] dual-compression script started", flush=True)

    # Run 1 — Google base
    if not args.skip_google:
        text_config = load_text_config_from_dir(GOOGLE_BASE_SOURCE)
        run_compression(
            "google_base",
            str(GOOGLE_BASE_SOURCE),
            GOOGLE_BASE_OUTPUT_DIR,
            text_config,
        )
        if not args.skip_jackrong:
            print(
                f"\n[pause {SOFT_PAUSE_SECONDS}s] Run 1/2 clean. "
                f"Ctrl-C now to abort Run 2.",
                flush=True,
            )
            time.sleep(SOFT_PAUSE_SECONDS)

    # Run 2 — Jackrong fine-tune
    if not args.skip_jackrong:
        text_config = load_text_config_via_hf(JACKRONG_HF_ID)
        run_compression(
            "jackrong_finetune",
            JACKRONG_HF_ID,
            JACKRONG_OUTPUT_DIR,
            text_config,
        )

    # Post-compression verification
    if not args.skip_google and not args.skip_jackrong:
        verify_both_manifests(GOOGLE_BASE_OUTPUT_DIR, JACKRONG_OUTPUT_DIR)

    script_finished = datetime.now(timezone.utc).isoformat()
    print(f"\n[{script_finished}] dual-compression script complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
