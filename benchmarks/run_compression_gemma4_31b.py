"""
Gemma 4 31B dense compression — Thursday 2026-05-07 single-model run.

Uses existing Gemma4Adapter directly (verified during pre-flight Thursday AM:
31B is dense, num_experts=None, enable_moe_block=False, no per-expert
slicing needed). Configurational fidelity assertion derived from HF
text_config per Q5 design — same architectural-ground-truth pattern as
the dual-compression script for 26B-A4B, simplified for the dense case.

Single-model template form: future single-model compressions (Phi-4
dense, Qwen3-30B-A3B MoE, etc.) can copy and modify the constants
section + compute function for their architecture.

Usage:
    python benchmarks/run_compression_gemma4_31b.py

Wall-clock estimate: ~45-60 min on M4 Pro (60 layers vs 30 for 26B-A4B,
roughly 2x tensor count post-classification at 410 + protected; offset by
no per-expert slicing overhead).

Copyright (c) 2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from terncore.convert import full_convert
from terncore.tern_model import TernModelReader


# ── Sprint constants ────────────────────────────────────────────────

SOURCE_HF_ID = "mlx-community/gemma-4-31b-it-bf16"
OUTPUT_DIR = Path(
    "/Volumes/Syn Archive/models/compressed/gemma4-31b/"
    "gemma4_31b_ternary_v0.1.0.tern-model"
)
THRESHOLD = 0.7
TOLERANCE = 65  # ±tolerance band; matches dual-compression-script convention


# ── Architectural ground truth ──────────────────────────────────────


def compute_expected_ternary_count(text_config: dict) -> int:
    """Architectural ground truth for dense Gemma 4 ternary-eligible count.

    Per-layer breakdown for dense Gemma 4:
    - 3 dense MLP weights (gate_proj, up_proj, down_proj)
    - 4 attention weights for sliding_attention (q, k, v, o)
      OR 3 attention weights for full_attention (q, k, o — no v_proj
      per ``docs/backlog.md`` "Known architecture quirk: full_attention
      v_proj absence", confirmed for 31B in pre-flight inspection)

    Generalises across Gemma 4 dense variants by reading ``layer_types``
    from text_config. For 31B (60 layers: 50 sliding + 10 full):
    50*7 + 10*6 = 410.
    """
    layer_types = text_config["layer_types"]
    sliding = layer_types.count("sliding_attention")
    full = layer_types.count("full_attention")

    dense_mlp = 3
    sliding_per_layer = dense_mlp + 4
    full_per_layer = dense_mlp + 3

    return sliding * sliding_per_layer + full * full_per_layer


def assert_configurational_fidelity(
    report: dict, expected: int, tolerance: int = TOLERANCE,
) -> None:
    """Hard guard on ``ternary_layers`` count vs HF-derived expectation."""
    actual = report["ternary_layers"]
    if not (expected - tolerance <= actual <= expected + tolerance):
        raise ValueError(
            f"Configurational fidelity violation: expected ~{expected} "
            f"(±{tolerance}), got {actual}. Full report: {report}"
        )


def load_text_config_via_hf(hf_id: str) -> dict:
    """Resolve text_config via huggingface_hub (config-only, cache-aware)."""
    from huggingface_hub import snapshot_download
    config_dir = snapshot_download(hf_id, allow_patterns=["config.json"])
    with open(Path(config_dir) / "config.json") as f:
        cfg = json.load(f)
    return cfg.get("text_config", cfg)


def write_run_summary(
    report: dict,
    expected: int,
    output_dir: Path,
    label: str,
    started_at: str,
    finished_at: str,
    wall_clock_seconds: float,
) -> None:
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


# ── Main ────────────────────────────────────────────────────────────


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    t_start = time.perf_counter()
    print(
        f"\n[{started_at}] Run starting: gemma4_31b_dense\n"
        f"  source: {SOURCE_HF_ID}\n"
        f"  output: {OUTPUT_DIR}",
        flush=True,
    )

    text_config = load_text_config_via_hf(SOURCE_HF_ID)
    expected = compute_expected_ternary_count(text_config)
    print(
        f"  Expected ternary count: ~{expected} "
        f"(±{TOLERANCE}; layer_types: "
        f"{text_config['layer_types'].count('sliding_attention')} sliding + "
        f"{text_config['layer_types'].count('full_attention')} full)",
        flush=True,
    )

    report = full_convert(
        model_id=SOURCE_HF_ID,
        adapter_name="gemma4",
        output_dir=str(OUTPUT_DIR),
        threshold=THRESHOLD,
        verbose=True,
    )

    wall_clock = time.perf_counter() - t_start
    finished_at = datetime.now(timezone.utc).isoformat()
    assert_configurational_fidelity(report, expected)
    write_run_summary(
        report, expected, OUTPUT_DIR, "gemma4_31b_dense",
        started_at, finished_at, wall_clock,
    )
    print(
        f"\n[{finished_at}] Run complete: gemma4_31b_dense "
        f"({wall_clock / 60:.1f} min wall)\n"
        f"  ternary_layers: {report['ternary_layers']} "
        f"(expected ~{expected}, ±{TOLERANCE}) — PASS",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
