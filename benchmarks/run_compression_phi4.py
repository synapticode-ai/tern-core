"""
Phi-4 dense compression — Thursday 2026-05-07 cross-architecture sprint.

Uses the new Phi3Adapter (PR #16) — declares Phi3ForCausalLM allow-list
(Phi-4 retains Phi3 architecture class). Classification logic mirrors
LlamaAdapter, validated by April 2026 prior production compression of
microsoft/phi-4 (160 ternary entries, 83 FP16, file size 6,835 MB —
artefact at /Volumes/Syn Archive/models/compressed/phi4-14b/).

Fused QKV (qkv_proj.weight) and fused gate+up (gate_up_proj.weight)
treated as single ternary tensors — matches April production behaviour
and forward-pass shape (one nn.Linear per fused tensor).

Configurational fidelity assertion derived from HF config (FLAT layout
— no text_config nesting). For Phi-4 (40 layers, dense, 4 ternary
projections per layer): 40 × 4 = 160 expected ternary entries. This
matches April production exactly.

Single-model template form copied from
``run_compression_gemma4_31b.py`` with Phi-4-specific constants +
formula.

Usage:
    python benchmarks/run_compression_phi4.py

Wall-clock estimate: ~25-40 min on M4 Pro. Phi-4 is ~14B (smaller
than 26B-A4B's 26B); only 160 ternary entries (vs 7,875 for 26B-A4B);
per-tensor processing should dominate but with far fewer entries.

Copyright (c) 2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from terncore.convert import full_convert


# ── Sprint constants ────────────────────────────────────────────────

SOURCE_HF_ID = "microsoft/phi-4"
OUTPUT_DIR = Path(
    "/Volumes/Syn Archive/models/compressed/phi-4/"
    "phi4_14b_ternary_v0.1.1.tern-model"
)
ADAPTER_NAME = "phi3"
THRESHOLD = 0.7
# Smaller absolute tolerance for Phi-4 (smaller total count means ±65
# would be ~40% of expected — too loose). ±10 = ~6%.
TOLERANCE = 10


# ── Architectural ground truth ──────────────────────────────────────


def compute_expected_ternary_count(text_config: dict) -> int:
    """Architectural ground truth for Phi-4 (dense, fused projections).

    Per-layer breakdown:
    - 1 fused QKV (qkv_proj.weight)
    - 1 separate o_proj.weight
    - 1 fused gate_up (gate_up_proj.weight)
    - 1 separate down_proj.weight
    = 4 ternary-eligible tensors per layer

    For Phi-4 (40 layers): 40 × 4 = 160. Matches April 2026
    production compression exactly (verified in
    /Volumes/Syn Archive/models/compressed/phi4-14b/
    phi4_14b_ternary_v0.1.0_conversion_report.json — ternary_layers: 160).
    """
    num_layers = text_config["num_hidden_layers"]
    per_layer = 4  # qkv_proj + o_proj + gate_up_proj + down_proj
    return num_layers * per_layer


def assert_configurational_fidelity(
    report: dict, expected: int, tolerance: int = TOLERANCE,
) -> None:
    """Hard guard on ternary_layers count vs HF-derived expectation."""
    actual = report["ternary_layers"]
    if not (expected - tolerance <= actual <= expected + tolerance):
        raise ValueError(
            f"Configurational fidelity violation: expected ~{expected} "
            f"(±{tolerance}), got {actual}. Full report: {report}"
        )


def load_text_config_via_hf(hf_id: str) -> dict:
    """Resolve text_config via huggingface_hub (cache-aware).

    Falls back to top-level config when text_config isn't nested
    (Phi-4's config layout is flat).
    """
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
        f"\n[{started_at}] Run starting: phi4_14b_dense\n"
        f"  source: {SOURCE_HF_ID}\n"
        f"  output: {OUTPUT_DIR}\n"
        f"  adapter: {ADAPTER_NAME}",
        flush=True,
    )

    text_config = load_text_config_via_hf(SOURCE_HF_ID)
    expected = compute_expected_ternary_count(text_config)
    print(
        f"  Expected ternary count: ~{expected} (±{TOLERANCE}; "
        f"{text_config['num_hidden_layers']} layers × 4 ternary "
        f"per layer — matches April 2026 production validation)",
        flush=True,
    )

    report = full_convert(
        model_id=SOURCE_HF_ID,
        adapter_name=ADAPTER_NAME,
        output_dir=str(OUTPUT_DIR),
        threshold=THRESHOLD,
        verbose=True,
    )

    wall_clock = time.perf_counter() - t_start
    finished_at = datetime.now(timezone.utc).isoformat()
    assert_configurational_fidelity(report, expected)
    write_run_summary(
        report, expected, OUTPUT_DIR, "phi4_14b_dense",
        started_at, finished_at, wall_clock,
    )
    print(
        f"\n[{finished_at}] Run complete: phi4_14b_dense "
        f"({wall_clock / 60:.1f} min wall)\n"
        f"  ternary_layers: {report['ternary_layers']} "
        f"(expected ~{expected}, ±{TOLERANCE}) — PASS",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
