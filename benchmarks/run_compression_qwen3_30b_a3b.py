"""
Qwen3-30B-A3B MoE compression — Thursday 2026-05-07 cross-architecture sprint.

Uses the new Qwen3MoeAdapter (PR #16) — declares Qwen3MoeForCausalLM
allow-list and handles per-expert 2-D indexed tensors directly (no
expand_stacked needed since Qwen3 stores experts as separate
``mlp.experts.K.{gate,up,down}_proj.weight`` entries in safetensors,
unlike Gemma 4's 3-D stacked pattern).

Configurational fidelity assertion derived from HF config per Q5
design — Qwen3's config is FLAT (no ``text_config`` nesting like
Gemma 4's multimodal config); the helper handles both layouts via
``cfg.get("text_config", cfg)`` fallback.

For Qwen3-30B-A3B (128 experts, 48 layers, no parallel dense MLP):
48 × (128 × 3 + 4) = 48 × 388 = 18,624 expected ternary entries.

Single-model template form copied from
``run_compression_gemma4_31b.py`` with Qwen3-specific constants +
formula. Future single-model MoE compressions (different num_experts,
num_layers) can copy this script and modify the constants section.

Usage:
    python benchmarks/run_compression_qwen3_30b_a3b.py

Wall-clock estimate: ~45-90 min on M4 Pro. Qwen3 has 18,624 ternary
entries (vs Gemma 4 26B-A4B's ~7,875), so per-tensor processing
dominates wall-clock — likely longer than 26B-A4B's 42 min.

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

SOURCE_HF_ID = "Qwen/Qwen3-30B-A3B"
OUTPUT_DIR = Path(
    "/Volumes/Syn Archive/models/compressed/qwen3-30b-a3b/"
    "qwen3_30b_a3b_ternary_v0.1.0.tern-model"
)
ADAPTER_NAME = "qwen3_moe"
THRESHOLD = 0.7
TOLERANCE = 65


# ── Architectural ground truth ──────────────────────────────────────


def compute_expected_ternary_count(text_config: dict) -> int:
    """Architectural ground truth for Qwen3MoE ternary-eligible count.

    Per-layer breakdown (no parallel dense MLP — pure MoE FFN):
    - ``num_experts × 3`` per-expert weights (gate_proj, up_proj,
      down_proj — separate tensors per expert, NOT fused like Gemma 4)
    - 4 attention weights (q, k, v, o; no sliding/full split for
      Qwen3 — uniform attention across layers)

    Generalises across Qwen3MoE family by reading num_experts +
    num_hidden_layers from config. For 30B-A3B (128 experts, 48
    layers): 48 * (128*3 + 4) = 48 * 388 = 18,624.
    """
    num_experts = text_config["num_experts"]
    num_layers = text_config["num_hidden_layers"]
    per_layer_expert = 3 * num_experts
    per_layer_attn = 4
    return num_layers * (per_layer_expert + per_layer_attn)


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

    Falls back to top-level config when ``text_config`` isn't nested
    (Qwen3's config layout, vs Gemma 4's nested multimodal layout).
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
        f"\n[{started_at}] Run starting: qwen3_30b_a3b_moe\n"
        f"  source: {SOURCE_HF_ID}\n"
        f"  output: {OUTPUT_DIR}\n"
        f"  adapter: {ADAPTER_NAME}",
        flush=True,
    )

    text_config = load_text_config_via_hf(SOURCE_HF_ID)
    expected = compute_expected_ternary_count(text_config)
    print(
        f"  Expected ternary count: ~{expected} (±{TOLERANCE}; "
        f"{text_config['num_experts']} experts × 3 + 4 attn = "
        f"{3 * text_config['num_experts'] + 4} per layer × "
        f"{text_config['num_hidden_layers']} layers)",
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
        report, expected, OUTPUT_DIR, "qwen3_30b_a3b_moe",
        started_at, finished_at, wall_clock,
    )
    print(
        f"\n[{finished_at}] Run complete: qwen3_30b_a3b_moe "
        f"({wall_clock / 60:.1f} min wall)\n"
        f"  ternary_layers: {report['ternary_layers']} "
        f"(expected ~{expected}, ±{TOLERANCE}) — PASS",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
