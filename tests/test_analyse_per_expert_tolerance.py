"""Synthetic tests for per-expert tolerance analysis script.

Tests both balanced and imbalanced group cases. Uses mock manifest entries
(no actual .tern-model file required) to verify:
- Group classification correctness
- Statistical computation produces expected directional results
- Plot generation completes without error
- JSON output schema parses cleanly
- Mann-Whitney robustness to unequal group sizes (imbalanced case)

Copyright (c) 2026 Gamma Seeds Pte Ltd. All rights reserved.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

# Ensure benchmarks/ analysis script is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
import analyse_per_expert_tolerance as ana  # noqa: E402


def _make_manifest_entries(layers: list[dict]) -> list[dict]:
    """Build a manifest entry list with name + dtype=ternary2 + sparsity."""
    return [
        {"name": layer["name"], "dtype": "ternary2",
         "sparsity": layer["sparsity"], "alpha": layer.get("alpha", 1.0)}
        for layer in layers
    ]


def _build_balanced_synthetic(seed: int = 42) -> list[dict]:
    """Balanced groups: 16 expert weights vs 6 dense vs 4 attention."""
    rng = np.random.default_rng(seed)
    entries: list[dict] = []
    # 2 MoE layers × 4 experts × 2 projections = 16 expert weights, high sparsity
    for layer_idx in range(2):
        for expert_idx in range(4):
            for proj in ("gate_up_proj", "down_proj"):
                entries.append({
                    "name": f"model.layers.{layer_idx}.experts.{expert_idx}.{proj}.weight",
                    "sparsity": float(rng.uniform(0.6, 0.9)),
                })
    # 2 dense layers × 3 mlp projections = 6 dense weights, low sparsity
    for layer_idx in range(2, 4):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            entries.append({
                "name": f"model.layers.{layer_idx}.mlp.{proj}.weight",
                "sparsity": float(rng.uniform(0.2, 0.4)),
            })
    # 4 attention weights in layer 0, low sparsity
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        entries.append({
            "name": f"model.layers.0.self_attn.{proj}.weight",
            "sparsity": float(rng.uniform(0.2, 0.4)),
        })
    return _make_manifest_entries(entries)


def _build_imbalanced_synthetic(seed: int = 43) -> list[dict]:
    """Realistic ratios: 100 expert weights vs 10 dense vs 10 attention.

    Mirrors the gemopus-4-26B-A4B scale ratio (7680 expert vs much smaller
    dense + attention populations in real compression). Verifies Mann-Whitney
    is robust to unequal group sizes in practice."""
    rng = np.random.default_rng(seed)
    entries: list[dict] = []
    # 5 MoE layers × 10 experts × 2 projections = 100 expert weights
    for layer_idx in range(5):
        for expert_idx in range(10):
            for proj in ("gate_up_proj", "down_proj"):
                entries.append({
                    "name": f"model.layers.{layer_idx}.experts.{expert_idx}.{proj}.weight",
                    "sparsity": float(rng.uniform(0.6, 0.9)),
                })
    # 10 dense MLP weights (4 layers × ~3, capped at 10)
    dense_count = 0
    for layer_idx in range(5, 9):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            if dense_count >= 10:
                break
            entries.append({
                "name": f"model.layers.{layer_idx}.mlp.{proj}.weight",
                "sparsity": float(rng.uniform(0.2, 0.4)),
            })
            dense_count += 1
        if dense_count >= 10:
            break
    # 10 attention weights (~2.5 layers' worth)
    attn_count = 0
    for layer_idx in range(3):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if attn_count >= 10:
                break
            entries.append({
                "name": f"model.layers.{layer_idx}.self_attn.{proj}.weight",
                "sparsity": float(rng.uniform(0.2, 0.4)),
            })
            attn_count += 1
        if attn_count >= 10:
            break
    return _make_manifest_entries(entries)


def _run_analysis(entries: list[dict], out_dir: Path) -> dict:
    """Run the analysis pipeline against in-memory entries; return JSON output."""
    records = ana.load_records_from_manifest_entries(entries, "gemma4_moe")

    expert_stats = ana.compute_group_stats(records, "expert_ffn")
    dense_stats = ana.compute_group_stats(records, "dense_ffn")
    attn_stats = ana.compute_group_stats(records, "attention")

    cmp_primary = ana.run_comparison(records, "expert_ffn", "dense_ffn")
    per_layer = ana.per_layer_breakdown(records, "expert_ffn", "dense_ffn")

    out_dir.mkdir(parents=True, exist_ok=True)
    ana.write_json(records, [expert_stats, dense_stats, attn_stats],
                   [cmp_primary], per_layer, None,
                   out_dir / "result.json")
    ana.plot_distribution_overlay(expert_stats, dense_stats,
                                  out_dir / "dist.png")
    ana.plot_per_layer_scatter(per_layer, "expert_ffn", "dense_ffn",
                               out_dir / "scatter.png")
    ana.plot_per_expert_heatmap(records, out_dir / "heatmap.png")
    ana.plot_external_reference(expert_stats.median, dense_stats.median,
                                None, out_dir / "ext_ref.png")
    ana.write_summary_table([cmp_primary], None, out_dir / "summary.md")

    with open(out_dir / "result.json") as f:
        return json.load(f)


class TestSyntheticBalanced:
    """Balanced groups: high-sparsity experts vs low-sparsity dense + attention."""

    def test_balanced_synthetic_groups(self):
        entries = _build_balanced_synthetic()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "balanced"
            result = _run_analysis(entries, out_dir)
            # All output artefacts exist + non-empty
            for fname in ("result.json", "dist.png", "scatter.png",
                          "heatmap.png", "ext_ref.png", "summary.md"):
                p = out_dir / fname
                assert p.exists(), f"missing: {p}"
                assert p.stat().st_size > 0, f"empty: {p}"

        # Group counts (16 expert + 6 dense + 4 attention)
        by_group = {g["name"]: g["n"] for g in result["group_stats"]}
        assert by_group["expert_ffn"] == 16
        assert by_group["dense_ffn"] == 6
        assert by_group["attention"] == 4

        # Primary comparison: expert >> dense
        primary = result["comparisons"][0]
        assert primary["group_a"] == "expert_ffn"
        assert primary["group_b"] == "dense_ffn"
        assert primary["median_diff"] > 0.3, (
            f"expected expert >> dense; got median_diff={primary['median_diff']}")
        assert primary["mwu_p"] < 0.001
        assert primary["rank_biserial"] > 0.8
        assert primary["cohens_d"] > 1.5


class TestSyntheticImbalanced:
    """Realistic scale ratios: 100 expert vs 10 dense vs 10 attention.

    Mann-Whitney is robust to unequal sizes; verify behaviour preserves
    direction + significance despite the ~10× imbalance."""

    def test_imbalanced_synthetic_groups(self):
        entries = _build_imbalanced_synthetic()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "imbalanced"
            result = _run_analysis(entries, out_dir)
            for fname in ("result.json", "dist.png", "scatter.png",
                          "heatmap.png", "ext_ref.png", "summary.md"):
                p = out_dir / fname
                assert p.exists()
                assert p.stat().st_size > 0

        by_group = {g["name"]: g["n"] for g in result["group_stats"]}
        assert by_group["expert_ffn"] == 100
        assert by_group["dense_ffn"] == 10
        assert by_group["attention"] == 10

        primary = result["comparisons"][0]
        assert primary["n_a"] == 100
        assert primary["n_b"] == 10
        assert primary["median_diff"] > 0.3
        assert primary["mwu_p"] < 0.001, (
            f"Mann-Whitney should detect difference despite imbalance; "
            f"got p={primary['mwu_p']}")
        assert primary["rank_biserial"] > 0.8


class TestClassification:
    def test_classify_weight_pattern_dispatch(self):
        patterns = ana.MODEL_GROUP_PATTERNS["gemma4_moe"]
        cases = [
            ("model.language_model.layers.5.experts.42.gate_up_proj.weight", "expert_ffn"),
            ("model.language_model.layers.5.experts.0.down_proj.weight", "expert_ffn"),
            ("model.language_model.layers.7.mlp.gate_proj.weight", "dense_ffn"),
            ("model.language_model.layers.7.mlp.up_proj.weight", "dense_ffn"),
            ("model.language_model.layers.7.mlp.down_proj.weight", "dense_ffn"),
            ("model.language_model.layers.10.self_attn.q_proj.weight", "attention"),
            ("model.language_model.layers.10.self_attn.k_proj.weight", "attention"),
            ("model.language_model.layers.10.self_attn.v_proj.weight", "attention"),
            ("model.language_model.layers.10.self_attn.o_proj.weight", "attention"),
            ("model.language_model.embed_tokens.weight", "other_protected"),
            ("model.language_model.norm.weight", "other_protected"),
            ("model.language_model.layers.5.input_layernorm.weight", "other_protected"),
            ("model.language_model.layers.5.router.proj.weight", "other_protected"),
        ]
        for name, expected in cases:
            actual = ana.classify_weight(name, patterns)
            assert actual == expected, (
                f"{name!r}: expected {expected}, got {actual}")

    def test_layer_idx_extraction(self):
        assert ana.parse_layer_idx(
            "model.layers.5.experts.0.gate_up_proj.weight") == 5
        assert ana.parse_layer_idx(
            "model.layers.42.self_attn.q_proj.weight") == 42
        assert ana.parse_layer_idx("model.embed_tokens.weight") is None

    def test_expert_idx_extraction(self):
        assert ana.parse_expert_idx(
            "model.layers.5.experts.42.gate_up_proj.weight") == 42
        assert ana.parse_expert_idx(
            "model.layers.5.experts.0.down_proj.weight") == 0
        assert ana.parse_expert_idx(
            "model.layers.5.mlp.gate_proj.weight") is None

    def test_skip_non_ternary_entries(self):
        """float16 and int4_block32 entries should be excluded from records."""
        entries = [
            {"name": "model.layers.0.experts.0.gate_up_proj.weight",
             "dtype": "ternary2", "sparsity": 0.7, "alpha": 1.0},
            {"name": "model.embed_tokens.weight",
             "dtype": "float16", "sparsity": 0.0, "alpha": 0.0},
            {"name": "model.layers.1.experts.0.gate_up_proj.weight",
             "dtype": "int4_block32", "sparsity": 0.0, "alpha": 0.0},
        ]
        records = ana.load_records_from_manifest_entries(entries, "gemma4_moe")
        assert len(records) == 1
        assert records[0].name.endswith("experts.0.gate_up_proj.weight")
        assert records[0].sparsity == 0.7
