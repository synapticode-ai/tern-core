"""Per-expert ternary tolerance analysis for MoE compression artefacts.

Loads a .tern-model manifest, classifies weights into expert / dense / attention
groups via configurable name patterns, computes per-group sparsity (zero-state
ratio) statistics, runs Mann-Whitney U + t-test comparisons with effect sizes,
and produces 4 plots + JSON output + markdown summary table.

Tests the IP hypothesis from tern-core/CLAUDE.md: inactive expert weights
cluster toward zero-state at higher rates than dense layer weights within
the same MoE model.

Usage:
    python benchmarks/analyse_per_expert_tolerance.py \\
      --manifest /path/to/model.tern-model \\
      --arch gemma4_moe \\
      --output-dir benchmarks/<phase_dir>/ \\
      --llama70b-manifest /path/to/llama70b.tern-model  # optional external ref

Copyright (c) 2026 Gamma Seeds Pte Ltd. All rights reserved.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# ── Group classification patterns (extensible per architecture) ────────
# Add new architectures by mirroring the gemma4_moe shape.
# Patterns are anchored regexes matched via re.search against full weight names.
MODEL_GROUP_PATTERNS: dict[str, dict[str, list[str]]] = {
    "gemma4_moe": {
        "expert_ffn":  [r"\.experts\.\d+\.gate_up_proj\.weight$",
                        r"\.experts\.\d+\.down_proj\.weight$"],
        "dense_ffn":   [r"\.mlp\.gate_proj\.weight$",
                        r"\.mlp\.up_proj\.weight$",
                        r"\.mlp\.down_proj\.weight$"],
        "attention":   [r"\.self_attn\.[qkvo]_proj\.weight$"],
    },
    "llama_dense": {
        "expert_ffn":  [],  # Llama-style dense; no experts
        "dense_ffn":   [r"\.mlp\.gate_proj\.weight$",
                        r"\.mlp\.up_proj\.weight$",
                        r"\.mlp\.down_proj\.weight$"],
        "attention":   [r"\.self_attn\.[qkvo]_proj\.weight$"],
    },
    # Extend for new architectures by mirroring the gemma4_moe shape.
}

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
_EXPERT_RE = re.compile(r"\.experts\.(\d+)\.")


# ── Data classes ─────────────────────────────────────────────────────

@dataclass
class WeightRecord:
    name: str
    group: str  # expert_ffn / dense_ffn / attention / other_protected
    sparsity: float
    alpha: float
    layer_idx: Optional[int]
    expert_idx: Optional[int]


@dataclass
class GroupStats:
    name: str
    n: int
    median: float
    mean: float
    stddev: float
    distribution: list[float]


@dataclass
class ComparisonResult:
    group_a: str
    group_b: str
    n_a: int
    n_b: int
    median_a: float
    median_b: float
    median_diff: float
    mwu_u: float
    mwu_p: float
    rank_biserial: float
    t_p: float
    cohens_d: float


# ── Classification ────────────────────────────────────────────────────

def classify_weight(name: str, patterns: dict[str, list[str]]) -> str:
    """Return group label for a weight name, or 'other_protected' if no match."""
    for group_name, group_patterns in patterns.items():
        for pat in group_patterns:
            if re.search(pat, name):
                return group_name
    return "other_protected"


def parse_layer_idx(name: str) -> Optional[int]:
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def parse_expert_idx(name: str) -> Optional[int]:
    m = _EXPERT_RE.search(name)
    return int(m.group(1)) if m else None


# ── Loading ───────────────────────────────────────────────────────────

def load_records_from_manifest_entries(
    entries: list[dict],
    arch: str,
) -> list[WeightRecord]:
    """Pure function: turn manifest entries into WeightRecord objects.

    Skips non-ternary entries (float16, int4_block32 — different metric).
    Used directly by tests with mock entries; load_manifest_records wraps
    this with file I/O for production use.
    """
    if arch not in MODEL_GROUP_PATTERNS:
        raise ValueError(
            f"Unknown arch {arch!r}; available: {list(MODEL_GROUP_PATTERNS)}"
        )
    patterns = MODEL_GROUP_PATTERNS[arch]
    records: list[WeightRecord] = []
    for entry in entries:
        if entry.get("dtype") != "ternary2":
            continue
        name = entry["name"]
        records.append(WeightRecord(
            name=name,
            group=classify_weight(name, patterns),
            sparsity=float(entry.get("sparsity", 0.0)),
            alpha=float(entry.get("alpha", 0.0)),
            layer_idx=parse_layer_idx(name),
            expert_idx=parse_expert_idx(name),
        ))
    return records


def load_manifest_records(path: str | Path, arch: str) -> list[WeightRecord]:
    """File-based wrapper: load .tern-model and extract WeightRecord list."""
    from terncore.tern_model import TernModelReader
    reader = TernModelReader(str(path))
    return load_records_from_manifest_entries(reader.manifest["layers"], arch)


# ── Statistics ────────────────────────────────────────────────────────

def compute_group_stats(records: list[WeightRecord], group: str) -> GroupStats:
    sparsities = [r.sparsity for r in records if r.group == group]
    arr = np.asarray(sparsities) if sparsities else np.array([])
    return GroupStats(
        name=group,
        n=len(arr),
        median=float(np.median(arr)) if arr.size else 0.0,
        mean=float(np.mean(arr)) if arr.size else 0.0,
        stddev=float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        distribution=arr.tolist(),
    )


def run_comparison(
    records: list[WeightRecord],
    group_a: str,
    group_b: str,
) -> ComparisonResult:
    """Mann-Whitney U + Welch's t-test + effect sizes for group_a vs group_b."""
    a = np.asarray([r.sparsity for r in records if r.group == group_a])
    b = np.asarray([r.sparsity for r in records if r.group == group_b])
    if a.size == 0 or b.size == 0:
        return ComparisonResult(
            group_a=group_a, group_b=group_b,
            n_a=int(a.size), n_b=int(b.size),
            median_a=float(np.median(a)) if a.size else 0.0,
            median_b=float(np.median(b)) if b.size else 0.0,
            median_diff=0.0,
            mwu_u=0.0, mwu_p=1.0,
            rank_biserial=0.0,
            t_p=1.0, cohens_d=0.0,
        )

    mwu = stats.mannwhitneyu(a, b, alternative="two-sided")
    # Rank-biserial = 1 - 2U/(n_a*n_b); sign indicates direction (a > b positive)
    rb_magnitude = abs(1.0 - 2.0 * mwu.statistic / (a.size * b.size))
    rb = rb_magnitude if np.median(a) >= np.median(b) else -rb_magnitude

    t = stats.ttest_ind(a, b, equal_var=False)  # Welch's
    pooled_sd = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0) if (
        a.size > 1 and b.size > 1) else 0.0
    d = (np.mean(a) - np.mean(b)) / pooled_sd if pooled_sd > 0 else 0.0

    return ComparisonResult(
        group_a=group_a, group_b=group_b,
        n_a=int(a.size), n_b=int(b.size),
        median_a=float(np.median(a)),
        median_b=float(np.median(b)),
        median_diff=float(np.median(a) - np.median(b)),
        mwu_u=float(mwu.statistic),
        mwu_p=float(mwu.pvalue),
        rank_biserial=float(rb),
        t_p=float(t.pvalue),
        cohens_d=float(d),
    )


def per_layer_breakdown(
    records: list[WeightRecord],
    group_a: str,
    group_b: str,
) -> list[dict]:
    """Per-layer stats for groups A and B (where each group has data)."""
    by_layer: dict[int, list[WeightRecord]] = {}
    for r in records:
        if r.layer_idx is None:
            continue
        by_layer.setdefault(r.layer_idx, []).append(r)

    out: list[dict] = []
    for layer_idx in sorted(by_layer.keys()):
        layer_records = by_layer[layer_idx]
        a_sparsity = [r.sparsity for r in layer_records if r.group == group_a]
        b_sparsity = [r.sparsity for r in layer_records if r.group == group_b]
        out.append({
            "layer_idx": layer_idx,
            "n_a": len(a_sparsity),
            "n_b": len(b_sparsity),
            "median_a": float(np.median(a_sparsity)) if a_sparsity else None,
            "median_b": float(np.median(b_sparsity)) if b_sparsity else None,
        })
    return out


# ── Plotting ─────────────────────────────────────────────────────────

def plot_distribution_overlay(
    stats_a: GroupStats,
    stats_b: GroupStats,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0.0, 1.0, 41)
    if stats_a.n:
        ax.hist(stats_a.distribution, bins=bins, alpha=0.5,
                label=f"{stats_a.name} (n={stats_a.n}, median={stats_a.median:.3f})",
                color="tab:orange")
        ax.axvline(stats_a.median, color="tab:orange", linestyle="--", linewidth=1)
    if stats_b.n:
        ax.hist(stats_b.distribution, bins=bins, alpha=0.5,
                label=f"{stats_b.name} (n={stats_b.n}, median={stats_b.median:.3f})",
                color="tab:blue")
        ax.axvline(stats_b.median, color="tab:blue", linestyle="--", linewidth=1)
    ax.set_xlabel("Per-tensor zero-state ratio (sparsity)")
    ax.set_ylabel("Weight count")
    ax.set_title(f"Sparsity distribution: {stats_a.name} vs {stats_b.name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_per_layer_scatter(
    per_layer: list[dict],
    group_a: str,
    group_b: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    layers = [d["layer_idx"] for d in per_layer]
    a_med = [d["median_a"] for d in per_layer]
    b_med = [d["median_b"] for d in per_layer]
    ax.plot(layers, a_med, "o-", color="tab:orange",
            label=f"{group_a} median", markersize=6)
    ax.plot(layers, b_med, "s-", color="tab:blue",
            label=f"{group_b} median", markersize=6)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Per-layer median zero-state ratio")
    ax.set_title(f"Per-layer median sparsity: {group_a} vs {group_b}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_per_expert_heatmap(records: list[WeightRecord], out_path: Path) -> None:
    """Heatmap of per-(layer, expert) median sparsity across gate_up + down."""
    cell: dict[tuple[int, int], list[float]] = {}
    for r in records:
        if r.group != "expert_ffn" or r.layer_idx is None or r.expert_idx is None:
            continue
        cell.setdefault((r.layer_idx, r.expert_idx), []).append(r.sparsity)

    if not cell:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.text(0.5, 0.5, "No expert weights to plot",
                ha="center", va="center", transform=ax.transAxes)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return

    layers = sorted({k[0] for k in cell.keys()})
    experts = sorted({k[1] for k in cell.keys()})
    grid = np.full((len(layers), len(experts)), np.nan)
    for (l, e), values in cell.items():
        grid[layers.index(l), experts.index(e)] = float(np.median(values))

    fig, ax = plt.subplots(figsize=(max(8, len(experts) * 0.08),
                                     max(4, len(layers) * 0.25)))
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", origin="lower")
    ax.set_xlabel("Expert index")
    ax.set_ylabel("Layer index")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers)
    ax.set_title("Per-expert per-layer median sparsity (gate_up + down combined)")
    fig.colorbar(im, ax=ax, label="median zero-state ratio")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_external_reference(
    expert_median: float,
    dense_median: float,
    llama_dense_median: Optional[float],
    out_path: Path,
) -> None:
    """Three-bar comparison: this model expert / dense / Llama-70B dense.

    If llama_dense_median is None, only the two within-model bars render.
    """
    if llama_dense_median is None:
        labels = ["This model:\nexpert_ffn", "This model:\ndense_ffn"]
        values = [expert_median, dense_median]
        colors = ["tab:orange", "tab:blue"]
    else:
        labels = ["This model:\nexpert_ffn",
                  "This model:\ndense_ffn",
                  "Llama-3.1-70B:\ndense"]
        values = [expert_median, dense_median, llama_dense_median]
        colors = ["tab:orange", "tab:blue", "tab:gray"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom")
    ax.set_ylabel("Median per-tensor zero-state ratio")
    ax.set_title("Cross-model zero-state ratio reference")
    ax.set_ylim(0, max(values) * 1.15 if values else 1)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ── Output ────────────────────────────────────────────────────────────

def write_json(
    records: list[WeightRecord],
    group_stats: list[GroupStats],
    comparisons: list[ComparisonResult],
    per_layer: list[dict],
    llama_dense_median: Optional[float],
    out_path: Path,
) -> None:
    payload = {
        "n_records": len(records),
        "group_stats": [asdict(g) for g in group_stats],
        "comparisons": [asdict(c) for c in comparisons],
        "per_layer": per_layer,
        "external_reference": {
            "llama_3_1_70b_dense_median": llama_dense_median,
        },
        "records": [asdict(r) for r in records],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def write_summary_table(
    comparisons: list[ComparisonResult],
    llama_dense_median: Optional[float],
    out_path: Path,
) -> None:
    lines = ["# Per-expert tolerance analysis — summary", ""]
    lines.append("| Comparison | n_a | n_b | median_a | median_b | Δ median | "
                 "Mann-Whitney p | rank-biserial | Cohen's d |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in comparisons:
        lines.append(
            f"| {c.group_a} vs {c.group_b} | {c.n_a} | {c.n_b} | "
            f"{c.median_a:.4f} | {c.median_b:.4f} | {c.median_diff:+.4f} | "
            f"{c.mwu_p:.3e} | {c.rank_biserial:+.3f} | {c.cohens_d:+.3f} |"
        )
    if llama_dense_median is not None:
        lines.append("")
        lines.append(f"**External reference**: Llama-3.1-70B dense_ffn median "
                     f"zero-state ratio = **{llama_dense_median:.4f}**")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Per-expert ternary tolerance analysis")
    parser.add_argument("--manifest", required=True,
                        help="Path to .tern-model artefact under analysis")
    parser.add_argument("--arch", default="gemma4_moe",
                        choices=list(MODEL_GROUP_PATTERNS.keys()))
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for JSON + plots + summary")
    parser.add_argument("--llama70b-manifest", default=None,
                        help="Optional Llama-3.1-70B .tern-model for external ref")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading manifest: {args.manifest}", file=sys.stderr)
    records = load_manifest_records(args.manifest, args.arch)
    print(f"  {len(records)} ternary weight records", file=sys.stderr)

    expert_stats = compute_group_stats(records, "expert_ffn")
    dense_stats = compute_group_stats(records, "dense_ffn")
    attn_stats = compute_group_stats(records, "attention")

    # Synthetic non_expert grouping for secondary comparison
    non_expert_records = [
        WeightRecord(name=r.name, group="non_expert",
                     sparsity=r.sparsity, alpha=r.alpha,
                     layer_idx=r.layer_idx, expert_idx=r.expert_idx)
        for r in records if r.group in ("dense_ffn", "attention")
    ]
    non_expert_stats = compute_group_stats(non_expert_records, "non_expert")

    cmp_primary = run_comparison(records, "expert_ffn", "dense_ffn")
    cmp_vs_attn = run_comparison(records, "expert_ffn", "attention")
    cmp_vs_non_expert = run_comparison(records + non_expert_records,
                                       "expert_ffn", "non_expert")
    comparisons = [cmp_primary, cmp_vs_attn, cmp_vs_non_expert]

    per_layer = per_layer_breakdown(records, "expert_ffn", "dense_ffn")

    llama_dense_median: Optional[float] = None
    if args.llama70b_manifest:
        print(f"Loading Llama-70B reference: {args.llama70b_manifest}",
              file=sys.stderr)
        llama_records = load_manifest_records(args.llama70b_manifest, "llama_dense")
        llama_dense = compute_group_stats(llama_records, "dense_ffn")
        llama_dense_median = llama_dense.median if llama_dense.n else None
        print(f"  Llama-70B dense_ffn: n={llama_dense.n} "
              f"median={llama_dense.median:.4f}", file=sys.stderr)

    write_json(records,
               [expert_stats, dense_stats, attn_stats, non_expert_stats],
               comparisons, per_layer, llama_dense_median,
               out_dir / "per_expert_tolerance_analysis.json")

    plot_distribution_overlay(expert_stats, dense_stats,
                              out_dir / "plot_distribution_overlay.png")
    plot_per_layer_scatter(per_layer, "expert_ffn", "dense_ffn",
                          out_dir / "plot_per_layer_scatter.png")
    plot_per_expert_heatmap(records, out_dir / "plot_per_expert_heatmap.png")
    plot_external_reference(expert_stats.median, dense_stats.median,
                           llama_dense_median,
                           out_dir / "plot_external_reference.png")

    write_summary_table(comparisons, llama_dense_median,
                       out_dir / "summary_table.md")

    print(f"\nResults written to {out_dir}", file=sys.stderr)
    print(f"  Primary comparison: expert_ffn (n={cmp_primary.n_a}) vs "
          f"dense_ffn (n={cmp_primary.n_b})", file=sys.stderr)
    print(f"  Median diff: {cmp_primary.median_diff:+.4f}", file=sys.stderr)
    print(f"  Mann-Whitney p: {cmp_primary.mwu_p:.3e}", file=sys.stderr)
    print(f"  Rank-biserial: {cmp_primary.rank_biserial:+.3f}", file=sys.stderr)
    print(f"  Cohen's d: {cmp_primary.cohens_d:+.3f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
