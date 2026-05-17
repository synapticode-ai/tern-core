"""Unit tests for tern_r12_sweep_aggregate.

Aggregator is a pure function over per-point JSONs in a sweep directory.
Tests use fixture per-point JSON dicts written to tmp_path; no real subprocess
invocation, no model load.

Coverage:
  - happy path: 6-point grid → grid_exhausted termination
  - ceiling_crossed: terminated_at_ceiling=True on point 3 → halt
  - hook_construction_failure: hook_construction_failed=True on point 1
  - ppl_eval_failure: ppl_kv_compressed=None without hook_construction_failed
  - incomplete: 3 points present, 6-point grid expected → 'incomplete'
  - recommended_operating_point selection: minimum b_mse under calibration gate
  - recommended_operating_point null-stub: no qualifying points
  - point ordering: out-of-order JSON filenames re-sort by point_index

Copyright (c) 2025-2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure tools/ on path
TOOLS = Path(__file__).resolve().parent.parent / "tools"
sys.path.insert(0, str(TOOLS))

import tern_r12_sweep_aggregate as agg  # noqa: E402


# ── Fixture helpers ────────────────────────────────────────────────────────


def _make_point_json(
    *,
    point_index: int,
    b_mse: int,
    diagnostic_run_id: str = "deadbeef-1234-5678-9abc-def012345678",
    ppl_kv_compressed: float | None = 8.0,
    ppl_headroom: float | None = 0.005,
    ppl_headroom_band: str | None = "Excellent",
    terminated_at_ceiling: bool = False,
    hook_construction_failed: bool = False,
    notes: str = "test fixture",
    model_id: str = "test/tinyllama",
    num_sequences: int = 16,
    seq_len: int = 2048,
) -> dict:
    return {
        "schema_version": "ppl_headroom_kv_cache_point/1.0",
        "diagnostic_run_id": diagnostic_run_id,
        "point_index": point_index,
        "config": {
            "b_mse": b_mse,
            "weight_threshold": "FP16",
            "kv_hook_application_scope": "all_layers",
            "num_sequences": num_sequences,
            "seq_len": seq_len,
            "model_id": model_id,
        },
        "ppl_eval_run_id": f"20260518T00000{point_index}Z",
        "baseline_ppl_r7b": 7.9367,
        "ppl_kv_compressed": ppl_kv_compressed,
        "ppl_headroom": ppl_headroom,
        "ppl_headroom_band": ppl_headroom_band,
        "kv_cache_compression_ratio": 16.0 / b_mse,
        "terminated_at_ceiling": terminated_at_ceiling,
        "hook_construction_failed": hook_construction_failed,
        "notes": notes,
    }


def _write_point(sweep_dir: Path, payload: dict) -> Path:
    idx = payload["point_index"]
    b = payload["config"]["b_mse"]
    runid = payload["ppl_eval_run_id"]
    fn = sweep_dir / f"point_{idx:02d}_b_mse_{b}_{runid}.json"
    fn.write_text(json.dumps(payload, indent=2) + "\n")
    return fn


def _default_inputs(b_mse_grid: list[int]) -> dict:
    return {
        "source_model": "test/tinyllama",
        "baseline_ppl_r7b": 7.9367,
        "baseline_run_id": "test-baseline-runid",
        "ppl_headroom_ceiling": 1.0,
        "sweep_grid": {"b_mse": b_mse_grid},
        "num_sequences": 16,
        "seq_len": 2048,
        "seed": 1337,
        "continue_past_ceiling": False,
    }


def _aggregate(
    sweep_dir: Path,
    b_mse_grid: list[int],
    *,
    continue_past_ceiling: bool = False,
) -> dict:
    manifest_path = agg.aggregate_sweep_directory(
        sweep_dir=sweep_dir,
        diagnostic_run_id="deadbeef-1234-5678-9abc-def012345678",
        inputs=_default_inputs(b_mse_grid),
        calibration_gate_pct=1.68,
        device="mps",
        notes="test",
        total_eval_wall_time_seconds=42.0,
        expected_b_mse_grid=b_mse_grid,
        continue_past_ceiling=continue_past_ceiling,
    )
    return json.loads(manifest_path.read_text())


# ── Tests ──────────────────────────────────────────────────────────────────


def test_happy_path_grid_exhausted(tmp_path: Path) -> None:
    """6-point clean grid → grid_exhausted; recommended = lowest b_mse under gate."""
    grid = [6, 5, 4, 3, 2, 1]
    for idx, b in enumerate(grid):
        # ppl_headroom = 0.005 = 0.5% — under 1.68% gate
        _write_point(tmp_path, _make_point_json(point_index=idx, b_mse=b))

    manifest = _aggregate(tmp_path, grid)

    assert manifest["schema_version"] == "ppl_headroom_kv_cache_sweep/1.0"
    assert manifest["termination"]["terminated_reason"] == "grid_exhausted"
    assert manifest["termination"]["terminated_at_point_index"] is None
    assert len(manifest["points"]) == 6
    assert len(manifest["frontier"]) == 6
    rec = manifest["recommended_operating_point"]
    assert rec["b_mse"] == 1  # lowest qualifying b_mse
    assert manifest["hardware"]["ppl_eval_wall_time_seconds"] == 42.0


def test_ceiling_crossed_halt(tmp_path: Path) -> None:
    """terminated_at_ceiling on point 3 → ceiling_crossed termination."""
    points = [
        _make_point_json(point_index=0, b_mse=6),
        _make_point_json(point_index=1, b_mse=5),
        _make_point_json(point_index=2, b_mse=4),
        _make_point_json(point_index=3, b_mse=3, terminated_at_ceiling=True,
                         ppl_kv_compressed=16.0, ppl_headroom=1.5,
                         ppl_headroom_band="Fail"),
    ]
    for p in points:
        _write_point(tmp_path, p)

    manifest = _aggregate(tmp_path, [6, 5, 4, 3, 2, 1])
    # incomplete grid (4 of 6) but ceiling_crossed is the explicit halt
    # cause and takes precedence over missing-points inference.
    assert manifest["termination"]["terminated_reason"] == "ceiling_crossed"
    assert manifest["termination"]["terminated_at_point_index"] == 3
    assert manifest["termination"]["failed_at_b_mse"] == 3


def test_ceiling_crossed_continues_when_flag_set(tmp_path: Path) -> None:
    """With continue_past_ceiling=True, ceiling_crossed does NOT halt."""
    points = [
        _make_point_json(point_index=0, b_mse=4),
        _make_point_json(point_index=1, b_mse=2, terminated_at_ceiling=True,
                         ppl_kv_compressed=16.0, ppl_headroom=1.5,
                         ppl_headroom_band="Fail"),
    ]
    for p in points:
        _write_point(tmp_path, p)

    manifest = _aggregate(tmp_path, [4, 2], continue_past_ceiling=True)
    assert manifest["termination"]["terminated_reason"] == "grid_exhausted"


def test_hook_construction_failure(tmp_path: Path) -> None:
    """hook_construction_failed=True → hook_construction_failure termination."""
    points = [
        _make_point_json(point_index=0, b_mse=6),
        _make_point_json(
            point_index=1, b_mse=5,
            ppl_kv_compressed=None, ppl_headroom=None, ppl_headroom_band=None,
            hook_construction_failed=True,
        ),
    ]
    for p in points:
        _write_point(tmp_path, p)

    manifest = _aggregate(tmp_path, [6, 5, 4, 3, 2, 1])
    assert manifest["termination"]["terminated_reason"] == "hook_construction_failure"
    assert manifest["termination"]["terminated_at_point_index"] == 1
    assert manifest["termination"]["failed_at_b_mse"] == 5


def test_ppl_eval_failure(tmp_path: Path) -> None:
    """ppl_kv_compressed=None without hook_construction_failed → ppl_eval_failure.

    Sweep continued past the failure point (less-aggressive b_mse may succeed),
    so all 6 points are present on disk.
    """
    grid = [6, 5, 4, 3, 2, 1]
    for idx, b in enumerate(grid):
        if b == 3:
            _write_point(tmp_path, _make_point_json(
                point_index=idx, b_mse=b,
                ppl_kv_compressed=None, ppl_headroom=None, ppl_headroom_band=None,
            ))
        else:
            _write_point(tmp_path, _make_point_json(point_index=idx, b_mse=b))

    manifest = _aggregate(tmp_path, grid)
    assert manifest["termination"]["terminated_reason"] == "ppl_eval_failure"
    assert manifest["termination"]["failed_at_b_mse"] == 3


def test_incomplete_subprocess_death(tmp_path: Path) -> None:
    """3 points present, 6 expected, no halt-marker → 'incomplete' termination.

    Simulates the libdispatch-trap failure mode: subprocess killed by OS
    before a halt-marker (ceiling/hook-fail) was reached.
    """
    grid = [6, 5, 4, 3, 2, 1]
    for idx in range(3):
        _write_point(tmp_path, _make_point_json(
            point_index=idx, b_mse=grid[idx],
        ))

    manifest = _aggregate(tmp_path, grid)
    assert manifest["termination"]["terminated_reason"] == "incomplete"
    assert manifest["termination"]["terminated_at_point_index"] == 2
    assert manifest["termination"]["failed_at_b_mse"] == 4


def test_recommended_operating_point_null_when_no_qualifier(tmp_path: Path) -> None:
    """All points outside calibration gate → null-stub recommended_operating_point."""
    # ppl_headroom = 0.05 = 5% (above 1.68% gate)
    for idx, b in enumerate([6, 5, 4]):
        _write_point(tmp_path, _make_point_json(
            point_index=idx, b_mse=b, ppl_headroom=0.05,
            ppl_headroom_band="Marginal",
        ))

    manifest = _aggregate(tmp_path, [6, 5, 4])
    rec = manifest["recommended_operating_point"]
    assert rec["b_mse"] is None
    assert rec["point_index"] is None
    assert "no sweep point" in rec["rationale"]


def test_point_ordering_by_point_index(tmp_path: Path) -> None:
    """Files written out of order → manifest points sorted by point_index."""
    # Write in reverse order
    for idx, b in [(2, 4), (0, 6), (1, 5)]:
        _write_point(tmp_path, _make_point_json(point_index=idx, b_mse=b))

    manifest = _aggregate(tmp_path, [6, 5, 4])
    indices = [pt["point_index"] for pt in manifest["points"]]
    assert indices == [0, 1, 2]
    b_mse_seq = [fr["b_mse"] for fr in manifest["frontier"]]
    assert b_mse_seq == [6, 5, 4]


def test_idempotent_re_aggregation(tmp_path: Path) -> None:
    """Re-running aggregator on same dir overwrites manifest cleanly."""
    for idx, b in enumerate([6, 5]):
        _write_point(tmp_path, _make_point_json(point_index=idx, b_mse=b))

    _aggregate(tmp_path, [6, 5])
    manifest_path = tmp_path / "sweep_manifest.json"
    assert manifest_path.exists()
    first = json.loads(manifest_path.read_text())

    # Add a point and re-aggregate
    _write_point(tmp_path, _make_point_json(point_index=2, b_mse=4))
    second = _aggregate(tmp_path, [6, 5, 4])
    assert len(first["points"]) == 2
    assert len(second["points"]) == 3


def test_manifest_schema_fields_present(tmp_path: Path) -> None:
    """Aggregate manifest carries all §6.2 required top-level fields."""
    _write_point(tmp_path, _make_point_json(point_index=0, b_mse=4))
    manifest = _aggregate(tmp_path, [4])

    required = {
        "schema_version", "diagnostic_run_id", "timestamp_utc",
        "tern_core_version", "tern_core_git_commit", "spec_version",
        "methodology_consumed", "inputs", "points", "frontier",
        "recommended_operating_point", "termination", "hardware", "notes",
    }
    assert required.issubset(manifest.keys())
    assert manifest["spec_version"].startswith("kv_cache_compression_ppl_headroom_diagnostic")


@pytest.mark.parametrize("b_mse,expected_ratio", [(1, 16.0), (2, 8.0), (4, 4.0), (8, 2.0)])
def test_compression_ratio_in_frontier(tmp_path: Path, b_mse: int, expected_ratio: float) -> None:
    """Frontier entry carries kv_cache_compression_ratio from per-point JSON."""
    _write_point(tmp_path, _make_point_json(point_index=0, b_mse=b_mse))
    manifest = _aggregate(tmp_path, [b_mse])
    assert manifest["frontier"][0]["kv_cache_compression_ratio"] == expected_ratio
