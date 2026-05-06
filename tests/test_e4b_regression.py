"""
E4B regression tests against the on-disk Gemopus-4-E4B baseline artefact.

Verifies that pre-rework .tern-model artefacts continue to read cleanly
after the Session 3 per-expert slicing rework. The baseline at
``/Volumes/Syn Archive/models/compressed/gemopus-4-e4b/`` was sealed
before the rework landed, so:

- It carries no ``stacked_parent`` metadata on any entry.
- Reading via the post-rework ``TernModelReader.reconstruct_all`` must
  produce a state_dict identical in structure to what the pre-rework
  reader would have produced (no accidental restacking-path entry).

Tests 1-3 run by default and auto-skip if the archive isn't mounted
(catches schema-breaking changes on every test cycle for developers
with the archive available).

Test 4 is opt-in via ``RUN_E4B_REGRESSION=1`` env var AND requires
E4B source weights at ``/Volumes/Syn Archive/models/source/gemma-4-e4b-it/``.
When enabled it re-runs ``full_convert`` against the source and diffs
the resulting manifest against the baseline. This is the canonical
heavy validation banked for Substep 2g manual execution.

Copyright (c) 2025–2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from terncore.tern_model import TernModelReader


BASELINE_PATH = Path(
    "/Volumes/Syn Archive/models/compressed/gemopus-4-e4b/"
    "gemopus_4_e4b_ternary_v0.1.0.tern-model/model.tern-model"
)
E4B_SOURCE_PATH = Path("/Volumes/Syn Archive/models/source/gemma-4-e4b-it")


_baseline_skip = pytest.mark.skipif(
    not BASELINE_PATH.exists(),
    reason=f"E4B baseline artefact not present at {BASELINE_PATH} "
           f"(Syn Archive likely not mounted on this host).",
)


# ── Tests 1–3: baseline backwards-compat (auto-skip if archive missing) ─


@_baseline_skip
def test_baseline_manifest_reads_cleanly():
    """Pre-rework E4B baseline opens via post-rework TernModelReader.

    Catches schema-breaking changes that would prevent loading existing
    .tern-model artefacts. The baseline is a real production artefact
    sealed before the Session 3 rework.
    """
    reader = TernModelReader(str(BASELINE_PATH))
    entries = reader.manifest["layers"]
    assert len(entries) > 0, (
        f"Baseline manifest has no entries: {BASELINE_PATH}"
    )


@_baseline_skip
def test_baseline_has_no_stacking_metadata():
    """Pre-rework baseline must carry no ``stacked_parent`` field.

    Two purposes:
    - Confirms the baseline is genuinely pre-rework (sanity check).
    - Confirms pre-rework manifests don't accidentally trigger the new
      restacking path on read — the restacking branch is gated on
      ``entry.get("stacked_parent")`` returning a non-None value.
    """
    reader = TernModelReader(str(BASELINE_PATH))
    entries = reader.manifest["layers"]
    offenders = [e["name"] for e in entries if "stacked_parent" in e]
    assert not offenders, (
        f"Baseline manifest contains {len(offenders)} entries with stacking "
        f"metadata, but baseline was sealed pre-rework: {offenders[:5]}"
    )


@pytest.mark.slow
@_baseline_skip
def test_baseline_reconstruct_all_succeeds():
    """End-to-end backwards-compat: reconstruct_all walks the entire
    baseline manifest and emits a state_dict.

    Catches restacking-path crashes on real artefacts (the in-flight
    accumulator should never enter the stacked branch since no entry
    carries ``stacked_parent``, but a regression in the gating logic
    would surface here).
    """
    reader = TernModelReader(str(BASELINE_PATH))
    state_dict = reader.reconstruct_all()
    # Sanity bound: E4B is a multi-billion-parameter model; expect
    # hundreds of state_dict entries (layers × projections + embeds + norms).
    assert len(state_dict) > 100, (
        f"reconstruct_all produced only {len(state_dict)} state_dict entries "
        f"from baseline — suspiciously low for E4B."
    )


# ── Test 4: active re-compression diff (opt-in, slow) ───────────────


@pytest.mark.slow
@pytest.mark.skipif(
    os.getenv("RUN_E4B_REGRESSION") != "1",
    reason="Set RUN_E4B_REGRESSION=1 to enable the slow active "
           "re-compression diff (~45 min wall clock).",
)
@pytest.mark.skipif(
    not E4B_SOURCE_PATH.exists(),
    reason=f"E4B source weights not present at {E4B_SOURCE_PATH}; "
           f"required for active re-compression.",
)
@_baseline_skip
def test_e4b_active_recompression_diff(tmp_path: Path):
    """Re-compress E4B source through the post-rework pipeline and diff
    the resulting manifest against the baseline.

    Modulo timestamp + non-deterministic metadata, the manifest entry
    list (names + dtypes + shapes) must match the baseline entry-for-entry.
    Any divergence indicates the rework changed dense-model behaviour.

    This is the canonical heavy regression test for the rework. Banked
    for Substep 2g manual execution; not in default CI.
    """
    from terncore.convert import full_convert

    out_dir = tmp_path / "e4b_recompress"
    full_convert(
        model_id=str(E4B_SOURCE_PATH),
        adapter_name="gemma4",
        output_dir=str(out_dir),
        threshold=0.7,
        verbose=False,
    )

    fresh = TernModelReader(str(out_dir / "model.tern-model"))
    baseline = TernModelReader(str(BASELINE_PATH))

    fresh_keys = sorted(
        (e["name"], e["dtype"], tuple(e["shape"])) for e in fresh.manifest["layers"]
    )
    baseline_keys = sorted(
        (e["name"], e["dtype"], tuple(e["shape"])) for e in baseline.manifest["layers"]
    )
    assert fresh_keys == baseline_keys, (
        f"E4B re-compression diff: {len(set(fresh_keys) ^ set(baseline_keys))} "
        f"entries differ between fresh and baseline manifests."
    )
