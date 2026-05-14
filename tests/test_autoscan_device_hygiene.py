"""
R9-α (2026-05-14) — autoscan device hygiene tests.

Pins the contract added in commit (to-be) on the
``feat/r9-autoscan-device-hygiene-gemma4-retry-2026-05-14`` branch:

* ``auto_scan`` / ``streaming_scan`` accept an explicit ``device`` kwarg.
* ``device=None`` preserves the historical accelerate dispatch path
  (``device_map="auto"`` + 50 GiB CPU spill ceiling).
* Explicit ``device="mps"`` / ``"cuda"`` / ``"cpu"`` forces direct
  placement; missing backends raise ``RuntimeError`` up front.
* ``ScanResult`` gains ``device_used`` for provenance.
* Cache file gains a top-level ``schema_version`` field (v2.0); v1.0
  legacy caches auto-migrate on next write; v1.0 entries lacking
  ``device_used`` load as empty-string (unknown).

The tests deliberately avoid real model loads. Each test exercises a
single decision surface (load-kwargs helper, device validator, cache
schema, ScanResult round-trip).

Copyright (c) 2025-2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from terncore import autoscan as _autoscan


# ── _resolve_load_kwargs (R9-α invariant: device=None preserves legacy) ───


def test_resolve_load_kwargs_device_none_preserves_accelerate_dispatch():
    kwargs = _autoscan._resolve_load_kwargs(device=None)
    assert kwargs["device_map"] == "auto"
    assert kwargs["max_memory"] == {"cpu": "50GiB"}
    assert kwargs["low_cpu_mem_usage"] is True
    assert kwargs["dtype"] == torch.float16


def test_resolve_load_kwargs_explicit_cpu_disables_device_map():
    kwargs = _autoscan._resolve_load_kwargs(device="cpu")
    assert kwargs["device_map"] is None
    assert "max_memory" not in kwargs
    assert kwargs["low_cpu_mem_usage"] is True
    assert kwargs["dtype"] == torch.float16


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS unavailable on this host",
)
def test_resolve_load_kwargs_explicit_mps_disables_device_map():
    kwargs = _autoscan._resolve_load_kwargs(device="mps")
    assert kwargs["device_map"] is None
    assert "max_memory" not in kwargs


def test_resolve_load_kwargs_explicit_cuda_runtime_check():
    """When CUDA is unavailable, requesting cuda raises up front."""
    if torch.cuda.is_available():
        kwargs = _autoscan._resolve_load_kwargs(device="cuda")
        assert kwargs["device_map"] is None
    else:
        with pytest.raises(RuntimeError, match="cuda"):
            _autoscan._resolve_load_kwargs(device="cuda")


def test_resolve_load_kwargs_respects_explicit_dtype():
    kwargs = _autoscan._resolve_load_kwargs(device="cpu", dtype=torch.float32)
    assert kwargs["dtype"] == torch.float32


# ── _validate_device_available (R9-α invariant: clean error if unavailable) ─


def test_validate_device_cpu_always_ok():
    # No exception expected.
    _autoscan._validate_device_available("cpu")


def test_validate_device_cuda_raises_when_unavailable():
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this host; opposite-direction test")
    with pytest.raises(RuntimeError, match="cuda"):
        _autoscan._validate_device_available("cuda")
    with pytest.raises(RuntimeError, match="cuda"):
        _autoscan._validate_device_available("cuda:0")


def test_validate_device_mps_raises_when_unavailable():
    if torch.backends.mps.is_available():
        pytest.skip("MPS is available on this host; opposite-direction test")
    with pytest.raises(RuntimeError, match="mps"):
        _autoscan._validate_device_available("mps")


# ── _post_load_to_device ──────────────────────────────────────────────────


def test_post_load_to_device_none_is_noop():
    calls: list[str] = []

    class _Model:
        def to(self, dev):
            calls.append(dev)
            return self

    m = _Model()
    out = _autoscan._post_load_to_device(m, device=None)
    assert out is m
    assert calls == [], "device=None must not invoke .to()"


def test_post_load_to_device_explicit_calls_to():
    calls: list[str] = []

    class _Model:
        def to(self, dev):
            calls.append(dev)
            return self

    m = _Model()
    _autoscan._post_load_to_device(m, device="cpu")
    assert calls == ["cpu"]


# ── _resolve_device_used (provenance recorder) ────────────────────────────


def test_resolve_device_used_returns_explicit_when_provided():
    assert _autoscan._resolve_device_used(model=None, device="cpu") == "cpu"
    assert _autoscan._resolve_device_used(model=None, device="mps") == "mps"


def test_resolve_device_used_inspects_model_when_implicit():
    """When device=None (accelerate dispatch), record where parameters landed."""
    # Build a tiny real torch model so .parameters() is genuine.
    model = torch.nn.Linear(2, 2)
    used = _autoscan._resolve_device_used(model=model, device=None)
    # On a CI host without MPS/CUDA this will be "cpu". Just assert it's a
    # device-string shape, not the literal "unknown" fallback.
    assert isinstance(used, str) and used != ""
    assert used != "unknown"


# ── ScanResult.device_used ────────────────────────────────────────────────


def test_scan_result_device_used_default_empty_string():
    """Backward-compat: ScanResult constructed without device_used gets ''."""
    r = _autoscan.ScanResult(
        model_id="x", baseline_ppl=10.0, best_ppl=10.5, ppl_ceiling=12.0,
        ppl_headroom=0.2, total_eligible=100, layers_converted=50,
        pct_converted=50.0, compression_ratio=2.0,
        protection_list=[], converted_list=[],
    )
    assert r.device_used == ""


def test_scan_result_device_used_round_trip_via_cache(tmp_path, monkeypatch):
    """Save a ScanResult with device_used="mps", reload, get device_used="mps"."""
    monkeypatch.setattr(_autoscan, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(_autoscan, "CACHE_FILE", tmp_path / "model_cache.json")

    r = _autoscan.ScanResult(
        model_id="probe/model", baseline_ppl=10.0, best_ppl=10.5, ppl_ceiling=12.0,
        ppl_headroom=0.2, total_eligible=100, layers_converted=50,
        pct_converted=50.0, compression_ratio=2.0,
        protection_list=["p1"], converted_list=["c1"], device_used="mps",
    )
    _autoscan._save_result(r, threshold=0.7)

    loaded = _autoscan.load_cached_result(
        "probe/model", threshold=0.7, ppl_headroom=0.2,
    )
    assert loaded is not None
    assert loaded.device_used == "mps"
    assert loaded.cached is True


# ── Cache schema v2.0 ─────────────────────────────────────────────────────


def test_cache_file_written_in_v2_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(_autoscan, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(_autoscan, "CACHE_FILE", tmp_path / "model_cache.json")

    r = _autoscan.ScanResult(
        model_id="probe/v2", baseline_ppl=10.0, best_ppl=10.5, ppl_ceiling=12.0,
        ppl_headroom=0.2, total_eligible=100, layers_converted=50,
        pct_converted=50.0, compression_ratio=2.0,
        protection_list=[], converted_list=[], device_used="cpu",
    )
    _autoscan._save_result(r, threshold=0.7)

    raw = json.loads((tmp_path / "model_cache.json").read_text())
    assert raw["schema_version"] == "autoscan_cache/2.0"
    assert "entries" in raw
    assert "probe/v2|t=0.7|h=0.2" in raw["entries"]
    entry = raw["entries"]["probe/v2|t=0.7|h=0.2"]
    assert entry["device_used"] == "cpu"


def test_cache_v1_legacy_format_loads_transparently(tmp_path, monkeypatch):
    """A flat-top-level v1.0 cache file loads as if it were v2.0 entries."""
    monkeypatch.setattr(_autoscan, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(_autoscan, "CACHE_FILE", tmp_path / "model_cache.json")

    # Hand-write a v1.0-shaped cache: entries flat at top level, no
    # device_used field, no schema_version.
    legacy = {
        "legacy/model|t=0.7|h=0.2": {
            "model_id": "legacy/model",
            "baseline_ppl": 9.0,
            "best_ppl": 9.5,
            "ppl_ceiling": 10.8,
            "ppl_headroom": 0.2,
            "total_eligible": 50,
            "layers_converted": 30,
            "pct_converted": 60.0,
            "compression_ratio": 1.8,
            "protection_list": [],
            "converted_list": [],
        },
    }
    (tmp_path / "model_cache.json").write_text(json.dumps(legacy))

    loaded = _autoscan.load_cached_result(
        "legacy/model", threshold=0.7, ppl_headroom=0.2,
    )
    assert loaded is not None
    # v1.0 entry without device_used → empty-string (unknown) per R9-α
    assert loaded.device_used == ""
    assert loaded.baseline_ppl == 9.0
    assert loaded.cached is True


def test_cache_v1_migrates_to_v2_on_next_write(tmp_path, monkeypatch):
    """Writing after a v1.0 read materialises the v2.0 wrapper."""
    monkeypatch.setattr(_autoscan, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(_autoscan, "CACHE_FILE", tmp_path / "model_cache.json")

    legacy = {
        "legacy/model|t=0.7|h=0.2": {
            "model_id": "legacy/model",
            "baseline_ppl": 9.0,
            "best_ppl": 9.5,
            "ppl_ceiling": 10.8,
            "ppl_headroom": 0.2,
            "total_eligible": 50,
            "layers_converted": 30,
            "pct_converted": 60.0,
            "compression_ratio": 1.8,
            "protection_list": [],
            "converted_list": [],
        },
    }
    (tmp_path / "model_cache.json").write_text(json.dumps(legacy))

    # Adding a fresh entry triggers a write through the v2.0 path.
    r = _autoscan.ScanResult(
        model_id="new/model", baseline_ppl=8.0, best_ppl=8.4, ppl_ceiling=9.6,
        ppl_headroom=0.2, total_eligible=20, layers_converted=10,
        pct_converted=50.0, compression_ratio=1.5,
        protection_list=[], converted_list=[], device_used="mps",
    )
    _autoscan._save_result(r, threshold=0.7)

    raw = json.loads((tmp_path / "model_cache.json").read_text())
    assert raw["schema_version"] == "autoscan_cache/2.0"
    # Both legacy and new entries present under the new wrapper.
    assert "legacy/model|t=0.7|h=0.2" in raw["entries"]
    assert "new/model|t=0.7|h=0.2" in raw["entries"]


# ── auto_scan signature accepts device kwarg ──────────────────────────────


def test_auto_scan_signature_has_device_kwarg():
    """Smoke check on the public signature so callers can target it."""
    import inspect

    sig = inspect.signature(_autoscan.auto_scan)
    assert "device" in sig.parameters
    assert sig.parameters["device"].default is None


def test_streaming_scan_signature_has_device_kwarg():
    import inspect

    sig = inspect.signature(_autoscan.streaming_scan)
    assert "device" in sig.parameters
    assert sig.parameters["device"].default is None


# ── CLI smoke (no model load — argparse only) ─────────────────────────────


def test_cli_accepts_device_flag(monkeypatch):
    """tern-autoscan --device <x> parses cleanly + threads through to auto_scan."""
    captured: dict = {}

    def _fake_auto_scan(**kw):
        captured.update(kw)
        return _autoscan.ScanResult(
            model_id="x", baseline_ppl=10.0, best_ppl=10.5, ppl_ceiling=12.0,
            ppl_headroom=0.2, total_eligible=0, layers_converted=0,
            pct_converted=0.0, compression_ratio=1.0,
            protection_list=[], converted_list=[], device_used="cpu",
        )

    monkeypatch.setattr(_autoscan, "auto_scan", _fake_auto_scan)
    monkeypatch.setattr(
        "sys.argv",
        ["tern-autoscan", "--model", "x", "--device", "cpu"],
    )
    _autoscan.main()
    assert captured["device"] == "cpu"
    assert captured["model_id"] == "x"
