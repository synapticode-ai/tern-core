# Copyright 2026 Gamma Seeds Pte Ltd. Inventor: Robert Lakelin.
"""Tests for harness.checkpointer — JSON .see3 persistence."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import mlx.core as mx

HARNESS_ROOT = Path(__file__).resolve().parents[2]
if str(HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(HARNESS_ROOT))


from harness.annotator import EpistemicAnnotator
from harness.checkpointer import (
    SCHEMA_VERSION,
    CheckpointData,
    HarnessCheckpointer,
)
from harness.epistemic_state import Domain, EpistemicLabel, EpistemicState
from harness.objective import ConfidenceObjective
from harness.projector import TernaryProjector
from harness.scheduler import AdaptationScheduler
from harness.trainer import TernaryTrainer


# ---------------------------------------------------------------------------
# Fixtures — produce a real TrainStepResult to feed save()
# ---------------------------------------------------------------------------

def _trivial_loss_fn(params, x, y):
    return mx.mean((params["w"] @ x - y) ** 2)


def _make_trainer():
    return TernaryTrainer(
        projector=TernaryProjector(),
        annotator=EpistemicAnnotator(),
        objective=ConfidenceObjective(sparsity_target=0.5),
        scheduler=AdaptationScheduler(total_steps=100, alpha_warmup_steps=0),
        loss_fn=_trivial_loss_fn,
        config={},
    )


def _make_step_result(step: int = 0):
    trainer = _make_trainer()
    params = {"w": mx.random.normal(shape=(4, 3), key=mx.random.key(7))}
    x = mx.array([[0.5], [0.2], [0.7]])
    y = mx.array([[1.0], [0.5], [0.3], [0.8]])
    label = EpistemicLabel(
        epistemic_state=EpistemicState.UNCERTAIN,
        confidence_score=0.5,
        escalate=False,
        domain=Domain.FACTUAL,
        source_reliability=0.7,
    )
    return trainer.train_step(params, x, y, [label], step=step), params


def _harness_config():
    return {
        "model": {"name": "test", "format": "tern-pkg/1.0"},
        "training": {"batch_size": 4, "max_steps": 100},
        "ternary_projector": {"initial_temperature": 1.0, "final_temperature": 0.01},
    }


# ---------------------------------------------------------------------------
# save creates the file
# ---------------------------------------------------------------------------

def test_save_creates_see3_file(tmp_path):
    chk = HarnessCheckpointer(output_dir=tmp_path)
    result, params = _make_step_result(step=42)
    written = chk.save(step=42, model_params=params,
                        trainer_result=result, config=_harness_config())

    assert written.exists()
    assert written.suffix == ".see3"
    assert "tfh_step_000042" in written.name
    assert written.parent == tmp_path


def test_filename_format_is_correct(tmp_path):
    """Filename must zero-pad the step to 6 digits with the
    tfh_step_ prefix and .see3 suffix."""
    chk = HarnessCheckpointer(output_dir=tmp_path)
    result, params = _make_step_result(step=7)
    path = chk.save(step=7, model_params=params,
                    trainer_result=result, config={})
    assert path.name == "tfh_step_000007.see3"


def test_save_creates_output_dir_if_missing(tmp_path):
    """A non-existent output_dir is created on first save."""
    target_dir = tmp_path / "nested" / "checkpoints"
    assert not target_dir.exists()
    chk = HarnessCheckpointer(output_dir=target_dir)
    result, params = _make_step_result(step=0)
    chk.save(step=0, model_params=params, trainer_result=result, config={})
    assert target_dir.exists()


# ---------------------------------------------------------------------------
# load round-trips
# ---------------------------------------------------------------------------

def test_load_round_trips_step_and_config(tmp_path):
    chk = HarnessCheckpointer(output_dir=tmp_path)
    result, params = _make_step_result(step=99)
    config = _harness_config()
    path = chk.save(step=99, model_params=params,
                    trainer_result=result, config=config)

    loaded = chk.load(path)
    assert isinstance(loaded, CheckpointData)
    assert loaded.step == 99
    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.harness_config == config
    # confidence_metadata should carry the annotation summary and the
    # objective fields the checkpointer extracted from trainer_result
    cm = loaded.confidence_metadata
    assert "annotation_summary" in cm
    assert "total_loss" in cm
    assert "tau" in cm
    assert "alpha" in cm
    assert "grad_norm" in cm
    # model_params survive as nested lists, with the original keys
    assert set(loaded.model_params.keys()) == set(params.keys())
    # And the values reconstruct into mx.arrays of the original shape
    reconstructed = mx.array(loaded.model_params["w"])
    assert reconstructed.shape == params["w"].shape


def test_load_rejects_missing_file(tmp_path):
    chk = HarnessCheckpointer(output_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        chk.load(tmp_path / "does_not_exist.see3")


def test_load_rejects_unknown_schema_version(tmp_path):
    """A checkpoint whose schema_version differs from SCHEMA_VERSION
    must be refused with a clear error — silent acceptance is exactly
    how training-time / inference-time format drift propagates."""
    bad_path = tmp_path / "bad.see3"
    bad_path.write_text(json.dumps({
        "step": 0,
        "schema_version": "tfh/2.0",
        "model_params": {},
        "confidence_metadata": {},
        "harness_config": {},
        "saved_at": "2026-04-08T00:00:00+00:00",
    }))
    chk = HarnessCheckpointer(output_dir=tmp_path)
    with pytest.raises(ValueError, match="Unsupported checkpoint schema"):
        chk.load(bad_path)


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------

def test_schema_version_is_tfh_1_0():
    assert SCHEMA_VERSION == "tfh/1.0"


def test_saved_payload_carries_schema_version(tmp_path):
    chk = HarnessCheckpointer(output_dir=tmp_path)
    result, params = _make_step_result(step=1)
    path = chk.save(step=1, model_params=params,
                    trainer_result=result, config={})
    raw = json.loads(path.read_text())
    assert raw["schema_version"] == "tfh/1.0"


# ---------------------------------------------------------------------------
# exists / delete
# ---------------------------------------------------------------------------

def test_exists_true_after_save(tmp_path):
    chk = HarnessCheckpointer(output_dir=tmp_path)
    result, params = _make_step_result(step=5)
    path = chk.save(step=5, model_params=params,
                    trainer_result=result, config={})
    assert chk.exists(path) is True


def test_exists_false_for_missing(tmp_path):
    chk = HarnessCheckpointer(output_dir=tmp_path)
    assert chk.exists(tmp_path / "ghost.see3") is False


def test_delete_removes_file(tmp_path):
    chk = HarnessCheckpointer(output_dir=tmp_path)
    result, params = _make_step_result(step=3)
    path = chk.save(step=3, model_params=params,
                    trainer_result=result, config={})
    assert path.exists()

    deleted = chk.delete(path)
    assert deleted is True
    assert not path.exists()
    # Idempotent: deleting a missing file returns False, no error
    assert chk.delete(path) is False


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------

def test_atomic_write_uses_tmp_then_rename(tmp_path):
    """Save must write to a .tmp path first then rename. Mock
    Path.rename and confirm the rename happened with the correct
    .tmp source path."""
    chk = HarnessCheckpointer(output_dir=tmp_path)
    result, params = _make_step_result(step=10)

    rename_calls: list[tuple[Path, Path]] = []
    original_rename = Path.rename

    def spy_rename(self, target):
        rename_calls.append((self, Path(target)))
        return original_rename(self, target)

    with patch.object(Path, "rename", spy_rename):
        chk.save(step=10, model_params=params,
                 trainer_result=result, config={})

    assert len(rename_calls) == 1
    src, dst = rename_calls[0]
    assert str(src).endswith(".see3.tmp"), f"src must be .tmp, got {src}"
    assert str(dst).endswith("tfh_step_000010.see3"), f"dst must be the final .see3, got {dst}"


def test_atomic_write_does_not_leave_tmp_behind(tmp_path):
    """After a successful save, the .tmp staging file must be gone."""
    chk = HarnessCheckpointer(output_dir=tmp_path)
    result, params = _make_step_result(step=20)
    path = chk.save(step=20, model_params=params,
                    trainer_result=result, config={})
    assert path.exists()
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == [], f"Temp files left behind: {tmp_files}"


def test_save_overwrites_existing_checkpoint(tmp_path):
    """Saving the same step twice overwrites cleanly — no duplicate
    files, no leftover .tmp."""
    chk = HarnessCheckpointer(output_dir=tmp_path)
    result1, params1 = _make_step_result(step=15)
    result2, params2 = _make_step_result(step=15)

    path1 = chk.save(step=15, model_params=params1,
                     trainer_result=result1, config={"run": "first"})
    path2 = chk.save(step=15, model_params=params2,
                     trainer_result=result2, config={"run": "second"})

    assert path1 == path2
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["tfh_step_000015.see3"]

    loaded = chk.load(path2)
    assert loaded.harness_config == {"run": "second"}
