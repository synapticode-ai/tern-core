# Copyright 2026 Gamma Seeds Pte Ltd. Inventor: Robert Lakelin.
# Patent alignment: candidate new provisional — HarnessCheckpointer³
# training→inference continuity (flag to Rod).
"""
HarnessCheckpointer — .see3-wrapped JSON checkpoint persistence.

Follows the persistence.py pattern from terncore (save / load /
exists / delete, version field, atomic writes), but is NOT a literal
wrap of GuardianPersistence — that module serialises Guardian state,
which is unrelated to model checkpoints. The pattern is what we
inherit; the contents are TFH-specific.

CRITICAL CONTINUITY PROPERTY
============================
A checkpoint saved by HarnessCheckpointer must be loadable by
tern-runtime without conversion. The model_params dict uses the same
key names as the tern-runtime loader expects so a TFH-trained
checkpoint deploys directly to LIS via tern-runtime/loader/pkg_loader.py.

Phase 1 honesty: tern-runtime/loader/pkg_loader.py currently consumes
the .tern-pkg CoreML container format produced by tern-compiler, NOT a
raw .see3 JSON. The continuity claim depends on a future tern-compiler
v0.4+ that can ingest a .see3 checkpoint and emit a .tern-pkg without
re-running quantisation. The .see3 format defined here is the input
to that compiler step. The key naming convention preserved here
matches what the LIS loader's downstream consumer expects, so when
the compiler step lands the format will already be aligned.

Storage format
==============
- One JSON file per checkpoint, named ``tfh_step_{step:06d}.see3``
- Atomic write: payload is written to ``<name>.tmp`` first, then
  ``Path.rename`` moves it into place. A crash mid-write leaves any
  prior checkpoint intact.
- Configurable output directory (default: ``./checkpoints/``)

Schema version
==============
``tfh/1.0`` — locked at module load time. Bump simultaneously with
the tern-runtime loader if the field set ever changes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import mlx.core as mx

from harness.trainer import TrainStepResult


SCHEMA_VERSION = "tfh/1.0"


# ---------------------------------------------------------------------------
# CheckpointData — what lives on disk
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CheckpointData:
    """Frozen view of one persisted training checkpoint.

    All fields are JSON-serialisable. ``model_params`` stores weight
    tensors as nested Python lists (via ``mx.array.tolist()``) — the
    loader reconstructs them via ``mx.array(...)`` on read.
    """

    step: int
    schema_version: str
    model_params: dict           # name → nested list of floats
    confidence_metadata: dict    # annotation_summary + objective fields
    harness_config: dict
    saved_at: str                # ISO 8601 UTC


# ---------------------------------------------------------------------------
# Checkpointer
# ---------------------------------------------------------------------------

class HarnessCheckpointer:
    """JSON-backed checkpoint store with atomic writes.

    Args:
        output_dir: Directory for ``.see3`` files. Created on first
            ``save()`` if absent. Default ``./checkpoints``.
    """

    DEFAULT_OUTPUT_DIR = Path("checkpoints")
    FILENAME_TEMPLATE = "tfh_step_{step:06d}.see3"

    def __init__(self, output_dir: Optional[Path | str] = None) -> None:
        self._output_dir = Path(output_dir) if output_dir is not None else self.DEFAULT_OUTPUT_DIR

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    # ----------------------------------------------------------- save

    def save(
        self,
        step: int,
        model_params: dict[str, mx.array],
        trainer_result: TrainStepResult,
        config: dict,
    ) -> Path:
        """Persist a checkpoint atomically.

        Args:
            step: Training step index. Drives the filename.
            model_params: Mapping from param name to MLX array. Each
                array is serialised via ``.tolist()`` so it survives
                JSON encoding.
            trainer_result: Output of ``TernaryTrainer.train_step()``
                for this step. The annotation summary and objective
                fields populate ``confidence_metadata``.
            config: The harness.yaml config dict that produced this
                run. Persisted verbatim so a reload can rebuild the
                exact same training context.

        Returns:
            Path to the written ``.see3`` file.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        target = self._output_dir / self.FILENAME_TEMPLATE.format(step=step)
        tmp = target.with_suffix(target.suffix + ".tmp")

        data = CheckpointData(
            step=int(step),
            schema_version=SCHEMA_VERSION,
            model_params=self._serialise_params(model_params),
            confidence_metadata=self._build_confidence_metadata(trainer_result),
            harness_config=dict(config),
            saved_at=datetime.now(timezone.utc).isoformat(),
        )

        # Atomic write: payload to .tmp, then rename. A crash before
        # the rename leaves any pre-existing target untouched and
        # only orphans a .tmp file (cleaned up on the next save).
        tmp.write_text(json.dumps(asdict(data), indent=2), encoding="utf-8")
        if target.exists():
            target.unlink()
        tmp.rename(target)
        return target

    # ----------------------------------------------------------- load

    def load(self, path: Path | str) -> CheckpointData:
        """Read a checkpoint from disk.

        Args:
            path: Filesystem path to the ``.see3`` file.

        Returns:
            ``CheckpointData`` with every field populated. Weight
            tensors are returned as nested lists — call
            ``mx.array(...)`` to lift them into MLX arrays.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        raw = json.loads(p.read_text(encoding="utf-8"))

        # Strict schema check — refuse to load a checkpoint whose
        # version we don't recognise. A future loader can grow
        # multi-version support; v1.0 is the only valid value today.
        version = raw.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported checkpoint schema {version!r}. "
                f"This loader supports {SCHEMA_VERSION!r}."
            )

        return CheckpointData(
            step=int(raw["step"]),
            schema_version=raw["schema_version"],
            model_params=dict(raw["model_params"]),
            confidence_metadata=dict(raw["confidence_metadata"]),
            harness_config=dict(raw["harness_config"]),
            saved_at=str(raw["saved_at"]),
        )

    # ----------------------------------------------------------- exists

    def exists(self, path: Path | str) -> bool:
        return Path(path).exists()

    # ----------------------------------------------------------- delete

    def delete(self, path: Path | str) -> bool:
        """Remove a checkpoint file. Returns True if it existed."""
        p = Path(path)
        if not p.exists():
            return False
        p.unlink()
        return True

    # ----------------------------------------------------------- helpers

    @staticmethod
    def _serialise_params(params: dict[str, mx.array]) -> dict:
        """Convert each MLX array to a nested Python list for JSON."""
        return {name: array.tolist() for name, array in params.items()}

    @staticmethod
    def _build_confidence_metadata(result: TrainStepResult) -> dict:
        """Pull the annotation summary and objective fields out of the
        TrainStepResult into a flat dict suitable for the
        ``confidence_metadata`` slot.
        """
        return {
            "annotation_summary": dict(result.annotation_summary),
            "total_loss": float(result.total_loss),
            "task_loss": float(result.task_loss),
            "calibration_penalty": float(result.calibration_penalty),
            "sparsity_penalty": float(result.sparsity_penalty),
            "tau": float(result.tau),
            "alpha": float(result.alpha),
            "mean_sparsity": float(result.mean_sparsity),
            "grad_norm": float(result.grad_norm),
        }
