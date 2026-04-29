"""FP16 cast guards for the CoreML export path.

Extracted from coreml_export.py to avoid pulling coremltools at
import time when the guards are exercised in isolation (e.g.,
unit tests). The guards are pure numpy logic; the original
co-location with coreml_export.py was incidental.
"""

from __future__ import annotations

import numpy as np

FP16_MAX = 65504.0


def _validate_ternary2_alpha(alpha: float, name: str) -> None:
    """Guard a ternary2 alpha against silent FP16 overflow.

    Non-finite alpha indicates a quantiser bug upstream; out-of-range
    alpha indicates a degenerate quantisation case. Either would cast
    to Inf in FP16 and break downstream palettisation. Both raise
    rather than silently clamp — clamping would mask the upstream bug.
    """
    if not np.isfinite(alpha):
        raise ValueError(
            f"Non-finite alpha {alpha} for ternary2 layer "
            f"{name}. This indicates a quantiser bug upstream "
            f"(likely in convert.py ternary2 path); the cast "
            f"to FP16 would silently produce Inf/NaN and break "
            f"downstream palettisation. Refusing to emit."
        )
    if abs(alpha) > FP16_MAX:
        raise ValueError(
            f"Alpha {alpha} for ternary2 layer {name} exceeds "
            f"FP16 range (±{int(FP16_MAX)}). This indicates a "
            f"degenerate quantisation case (e.g., sparse or "
            f"all-zero block); the cast to FP16 would silently "
            f"produce Inf. Refusing to emit."
        )


def _cast_fp16_retain_with_guards(
    weight_fp32: np.ndarray, name: str
) -> np.ndarray:
    """Cast a protected FP32 weight to FP16 with finite/range guards.

    Three-way handling distinguishes input-bug (raise) from
    representation-bug (clamp):
      - Source NaN/Inf: raise (source-model corruption upstream).
      - Finite but |value| > FP16_MAX: clamp to ±FP16_MAX with
        operator-visible WARNING. Preserves the run on legitimate
        outliers while surfacing them for investigation.
      - Finite within range: cast as-is.
    """
    if not np.all(np.isfinite(weight_fp32)):
        n_nan = int(np.isnan(weight_fp32).sum())
        n_inf = int(np.isinf(weight_fp32).sum())
        raise ValueError(
            f"Non-finite values in fp16_retain layer {name}: "
            f"{n_nan} NaN, {n_inf} Inf in source FP32 weight. "
            f"This indicates a source-model corruption or an "
            f"upstream bug; refusing to emit silently."
        )
    abs_max = float(np.abs(weight_fp32).max())
    if abs_max > FP16_MAX:
        n_clamped_high = int((weight_fp32 >  FP16_MAX).sum())
        n_clamped_low  = int((weight_fp32 < -FP16_MAX).sum())
        print(
            f"WARNING: fp16_retain layer {name} has values "
            f"outside FP16 range (abs_max={abs_max:.3e}); "
            f"clamping {n_clamped_high} values to +{int(FP16_MAX)} "
            f"and {n_clamped_low} values to -{int(FP16_MAX)}. "
            f"Source weight may have an outlier worth investigating.",
            flush=True,
        )
        weight_fp32 = np.clip(weight_fp32, -FP16_MAX, FP16_MAX)
    return weight_fp32.astype(np.float16)
