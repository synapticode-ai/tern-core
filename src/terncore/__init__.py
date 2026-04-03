"""
terncore — Ternary execution engine for NPU inference.

CNS Synaptic™ by Synapticode Co., Ltd.
Patents 1-9: Foundation layer.

Core principle: all neural network weights are {-1, 0, +1}.
All arithmetic is compare-and-add. No multiply-accumulate.
Determinism is non-negotiable: same input + same model = bit-identical output.
"""

__version__ = "0.2.0"

from terncore.arithmetic.quantizer import TernaryQuantizer
from terncore.arithmetic.linear import TernaryLinear
from terncore.engine.inference import TernaryInferenceEngine
from terncore.model_loader.tern_model import TernModelWriter, TernModelReader
from terncore.autoscan import auto_scan, ScanResult

# v0.2.0 — Ternary Confidence Layer
from terncore.confidence import RoutingConfidence, stack_confidence
from terncore.routing import TernaryRouter, RouteDecision
from terncore.queue import ConfidenceQueue, QueuedRoute, ReleasedRoute, ReleaseReason
from terncore.meta import MetaAgent, UncertaintyReport, ResolutionStrategy

__all__ = [
    # v0.1.0 — Ternary execution engine
    "TernaryQuantizer",
    "TernaryLinear",
    "TernaryInferenceEngine",
    "TernModelWriter",
    "TernModelReader",
    "auto_scan",
    "ScanResult",
    # v0.2.0 — Ternary confidence layer
    "RoutingConfidence",
    "stack_confidence",
    "TernaryRouter",
    "RouteDecision",
    "ConfidenceQueue",
    "QueuedRoute",
    "ReleasedRoute",
    "ReleaseReason",
    "MetaAgent",
    "UncertaintyReport",
    "ResolutionStrategy",
]
