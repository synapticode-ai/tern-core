"""
terncore — Ternary execution engine for NPU inference.

CNS Synaptic™ by Synapticode Co., Ltd.
Patents 1-9: Foundation layer.

Core principle: all neural network weights are {-1, 0, +1}.
All arithmetic is compare-and-add. No multiply-accumulate.
Determinism is non-negotiable: same input + same model = bit-identical output.
"""

__version__ = "0.4.0"

from terncore.arithmetic.quantizer import TernaryQuantizer
from terncore.arithmetic.linear import TernaryLinear
from terncore.engine.inference import TernaryInferenceEngine
from terncore.tern_model import TernModelWriter, TernModelReader
from terncore.autoscan import auto_scan, ScanResult

# v0.2.0 — Ternary Confidence Layer
from terncore.confidence import RoutingConfidence, stack_confidence
from terncore.routing import TernaryRouter, RouteDecision
from terncore.queue import ConfidenceQueue, QueuedRoute, ReleasedRoute, ReleaseReason
from terncore.meta import MetaAgent, UncertaintyReport, ResolutionStrategy

# v0.3.0 — Model Routing
from terncore.model_router import TernaryModelRouter, ModelSpec, ModelResponse
from terncore.model_specs import tinyllama_spec, mistral_spec

# v0.4.0 — CubeAction Address Protocol
from terncore.cube import (
    CubeAction,
    CubeyClient,
    Guardian,
    GuardianVerdict,
    CUBE_ADDRESS_SPACE,
    validate_address,
)
from terncore.persistence import GuardianPersistence, CubeySessionPersistence
from terncore.analytics import analyze as guardian_analyze, GuardianAnalytics, AnalyticsWindow

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
    # v0.3.0 — Model routing
    "TernaryModelRouter",
    "ModelSpec",
    "ModelResponse",
    "tinyllama_spec",
    "mistral_spec",
    # v0.4.0 — CubeAction Address Protocol
    "CubeAction",
    "CubeyClient",
    "Guardian",
    "GuardianVerdict",
    "CUBE_ADDRESS_SPACE",
    "validate_address",
    # v0.4.1 — Persistence
    "GuardianPersistence",
    "CubeySessionPersistence",
    # v0.4.2 — Analytics
    "guardian_analyze",
    "GuardianAnalytics",
    "AnalyticsWindow",
]
