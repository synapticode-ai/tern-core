"""
terncore — Ternary execution engine for NPU inference.

CNS Synaptic™ by Synapticode Co., Ltd.
Patents 1-9: Foundation layer.

Core principle: all neural network weights are {-1, 0, +1}.
All arithmetic is compare-and-add. No multiply-accumulate.
Determinism is non-negotiable: same input + same model = bit-identical output.
"""

__version__ = "0.1.0"

from terncore.arithmetic.quantizer import TernaryQuantizer
from terncore.arithmetic.linear import TernaryLinear
from terncore.engine.inference import TernaryInferenceEngine
from terncore.model_loader.tern_model import TernModelWriter, TernModelReader
from terncore.autoscan import auto_scan, ScanResult

__all__ = [
    "TernaryQuantizer",
    "TernaryLinear",
    "TernaryInferenceEngine",
    "TernModelWriter",
    "TernModelReader",
    "auto_scan",
    "ScanResult",
]
