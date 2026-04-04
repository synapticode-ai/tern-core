"""
terncore.model_specs — Convenience constructors for common model tiers.

CNS Synaptic™ by Synapticode Co., Ltd.
"""

from pathlib import Path

from terncore.confidence import RoutingConfidence
from terncore.model_router import ModelSpec


def tinyllama_spec(model_path: Path) -> ModelSpec:
    """TinyLlama 1.1B — fast tier, SURE queries."""
    return ModelSpec(
        name="tinyllama-1.1b",
        path=model_path,
        confidence=RoutingConfidence.SURE,
        weight_min=0.85,
        max_tokens=256,
        temperature=0.3,
        scorer=lambda p: 0.90 if len(p.split()) < 50 else 0.70,
    )


def mistral_spec(model_path: Path) -> ModelSpec:
    """Mistral 7B ternary — large tier, UNSURE queries."""
    return ModelSpec(
        name="mistral-7b-ternary",
        path=model_path,
        confidence=RoutingConfidence.UNSURE,
        weight_min=0.30,
        max_tokens=512,
        temperature=0.7,
        scorer=lambda p: 0.60 if len(p.split()) >= 50 else 0.40,
    )
