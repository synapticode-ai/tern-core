"""
Tests for tern-core v0.3.0 — Model Routing.

TernaryModelRouter, ModelSpec, ModelResponse.
Uses mock engines — no real model files needed.

CNS Synaptic™ by Synapticode Co., Ltd.
"""

from dataclasses import dataclass
from pathlib import Path

import pytest

from terncore.confidence import RoutingConfidence
from terncore.model_router import ModelSpec, ModelResponse, TernaryModelRouter
from terncore.model_specs import tinyllama_spec, mistral_spec
from terncore.routing import TernaryRouter, RouteDecision
from terncore.meta import MetaAgent


# ── Mock inference engine ───────────────────────────────────────────


@dataclass
class MockResult:
    text: str
    perplexity: float
    tokens_used: int
    latency_ms: float


class MockEngine:
    """Mock inference engine — returns canned results based on perplexity."""

    def __init__(self, name: str, perplexity: float = 5.0):
        self.name = name
        self.perplexity = perplexity
        self.call_count = 0

    def generate(
        self, prompt: str, max_tokens: int = 512, temperature: float = 0.7
    ) -> MockResult:
        self.call_count += 1
        return MockResult(
            text=f"[{self.name}] response to: {prompt[:50]}",
            perplexity=self.perplexity,
            tokens_used=len(prompt.split()) * 2,
            latency_ms=10.0 if "fast" in self.name else 50.0,
        )


# ── Helpers ─────────────────────────────────────────────────────────


def make_router(
    fast_ppl: float = 5.0,
    large_ppl: float = 8.0,
    fast_scorer=None,
    large_scorer=None,
):
    """Build a TernaryModelRouter with mock engines."""
    fast_spec = ModelSpec(
        name="fast-model",
        path=Path("models/fast.tern"),
        confidence=RoutingConfidence.SURE,
        weight_min=0.85,
        max_tokens=256,
        temperature=0.3,
        scorer=fast_scorer or (lambda p: 0.90),
    )
    large_spec = ModelSpec(
        name="large-model",
        path=Path("models/large.tern"),
        confidence=RoutingConfidence.UNSURE,
        weight_min=0.30,
        max_tokens=512,
        temperature=0.7,
        scorer=large_scorer or (lambda p: 0.50),
    )

    fast_engine = MockEngine("fast-model", perplexity=fast_ppl)
    large_engine = MockEngine("large-model", perplexity=large_ppl)

    router = TernaryModelRouter(
        fast_model=fast_spec,
        large_model=large_spec,
    )
    router.register_engine("fast-model", fast_engine)
    router.register_engine("large-model", large_engine)

    return router, fast_engine, large_engine


# ── TestTernaryModelRouter ──────────────────────────────────────────


class TestTernaryModelRouter:
    def test_sure_path_dispatches_fast_model(self):
        """SURE prompt → fast model."""
        router, fast, large = make_router()
        response = router.generate("short prompt")

        assert response.model == "fast-model"
        assert response.routed_via == "sure_path"
        assert response.confidence == RoutingConfidence.SURE
        assert fast.call_count == 1

    def test_unsure_path_dispatches_large_model(self):
        """UNSURE prompt → large model."""
        router, fast, large = make_router(
            fast_scorer=lambda p: 0.40,  # deferral band
            large_scorer=lambda p: 0.50,
        )
        response = router.generate("complex query")

        assert response.model == "large-model"
        assert response.routed_via == "unsure_path"
        assert large.call_count == 1

    def test_unknown_path_dispatches_meta_agent(self):
        """UNKNOWN prompt → MetaAgent → surfaced uncertainty."""
        router, fast, large = make_router(
            fast_scorer=lambda p: -0.50,
            large_scorer=lambda p: -0.50,
        )
        response = router.generate("opaque")

        assert response.model == "meta_agent"
        assert response.routed_via == "meta_agent"
        assert response.confidence == RoutingConfidence.UNKNOWN
        assert "[UNCERTAIN]" in response.text

    def test_fast_model_uncertain_retries_large(self):
        """Fast model returns high PPL → UNSURE stacking → retry with large."""
        router, fast, large = make_router(
            fast_ppl=30.0,  # UNSURE PPL
            large_ppl=5.0,  # SURE PPL
        )
        response = router.generate("ambiguous prompt")

        # SURE route + UNSURE agent → stacked UNSURE → retry large
        assert response.model == "large-model"
        assert fast.call_count == 1  # fast tried first
        assert large.call_count == 1  # then large

    def test_large_model_unknown_dispatches_meta(self):
        """Large model returns very high PPL → UNKNOWN → MetaAgent."""
        router, fast, large = make_router(
            fast_scorer=lambda p: 0.50,  # UNSURE
            large_scorer=lambda p: 0.50,
            large_ppl=100.0,  # UNKNOWN PPL
        )
        response = router.generate("very complex")

        # UNSURE route + UNKNOWN agent → UNKNOWN → meta
        assert response.routed_via == "meta_agent"

    def test_meta_agent_resolved_re_enters_router(self):
        """MetaAgent resolves via reframe → re-enters router."""
        # Build router where reframed prompts score high
        fast_spec = ModelSpec(
            name="fast-model",
            path=Path("f.tern"),
            confidence=RoutingConfidence.SURE,
            weight_min=0.85,
            scorer=lambda p: 0.90 if "Alternative framing" in p else -0.5,
        )
        large_spec = ModelSpec(
            name="large-model",
            path=Path("l.tern"),
            confidence=RoutingConfidence.UNSURE,
            weight_min=0.30,
            scorer=lambda p: 0.50 if "Alternative framing" in p else -0.5,
        )

        router = TernaryModelRouter(fast_model=fast_spec, large_model=large_spec)
        router.register_engine("fast-model", MockEngine("fast-model", 5.0))
        router.register_engine("large-model", MockEngine("large-model", 8.0))

        response = router.generate("opaque prompt")

        # MetaAgent reframes → re-enters → fast model handles reframed
        assert response.model == "fast-model"
        assert response.confidence == RoutingConfidence.SURE

    def test_meta_agent_surfaced_returns_uncertain(self):
        """MetaAgent surfaces → UNCERTAIN response."""
        router, _, _ = make_router(
            fast_scorer=lambda p: -0.50,
            large_scorer=lambda p: -0.50,
        )
        response = router.generate("irreducible")

        assert "[UNCERTAIN]" in response.text
        assert response.tokens_used == 0
        assert response.latency_ms == 0.0

    def test_lazy_loading_fast_only_on_sure(self):
        """SURE path only loads fast model."""
        fast_spec = ModelSpec(
            name="fast", path=Path("f.tern"),
            confidence=RoutingConfidence.SURE, weight_min=0.85,
            scorer=lambda p: 0.90,
        )
        large_spec = ModelSpec(
            name="large", path=Path("l.tern"),
            confidence=RoutingConfidence.UNSURE, weight_min=0.30,
            scorer=lambda p: 0.40,
        )

        router = TernaryModelRouter(fast_model=fast_spec, large_model=large_spec)
        router.register_engine("fast", MockEngine("fast", 5.0))
        # large NOT registered — should not be needed

        response = router.generate("short")
        assert response.model == "fast"
        assert "large" not in router.loaded_models

    def test_lazy_loading_large_not_loaded_on_sure(self):
        """SURE path should not touch the large model."""
        router, fast, large = make_router()
        router.generate("short prompt")

        assert fast.call_count == 1
        assert large.call_count == 0

    def test_model_response_routed_via_correct(self):
        """Response routed_via field reflects actual path taken."""
        router, _, _ = make_router()
        resp = router.generate("test")
        assert resp.routed_via in ("sure_path", "unsure_path", "meta_agent")

    def test_perplexity_to_confidence_thresholds(self):
        """PPL-gated confidence thresholds."""
        assert TernaryModelRouter._perplexity_to_confidence(5.0) == RoutingConfidence.SURE
        assert TernaryModelRouter._perplexity_to_confidence(9.9) == RoutingConfidence.SURE
        assert TernaryModelRouter._perplexity_to_confidence(10.0) == RoutingConfidence.UNSURE
        assert TernaryModelRouter._perplexity_to_confidence(49.9) == RoutingConfidence.UNSURE
        assert TernaryModelRouter._perplexity_to_confidence(50.0) == RoutingConfidence.UNKNOWN
        assert TernaryModelRouter._perplexity_to_confidence(100.0) == RoutingConfidence.UNKNOWN

    def test_build_prompt_with_context(self):
        """Context from prior attempt prepended to prompt."""
        prompt = TernaryModelRouter._build_prompt("query", ["prior output"])
        assert "[Prior attempt]: prior output" in prompt
        assert "UNSURE" in prompt
        assert "query" in prompt

    def test_build_prompt_without_context(self):
        """No context → plain prompt."""
        prompt = TernaryModelRouter._build_prompt("query", None)
        assert prompt == "query"

    def test_stacking_fast_unsure_retries_large(self):
        """SURE route + UNSURE agent = UNSURE → retry large model."""
        router, fast, large = make_router(fast_ppl=25.0, large_ppl=5.0)
        response = router.generate("test")

        assert fast.call_count == 1
        assert large.call_count == 1
        assert response.model == "large-model"


# ── TestModelSpec ───────────────────────────────────────────────────


class TestModelSpec:
    def test_tinyllama_spec_defaults(self):
        spec = tinyllama_spec(Path("models/tinyllama.tern"))
        assert spec.name == "tinyllama-1.1b"
        assert spec.max_tokens == 256
        assert spec.temperature == 0.3
        assert spec.confidence == RoutingConfidence.SURE
        assert spec.weight_min == 0.85

    def test_mistral_spec_defaults(self):
        spec = mistral_spec(Path("models/mistral.tern"))
        assert spec.name == "mistral-7b-ternary"
        assert spec.max_tokens == 512
        assert spec.temperature == 0.7
        assert spec.confidence == RoutingConfidence.UNSURE
        assert spec.weight_min == 0.30

    def test_scorer_short_prompt_fast(self):
        """Short prompts score higher on fast model."""
        spec = tinyllama_spec(Path("t.tern"))
        short_score = spec.scorer("Short prompt here")
        long_score = spec.scorer(" ".join(["word"] * 60))
        assert short_score > long_score

    def test_scorer_long_prompt_large(self):
        """Long prompts score higher on large model."""
        spec = mistral_spec(Path("m.tern"))
        long_score = spec.scorer(" ".join(["word"] * 60))
        short_score = spec.scorer("Short")
        assert long_score > short_score
