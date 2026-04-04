"""
terncore.model_router — Routes inference across model tiers using ternary confidence.

Fast model  → SURE queries   (weight ≥ 0.85)
Large model → UNSURE queries (weight 0.30–0.85)
MetaAgent   → UNKNOWN        (weight < 0.30)

The router scores the prompt before loading the model.
Expensive models only load when confidence warrants it.

CNS Synaptic™ by Synapticode Co., Ltd.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Protocol

from terncore.confidence import RoutingConfidence, stack_confidence
from terncore.meta import MetaAgent, ResolutionResult
from terncore.queue import ConfidenceQueue
from terncore.routing import RouteDecision, TernaryRouter


# MARK: - Engine Protocol

class InferenceResult(Protocol):
    """Protocol for inference engine output."""

    @property
    def text(self) -> str: ...
    @property
    def perplexity(self) -> float: ...
    @property
    def tokens_used(self) -> int: ...
    @property
    def latency_ms(self) -> float: ...


class InferenceEngine(Protocol):
    """Protocol for inference engines — real or mock."""

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> InferenceResult: ...


# MARK: - ModelSpec


@dataclass
class ModelSpec:
    """Specification for a model tier."""

    name: str
    path: Path
    confidence: RoutingConfidence  # which tier this model serves
    weight_min: float  # minimum routing weight to use this model
    max_tokens: int = 512
    temperature: float = 0.7
    scorer: Optional[Callable[[str], float]] = None


# MARK: - ModelResponse


@dataclass(frozen=True)
class ModelResponse:
    """Response from TernaryModelRouter with full routing audit trail."""

    text: str
    model: str  # which model produced this
    confidence: RoutingConfidence
    weight: float
    routed_via: str  # "sure_path" / "unsure_path" / "meta_agent"
    tokens_used: int
    latency_ms: float


# MARK: - TernaryModelRouter


class TernaryModelRouter:
    """
    Routes inference requests across multiple model tiers
    based on ternary confidence scoring.

    Fast model  → SURE queries   (weight ≥ 0.85)
    Large model → UNSURE queries (weight 0.30–0.85)
    MetaAgent   → UNKNOWN        (weight < 0.30)
    """

    def __init__(
        self,
        fast_model: ModelSpec,
        large_model: ModelSpec,
        queue: Optional[ConfidenceQueue] = None,
        meta: Optional[MetaAgent] = None,
        engine_factory: Optional[Callable[[ModelSpec], InferenceEngine]] = None,
    ):
        self._fast = fast_model
        self._large = large_model
        self._queue = queue or ConfidenceQueue()
        self._engines: dict[str, InferenceEngine] = {}
        self._engine_factory = engine_factory
        self._router = TernaryRouter(
            escalation_threshold=0.30,
            deferral_band=(0.30, 0.85),
        )

        # Register model scorers
        if fast_model.scorer:
            self._router.register(fast_model.name, fast_model.scorer)
        if large_model.scorer:
            self._router.register(large_model.name, large_model.scorer)

        # MetaAgent wraps the router
        self._meta = meta or MetaAgent(
            router=self._router,
            assess=lambda p: self._router.route(p),
        )

    def generate(
        self,
        prompt: str,
        context: Optional[list[str]] = None,
        max_tokens: Optional[int] = None,
    ) -> ModelResponse:
        """Route prompt → score → dispatch to correct model tier."""
        decision = self._router.route(prompt)

        if decision.dispatchable:
            return self._dispatch_fast(prompt, decision, max_tokens)
        elif decision.should_defer:
            return self._dispatch_large(prompt, decision, context, max_tokens)
        else:
            return self._dispatch_meta(prompt, decision)

    def _dispatch_fast(
        self,
        prompt: str,
        decision: RouteDecision,
        max_tokens: Optional[int],
    ) -> ModelResponse:
        engine = self._load_engine(self._fast)
        result = engine.generate(
            prompt,
            max_tokens=max_tokens or self._fast.max_tokens,
            temperature=self._fast.temperature,
        )

        agent_conf = self._perplexity_to_confidence(result.perplexity)
        stacked = stack_confidence(decision.confidence, agent_conf)

        if stacked == RoutingConfidence.UNSURE:
            # Fast model uncertain — retry with large
            return self._dispatch_large(
                prompt, decision, [result.text], max_tokens
            )

        return ModelResponse(
            text=result.text,
            model=self._fast.name,
            confidence=stacked,
            weight=decision.weight,
            routed_via="sure_path",
            tokens_used=result.tokens_used,
            latency_ms=result.latency_ms,
        )

    def _dispatch_large(
        self,
        prompt: str,
        decision: RouteDecision,
        context: Optional[list[str]],
        max_tokens: Optional[int],
    ) -> ModelResponse:
        engine = self._load_engine(self._large)
        full_prompt = self._build_prompt(prompt, context)
        result = engine.generate(
            full_prompt,
            max_tokens=max_tokens or self._large.max_tokens,
            temperature=self._large.temperature,
        )

        agent_conf = self._perplexity_to_confidence(result.perplexity)
        stacked = stack_confidence(decision.confidence, agent_conf)

        if stacked == RoutingConfidence.UNKNOWN:
            return self._dispatch_meta(prompt, decision)

        return ModelResponse(
            text=result.text,
            model=self._large.name,
            confidence=stacked,
            weight=decision.weight,
            routed_via="unsure_path",
            tokens_used=result.tokens_used,
            latency_ms=result.latency_ms,
        )

    def _dispatch_meta(
        self,
        prompt: str,
        decision: RouteDecision,
    ) -> ModelResponse:
        outcome = self._meta.handle(prompt, decision)

        if outcome.outcome == "resolved" and outcome.prompt:
            return self.generate(outcome.prompt)

        report = outcome.report
        summary = report.summary if report else "Unresolvable"
        return ModelResponse(
            text=f"[UNCERTAIN] {summary}",
            model="meta_agent",
            confidence=RoutingConfidence.UNKNOWN,
            weight=decision.weight,
            routed_via="meta_agent",
            tokens_used=0,
            latency_ms=0.0,
        )

    def _load_engine(self, spec: ModelSpec) -> InferenceEngine:
        """Lazy-load — expensive models only when needed."""
        if spec.name not in self._engines:
            if self._engine_factory:
                self._engines[spec.name] = self._engine_factory(spec)
            else:
                raise RuntimeError(
                    f"No engine factory configured and model '{spec.name}' not pre-loaded. "
                    f"Provide engine_factory or pre-register engines via register_engine()."
                )
        return self._engines[spec.name]

    def register_engine(self, name: str, engine: InferenceEngine) -> None:
        """Pre-register an engine — used for testing and pre-loading."""
        self._engines[name] = engine

    @property
    def loaded_models(self) -> list[str]:
        """Names of currently loaded models."""
        return list(self._engines.keys())

    @staticmethod
    def _perplexity_to_confidence(ppl: float) -> RoutingConfidence:
        """PPL-gated confidence — existing tern-core mapping."""
        if ppl < 10:
            return RoutingConfidence.SURE
        if ppl < 50:
            return RoutingConfidence.UNSURE
        return RoutingConfidence.UNKNOWN

    @staticmethod
    def _build_prompt(prompt: str, context: Optional[list[str]]) -> str:
        if not context:
            return prompt
        ctx = "\n".join(f"[Prior attempt]: {c}" for c in context)
        return f"{ctx}\n\nRouting confidence: UNSURE — exercise extra caution.\n{prompt}"
