"""
Quantisation-Aware Training (QAT) trainer using Straight-Through Estimator.

Patent 36: Biological neural mapping — STE training mimics synaptic plasticity
           where discrete ternary states are refined by continuous gradient signals.
Patent 1:  Ternary weight encoding {-1, 0, +1} — forward pass is pure
           compare-and-add during QAT.

This trainer:
    1. Replaces eligible nn.Linear layers with TernaryLinearSTE
    2. Freezes non-ternary parameters (embeddings, LayerNorm, LM head)
    3. Trains with SGD + gradient checkpointing for memory efficiency
    4. Logs per-step loss and ternary statistics

Designed for 16GB RAM constraint: SGD (no momentum states), gradient
checkpointing, batch size 1 with gradient accumulation.
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from terncore.ste import TernaryLinearSTE
from terncore.engine.inference import TernaryInferenceEngine

logger = logging.getLogger(__name__)


PROTECT_PATTERNS = (
    "embed", "layernorm", "layer_norm", "rmsnorm",
    "lm_head", "output", "classifier",
)


@dataclass
class TrainStep:
    """Record of a single training step."""

    step: int
    loss: float
    lr: float
    time_s: float
    avg_sparsity: float = 0.0


@dataclass
class TrainResult:
    """Complete training result."""

    total_steps: int
    total_time_s: float
    final_loss: float
    initial_loss: float
    loss_reduction: float  # (initial - final) / initial
    steps: list[TrainStep] = field(default_factory=list)
    converted_layers: int = 0
    protected_layers: int = 0
    trainable_params: int = 0
    total_params: int = 0


class STETrainer:
    """
    QAT trainer for ternary models using Straight-Through Estimator.

    Converts eligible Linear layers to TernaryLinearSTE and trains with
    SGD + gradient checkpointing. Designed for memory-constrained environments.

    Args:
        model:              HuggingFace causal LM model.
        threshold:          Ternary quantisation threshold (default 0.7).
        lr:                 Learning rate for SGD (default 1e-4).
        grad_accum_steps:   Gradient accumulation steps (default 1).
        log_every:          Log every N steps (default 10).
    """

    def __init__(
        self,
        model: nn.Module,
        threshold: float = 0.7,
        lr: float = 1e-4,
        grad_accum_steps: int = 1,
        log_every: int = 10,
    ) -> None:
        self.model = model
        self.threshold = threshold
        self.lr = lr
        self.grad_accum_steps = grad_accum_steps
        self.log_every = log_every
        self.ste_params: list[nn.Parameter] = []
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self._converted = False

    def setup(self) -> tuple[int, int]:
        """
        Convert model to STE training mode.

        Returns:
            (converted_layers, protected_layers)
        """
        converted = 0
        protected = 0

        # Convert eligible Linear layers to TernaryLinearSTE
        for name, module in list(self.model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            name_lower = name.lower()
            if any(p in name_lower for p in PROTECT_PATTERNS):
                protected += 1
                continue
            ste = TernaryLinearSTE.from_linear(module, threshold=self.threshold)
            TernaryInferenceEngine._replace_module(self.model, name, ste)
            converted += 1

        # Freeze all, then enable grad on STE layers only
        for p in self.model.parameters():
            p.requires_grad = False
        for m in self.model.modules():
            if isinstance(m, TernaryLinearSTE):
                for p in m.parameters():
                    p.requires_grad = True
                    self.ste_params.append(p)

        # SGD: no momentum states, minimal memory
        self.optimizer = torch.optim.SGD(self.ste_params, lr=self.lr)

        # Enable gradient checkpointing + input requires grads
        self.model.train()
        self.model.gradient_checkpointing_enable()
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

        gc.collect()
        self._converted = True
        return converted, protected

    def train(
        self,
        data_iterator,
        num_steps: int,
        quiet: bool = False,
    ) -> TrainResult:
        """
        Run QAT training for a fixed number of steps.

        Args:
            data_iterator:  Iterable yielding input_ids tensors (batch, seq_len).
            num_steps:      Total training steps (after gradient accumulation).
            quiet:          Suppress per-step logging.

        Returns:
            TrainResult with per-step loss history.
        """
        if not self._converted:
            raise RuntimeError("Call setup() before train()")

        result = TrainResult(
            total_steps=num_steps,
            total_time_s=0.0,
            final_loss=0.0,
            initial_loss=0.0,
            loss_reduction=0.0,
            converted_layers=sum(
                1 for m in self.model.modules() if isinstance(m, TernaryLinearSTE)
            ),
            protected_layers=0,
            trainable_params=sum(p.numel() for p in self.ste_params),
            total_params=sum(p.numel() for p in self.model.parameters()),
        )

        self.optimizer.zero_grad(set_to_none=True)
        step = 0
        accum_loss = 0.0
        accum_count = 0
        t_start = time.perf_counter()

        data_iter = iter(data_iterator)

        while step < num_steps:
            # Accumulate gradients
            for _ in range(self.grad_accum_steps):
                try:
                    input_ids = next(data_iter)
                except StopIteration:
                    # Wrap around if dataset is exhausted
                    data_iter = iter(data_iterator)
                    input_ids = next(data_iter)

                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs.loss / self.grad_accum_steps
                loss.backward()
                accum_loss += outputs.loss.item()
                accum_count += 1

            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            step += 1
            avg_loss = accum_loss / accum_count
            t_elapsed = time.perf_counter() - t_start

            step_record = TrainStep(
                step=step,
                loss=avg_loss,
                lr=self.lr,
                time_s=t_elapsed,
            )
            result.steps.append(step_record)

            if step == 1:
                result.initial_loss = avg_loss

            # Log progress
            if not quiet and (step % self.log_every == 0 or step == 1 or step == num_steps):
                logger.info(
                    "  Step %4d/%d | loss=%.4f | time=%.1fs",
                    step, num_steps, avg_loss, t_elapsed,
                )

            accum_loss = 0.0
            accum_count = 0

        t_total = time.perf_counter() - t_start
        result.total_time_s = t_total
        result.final_loss = result.steps[-1].loss if result.steps else 0.0
        if result.initial_loss > 0:
            result.loss_reduction = (
                (result.initial_loss - result.final_loss) / result.initial_loss
            )

        return result

    def get_avg_sparsity(self) -> float:
        """Average sparsity across all STE layers."""
        sparsities = []
        for m in self.model.modules():
            if isinstance(m, TernaryLinearSTE):
                sparsities.append(m.sparsity)
        return sum(sparsities) / len(sparsities) if sparsities else 0.0
