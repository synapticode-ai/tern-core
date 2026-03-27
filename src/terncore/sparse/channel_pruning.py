"""
Channel pruning for structured sparsity on ANE.

The ANE executes dense matrix multiplications — it cannot skip individual
zero weights. To exploit ternary sparsity on ANE, we must create *structured*
zeros: entire output channels (rows) or input channels (columns) that are
fully zero, then physically remove them to produce smaller matmuls.

Strategy:
  1. Score each channel by importance (L1 norm of ternary weights)
  2. Identify low-importance channels that contribute least
  3. Prune entire channels → creates structurally sparse layers
  4. Build pruned model with physically smaller Linear layers
  5. Smaller matmuls → faster ANE dispatch, lower power

Patent 37: Zero-weight clock-gating → channel-level zero-skip.
Patent 7:  Sparsity-aware execution → structured pruning for hardware.

Terncore · Cubey/Synapticode · 2026
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ChannelPruneStats:
    """Statistics from channel pruning a single layer."""
    layer_name: str
    original_out: int
    original_in: int
    pruned_out: int
    pruned_in: int
    channels_removed: int
    prune_ratio: float
    weight_sparsity_before: float
    weight_sparsity_after: float  # should be 0 for removed channels


def score_channel_importance(weight: torch.Tensor) -> torch.Tensor:
    """Score each output channel by L1 norm.

    Args:
        weight: (out_features, in_features) weight tensor.

    Returns:
        (out_features,) importance scores — higher = more important.
    """
    return weight.abs().sum(dim=1).float()


def detect_prunable_channels(
    weight: torch.Tensor,
    prune_ratio: float = 0.0,
    min_importance: float = 0.0,
) -> torch.Tensor:
    """Identify output channels to prune.

    Channels are pruned if they are either:
      - Entirely zero (always pruned regardless of ratio), OR
      - Below the prune_ratio threshold by L1 importance

    Args:
        weight:         (out_features, in_features) weight tensor.
        prune_ratio:    Fraction of channels to prune (0.0 to 1.0).
        min_importance: Absolute threshold — prune channels below this.

    Returns:
        Boolean mask (out_features,) — True = KEEP, False = PRUNE.
    """
    scores = score_channel_importance(weight)
    keep = torch.ones(weight.shape[0], dtype=torch.bool)

    # Always prune fully-zero channels
    keep[scores == 0] = False

    # Prune by ratio (lowest importance)
    if prune_ratio > 0:
        n_prune = int(weight.shape[0] * prune_ratio)
        if n_prune > 0:
            _, indices = scores.sort()
            keep[indices[:n_prune]] = False

    # Prune by absolute threshold
    if min_importance > 0:
        keep[scores < min_importance] = False

    return keep


def prune_linear_output(
    linear: nn.Linear,
    keep_mask: torch.Tensor,
) -> nn.Linear:
    """Create a new Linear layer with pruned output channels.

    Args:
        linear:    Original nn.Linear layer.
        keep_mask: Boolean (out_features,) — True = keep channel.

    Returns:
        New nn.Linear with fewer output features.
    """
    kept_indices = keep_mask.nonzero(as_tuple=True)[0]
    new_out = kept_indices.shape[0]

    new_linear = nn.Linear(
        linear.in_features, new_out,
        bias=linear.bias is not None,
    )
    new_linear.weight = nn.Parameter(linear.weight.data[kept_indices])
    if linear.bias is not None:
        new_linear.bias = nn.Parameter(linear.bias.data[kept_indices])

    return new_linear


def prune_linear_input(
    linear: nn.Linear,
    keep_mask: torch.Tensor,
) -> nn.Linear:
    """Create a new Linear layer with pruned input channels.

    Args:
        linear:    Original nn.Linear layer.
        keep_mask: Boolean (in_features,) — True = keep channel.

    Returns:
        New nn.Linear with fewer input features.
    """
    kept_indices = keep_mask.nonzero(as_tuple=True)[0]
    new_in = kept_indices.shape[0]

    new_linear = nn.Linear(
        new_in, linear.out_features,
        bias=linear.bias is not None,
    )
    new_linear.weight = nn.Parameter(linear.weight.data[:, kept_indices])
    if linear.bias is not None:
        new_linear.bias = nn.Parameter(linear.bias.data.clone())

    return new_linear


def prune_mlp_channels(
    gate_proj: nn.Linear,
    up_proj: nn.Linear,
    down_proj: nn.Linear,
    prune_ratio: float = 0.3,
) -> Tuple[nn.Linear, nn.Linear, nn.Linear, ChannelPruneStats]:
    """Prune the MLP intermediate dimension (gate/up output, down input).

    The MLP computes: down_proj(silu(gate_proj(x)) * up_proj(x))
    gate_proj and up_proj share the same output dimension (intermediate_size).
    down_proj takes that as input. We prune channels from the intermediate dim.

    Importance is scored jointly: a channel is important only if BOTH
    gate_proj and up_proj have significant weights for it.

    Args:
        gate_proj:   Linear(hidden, intermediate).
        up_proj:     Linear(hidden, intermediate).
        down_proj:   Linear(intermediate, hidden).
        prune_ratio: Fraction of intermediate channels to prune.

    Returns:
        (pruned_gate, pruned_up, pruned_down, stats)
    """
    # Joint importance: geometric mean of gate and up channel norms
    gate_scores = score_channel_importance(gate_proj.weight)
    up_scores = score_channel_importance(up_proj.weight)
    joint_scores = (gate_scores * up_scores).sqrt()

    # Build keep mask from joint scores
    n_channels = gate_proj.out_features
    n_prune = int(n_channels * prune_ratio)
    keep_mask = torch.ones(n_channels, dtype=torch.bool)
    if n_prune > 0:
        _, indices = joint_scores.sort()
        keep_mask[indices[:n_prune]] = False

    n_kept = keep_mask.sum().item()

    # Prune all three layers
    pruned_gate = prune_linear_output(gate_proj, keep_mask)
    pruned_up = prune_linear_output(up_proj, keep_mask)
    pruned_down = prune_linear_input(down_proj, keep_mask)

    stats = ChannelPruneStats(
        layer_name="mlp",
        original_out=n_channels,
        original_in=down_proj.out_features,
        pruned_out=n_kept,
        pruned_in=down_proj.out_features,
        channels_removed=n_prune,
        prune_ratio=n_prune / n_channels,
        weight_sparsity_before=(gate_proj.weight == 0).float().mean().item(),
        weight_sparsity_after=(pruned_gate.weight == 0).float().mean().item(),
    )

    return pruned_gate, pruned_up, pruned_down, stats


def prune_attention_channels(
    q_proj: nn.Linear,
    o_proj: nn.Linear,
    prune_ratio: float = 0.2,
) -> Tuple[nn.Linear, nn.Linear, ChannelPruneStats]:
    """Prune the attention internal dimension (q output, o input).

    In the simplified benchmark: o_proj(q_proj(x)) + x
    q_proj output dim = o_proj input dim. Prune this shared dimension.

    Args:
        q_proj:      Linear(hidden, head_dim).
        o_proj:      Linear(head_dim, hidden).
        prune_ratio: Fraction of channels to prune.

    Returns:
        (pruned_q, pruned_o, stats)
    """
    scores = score_channel_importance(q_proj.weight)
    n_channels = q_proj.out_features
    n_prune = int(n_channels * prune_ratio)
    keep_mask = torch.ones(n_channels, dtype=torch.bool)
    if n_prune > 0:
        _, indices = scores.sort()
        keep_mask[indices[:n_prune]] = False

    n_kept = keep_mask.sum().item()
    pruned_q = prune_linear_output(q_proj, keep_mask)
    pruned_o = prune_linear_input(o_proj, keep_mask)

    stats = ChannelPruneStats(
        layer_name="attention",
        original_out=n_channels,
        original_in=o_proj.out_features,
        pruned_out=n_kept,
        pruned_in=o_proj.out_features,
        channels_removed=n_prune,
        prune_ratio=n_prune / n_channels,
        weight_sparsity_before=(q_proj.weight == 0).float().mean().item(),
        weight_sparsity_after=(pruned_q.weight == 0).float().mean().item(),
    )

    return pruned_q, pruned_o, stats
