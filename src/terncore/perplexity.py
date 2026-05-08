"""Standalone perplexity computation utility for tern-core.

Public API for measuring language model perplexity on benchmark text
datasets. Used by TN-003 measurement orchestration to evaluate
compressed ``.tern-model`` artefacts under varied KV-cache compression
configurations.

The implementation uses HuggingFace's canonical sliding-window
perplexity pattern (per https://huggingface.co/docs/transformers/perplexity):
slide a context window of ``max_length`` tokens with ``stride`` tokens
between window starts; mask tokens outside each new window's
contribution slice with ``-100`` so the cross-entropy loss only counts
genuinely new tokens; report ``PPL = exp(sum(NLL) / total_tokens)``.

Variant choice rationale: HF canonical is the convention TurboQuant
(ICLR 2026), KIVI (ICML 2024), and KVQuant cite when reporting
WikiText-2 PPL numbers. Using the same variant ensures published
numbers are directly comparable to ours.

Variant ALTERNATIVES (not implemented; could be added behind a
``mode=`` parameter if cross-paper reproducibility requires):
- Mean-loss-per-token (equivalent to canonical for constant-stride
  windows; differs subtly on the final partial window)
- Non-overlapping windows (faster but loses context at chunk
  boundaries; reports higher PPL; common in pre-2020 literature)

Default parameters match HuggingFace docs:
- ``stride=512`` — window step size; smaller stride → lower PPL but
  slower wall-clock
- ``max_length=2048`` — context window; conservative for most modern
  models; raise to model context limit for tighter PPL
- WikiText-2 raw v1 validation split — convention per
  papers-with-code WikiText-2 PPL benchmark

Empty-line handling: WikiText raw includes ~33% empty rows as
paragraph separators (1299 of 3760 in validation split, verified
2026-05-08). These are filtered before joining to avoid inflating
effective text density with ``\\n\\n\\n\\n`` doubles.

Copyright (c) 2025-2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

from typing import Optional

import torch


def compute_perplexity(
    model,
    tokenizer,
    dataset_name: str = "wikitext-2-raw-v1",
    split: str = "validation",
    stride: int = 512,
    max_length: int = 2048,
    device: Optional[str] = None,
) -> float:
    """Compute language model perplexity on a benchmark text dataset.

    Implements the HuggingFace canonical sliding-window perplexity
    pattern. See module docstring for variant rationale + parameter
    conventions.

    Caller is responsible for setting the model to eval mode
    (``model.eval()``) before calling this function. The function does
    not modify model state.

    Args:
        model: HuggingFace causal LM. Must accept
            ``model(input_ids, labels=...)`` and return an object with
            a ``.loss`` attribute (mean cross-entropy over non-masked
            tokens).
        tokenizer: Matching HuggingFace tokenizer.
        dataset_name: HF datasets identifier. WikiText configs
            (``wikitext-2-*``, ``wikitext-103-*``) loaded via the
            ``wikitext`` parent dataset; other names passed directly to
            ``load_dataset``.
        split: Dataset split (default: ``validation``).
        stride: Sliding-window step size (default: ``512``).
        max_length: Context window length (default: ``2048``).
        device: Optional device override; defaults to model's device.

    Returns:
        Perplexity. Lower is better; typical values 5-50 for modern
        LMs on WikiText-2 validation.
    """
    from datasets import load_dataset

    if device is None:
        device = next(model.parameters()).device

    # WikiText configs share the parent dataset "wikitext"; other
    # datasets pass their identifier directly to load_dataset.
    if dataset_name.startswith("wikitext-"):
        ds = load_dataset("wikitext", dataset_name, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)

    text = "\n\n".join(row["text"] for row in ds if row["text"].strip())

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        window_input_ids = input_ids[:, begin_loc:end_loc]
        target_ids = window_input_ids.clone()
        # Mask tokens already counted by previous windows so loss only
        # counts new contributions from this window.
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(window_input_ids, labels=target_ids)
            # outputs.loss is mean over trg_len non-masked positions;
            # multiply back to get total NLL contribution.
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    return float(torch.exp(torch.stack(nlls).sum() / end_loc))
