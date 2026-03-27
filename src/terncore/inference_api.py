"""
inference_api.py — High-level text generation API for ternary TinyLlama.

Provides a single-function interface for the orchestrator and other callers:

    from terncore.inference_api import generate

    result = generate("Summarise this brief: ...", max_tokens=100)
    print(result["text"])

Uses the v_proj_late3 mixed-precision configuration (3 ternary layers in
blocks 19-21) which is the only validated deployment config for TinyLlama
within <5% quality target (+4.1% PPL gap).

The model is loaded lazily on first call and cached for subsequent calls.

For raw tensor inference via CoreML/ANE, use coreml_predict() instead.

Synapticode Co., Ltd. — Patent 36 (deterministic inference).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Lazy-loaded singleton model cache
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict = {}

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# v_proj_late3: the 3 layers safe to ternize (Day 3 sensitivity analysis)
V_PROJ_LATE3_LAYERS = [
    "model.layers.19.self_attn.v_proj",
    "model.layers.20.self_attn.v_proj",
    "model.layers.21.self_attn.v_proj",
]

COREML_MODEL_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "output" / "coreml_models" / "ternstack_ternary_2bit.mlpackage"
)


@dataclass
class GenerationResult:
    """Result from a text generation call."""
    text: str
    tokens: int
    latency_ms: float
    tokens_per_second: float
    model: str = MODEL_ID
    ternary_layers: int = 3

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "tokens": self.tokens,
            "latency_ms": round(self.latency_ms, 1),
            "tokens_per_second": round(self.tokens_per_second, 1),
            "model": self.model,
            "ternary_layers": self.ternary_layers,
        }


def _load_model():
    """Load TinyLlama with v_proj_late3 ternary conversion. Cached."""
    if "model" in _MODEL_CACHE:
        return _MODEL_CACHE["model"], _MODEL_CACHE["tokenizer"]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from terncore.mixed_precision import MixedPrecisionConverter

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Protect everything except v_proj_late3
    all_linears = [
        name for name, m in model.named_modules()
        if isinstance(m, torch.nn.Linear)
    ]
    protection_list = [
        name for name in all_linears if name not in V_PROJ_LATE3_LAYERS
    ]

    converter = MixedPrecisionConverter(
        threshold=0.7,
        protection_list=protection_list,
    )
    converter.convert(model)
    model.eval()

    _MODEL_CACHE["model"] = model
    _MODEL_CACHE["tokenizer"] = tokenizer
    return model, tokenizer


def generate(
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.0,
) -> GenerationResult:
    """Generate text using ternary TinyLlama.

    Args:
        prompt: Input text prompt.
        max_tokens: Maximum tokens to generate.
        temperature: 0.0 for greedy (deterministic), >0 for sampling.

    Returns:
        GenerationResult with text, latency, and throughput stats.
    """
    model, tokenizer = _load_model()

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generated_ids = input_ids.clone()
    tokens_generated = 0

    t0 = time.perf_counter()

    with torch.no_grad():
        for _ in range(max_tokens):
            outputs = model(generated_ids)
            logits = outputs.logits[:, -1, :]

            if temperature <= 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            if next_id.item() == tokenizer.eos_token_id:
                break

            generated_ids = torch.cat([generated_ids, next_id], dim=-1)
            tokens_generated += 1

    elapsed_ms = (time.perf_counter() - t0) * 1000
    tps = tokens_generated / (elapsed_ms / 1000) if elapsed_ms > 0 else 0

    full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    # Strip the original prompt to return only the generated portion
    generated_text = full_text[len(tokenizer.decode(input_ids[0], skip_special_tokens=True)):].strip()

    return GenerationResult(
        text=generated_text,
        tokens=tokens_generated,
        latency_ms=elapsed_ms,
        tokens_per_second=tps,
    )


def coreml_predict(
    input_array=None,
    compute_units: str = "CPU_AND_NE",
) -> dict:
    """Run a raw tensor forward pass via CoreML/ANE.

    This is the ternary linear stack (22 blocks × 7 layers = 154 matmuls),
    not the full LLM. Used for latency benchmarks and forward-pass testing.

    Args:
        input_array: numpy array of shape (1, 64, 2048), dtype float16.
                     If None, generates random input.
        compute_units: "CPU_AND_NE" (ANE), "CPU_AND_GPU", "CPU_ONLY", "ALL".

    Returns:
        Dict with output array, latency_ms, and compute_units.
    """
    import coremltools as ct
    import numpy as np

    cu_map = {
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "ALL": ct.ComputeUnit.ALL,
    }
    cu = cu_map.get(compute_units, ct.ComputeUnit.CPU_AND_NE)

    if not COREML_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"CoreML model not found: {COREML_MODEL_PATH}\n"
            f"Run benchmarks/bench_coreml_ane.py to generate it."
        )

    if "coreml_model" not in _MODEL_CACHE:
        _MODEL_CACHE["coreml_model"] = ct.models.MLModel(
            str(COREML_MODEL_PATH), compute_units=cu
        )

    mlmodel = _MODEL_CACHE["coreml_model"]

    if input_array is None:
        input_array = np.random.randn(1, 64, 2048).astype(np.float16)

    # Warmup
    mlmodel.predict({"input": input_array})

    t0 = time.perf_counter()
    output = mlmodel.predict({"input": input_array})
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "output": output["output"],
        "latency_ms": round(elapsed_ms, 2),
        "compute_units": compute_units,
        "shape": list(output["output"].shape),
    }
