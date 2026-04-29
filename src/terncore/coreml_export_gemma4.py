"""
CoreML exporter for Gemma 4 language model.

Exports the text-only language model portion of Gemma 4
(Gemma4ForConditionalGeneration) to a CoreML .mlpackage via
torch.export + coremltools conversion.

Vision and audio encoders are excluded — this targets text-only
inference benchmarking on CPU/ANE/GPU compute units.

Part of tern-core v0.6.0 compression stack.

Copyright (c) 2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def export_gemma4_coreml(
    model_id: str,
    output_path: str,
    seq_len: int = 64,
    verbose: bool = True,
) -> None:
    """Export Gemma 4 language model to CoreML .mlpackage.

    Args:
        model_id:    HuggingFace model ID or local path.
        output_path: Path for the .mlpackage output.
        seq_len:     Fixed sequence length.
        verbose:     Print progress.
    """
    import coremltools as ct

    def _log(msg):
        if verbose:
            print(msg, flush=True)

    t0 = time.time()
    _log(f"Exporting Gemma 4 to CoreML")
    _log(f"  Model:   {model_id}")
    _log(f"  Seq len: {seq_len}")
    _log(f"  Output:  {output_path}")

    # ── Load language model only ──
    _log("\n  Loading language model...")
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    _log(f"  Loaded: {type(model).__name__}")

    # ── Export the model ──
    _log("\n  Exporting model via torch.export...")
    example_input = torch.zeros(1, seq_len, dtype=torch.long)

    # Use torch.export.export which handles dynamic control flow via
    # graph breaks, unlike torch.jit.trace which fails on Gemma 4's
    # complex masking logic.
    from torch.export import export

    class LogitsWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids):
            outputs = self.model(input_ids=input_ids, use_cache=False)
            return outputs.logits

    wrapper = LogitsWrapper(model)
    wrapper.eval()

    with torch.no_grad():
        exported = export(wrapper, (example_input,), strict=False)

    # Decompose to ATEN dialect for coremltools compatibility
    exported = exported.run_decompositions({})
    _log("  Export complete")

    # Free original model to save memory before conversion
    del model, wrapper
    gc.collect()

    # ── Convert to CoreML ──
    _log("\n  Converting to CoreML mlprogram...")
    mlmodel = ct.convert(
        exported,
        inputs=[ct.TensorType(
            name="input_ids",
            shape=(1, seq_len),
            dtype=np.int32,
        )],
        outputs=[ct.TensorType(name="logits")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS18,
        compute_units=ct.ComputeUnit.ALL,
    )

    del traced
    gc.collect()

    # ── Save ──
    _log(f"\n  Saving to {output_path}...")
    mlmodel.save(output_path)

    elapsed = time.time() - t0
    _log(f"  Done in {elapsed:.1f}s")

    # ── Validate ──
    _log("\n  Validating...")
    spec = mlmodel.get_spec()
    _log(f"  Spec version: {spec.specificationVersion}")
    inp = spec.description.input[0]
    out = spec.description.output[0]
    _log(f"  Input:  {inp.name} {list(inp.type.multiArrayType.shape)}")
    _log(f"  Output: {out.name}")
    _log(f"  Compute units: ALL")

    return mlmodel


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export Gemma 4 to CoreML .mlpackage"
    )
    parser.add_argument("--model", required=True,
                        help="HuggingFace model ID or local path")
    parser.add_argument("--output", required=True,
                        help="Output .mlpackage path")
    parser.add_argument("--seq-len", type=int, default=64,
                        help="Sequence length (default: 64)")
    args = parser.parse_args()

    export_gemma4_coreml(args.model, args.output, seq_len=args.seq_len)
