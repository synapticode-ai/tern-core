#!/usr/bin/env python3
# Copyright (c) 2026 Gamma Seeds Pte Ltd. Inventor: Robert Lakelin.
"""
bench_gemopus_phase2.py — Gemopus-4-E4B-it / Gemma 4 E4B-it phase-2 benchmark harness.

Parameterised generation benchmark for the six-row Gemopus-4-E4B-it
comparison (rows 1, 1', 2, 3, 3a, 4). Methodology mirrors
energy_cleanroom.py: sudo-backed powermetrics, Combined Power parsing,
warmup-then-timed, deterministic seed, fixed prompt across rows.

Format dispatch:
  mlx_fp16              — mlx_lm.stream_generate on a HuggingFace model id.
                          Text-only models. Fails on multimodal architectures
                          (Gemma 4 family, etc.) — use mlx_vlm_bf16 instead.
  mlx_vlm_bf16          — mlx_vlm.stream_generate on an mlx-community
                          BF16-converted multimodal model. Required for
                          Gemma 4 family (multimodal-by-architecture, with
                          vision + audio encoders carried even in text-only
                          use). BF16 is the native HF-default for Gemma; the
                          mlx-community variant preserves it un-quantized.
  pytorch_mps_fp16      — transformers + torch.float16 on device='mps'.
  pytorch_mps_ternary   — transformers + TernaryInferenceEngine.convert()
                          (in-memory generic ternisation — see Caveat 1
                          for the divergence from the persisted .tern-model).
  llamacpp_gguf         — subprocess llama-cli with --seed and equivalent
                          measurement plumbing.

Output: ./gemopus_4_e4b_phase2/<label>.json (relative to this file's dir).

Standard prompt (do not change without re-baselining the entire sprint cluster):
  "Describe the process of photosynthesis at a high-school level. Cover the
   inputs, outputs, and where the reaction takes place. Keep your answer to
   roughly four sentences."

REVIEW CAVEATS — please validate before running:

  1. TernaryInferenceEngine signature (verified 2026-05-01 against
     engine/inference.py:74-117): ``TernaryInferenceEngine(threshold=)``
     — no ``adapter`` kwarg. ``engine.convert(model)`` mutates ``model``
     in place and returns a ``ConversionReport``. The ``--adapter`` CLI
     flag is retained for parity with ``terncore.convert`` invocations
     but is NOT consumed by ``TernaryInferenceEngine`` — the in-memory
     engine uses generic layer-protection rules (``_should_protect()``
     against generic name patterns). The persisted ``.tern-model``
     artefact (written by ``terncore.convert --adapter gemma4
     --mixed-int4``) runs through a separate adapter-aware code path
     in ``convert.py``. The two paths share methodology but produce
     different weight protection maps. This is **Benchmark Phase 1**:
     in-memory generic ternisation as the methodology-validation row
     set. **Benchmark Phase 2** (next sprint) will round-trip the
     persisted artefact via a TernModelReader for byte-identical
     numbers against the canonical Korean NPU deliverable. Phase 1 and
     Phase 2 numbers are directly comparable: prompt, max-tokens, seed,
     and harness shape are fixed across both phases.
  2. llama-cli flag set: ``--seed``, ``-no-cnv``, ``--temp 0`` are the
     forms in current llama.cpp; older / newer builds may rename these.
  3. mlx random reseed: warmup advances mlx RNG state, so the timed
     run reseeds before stream_generate. Confirm this matches the
     sprint's reproducibility convention.
  4. PyTorch two-phase prompt-eval split: prompt_eval_time captures one
     forward pass over the prompt tensor before generate(), with
     ``torch.mps.synchronize()`` between phases. This is an
     approximation of "time to first token", not a per-token disaggregation.

  5. MLX coverage verification + library selection (sprint-wide pre-flight
     pattern, banked 2026-05-01 from the gemma-4-E4B-it Row 1 hybrid-
     attention failure):

     (a) Verify the chosen MLX library can load the target model in
         Phase 0 (load only, no generation). If load fails on architecture
         coverage, the row plan adjusts before download — bump the library,
         find an ``mlx-community`` pre-converted variant, or defer the MLX
         reference row to a later Benchmark Phase.
     (b) Library selection: ``mlx_lm`` is the right tool for text-only
         architectures; multimodal architectures (Gemma 4 family carries
         vision + audio encoders even in text-only use) need ``mlx_vlm``.
         For any new model in the Gemopus → 26B MoE → 26B-A4B → 31B sprint
         cluster, probe ``mlx_vlm.load()`` first when the architecture is
         multimodal-flavoured; fall back to ``mlx_lm.load()`` only for
         purely text-only models. The processor returned by ``mlx_vlm.load``
         wraps the tokenizer — handle accordingly.
     (c) MLX maturity for newly-released model architectures often trails
         the HF release by weeks to months; pre-flight catches this cheaply.

Engine version stack at session start: mlx 0.31.2, mlx-lm 0.31.3,
mlx-metal 0.31.2, terncore 0.4.0 (commit f48a7e5, PR #8 Group A merge).
Record this in REPORT.md methodology section.

Usage:
  python bench_gemopus_phase2.py --format mlx_fp16 \\
      --model google/gemma-4-E4B-it \\
      --label row1_gemma4_e4b_mlx_fp16

  python bench_gemopus_phase2.py --format mlx_vlm_bf16 \\
      --model mlx-community/gemma-4-e4b-it-bf16 \\
      --label row1_gemma4_e4b_mlx_vlm_bf16

  python bench_gemopus_phase2.py --format pytorch_mps_fp16 \\
      --model google/gemma-4-E4B-it \\
      --label row1prime_gemma4_e4b_pytorch_mps_fp16

  python bench_gemopus_phase2.py --format pytorch_mps_ternary \\
      --model google/gemma-4-E4B-it --adapter gemma4 --threshold 0.7 \\
      --label row2_gemma4_e4b_pytorch_mps_ternary

  python bench_gemopus_phase2.py --format pytorch_mps_ternary \\
      --model Jackrong/Gemopus-4-E4B-it --adapter gemma4 --threshold 0.7 \\
      --label row3_gemopus_4_e4b_pytorch_mps_ternary_thinking_off

  python bench_gemopus_phase2.py --format pytorch_mps_ternary \\
      --model Jackrong/Gemopus-4-E4B-it --adapter gemma4 --threshold 0.7 \\
      --system-prompt-prefix "<|think|>" \\
      --label row3a_gemopus_4_e4b_pytorch_mps_ternary_thinking_on

  python bench_gemopus_phase2.py --format llamacpp_gguf \\
      --model /path/to/Gemopus-4-E4B-it.Q4_K_M.gguf \\
      --label row4_gemopus_4_e4b_llamacpp_gguf
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import re
import resource
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

PROMPT = (
    "Describe the process of photosynthesis at a high-school level. "
    "Cover the inputs, outputs, and where the reaction takes place. "
    "Keep your answer to roughly four sentences."
)

WARMUP_TOKENS = 50
DEFAULT_MAX_TOKENS = 200
DEFAULT_SEED = 42
DEFAULT_THRESHOLD = 0.7

OUTPUT_DIR = Path(__file__).resolve().parent / "gemopus_4_e4b_phase2"

POWER_SAMPLE_INTERVAL_MS = 500

# Match "Combined Power: 1234 mW" or "Package Power: 5.6 W" lines.
POWER_RE = re.compile(
    r'(?:Combined|Package)\s+Power.*?:\s*([\d.]+)\s*(m?W)',
    re.IGNORECASE,
)

# Powermetrics sample boundary marker:
#   *** Sampled system activity (Fri May  1 16:24:25 2026 +1000) (516.70ms elapsed) ***
SAMPLE_HEADER_RE = re.compile(
    r'\*\*\* Sampled system activity \(([^)]+)\) \(([\d.]+)ms elapsed\)'
)


# ──────────────────────────────────────────────────────────────────────
# Powermetrics
# ──────────────────────────────────────────────────────────────────────

def cache_sudo() -> None:
    """Ensure sudo can run powermetrics non-interactively.

    Probes ``sudo -n powermetrics --help`` first. If that succeeds (visudo
    NOPASSWD is scoped to powermetrics, as documented in tern-core/CLAUDE.md),
    return immediately — no prompt, harness runs unattended. Otherwise prompt
    for password to cache credentials for the duration of the run.
    """
    probe = subprocess.run(
        ["sudo", "-n", "powermetrics", "--help"],
        capture_output=True, timeout=5,
    )
    if probe.returncode == 0:
        print("  [info] sudo NOPASSWD active for powermetrics; proceeding "
              "without prompt.", file=sys.stderr)
        return
    print("\n" + "=" * 60, file=sys.stderr)
    print(" SUDO password requested for powermetrics", file=sys.stderr)
    print("=" * 60 + "\n", file=sys.stderr)
    sys.stderr.flush()
    rc = subprocess.run(["sudo", "-v"]).returncode
    if rc != 0:
        raise SystemExit("sudo refresh failed; powermetrics start blocked")


def start_powermetrics() -> tuple[subprocess.Popen, Path]:
    """Launch powermetrics in background; capture stdout to a temp file."""
    out_fd, out_path = tempfile.mkstemp(prefix="powermetrics_", suffix=".txt")
    proc = subprocess.Popen(
        [
            "sudo", "-n", "powermetrics",
            "--samplers", "cpu_power,gpu_power",
            "-i", str(POWER_SAMPLE_INTERVAL_MS),
        ],
        stdout=out_fd,
        stderr=subprocess.DEVNULL,
    )
    os.close(out_fd)  # parent drops its dup; child holds the live fd
    return proc, Path(out_path)


def stop_powermetrics(proc: subprocess.Popen) -> None:
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=5)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass


def parse_powermetrics(path: Path,
                       wall_t0: Optional[float] = None,
                       wall_t1: Optional[float] = None) -> dict:
    """Parse powermetrics samples; optionally filter to an active window.

    Each powermetrics sample block opens with a wall-clock timestamp:
      *** Sampled system activity (Fri May  1 16:24:25 2026 +1000) (...) ***
    followed by a Combined/Package Power line. We pair them, parse the
    timestamp, and bucket each sample as 'active' (within the [wall_t0,
    wall_t1] epoch interval) or 'idle' otherwise.

    When wall_t0 / wall_t1 are not supplied, every sample counts as active
    (back-compat with the original mixed-window methodology).

    Returns a dict with keys:
      all_samples_w, all_avg_w, n_all,
      active_samples_w, active_avg_w, n_active.
    """
    text = path.read_text(errors="ignore")
    blocks = re.split(r'(?=\*\*\* Sampled system activity)', text)

    all_samples: list[float] = []
    active_samples: list[float] = []

    for block in blocks:
        pm = POWER_RE.search(block)
        if not pm:
            continue
        v = float(pm.group(1))
        if pm.group(2).lower() == "mw":
            v /= 1000.0
        all_samples.append(v)

        if wall_t0 is None or wall_t1 is None:
            active_samples.append(v)
            continue

        header = SAMPLE_HEADER_RE.search(block)
        if not header:
            continue
        ts_str = re.sub(r'\s+', ' ', header.group(1)).strip()
        try:
            sample_dt = datetime.strptime(ts_str, "%a %b %d %H:%M:%S %Y %z")
            sample_epoch = sample_dt.timestamp()
        except ValueError:
            continue
        if wall_t0 <= sample_epoch <= wall_t1:
            active_samples.append(v)

    all_avg = sum(all_samples) / len(all_samples) if all_samples else 0.0
    active_avg = sum(active_samples) / len(active_samples) if active_samples else 0.0
    return {
        "all_samples_w": all_samples,
        "all_avg_w": all_avg,
        "n_all": len(all_samples),
        "active_samples_w": active_samples,
        "active_avg_w": active_avg,
        "n_active": len(active_samples),
    }


# ──────────────────────────────────────────────────────────────────────
# Memory
# ──────────────────────────────────────────────────────────────────────

def peak_rss_bytes() -> int:
    """Peak resident set size (macOS reports bytes; Linux reports KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


# ──────────────────────────────────────────────────────────────────────
# Chat-template helper
# ──────────────────────────────────────────────────────────────────────

def build_messages(prompt: str, system_prefix: str) -> list[dict]:
    msgs: list[dict] = []
    if system_prefix.strip():
        msgs.append({"role": "system", "content": system_prefix})
    msgs.append({"role": "user", "content": prompt})
    return msgs


# ──────────────────────────────────────────────────────────────────────
# Backend: mlx_fp16
# ──────────────────────────────────────────────────────────────────────

def run_mlx_fp16(model_id: str, prompt: str, max_tokens: int, seed: int,
                 system_prefix: str) -> dict:
    import mlx.core as mx
    import mlx_lm
    from mlx_lm import generate as mlx_generate, stream_generate

    mx.random.seed(seed)
    model, tokenizer = mlx_lm.load(model_id)

    messages = build_messages(prompt, system_prefix)
    chat = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False)
    prompt_token_ids = tokenizer.encode(chat)
    prompt_n = len(prompt_token_ids)

    # Warmup (no powermetrics)
    _ = mlx_generate(model, tokenizer, prompt=chat,
                     max_tokens=WARMUP_TOKENS, verbose=False)

    proc, ppath = start_powermetrics()
    try:
        mx.random.seed(seed)  # restore deterministic state
        t0 = time.perf_counter()
        wall_t0 = time.time()
        first_t: Optional[float] = None
        wall_first_t: Optional[float] = None
        text_chunks: list[str] = []
        n_tokens = 0
        for response in stream_generate(
                model, tokenizer, prompt=chat, max_tokens=max_tokens):
            if first_t is None:
                first_t = time.perf_counter()
                wall_first_t = time.time()
            text_chunks.append(getattr(response, "text", ""))
            n_tokens += 1
        t1 = time.perf_counter()
        wall_t1 = time.time()
    finally:
        stop_powermetrics(proc)

    output_text = "".join(text_chunks)
    prompt_eval_s = (first_t - t0) if first_t else 0.0
    gen_s = (t1 - first_t) if first_t else 0.0

    pm = parse_powermetrics(ppath, wall_t0=wall_first_t, wall_t1=wall_t1)
    avg_w = pm["active_avg_w"]
    energy_j = avg_w * gen_s
    result = _result_dict(prompt_n, n_tokens, prompt_eval_s, gen_s,
                          pm["active_samples_w"], avg_w, energy_j, output_text)
    result["all_power_samples_w"] = pm["all_samples_w"]
    result["all_avg_power_w"] = pm["all_avg_w"]
    result["n_power_samples_all"] = pm["n_all"]
    result["n_power_samples_active"] = pm["n_active"]
    return result


# ──────────────────────────────────────────────────────────────────────
# Backend: mlx_vlm_bf16  (multimodal MLX path; Gemma 4 family + cousins)
# ──────────────────────────────────────────────────────────────────────

def run_mlx_vlm_bf16(model_id: str, prompt: str, max_tokens: int, seed: int,
                     system_prefix: str) -> dict:
    import json
    import mlx.core as mx
    import mlx_vlm
    from huggingface_hub import hf_hub_download

    mx.random.seed(seed)
    model, processor = mlx_vlm.load(model_id)

    # apply_chat_template needs the model config dict (uses model_type to
    # route get_message_json). Load config.json from the HF cache directly —
    # this is the most reliable path across mlx_vlm versions.
    cfg_path = hf_hub_download(model_id, "config.json")
    with open(cfg_path) as f:
        config = json.load(f)

    messages = build_messages(prompt, system_prefix)
    chat_prompt = mlx_vlm.apply_chat_template(
        processor, config, messages, add_generation_prompt=True,
        num_images=0, num_audios=0,
    )

    # Warmup (no powermetrics)
    _ = mlx_vlm.generate(
        model, processor, prompt=chat_prompt,
        max_tokens=WARMUP_TOKENS, verbose=False, temperature=0.0,
    )

    proc_pm, ppath = start_powermetrics()
    try:
        mx.random.seed(seed)
        t0 = time.perf_counter()
        wall_t0 = time.time()
        first_t: Optional[float] = None
        wall_first_t: Optional[float] = None
        text_chunks: list[str] = []
        last_response = None
        n_tokens = 0
        for response in mlx_vlm.stream_generate(
                model, processor, prompt=chat_prompt,
                max_tokens=max_tokens, temperature=0.0):
            if first_t is None:
                first_t = time.perf_counter()
                wall_first_t = time.time()
            text_chunks.append(getattr(response, "text", ""))
            last_response = response
            n_tokens += 1
        t1 = time.perf_counter()
        wall_t1 = time.time()
    finally:
        stop_powermetrics(proc_pm)

    output_text = "".join(text_chunks)

    # Prefer mlx_vlm's own counts where it exposes them.
    prompt_n_lib = getattr(last_response, "prompt_tokens", 0) if last_response else 0
    n_tokens_lib = getattr(last_response, "generation_tokens", n_tokens) if last_response else n_tokens
    prompt_tps_lib = getattr(last_response, "prompt_tps", 0.0) if last_response else 0.0
    generation_tps_lib = getattr(last_response, "generation_tps", 0.0) if last_response else 0.0
    peak_memory_gib_lib = getattr(last_response, "peak_memory", 0.0) if last_response else 0.0

    prompt_n = prompt_n_lib if prompt_n_lib else 0
    n_tokens_final = n_tokens_lib if n_tokens_lib else n_tokens

    prompt_eval_s = (first_t - t0) if first_t else 0.0
    gen_s = (t1 - first_t) if first_t else 0.0

    pm = parse_powermetrics(ppath, wall_t0=wall_first_t, wall_t1=wall_t1)
    avg_w = pm["active_avg_w"]
    energy_j = avg_w * gen_s

    result = _result_dict(prompt_n, n_tokens_final, prompt_eval_s, gen_s,
                          pm["active_samples_w"], avg_w, energy_j, output_text)
    result["all_power_samples_w"] = pm["all_samples_w"]
    result["all_avg_power_w"] = pm["all_avg_w"]
    result["n_power_samples_all"] = pm["n_all"]
    result["n_power_samples_active"] = pm["n_active"]
    # Augment with mlx_vlm's library-reported metrics for cross-check
    result["mlx_vlm_prompt_tps"] = prompt_tps_lib
    result["mlx_vlm_generation_tps"] = generation_tps_lib
    result["mlx_vlm_peak_memory_gib"] = peak_memory_gib_lib
    return result


# ──────────────────────────────────────────────────────────────────────
# Backend: pytorch_mps_fp16
# ──────────────────────────────────────────────────────────────────────

def run_pytorch_mps_fp16(model_id: str, prompt: str, max_tokens: int,
                         seed: int, system_prefix: str) -> dict:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16).to("mps")
    model.eval()

    messages = build_messages(prompt, system_prefix)
    chat = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(chat, return_tensors="pt").to("mps")
    prompt_n = inputs["input_ids"].shape[1]

    # Warmup
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=WARMUP_TOKENS,
                           do_sample=False)
    torch.mps.synchronize()

    proc, ppath = start_powermetrics()
    try:
        torch.manual_seed(seed)
        t0 = time.perf_counter()
        wall_t0 = time.time()
        with torch.inference_mode():
            _ = model(**inputs)               # prefill (prompt eval)
            torch.mps.synchronize()
            first_t = time.perf_counter()
            wall_first_t = time.time()
            out = model.generate(**inputs, max_new_tokens=max_tokens,
                                 do_sample=False)
            torch.mps.synchronize()
        t1 = time.perf_counter()
        wall_t1 = time.time()
    finally:
        stop_powermetrics(proc)

    out_ids = out[0][prompt_n:]
    n_tokens = int(out_ids.shape[0])
    output_text = tokenizer.decode(out_ids, skip_special_tokens=True)

    prompt_eval_s = first_t - t0
    gen_s = t1 - first_t

    pm = parse_powermetrics(ppath, wall_t0=wall_first_t, wall_t1=wall_t1)
    avg_w = pm["active_avg_w"]
    energy_j = avg_w * gen_s
    result = _result_dict(prompt_n, n_tokens, prompt_eval_s, gen_s,
                          pm["active_samples_w"], avg_w, energy_j, output_text)
    result["all_power_samples_w"] = pm["all_samples_w"]
    result["all_avg_power_w"] = pm["all_avg_w"]
    result["n_power_samples_all"] = pm["n_all"]
    result["n_power_samples_active"] = pm["n_active"]
    return result


# ──────────────────────────────────────────────────────────────────────
# Backend: pytorch_mps_ternary
# ──────────────────────────────────────────────────────────────────────

def run_pytorch_mps_ternary(model_id: str, adapter: str, threshold: float,
                            prompt: str, max_tokens: int, seed: int,
                            system_prefix: str) -> dict:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from terncore.engine.inference import TernaryInferenceEngine

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Load FP16 first, ternise on CPU, then push to MPS.
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16)

    # In-memory engine.convert() uses GENERIC layer-protection rules
    # (walks nn.Linear / Conv2d / HF Conv1D against generic name patterns).
    # It is adapter-agnostic by construction — the gemma4 adapter is consumed
    # by the .tern-model write path in convert.py, not by this engine.
    # The harness measures the in-memory generic path (Benchmark Phase 1);
    # the .tern-model artefact (produced separately by terncore.convert
    # --adapter gemma4 --mixed-int4) remains the canonical NPU deliverable
    # and will be re-baselined in Benchmark Phase 2 via TernModelReader.
    # See Caveat 1 in module docstring.
    print(
        f"  [info] --adapter={adapter!r} retained for CLI parity with "
        f"terncore.convert, but is NOT consumed by TernaryInferenceEngine. "
        f"In-memory path uses generic layer-protection rules.",
        file=sys.stderr,
    )
    engine = TernaryInferenceEngine(threshold=threshold)
    conversion_report = engine.convert(model)  # mutates model; returns ConversionReport
    print(
        f"  [info] ConversionReport: "
        f"total_layers={getattr(conversion_report, 'total_layers', '?')} "
        f"converted={getattr(conversion_report, 'converted_layers', '?')} "
        f"skipped={getattr(conversion_report, 'skipped_layers', '?')} "
        f"ternary_params={getattr(conversion_report, 'ternary_params', '?')} "
        f"total_params={getattr(conversion_report, 'total_params', '?')}",
        file=sys.stderr,
    )

    model = model.to("mps")
    model.eval()

    messages = build_messages(prompt, system_prefix)
    chat = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(chat, return_tensors="pt").to("mps")
    prompt_n = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=WARMUP_TOKENS,
                           do_sample=False)
    torch.mps.synchronize()

    proc, ppath = start_powermetrics()
    try:
        torch.manual_seed(seed)
        t0 = time.perf_counter()
        wall_t0 = time.time()
        with torch.inference_mode():
            _ = model(**inputs)
            torch.mps.synchronize()
            first_t = time.perf_counter()
            wall_first_t = time.time()
            out = model.generate(**inputs, max_new_tokens=max_tokens,
                                 do_sample=False)
            torch.mps.synchronize()
        t1 = time.perf_counter()
        wall_t1 = time.time()
    finally:
        stop_powermetrics(proc)

    out_ids = out[0][prompt_n:]
    n_tokens = int(out_ids.shape[0])
    output_text = tokenizer.decode(out_ids, skip_special_tokens=True)

    prompt_eval_s = first_t - t0
    gen_s = t1 - first_t

    pm = parse_powermetrics(ppath, wall_t0=wall_first_t, wall_t1=wall_t1)
    avg_w = pm["active_avg_w"]
    energy_j = avg_w * gen_s
    result = _result_dict(prompt_n, n_tokens, prompt_eval_s, gen_s,
                          pm["active_samples_w"], avg_w, energy_j, output_text)
    result["all_power_samples_w"] = pm["all_samples_w"]
    result["all_avg_power_w"] = pm["all_avg_w"]
    result["n_power_samples_all"] = pm["n_all"]
    result["n_power_samples_active"] = pm["n_active"]
    return result


# ──────────────────────────────────────────────────────────────────────
# Backend: terncore_packed (persisted .tern-model + Metal-aware forward)
# ──────────────────────────────────────────────────────────────────────

def run_terncore_packed(artefact_path: str, hf_id: str, prompt: str,
                        max_tokens: int, seed: int,
                        system_prefix: str) -> dict:
    """Persisted .tern-model artefact + Metal-aware PackedTernaryLinear path.

    Loads the .tern-model artefact via the decoupled load path
    (load_as_model + key_mapping=GEMMA4_MULTIMODAL_TRANSFORMERS_5_5 +
    convert_model_to_packed), moves to MPS, and runs the standard prefill
    + generate timing with powermetrics. This is Benchmark Phase 2 per
    STATUS_PHASE2.md — the round-trip via the persisted artefact through
    the new Metal kernel integration (Phase 2.5 Stage 2).
    """
    import torch
    from transformers import AutoConfig, AutoTokenizer
    import transformers
    from terncore.tern_model import (
        TernModelReader, GEMMA4_MULTIMODAL_TRANSFORMERS_5_5,
    )
    from terncore.packed_linear import (
        PackedTernaryLinear, convert_model_to_packed,
    )

    torch.manual_seed(seed)
    gc.collect()

    # Architecture from cached HF config; instantiate via the class the
    # config declares (robust to future Gemma 4 class renames).
    config = AutoConfig.from_pretrained(hf_id)
    arch_name = config.architectures[0]
    if not hasattr(transformers, arch_name):
        raise SystemExit(
            f"transformers lacks {arch_name} (version mismatch)")
    ModelClass = getattr(transformers, arch_name)

    # Random-init in FP16; immediately overwritten by the .tern-model load.
    model = ModelClass._from_config(config, dtype=torch.float16)
    model.eval()

    # Decoupled load: state_dict via load_as_model + key_mapping, then
    # convert eligible Linear layers to PackedTernaryLinear (with
    # Metal-aware forward).
    reader = TernModelReader(str(artefact_path))
    reader.load_as_model(
        model, strict=False,
        key_mapping=GEMMA4_MULTIMODAL_TRANSFORMERS_5_5,
    )
    convert_stats = convert_model_to_packed(model, threshold=DEFAULT_THRESHOLD)
    print(
        f"  [info] convert_model_to_packed: "
        f"packed_layers={convert_stats['packed_layers']} "
        f"protected_layers={convert_stats['protected_layers']} "
        f"total_layers={convert_stats['total_layers']}",
        file=sys.stderr,
    )

    model = model.to("mps")

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    messages = build_messages(prompt, system_prefix)
    chat = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(chat, return_tensors="pt").to("mps")
    prompt_n = inputs["input_ids"].shape[1]

    # Warmup (no powermetrics)
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=WARMUP_TOKENS,
                           do_sample=False)
    torch.mps.synchronize()

    proc, ppath = start_powermetrics()
    try:
        torch.manual_seed(seed)
        t0 = time.perf_counter()
        wall_t0 = time.time()
        with torch.inference_mode():
            _ = model(**inputs)               # prefill
            torch.mps.synchronize()
            first_t = time.perf_counter()
            wall_first_t = time.time()
            out = model.generate(**inputs, max_new_tokens=max_tokens,
                                 do_sample=False)
            torch.mps.synchronize()
        t1 = time.perf_counter()
        wall_t1 = time.time()
    finally:
        stop_powermetrics(proc)

    out_ids = out[0][prompt_n:]
    n_tokens = int(out_ids.shape[0])
    output_text = tokenizer.decode(out_ids, skip_special_tokens=True)

    prompt_eval_s = first_t - t0
    gen_s = t1 - first_t

    pm = parse_powermetrics(ppath, wall_t0=wall_first_t, wall_t1=wall_t1)
    avg_w = pm["active_avg_w"]
    energy_j = avg_w * gen_s

    result = _result_dict(prompt_n, n_tokens, prompt_eval_s, gen_s,
                          pm["active_samples_w"], avg_w, energy_j, output_text)
    result["all_power_samples_w"] = pm["all_samples_w"]
    result["all_avg_power_w"] = pm["all_avg_w"]
    result["n_power_samples_all"] = pm["n_all"]
    result["n_power_samples_active"] = pm["n_active"]
    result["artefact_path"] = str(artefact_path)
    result["hf_id"] = hf_id
    result["convert_packed_layers"] = convert_stats["packed_layers"]
    result["convert_protected_layers"] = convert_stats["protected_layers"]
    return result


# ──────────────────────────────────────────────────────────────────────
# Backend: llamacpp_gguf
# ──────────────────────────────────────────────────────────────────────

def run_llamacpp_gguf(model_path: str, prompt: str, max_tokens: int,
                      seed: int, system_prefix: str) -> dict:
    import shutil
    bin_path = shutil.which("llama-cli") or shutil.which("llama")
    if not bin_path:
        raise SystemExit("llama-cli not found on PATH")

    base_args = [bin_path, "-m", model_path, "--seed", str(seed),
                 "--single-turn", "--temp", "0", "-ngl", "999", "-p", prompt]
    if system_prefix.strip():
        base_args = base_args[:-2] + ["--system-prompt", system_prefix,
                                      "-p", prompt]

    # Warmup
    subprocess.run(base_args + ["-n", str(WARMUP_TOKENS)],
                   capture_output=True, check=False)

    proc, ppath = start_powermetrics()
    try:
        t0 = time.perf_counter()
        wall_t0 = time.time()
        result = subprocess.run(
            base_args + ["-n", str(max_tokens)],
            capture_output=True, text=True)
        t1 = time.perf_counter()
        wall_t1 = time.time()
    finally:
        stop_powermetrics(proc)

    # Parse llama-cli's own timing report from stderr
    stderr = result.stderr or ""
    stdout = result.stdout or ""
    prompt_eval_re = re.compile(
        r'prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens')
    eval_re = re.compile(
        r'^\s*eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs',
        re.MULTILINE)
    # llama-cli >=8990 emits a single-line summary instead:
    # "[ Prompt: 130.2 t/s | Generation: 66.1 t/s ]"
    new_fmt_re = re.compile(
        r'\[\s*Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s\s*\]')

    prompt_eval_match = prompt_eval_re.search(stderr)
    eval_match = eval_re.search(stderr)
    new_match = new_fmt_re.search(stderr) or new_fmt_re.search(stdout)
    prompt_eval_s = float(prompt_eval_match.group(1)) / 1000.0 if prompt_eval_match else 0.0
    prompt_n = int(prompt_eval_match.group(2)) if prompt_eval_match else 0
    gen_s = float(eval_match.group(1)) / 1000.0 if eval_match else 0.0
    n_tokens = int(eval_match.group(2)) if eval_match else 0
    # Fall through to new-format if the legacy regex missed. Generation rate
    # from the new format is authoritative; n_tokens approximated from
    # max_tokens (llama.cpp 8990 omits the actual decoded count from this
    # summary line).
    llamacpp_generation_tps = None
    llamacpp_prompt_tps = None
    if new_match:
        llamacpp_prompt_tps = float(new_match.group(1))
        llamacpp_generation_tps = float(new_match.group(2))
        if not eval_match and llamacpp_generation_tps > 0:
            n_tokens = max_tokens
            gen_s = n_tokens / llamacpp_generation_tps

    # Decode-only window: wall_first_t = wall_t0 + prompt_eval_s
    # (subprocess doesn't yield first-token boundary directly; we use llama-cli's
    # own prompt_eval_s as the offset.)
    wall_first_t = wall_t0 + prompt_eval_s if prompt_eval_s > 0 else wall_t0
    pm = parse_powermetrics(ppath, wall_t0=wall_first_t, wall_t1=wall_t1)
    avg_w = pm["active_avg_w"]
    energy_j = avg_w * gen_s

    out = _result_dict(prompt_n, n_tokens, prompt_eval_s, gen_s,
                      pm["active_samples_w"], avg_w, energy_j, result.stdout)
    out["all_power_samples_w"] = pm["all_samples_w"]
    out["all_avg_power_w"] = pm["all_avg_w"]
    out["n_power_samples_all"] = pm["n_all"]
    out["n_power_samples_active"] = pm["n_active"]
    out["_llama_cli_stderr_tail"] = "\n".join(stderr.splitlines()[-30:])
    if llamacpp_generation_tps is not None:
        out["llamacpp_generation_tps"] = llamacpp_generation_tps
        out["llamacpp_prompt_tps"] = llamacpp_prompt_tps
    return out


# ──────────────────────────────────────────────────────────────────────
# Result assembly
# ──────────────────────────────────────────────────────────────────────

def _result_dict(prompt_n: int, n_tokens: int, prompt_eval_s: float,
                 gen_s: float, samples_w: list[float], avg_w: float,
                 energy_j: float, output_text: str) -> dict:
    return {
        "prompt_tokens": prompt_n,
        "tokens_generated": n_tokens,
        "prompt_eval_time_s": prompt_eval_s,
        "generation_time_s": gen_s,
        "tok_per_s": (n_tokens / gen_s) if gen_s > 0 else 0.0,
        "prompt_eval_ms_per_token":
            (prompt_eval_s * 1000.0 / prompt_n) if prompt_n else 0.0,
        "generation_ms_per_token":
            (gen_s * 1000.0 / n_tokens) if n_tokens else 0.0,
        "peak_memory_bytes": peak_rss_bytes(),
        "power_samples_w": samples_w,
        "avg_power_w": avg_w,
        "energy_j": energy_j,
        "j_per_token": (energy_j / n_tokens) if n_tokens else 0.0,
        "output_text": output_text,
    }


def capture_versions() -> dict:
    import importlib.metadata as _md
    versions: dict[str, str] = {}
    for mod_name in ("mlx", "mlx_lm", "mlx_vlm", "torch", "transformers", "terncore"):
        try:
            mod = __import__(mod_name)
            v = getattr(mod, "__version__", None)
            if v is None:
                # Fall back to package metadata for modules that don't expose __version__
                # (mlx, mlx_vlm sometimes hide the attribute on the top-level module).
                pkg = mod_name.replace("_", "-")
                try:
                    v = _md.version(pkg)
                except Exception:
                    v = "?"
            versions[mod_name] = v
        except Exception:
            versions[mod_name] = "(not installed)"
    # Distribution-only packages (no top-level importable module). mlx-metal
    # is the Metal backend wheel for mlx; its dist name is hyphenated and it
    # has no `mlx_metal` import path, so it must be queried via metadata.
    for dist_name in ("mlx-metal",):
        try:
            versions[dist_name] = _md.version(dist_name)
        except Exception:
            versions[dist_name] = "(not installed)"
    try:
        import shutil
        bin_path = shutil.which("llama-cli") or shutil.which("llama")
        if bin_path:
            r = subprocess.run([bin_path, "--version"],
                               capture_output=True, text=True, timeout=5)
            versions["llama_cli"] = (r.stdout or r.stderr).strip().splitlines()[0]
    except Exception:
        versions["llama_cli"] = "(unavailable)"
    return versions


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gemopus-4-E4B-it phase-2 benchmark harness")
    parser.add_argument("--format", required=True,
                        choices=["mlx_fp16", "mlx_vlm_bf16",
                                 "pytorch_mps_fp16",
                                 "pytorch_mps_ternary", "llamacpp_gguf",
                                 "terncore_packed"])
    parser.add_argument("--model", required=True,
                        help="HF id (mlx/pytorch), local .gguf path "
                             "(llamacpp_gguf), or local .tern-model path "
                             "(terncore_packed)")
    parser.add_argument("--hf-id", default=None,
                        help="HF id for AutoConfig + AutoTokenizer "
                             "(terncore_packed only; required for that format)")
    parser.add_argument("--label", required=True,
                        help="Filename prefix and report row identifier")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--system-prompt-prefix", default="")
    parser.add_argument("--adapter", default="gemma4",
                        help="Architecture adapter (pytorch_mps_ternary)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Quantisation threshold (pytorch_mps_ternary)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_sudo()

    print(f"[{datetime.now(timezone.utc).isoformat()}] start "
          f"label={args.label} format={args.format}", file=sys.stderr)

    if args.format == "mlx_fp16":
        result = run_mlx_fp16(
            args.model, args.prompt, args.max_tokens, args.seed,
            args.system_prompt_prefix)
    elif args.format == "mlx_vlm_bf16":
        result = run_mlx_vlm_bf16(
            args.model, args.prompt, args.max_tokens, args.seed,
            args.system_prompt_prefix)
    elif args.format == "pytorch_mps_fp16":
        result = run_pytorch_mps_fp16(
            args.model, args.prompt, args.max_tokens, args.seed,
            args.system_prompt_prefix)
    elif args.format == "pytorch_mps_ternary":
        result = run_pytorch_mps_ternary(
            args.model, args.adapter, args.threshold,
            args.prompt, args.max_tokens, args.seed,
            args.system_prompt_prefix)
    elif args.format == "llamacpp_gguf":
        result = run_llamacpp_gguf(
            args.model, args.prompt, args.max_tokens, args.seed,
            args.system_prompt_prefix)
    elif args.format == "terncore_packed":
        if not args.hf_id:
            parser.error("--hf-id is required when --format=terncore_packed")
        result = run_terncore_packed(
            args.model, args.hf_id, args.prompt, args.max_tokens, args.seed,
            args.system_prompt_prefix)
    else:
        raise SystemExit(f"unknown format: {args.format}")

    result["label"] = args.label
    result["format"] = args.format
    result["model"] = args.model
    result["prompt"] = args.prompt
    result["system_prompt_prefix"] = args.system_prompt_prefix
    result["max_tokens"] = args.max_tokens
    result["seed"] = args.seed
    result["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    result["host"] = platform.node()
    result["engine_versions"] = capture_versions()
    result["peak_memory_mib"] = result["peak_memory_bytes"] / (1024 * 1024)

    out_path = OUTPUT_DIR / f"{args.label}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== {args.label} results ===", file=sys.stderr)
    print(f"  tokens_generated:    {result['tokens_generated']}", file=sys.stderr)
    print(f"  generation_time_s:   {result['generation_time_s']:.2f}", file=sys.stderr)
    print(f"  tok_per_s:           {result['tok_per_s']:.2f}", file=sys.stderr)
    print(f"  prompt_eval_ms/tok:  {result['prompt_eval_ms_per_token']:.2f}", file=sys.stderr)
    print(f"  generation_ms/tok:   {result['generation_ms_per_token']:.2f}", file=sys.stderr)
    print(f"  peak_memory_mib:     {result['peak_memory_mib']:.1f}", file=sys.stderr)
    print(f"  avg_power_w:         {result['avg_power_w']:.2f}", file=sys.stderr)
    print(f"  energy_j:            {result['energy_j']:.2f}", file=sys.stderr)
    print(f"  j_per_token:         {result['j_per_token']:.4f}", file=sys.stderr)
    print(f"  → {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
