# tern-core — CLAUDE.md
_Last updated: 21 April 2026_

Global conventions (corporate identity, canonical vocabulary, DOCX
formatting, editorial voice) live in `~/.claude/CLAUDE.md` and apply
here by default. Workspace-level stack context lives in
`~/synapticode/CLAUDE.md`. This file adds tern-core-specific
architecture, sprint state, and benchmark detail.

---

## Project Identity

**Repo:** github.com/synapticode-ai/tern-core (public)  
**Working directory:** `/Users/syn/synapticode/tern-core`  
**Python env:** activate `.venv` before running (`source .venv/bin/activate`)

---

## What tern-core is

A ternary neural network compression and inference library. It
compiles standard model weights (safetensors, HuggingFace) into
mixed ternary/INT4 quantised format (`.tern-model`, `.tern-pkg`),
exports validated CoreML mlpackages for Apple silicon inference, and
benchmarks the result across compute units and energy profiles.

This work forms the technical foundation for an ARM-style IP
licensing model targeting Apple (CoreML integration partnership, May
2026) and Korean NPU vendors (Rebellions, FuriosaAI, KAIST/KSGC).

---

## Repository Layout

```
tern-core/
├── CLAUDE.md                                         ← this file
├── SPEC-TFH-001_Ternary_Fine_Tuning_Harness.docx    ← TFH specification
├── benchmarks/                ← phase-2 runners + JSON results across 10 models
├── configs/
├── data/
├── docs/                      ← TN-001, TN-002, architectural notes
├── harness/                   ← TFH components (Phase 1 + Phase 2 landed)
├── output/                    ← CoreML mlpackages (demo artefacts)
├── src/terncore/              ← library source
├── tests/
└── tools/
```

The KAIST delivery bundles have moved to `tern-runtime/packages/` —
they now live alongside the Local Inference Stack that serves them.

---

## Version Track

| Version | Highlight |
|---|---|
| v0.6.0 | Mixed ternary/INT4 quantisation, CoreML-native (TN-001, TN-002) |
| v0.5.1 | Streaming write + 70B compression analysis |
| v0.5.0 | Streaming shard-by-shard conversion pipeline |
| v0.3.0 | `.tern-pkg` delivery format established |
| v0.2.0 | Ternary confidence layer, model routing |
| v0.1.0 | Perplexity-gated autoscan, TurboQuant KV hook, energy benchmarks |

---

## Benchmark State — Phase 2 Runs

Phase-2 runners and JSON results live in `benchmarks/`. Models with
Phase-2 completion (per-model headline numbers and compute-unit
breakdown in `benchmarks/<name>_phase2.json`):

- Mistral-7B (reference exemplar — detail below)
- Llama-3.1-70B
- Llama-3.2 1B / 3B
- Gemma 3 4B / 12B
- DSR1 7B / 14B
- Phi-4 14B
- Qwen2.5 7B

Dryrun-only, full Phase 2 pending:

- Gemma 4 E4B — **blocked on coremltools 9.1 / 9.2** (see Model Sprint below)

### Mistral-7B — Phase B (Inference, complete)

| Compute Unit | Mean ms | tok/s | Stdev ms | Peak RSS |
|---|---|---|---|---|
| ALL | 222.60 | 287.5 | 225.2 | 28.5 GB |
| **CPU_AND_NE** | **215.07** | **297.6** | **9.7** | **31.0 GB** |
| CPU_AND_GPU | 226.55 | 282.5 | 245.2 | 31.0 GB |

CPU_AND_NE is the confirmed winner. The 9.7 ms stdev is the headline
stability number.

### Mistral-7B — Phase D (Energy Profile, CPU_AND_NE)

| Metric | Value |
|---|---|
| Mean package power | 5.39 W |
| Energy per inference | 1,122.6 mJ |
| Energy per token (512 tokens) | 2.19 mJ |
| Inferences in 15 s | 72 |
| Power stdev | 0.70 W |

### Mistral-7B — Phase C (Palettisation, non-fatal)

Phase C fails at op 295/298 with `ValueError: Input X contains
infinity or a value too large for dtype('float64')`. FP16 weight
tensors in the mlpackage contain Inf values. The palettisation block
is wrapped in try/except and stays non-fatal — Phase D proceeds
regardless. Filed as **tern-core #1** on GitHub.

Root cause: tern-compiler export bug. Inf values land in FP16-encoded
ternary weights. The fix belongs in the tern-core compiler path
rather than the benchmark runner.

### Pending — Energy Baseline

Phase D energy profile covers CPU_AND_NE only. CPU_ONLY and
CPU_AND_GPU baselines are still pending. Both anchor the 5.39 W
headline in the Apple brief — run them before the May conversation.

---

## Mistral-7B Compression Story

| Stage | Size | Ratio vs FP16 |
|---|---|---|
| FP16 source | ~14.0 GB | 1.0x |
| tern-core v0.3.0 (.tern-pkg, 96.4 % ternary) | 2.27 GB | 6.2x |
| CoreML mlpackage today (FP16-encoded) | 14.5 GB | 0.97x |
| CoreML mlpackage — native ternary (projected) | ~2.27 GB | ~6.2x |

At 2.27 GB, Mistral-7B runs on iPhone. The gap between 14.5 GB and
2.27 GB defines the CoreML native ternary argument.

---

## Llama-3.1-70B Compression Story (TN-001 final)

| Stage | Size | Ratio vs FP16 |
|---|---|---|
| FP16 source | 131.0 GB | 1.0x |
| tern-core v0.6.0 on-disk output | 35.0 GB | 6.62x |
| CoreML mlpackage (iOS 18 spec, validated) | 38.96 GB | 3.36x |
| Current in-memory footprint at load | ~116 GB | — |
| In-memory footprint — native ternary (projected) | ~39 GB | 3.0x |

The 70B mlpackage stays a **demo artefact only**. Inference on M4 Pro
64 GB remains out of reach — decompression expands to ~116 GB. M2/M3
Ultra (192 GB) is the target hardware for 70B inference. Source
safetensors have been removed from the workspace and backed up to the
MacBook Pro.

TN-001 (final on-disk results) and TN-002 (CoreML/ANE export
investigation — proof-of-concept validated) in `docs/` carry the
authoritative record.

---

## TFH Sprint — Ternary-Aware Fine-Tuning Harness

Spec: `SPEC-TFH-001_Ternary_Fine_Tuning_Harness.docx`. Components
land under `harness/`. Required dependency: `mlx>=0.18` (optional
`harness` extra).

| Phase | Components | Status |
|---|---|---|
| 1 | Epistemic state, scheduler, projector, annotator, objective | Landed |
| 2 | Trainer, checkpointer | Landed |
| 3 | ConfidenceEventLog³ extension, reference materials audit trail | Reference materials landed; audit trail ongoing |

---

## Model Sprint — Current Order

**Completed Phase-2 runs** (per-model detail in benchmark JSON):

1. Mistral-7B
2. Llama-3.1-70B (compression validated; inference out of reach on M4 Pro — Ultra hardware required)
3. Llama-3.2 1B / 3B
4. Gemma 3 4B / 12B
5. DSR1 7B / 14B (14B ran ahead of its original slot — April 2026 sequencing adjustment)
6. Phi-4 14B
7. Qwen2.5 7B

**Pending — blocked on coremltools 9.1 / 9.2 release:**

- **Gemma 4 E4B** — multimodal dense, Agent³ eOS reasoning engine target. Dryrun complete. Adapter session required before full compression (see below).
- **Gemma 4 26B MoE** — ternary + MoE IP angle, potential new provisional.
- **Gemma 4 31B dense** — Apple conversation benchmark, largest dense target.

**Pending — future:**

- **MiniMax M2.7 MoE** — awaiting the Biwin M350 + Acasis dock archive drive before downloading weights.

---

### Gemma 4 — Before Any Compression: Wire the Adapter

**Gemma 4 uses a different architecture to Llama.** tern-core's
current conversion pipeline is Llama-native. The Gemma 4 adapter
needs writing and validating before any Gemma 4 compression sprint
begins.

**Reference:** llama.cpp already carries Gemma 4 GGUF support — its
conversion logic is the reference to borrow from. Inspect
`llama.cpp/convert_hf_to_gguf.py` for the Gemma 4 architecture
handling.

A dedicated CC session covers the adapter work. Gemma 4 compression
waits for that session to complete.

Adapter session checklist:

1. Read `~/synapticode/CLAUDE.md` and `tern-core/CLAUDE.md`.
2. Pull llama.cpp Gemma 4 GGUF conversion as reference.
3. Write `tern_core/adapters/gemma4.py` — architecture-specific weight mapping.
4. Validate on E4B (smallest model, fastest feedback loop).
5. Run a dry-run conversion — confirm weight shapes, layer names, ternary tolerance scan.
6. Proceed to full compression once the dry-run is clean.

### Gemma 4 E4B — notes

4B parameters, dense transformer. Multimodal: text, audio, vision,
video. Directly relevant to Agent³ eOS as a local reasoning engine.
The multimodal capability is the headline — a ternary-compressed
multimodal model running locally on M4 Pro is a strong demo artefact
beyond the Apple conversation.

### Gemma 4 26B MoE — IP opportunity

26B total parameters, ~3.8B active per forward pass. Ternary
compression on MoE architecture remains largely unexplored. Key
hypothesis: inactive expert weights — those the router skips for a
given token — cluster toward the zero-state at higher rates than
dense layer weights. If confirmed, the ternary tolerance ratio for a
MoE model could significantly exceed the 22.3 % achieved on
Mistral-7B.

During compression, log ternary tolerance ratio **per-expert**, not
just the aggregate. The per-expert distribution is the IP
observation. If inactive experts show markedly higher zero-state
concentration, draft a new provisional covering ternary compression
of MoE sparse expert weights.

### Gemma 4 31B dense — Apple room number

Largest dense model in the Gemma 4 sprint. Compression ratio on a
31B model at ~62 GB FP16 is the Apple conversation benchmark —
demonstrates ternary compression scaling beyond 7B without quality
degradation.

### MiniMax M2.7 MoE — notes for when it arrives

MoE architecture. Only a subset of parameters activate per inference
pass. Same IP hypothesis as Gemma 4 26B — inactive experts as
zero-state candidates. Check total vs active parameter count before
committing disk space. Weights on HuggingFace. Download once the
Biwin M350 + Acasis dock archive drive is online.

---

## CoreML Export — Correct CLI

**Module name:** `terncore.coreml_export` (the form `terncore.export_coreml` is retired)  
**Surface flags:** `--model`, `--output`, `--arch-preset` — these three exclusively.

```bash
source .venv/bin/activate
python -m terncore.coreml_export \
  --model /Users/syn/synapticode/models/compressed/[model]/[name].tern-model \
  --output /Users/syn/synapticode/models/coreml/[model]/[name].mlpackage \
  --arch-preset [preset-name]
```

Available presets: `llama32-1b`, `llama32-3b`, `mistral-7b`,
`gemma3-4b`, `gemma3-12b`.

**Venv note:** `source .venv/bin/activate` activates
`/Users/syn/synapticode/venv/bin/python`, which carries terncore on
its path via the src layout. Use the activated shell — calling
`.venv/bin/python` directly bypasses the src-layout resolution and
typically fails to import `terncore`.

---

## Benchmark Runner — How to Run

```bash
cd /Users/syn/synapticode/tern-core
source .venv/bin/activate
python benchmarks/bench_mistral7b_phase2.py 2>&1 | tee benchmarks/mistral7b_phase2.stdout.log
```

Phase D (energy): the benchmark runner calls `sudo powermetrics`
internally. visudo grant is in place — runs without a password
prompt. Run the benchmark script itself without sudo — only
powermetrics needs elevation.

Results write to `benchmarks/[model]_phase2.json` incrementally —
safe to inspect mid-run.

**Watchdog threshold:** default is `WATCHDOG_COMPRESSOR_TRIP_PAGES =
1_800_000`. For models above 7B on the 64 GB M4 Pro, raise to
`3_000_000` in the runner before executing. The 12B benchmark needed
this — palettisation hit 44 GB RSS on the first attempt.

---

## Coding Conventions (repo-specific additions)

Global conventions in `~/.claude/CLAUDE.md` cover corporate name,
copyright headers, editorial voice, and canonical vocabulary.
tern-core adds:

- **Soft-delete policy**: move Swift and Python source files to `~/synapticode/_trash/` with a `~YYYY-MM-DD_` prefix — keep `rm` off source files.
- **Branch before editing** benchmark runners — they are primary Apple-brief evidence.
- **JSON results files** stay append-safe. Rename the original file before re-running; the existing result file is evidence.

---

## Key Contacts and Context

- **Rod** — patent attorney, all IP filings.
- **Apple conversation** — May 2026. Brief produced. Three-stage CoreML partnership ask.
- **KAIST / KSGC** — primary near-term Korean commercial engagement.
- **VibeVoice** — separate project, TTS/ASR for Remotion narration. See `~/synapticode/vibevoice/CLAUDE.md`.
- **Remotion** — separate project, Apple brief as animated video. See `~/synapticode/remotion/CLAUDE.md`.
- **Model library** — 4TB SSD archive. See `~/synapticode/model-library/CLAUDE.md`.
