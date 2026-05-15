# WikiText-2 Perplexity Evaluation Methodology

**Document status:** Canonical (v1.0)
**Authored:** 2026-05-14
**Repository path:** `tern-core/docs/wikitext2_ppl_methodology.md`
**Scope:** PPL evaluation methodology for FP16 baseline models and tern-core compressed (ternary / TQ-compressed) variants. Consumed by the PPL benchmark tooling and by the R8 `ppl_headroom` diagnostic.

---

## §1 Purpose

This document defines the canonical methodology for evaluating language-model perplexity (PPL) on the WikiText-2 benchmark within `tern-core`. It standardises:

1. Dataset selection and provenance
2. Tokenisation and special-token handling
3. Sliding-window evaluation protocol
4. Loss computation
5. FP16-baseline vs ternary-variant comparison protocol (`ppl_headroom`)
6. Results JSON schema for reproducible reporting

All PPL numbers reported in tern-core releases, patent empirical-evidence citations, and external collateral (engagement briefs, partner-facing technical material) MUST conform to this methodology. Deviations require explicit documentation per §9 (Reproducibility checklist) plus justification in the run's `results.notes` field.

---

## §2 Dataset specification

**Source:** HuggingFace `Salesforce/wikitext`, configuration `wikitext-2-raw-v1`, split `test`.

**Revision pin:** REQUIRED at first use. Capture the exact HuggingFace dataset revision (commit sha of the dataset repository at load time) and record it in every results JSON under `dataset.huggingface_revision`. Pinning prevents silent dataset drift; updates to the upstream WikiText-2 HF dataset MUST be re-baselined explicitly with a new revision pin and a deliberate re-evaluation of the FP16 baseline before any ternary variant is compared against it.

**Variant rationale:** The `raw` variant preserves natural-language input without `<unk>` substitution, allowing modern BPE/SentencePiece tokenisers to handle vocabulary mapping. The legacy tokenised variant (`wikitext-2-v1`) inserts `<unk>` placeholders incompatible with the tern-core target model families (Llama, Gemma, TinyLlama, Phi, Mistral).

**Split rationale:** Test split for all reported PPL numbers. Validation split MAY be used for hyperparameter calibration of ternary compression (e.g. selecting `b_mse`, internal `ppl_headroom` sweep tuning) but MUST NOT appear in headline PPL reports. Train split is not used for evaluation.

**Approximate token volume:** ~245k tokens under the Llama 3.2 tokeniser (reference figure; varies by tokeniser). Exact token count for each run is recorded in `dataset.total_tokens`.

---

## §3 Tokenisation

**Tokeniser source:** The exact tokeniser shipped with the target model. NO substitution, NO modification, NO vocabulary trimming. The tokeniser MUST be loaded from the same model artefact as the model weights being evaluated.

**Concatenation:** All entries in the test split are concatenated with `\n\n` between non-empty entries, preserving the dataset's natural paragraph structure:

```python
test_text = "\n\n".join(s for s in test_split["text"] if s.strip())
tokens = tokenizer(test_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
```

**`add_special_tokens=False`:** REQUIRED. Special-token insertion is controlled explicitly per §4, not delegated to the tokeniser's default behaviour (which varies across tokeniser families and would silently corrupt cross-family comparisons).

**Tokeniser state for `.tern-model` artefacts:** If a `.tern-model` artefact resolves to a tokeniser via its manifest (per R4-C Option C tokenizer resolution), the manifest-resolved tokeniser is the canonical evaluation tokeniser for that artefact. CLI `--tokenizer` override is permitted but MUST be recorded in `tokeniser.source` and flagged in `notes`.

---

## §4 Special-token handling (BOS / EOS)

**BOS (beginning of sequence):**

- Prepended ONCE at the very start of the concatenated test token stream.
- NOT prepended at the start of each sliding window.

Rationale: The test stream represents a continuous body of text. Modelling each sliding window as if it began a new document (by inserting BOS) corrupts the autoregressive conditioning and inflates PPL. The BOS at position 0 ensures the first window has the model's natural document-start signal; subsequent windows are continuations of the same stream.

**EOS (end of sequence):**

- NOT inserted. The WikiText-2 test stream is treated as a single open-ended document.

Rationale: WikiText-2 evaluation in published literature (Merity et al. 2016; Radford et al. 2019; common HuggingFace `evaluate` implementations) treats the stream as continuous, not as a sequence of independently-terminated documents.

**Per-tokeniser BOS handling:**

| Tokeniser family | BOS token | Action |
|---|---|---|
| Llama 1 / 2 / 3 / 3.2 | `<\|begin_of_text\|>` (id 128000 for Llama 3+) | Prepend once at position 0 |
| Gemma 3 / 4 | `<bos>` (id 2) | Prepend once at position 0 |
| TinyLlama | `<s>` (id 1) | Prepend once at position 0 |
| Mistral / Mixtral | `<s>` (id 1) | Prepend once at position 0 |
| Phi-4 | (no BOS convention) | Do NOT prepend; set `bos_prepended=false` |

For tokenisers without a documented BOS (e.g. Phi-4 convention), do NOT prepend. The choice MUST be recorded in `tokeniser.bos_prepended` (bool) in the results JSON.

---

## §5 Sliding-window evaluation protocol

**Window length (`seq_len`):** 2048 tokens.

Rationale: Matches the long-context regime exercised by tern-core's R6 long-sequence convergence work (2287 tok empirical validation, commit `df4c141`). Sufficient context to be representative of deployment usage; small enough to fit comfortably in 64 GB-class hardware across the full target model family up to 31B dense.

**Stride (`stride`):** 2048 tokens (non-overlapping).

Rationale: Non-overlapping evaluation is faster, deterministic, and the published convention for headline PPL numbers (Radford et al. 2019, GPT-2 paper; OpenAI evaluation harness; HuggingFace `evaluate` defaults for non-rolling PPL). Sliding-window with smaller stride (e.g. 1024) yields slightly lower PPL but the methodology MUST be explicit about which variant is reported. Cross-variant comparisons (`ppl_headroom`) require matched stride.

**Optional rolling-window variant:** For diagnostic comparison only, a rolling-window evaluation with `stride=1024` MAY be computed in parallel and reported alongside non-overlapping PPL as `results.ppl_rolling`. This is OPTIONAL and labelled distinctly in the results JSON. The non-overlapping number remains the headline figure.

**Last-window handling:** If the token stream length is not divisible by `seq_len`, the final partial window is discarded. The number of discarded tokens MUST be recorded in `dataset.tokens_discarded` in the results JSON.

Rationale: Including a short final window distorts the per-window-average loss because the final window has fewer scored positions. Discarding ≤2047 tokens out of ~245k is <1% loss of evaluation volume; acceptable in exchange for methodological cleanliness.

**Pseudocode (canonical implementation):**

```python
# Inputs: tokens (1D LongTensor, including prepended BOS if applicable),
#         model, tokenizer, device
seq_len = 2048
stride = 2048

total_loss_sum = 0.0
total_tokens_scored = 0
per_window_losses = []

model.eval()
for window_start in range(0, len(tokens) - seq_len + 1, stride):
    window_end = window_start + seq_len
    input_ids = tokens[window_start:window_end].unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        # outputs.loss = mean cross-entropy over (seq_len - 1) scored positions

    # Un-mean to enable correct aggregate across windows of equal size
    window_loss_sum = outputs.loss.item() * (seq_len - 1)
    total_loss_sum += window_loss_sum
    total_tokens_scored += (seq_len - 1)
    per_window_losses.append(outputs.loss.item())

mean_loss = total_loss_sum / total_tokens_scored
ppl = math.exp(mean_loss)
```

**Per-window vs aggregate:** Per-window losses MAY be stored for diagnostics (variance analysis, outlier detection, long-context degradation signal) but the headline number is the aggregate `ppl` computed from `mean_loss` over all scored positions across all windows.

---

## §6 Loss computation

**Loss function:** Standard causal-LM cross-entropy as implemented by HuggingFace `AutoModelForCausalLM.forward(input_ids, labels=input_ids)`. Internal mechanics:

- Labels are shifted by 1 internally (predict position N from positions 0..N-1)
- Loss is averaged over the `seq_len - 1` scored positions per window
- Ignored indices (e.g. `-100`) are skipped — but in WikiText-2 PPL eval, no positions are ignored

**Reduction:** `mean` over scored positions (HF default). Aggregate across windows by un-meaning per-window (`loss * (seq_len - 1)`), summing, dividing by total scored tokens (per §5 pseudocode).

**Numerical precision:**

| Model variant | Activation dtype | Loss accumulator dtype |
|---|---|---|
| FP16 baseline | `torch.float16` | `torch.float32` (HF default) |
| Ternary (tern-core) | model-internal mixed (ternary weights, fp16/fp32 activations per layer config) | `torch.float32` |

Loss accumulation in `float32` is REQUIRED. With ~120 windows × 2047 scored tokens each, fp16 accumulation accumulates rounding error sufficient to perturb the 4th significant figure of PPL.

---

## §7 FP16-baseline vs ternary comparison protocol

The headline tern-core PPL deliverable is the comparison between a model's FP16 baseline and its ternary-compressed variant.

**PPL pair:** A "PPL pair" consists of:

- **Baseline:** The model in its source FP16 form, loaded via HuggingFace `AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)`.
- **Variant:** The same model after tern-core compression, loaded via the `.tern-model` zero-copy loader (per R4-C `load_packed_model` path).

Both MUST evaluate against IDENTICAL:

- WikiText-2 test stream (same revision pin, same concatenation, same total token count)
- Tokeniser (the source model's bundled tokeniser; identical BOS handling)
- `seq_len`, `stride`, special-token handling
- Hardware device (same `torch.device`)
- Run cadence (run baseline and variant in the same session where feasible to control for environmental drift)

Differences that ARE permitted (and recorded explicitly in each run's JSON):

- Activation dtype mixing within ternary variant per its `.tern-model` config
- Batch size (typically 1 for both, but documented in `hardware`)
- Random seed (PPL eval is deterministic so seed is informational only)

**`ppl_headroom` metric:**

```
ppl_headroom = (ppl_ternary - ppl_fp16) / ppl_fp16
```

Expressed as a fraction (0.05 = 5% headroom). Lower is better. Reported to 4 significant figures.

**Threshold bands** (informational; not pass/fail gates unless promoted to gates by downstream tooling such as the R8 diagnostic):

| Band | `ppl_headroom` range | Disposition |
|---|---|---|
| Excellent | < 0.02 | Production-ready |
| Acceptable | 0.02 – 0.10 | Production candidate; document trade-offs in release notes |
| Marginal | 0.10 – 0.25 | Diagnostic compression; investigate before deployment |
| Fail | > 0.25 | Compression methodology requires revision |

The R8 diagnostic uses `ppl_headroom=0.50` as the upper sweep ceiling for parameter exploration; this is exploration scope, not a quality target.

**Reference calibration point:** TinyLlama-1.1B at b_mse=3, OPT-B packed bits: `ppl_fp16=7.82`, `ppl_ternary=8.14`, `ppl_headroom=0.0409` (Acceptable band). Source: `tq_bench_results.json`, 2026-03-30. Methodology version at calibration time was pre-v1.0 (this document); re-baselining under v1.0 is in R8 scope.

---

## §8 Results JSON schema

All PPL evaluation runs emit a structured JSON record. Schema:

```json
{
  "schema_version": "wikitext2_ppl/1.0",
  "run_id": "string (uuid4 or yyyymmddTHHMMSSZ timestamp)",
  "timestamp_utc": "ISO 8601 (e.g. 2026-05-14T01:13:25Z)",
  "tern_core_version": "string (e.g. 0.6.0)",
  "tern_core_git_commit": "string (10-char sha)",

  "model": {
    "model_id": "string (e.g. meta-llama/Llama-3.2-1B)",
    "variant": "fp16 | ternary",
    "source_path": "string (model artefact path or HF id)",
    "tern_model_manifest_sha256": "string | null (only present if variant=ternary)"
  },

  "tokeniser": {
    "source": "string (HF tokenizer id or local path)",
    "bos_token_id": "int | null",
    "bos_prepended": "bool"
  },

  "dataset": {
    "name": "wikitext-2-raw-v1",
    "split": "test",
    "huggingface_revision": "string (commit sha at load time)",
    "total_tokens": "int (after BOS prepend if applicable)",
    "tokens_discarded": "int (final partial window)"
  },

  "methodology": {
    "spec_version": "wikitext2_ppl_methodology v1.0",
    "seq_len": 2048,
    "stride": 2048,
    "rolling_variant_included": "bool"
  },

  "hardware": {
    "device": "string (cuda:0 | mps | cpu)",
    "dtype_activation": "string (float16 | mixed | float32)",
    "dtype_loss": "float32",
    "batch_size": "int"
  },

  "results": {
    "windows_evaluated": "int",
    "tokens_scored": "int",
    "mean_loss": "float (>= 4 sig fig)",
    "ppl": "float (>= 4 sig fig)",
    "ppl_rolling": "float | null (only if rolling variant computed)",
    "per_window_losses": "[float] | null (optional diagnostic array)"
  },

  "comparison": {
    "baseline_run_id": "string | null (links to paired FP16 run)",
    "baseline_ppl": "float | null",
    "ppl_headroom": "float | null (>= 4 sig fig)",
    "ppl_headroom_band": "Excellent | Acceptable | Marginal | Fail | null"
  },

  "notes": "string (free-form; methodology deviations or context)"
}
```

**Storage convention:** One JSON file per run. Filename pattern:

```
ppl_<model_short>_<variant>_<timestamp>.json
```

Examples:

- `ppl_llama32-1b_fp16_2026-05-14T011325Z.json`
- `ppl_llama32-1b_ternary_2026-05-14T013047Z.json`
- `ppl_gemma4-31b_ternary_2026-05-14T021500Z.json`

Stored in `tern-core/results/wikitext2_ppl/` (mkdir if not present). Both baseline and variant JSONs are committed alongside the analysis that consumed them.

**Comparison-record convention:** When emitting a `ternary` variant run that pairs to an existing `fp16` baseline, populate `comparison.baseline_run_id` with the baseline's `run_id` plus `comparison.baseline_ppl` and `comparison.ppl_headroom`. The baseline run JSON does NOT back-reference the variant; pairs are uni-directional (variant → baseline). This permits one baseline to anchor multiple variants without rewriting the baseline JSON.

---

## §9 Reproducibility checklist

For any PPL number cited in patent collateral, external briefs, or release notes, the following MUST be capturable from the results JSON alone:

- [ ] WikiText-2 HF revision pin (`dataset.huggingface_revision`)
- [ ] Tokeniser source + BOS handling (`tokeniser.source`, `tokeniser.bos_prepended`)
- [ ] `seq_len` and `stride` (`methodology.seq_len`, `methodology.stride`)
- [ ] Methodology spec version (`methodology.spec_version`)
- [ ] Hardware device + activation/loss dtype (`hardware.device`, `hardware.dtype_activation`)
- [ ] tern-core version + git commit (`tern_core_version`, `tern_core_git_commit`)
- [ ] Model artefact source path (FP16: HF model_id; ternary: `.tern-model` manifest sha256)
- [ ] Total tokens scored + tokens discarded (`dataset.total_tokens`, `dataset.tokens_discarded`)
- [ ] Comparison pair `baseline_run_id` if reporting `ppl_headroom`

Any deviation from §1–§8 MUST be documented in `results.notes` and flagged in any downstream citation. Citations that cannot be reconstructed from a single results JSON are not eligible for patent-evidence or external-collateral use under v1.0 methodology.

---

## §10 Hardware notes

**MPS (Apple Silicon, primary tern-core dev hardware):**

- `dtype_activation = float16` is supported and is the default for FP16 baseline
- BNNS cache effects may produce first-run variance; PPL eval itself is deterministic and unaffected by BNNS cache, but if also recording throughput / timing metrics, warm up with a discard window first
- `torch.mps.synchronize()` not required for PPL correctness but recommended before reading loss values
- 64 GB unified memory comfortably accommodates seq_len=2048 across the target model family up to 31B dense via tern-core zero-copy loader (per R4-C Phase 4 22.5 GiB RSS at load on Gemma 4 31B)

**CUDA:**

- Standard `torch.cuda` semantics apply
- Mixed-precision autocast NOT used in PPL eval — explicit dtype management only
- `torch.backends.cudnn.deterministic = True` recommended though PPL eval is forward-pass-only and deterministic regardless

**CPU:**

- Permitted for small models (TinyLlama 1.1B, Llama 3.2 1B); throughput-prohibitive for ≥7B models
- Loss values are numerically equivalent to CUDA/MPS within float32 tolerance

---

## §11 References

- Merity, S., Xiong, C., Bradbury, J., Socher, R. (2016). "Pointer Sentinel Mixture Models." arXiv:1609.07843. Original WikiText-2 dataset paper.
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I. (2019). "Language Models are Unsupervised Multitask Learners." OpenAI. GPT-2 paper — non-overlapping seq_len=1024 PPL convention.
- HuggingFace `Salesforce/wikitext` dataset card — canonical hosted source.
- tern-core R6 long-sequence convergence validation (commit `df4c141`, 2026-05-13) — anchors the `seq_len=2048` choice.
- tern-core R4-C native `.tern-model` bench harness (PR #19) — consuming pipeline for ternary-variant evaluation.
- tern-core `tq_bench_results.json` (2026-03-30) — TinyLlama-1.1B calibration data point referenced in §7.

---

*Generated 2026-05-14 — tern-core canonical methodology document. v1.0 supersedes any prior ad-hoc PPL eval conventions in the repository; pre-v1.0 results require re-baselining before consumption in v1.0-conformant downstream analysis.*
