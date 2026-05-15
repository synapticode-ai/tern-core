# WikiText-2 Autoregressive PPL Methodology — R7-B v1.0

**Document status:** Canonical (v1.0)
**Authored:** 2026-05-15
**Repository path:** `tern-core/docs/wikitext2_ppl_methodology_autoregressive.md`
**Companion documents:**
- `wikitext2_ppl_methodology.md` (R7-A v1.0) — teacher-forcing PPL methodology; sibling document sharing eval corpus + tokenisation conventions
- `weight_compression_ppl_headroom_diagnostic.md` (R8 v1.1) — weight-compression diagnostic consuming R7-A
- `r8_v1.1_disposition_note.md` — ratification record for the R8 split that surfaced R7-B as a separate methodology
**Scope:** Strict autoregressive perplexity evaluation methodology that measures language-model quality under token-by-token generation with accumulating KV cache. Enables the R12 KV-cache compression diagnostic by exposing the cache-mediated quality signal that teacher-forcing methodologies cannot reach.

---

## §1 Purpose

R7-B v1.0 is the **autoregressive sibling** to R7-A v1.0's teacher-forcing methodology. Both measure PPL on the same WikiText-2 test corpus using the same tokenisation conventions; they differ in *how the model processes the corpus*.

| Aspect | R7-A v1.0 (teacher-forcing) | R7-B v1.0 (autoregressive) |
|---|---|---|
| Forward-pass shape | Single forward pass per window (`seq_len` tokens at once) | Token-by-token, L-1 forward passes per sequence of length L |
| KV cache lifecycle | Within one forward call only (not user-visible) | Explicit accumulation across forward calls; user-injectable |
| What it isolates | Weight-compression effects on logits | KV-cache-compression effects on logits via accumulated cache |
| Primary consumer | R8 v1.1 weight-compression diagnostic | R12 KV-cache compression diagnostic (to draft) |
| Production cost | ~150s per eval on TinyLlama-1.1B MPS | ~30-60 min per eval on TinyLlama-1.1B MPS at N=16 sequences |

R7-B's purpose is to provide R12 with a methodologically-sound measurement hook: a procedure where KV-cache values exist as accumulating intermediate state across forward calls, and where compression of that cache is observable via PPL degradation. R7-A's teacher-forcing methodology cannot serve this purpose — its forward passes do not retain inter-call KV state, so KV-cache compression has nothing to corrupt.

### §1.1 The methodology invariant

**Under no KV-cache compression, R7-B and R7-A must produce equivalent PPL on the same model** (modulo float-precision differences between accumulated-cache attention and single-pass attention, expected <0.5%).

This invariant is the calibration gate: R7-B v1.0 first execution on TinyLlama-1.1B FP16 must produce baseline PPL within 0.5% of the R7-A v1.0 cached baseline (`8.0307`, run_id `20260514T031257Z`, commit `77532e8`). Material deviation flags an implementation bug — typically position-embedding mismatch, BOS handling, or attention-mask logic in the token-by-token loop.

### §1.2 Distinction from the 12 May TQ bench PPL anchor

The TinyLlama `b_mse=3` PPL anchor in `benchmarks/tq_bench_results.json` (2026-03-30, ppl_baseline=7.82 → ppl_ternary=8.14) was measured under "5 sentences, single forward pass each" — teacher-forcing on short isolated sequences with no inter-sentence cache reuse. That methodology does NOT exercise KV-cache compression effects. Its PPL signal reflects weight-compression error (similar to what R7-A measures), not KV-cache error.

**R7-B v1.0 supersedes the 12 May methodology for any future KV-cache PPL measurement.** The 12 May anchor is preserved as documentary record but is NOT R7-B-conformant calibration. R12's first execution will re-establish the b_mse anchor under R7-B v1.0 methodology, which is the only way to legitimately attribute PPL deltas to KV-cache compression.

---

## §2 Eval corpus

R7-B uses the **same corpus, split, and revision pin as R7-A v1.0 §2**:

- Dataset: `wikitext-2-raw-v1`, test split
- HuggingFace revision: pinned per-run in the output JSON's `dataset_revision` field
- Concatenation: test split's `text` column joined with empty separator into a single contiguous stream (matching R7-A §2 convention)

The shared corpus means R7-B and R7-A baseline runs on the same model are directly comparable per §1.1 invariant.

---

## §3 Tokenisation

R7-B uses the **same tokenisation conventions as R7-A v1.0 §3**:

- Per-model tokeniser via `AutoTokenizer.from_pretrained(model_id, revision=...)`
- `add_special_tokens=False` on the corpus tokenisation call
- BOS handling differs from R7-A; see §4

The tokenised stream is the same `tokens: List[int]` object R7-A produces; R7-B differs only in how the stream is segmented and processed.

---

## §4 Sequence construction

Where R7-A processes the entire tokenised stream as non-overlapping sliding windows of `seq_len=2048`, R7-B samples **N sequences of length L** from the stream and processes each as an independent autoregressive context.

**R7-B v1.0 canonical parameters:**

| Parameter | v1.0 value | Notes |
|---|---|---|
| `sequence_length` (L) | 2048 | Matches R7-A `seq_len` for direct comparability |
| `sequence_count` (N) | 16 | Pragmatic; ~27 min wall-clock on TinyLlama MPS |
| `stride_between_sequences` | L (non-overlapping) | First sequence starts at token 0; second at token L; etc. |
| `bos_handling` | Prepend BOS to EACH sequence | Each sequence has its own KV-cache lifecycle |
| `eos_insertion` | Not inserted | Same as R7-A |

**BOS handling distinction from R7-A:** R7-A prepends BOS once at the start of the entire tokenised stream (§4 of that document). R7-B prepends BOS to each of the N sequences independently, because each sequence is an independent autoregressive context with its own KV-cache lifecycle. A sequence starting mid-stream without BOS would feed the model an out-of-distribution prefix; BOS resets the model's notion of "start of context."

**Why N=16 not all of test:** WikiText-2 test contains ~245k tokens. Processing the entire stream autoregressively would require ~120 sequences × 2048 forward passes = 245k forward passes, ~6 hours of wall-clock on TinyLlama MPS. N=16 covers ~32k tokens (~13% of test), enough for statistically meaningful PPL while keeping eval feasible. Future v1.1 may increase N if compute budget allows.

**Sampling determinism:** Sequences are taken sequentially from the stream (first N×L tokens), NOT randomly sampled. This makes results bit-reproducible across runs without seed management for sequence selection.

---

## §5 Autoregressive generation loop (CANONICAL)

The heart of R7-B v1.0. Reference pseudocode in Python with HuggingFace conventions:

```python
def autoregressive_ppl(model, tokenizer, sequence: List[int],
                       kv_cache_hook=None) -> Tuple[float, int]:
    """Compute autoregressive PPL for a single sequence.

    Args:
        model:         HF causal LM with use_cache support.
        tokenizer:     Tokeniser (for BOS token id).
        sequence:      List of L token ids (corpus-derived, NOT BOS-prepended).
        kv_cache_hook: Optional callable(past_kv) -> past_kv applied between
                       forward calls. R12 injects KV-cache compression here.
                       Default: identity (no compression).

    Returns:
        (sum_log_loss, num_tokens_scored)
    """
    if kv_cache_hook is None:
        kv_cache_hook = lambda kv: kv

    bos_id = tokenizer.bos_token_id
    tokens = [bos_id] + sequence  # BOS prepended per §4
    L_eff = len(tokens)

    past_kv = None
    sum_loss = 0.0
    num_scored = 0

    for t in range(L_eff - 1):
        # Input: token at position t. Past KV holds positions 0..t-1.
        # Output: logits at position t, used to predict token t+1.
        input_ids = torch.tensor([[tokens[t]]], device=model.device)
        outputs = model(input_ids=input_ids,
                        past_key_values=past_kv,
                        use_cache=True)
        past_kv = outputs.past_key_values
        past_kv = kv_cache_hook(past_kv)  # R12 compression injection

        logits = outputs.logits[0, -1]  # last (only) position's logits
        target = torch.tensor(tokens[t + 1], device=model.device)
        loss = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0),
                               reduction='sum').item()
        sum_loss += loss
        num_scored += 1

    return sum_loss, num_scored
```

**Loss accumulator type:** `float` (Python double, 64-bit). The loss-per-token tensor is cast to `.item()` before accumulation, avoiding GPU float32 accumulator drift on long sequences.

**Final PPL:** Across all N sequences, total log-loss summed in float64, divided by total `num_scored`, then `exp()`:

```python
total_loss = sum(seq_loss for seq_loss, _ in sequence_results)
total_scored = sum(seq_n for _, seq_n in sequence_results)
ppl = math.exp(total_loss / total_scored)
```

For N=16 sequences of L=2048 with BOS prepended, total `num_scored = 16 × 2048 = 32,768` tokens.

### §5.1 Why one token at a time

The model receives ONLY the single new token per forward call. The KV cache holds all prior positions. This is mechanically identical to how transformers autoregressively generate at inference time — the difference is we provide the next "generated" token from the corpus (teacher-forcing in spirit) rather than sampling from the logits.

This contrasts with R7-A, which feeds the entire window's tokens in one forward call and uses a causal-mask-aware single-pass attention. R7-A's single-pass attention computes K and V for all positions in one matrix multiply; R7-B's accumulating cache builds K and V position-by-position. Under no compression, the resulting attention output should be numerically identical modulo float-rounding (typically <0.01% per-token loss difference; <0.5% aggregate PPL difference).

### §5.2 KV cache compression injection (R12 hook)

The `kv_cache_hook` callable is the R12 integration surface. R12's KV-cache compression diagnostic injects its compression operator here:

```python
def make_b_mse_hook(b_mse: int):
    """R12 compression hook factory."""
    def hook(past_kv):
        # Iterate (K, V) tensor pairs per layer; apply IncrementalTQCompressor
        # at the configured b_mse parameter; return updated past_kv.
        ...
    return hook

# Usage in R12 sweep:
for b_mse_value in [1, 2, 3, 4, 5, 6]:
    hook = make_b_mse_hook(b_mse_value)
    sum_loss, n = autoregressive_ppl(model, tokenizer, sequence, kv_cache_hook=hook)
    # record sum_loss / n per sweep point
```

The hook receives the full `past_key_values` tuple-of-tuples (HF format: outer tuple over layers, inner tuple of (K, V)). The compression operator may mutate or replace tensors; the returned object must be a valid `past_key_values` for the next forward call.

### §5.3 Memory accounting

Autoregressive PPL holds the full KV cache for one sequence in GPU/MPS memory throughout that sequence. For TinyLlama-1.1B at L=2048 in float16, the KV cache is ~50 MB; on Gemma 4 31B at L=2048 in float16, ~2.5 GB. Future scale-up (Gemma 4 31B, Llama 70B) will need explicit memory accounting and may require lower N or shorter L.

For v1.0, the canonical TinyLlama-1.1B target fits comfortably in M4 Pro 64 GB unified memory.

---

## §6 Threshold bands

R7-B uses the **same `ppl_headroom` bands as R7-A v1.0 §7**:

| Band | `ppl_headroom = (ppl_compressed - ppl_baseline) / ppl_baseline` |
|---|---|
| Excellent | < 2% |
| Acceptable | 2% – 10% |
| Marginal | 10% – 25% |
| Fail | > 25% |

R12 consumes these bands directly to classify KV-cache compression operating points. The bands are methodology-neutral — a `ppl_headroom` of 5% means the same thing for weight compression (R8 v1.1) and KV-cache compression (R12).

---

## §7 Output schema

R7-B emits per-run JSON conformant to `wikitext2_ppl_autoregressive/1.0`:

```json
{
  "schema_version": "wikitext2_ppl_autoregressive/1.0",
  "run_id": "string (timestamp ISO 8601 compact, e.g. 20260515T091500Z)",
  "tern_core_version": "string",
  "tern_core_git_commit": "string (10-char sha)",
  "spec_version": "wikitext2_ppl_methodology_autoregressive v1.0",

  "model": {
    "model_id": "string (HF model_id or local path)",
    "huggingface_revision": "string (full sha)",
    "dtype": "float16 | float32 | bfloat16",
    "device": "string (mps | cuda:0 | cpu)"
  },

  "tokeniser": {
    "tokenizer_id": "string (defaults to model_id)",
    "bos_token_id": "int",
    "vocab_size": "int"
  },

  "dataset": {
    "name": "wikitext-2-raw-v1",
    "split": "test",
    "dataset_revision": "string (full sha pinned at load)"
  },

  "config": {
    "sequence_length": "int (L)",
    "sequence_count": "int (N)",
    "stride_between_sequences": "int (typically L)",
    "bos_handling": "prepend_per_sequence",
    "eos_insertion": "none",
    "kv_cache_compression": {
      "enabled": "bool",
      "hook_spec": "string (e.g. 'b_mse_compressor_v1' or 'none')",
      "parameters": "object (hook-specific)"
    }
  },

  "results": {
    "total_loss_float64": "float (sum of per-token cross-entropy)",
    "tokens_scored": "int (N * L when BOS prepended)",
    "mean_loss": "float (4 sig fig)",
    "ppl_autoregressive": "float (4 sig fig)"
  },

  "comparison": {
    "baseline_run_id": "string | null (links to R7-B or R7-A baseline JSON)",
    "baseline_ppl": "float | null",
    "baseline_methodology": "wikitext2_ppl/1.0 | wikitext2_ppl_autoregressive/1.0 | null",
    "ppl_headroom": "float | null (computed when baseline given)",
    "ppl_headroom_band": "Excellent | Acceptable | Marginal | Fail | null"
  },

  "timing": {
    "model_load_seconds": "float",
    "eval_wall_time_seconds": "float",
    "tokens_per_second": "float (tokens_scored / eval_wall_time)"
  },

  "hardware": {
    "device": "string",
    "torch_version": "string",
    "transformers_version": "string"
  },

  "notes": "string"
}
```

Filename: `ppl_ar_<model_short>_<dtype>_<timestamp>.json` (e.g. `ppl_ar_tinyllama-1.1b-chat-v1.0_fp16_20260515T091500Z.json`).

Stored under `results/wikitext2_ppl_autoregressive/`. The directory split from R7-A's `results/wikitext2_ppl/` keeps teacher-forcing and autoregressive outputs cleanly separated.

---

## §8 R7-B v1.0 first execution — TinyLlama-1.1B FP16 baseline + calibration

The canonical first execution validates R7-B against R7-A per §1.1 invariant.

**Inputs:**

```yaml
model_id: TinyLlama/TinyLlama-1.1B-Chat-v1.0
huggingface_revision: b08601e04326c79dfdd32d625aee71d232d685c3  # same as R7-A baseline
dtype: float16
device: mps
sequence_length: 2048
sequence_count: 16
kv_cache_compression:
  enabled: false  # no compression — baseline run
seed: 1337  # for any tiebreak determinism; sequence selection is sequential, not random
```

**Acceptance gates:**

1. **§1.1 invariant**: produced `ppl_autoregressive` must be within 0.5% of R7-A v1.0 cached baseline (`8.0307`). I.e., result MUST be in `[7.99, 8.07]`. Outside this range = methodology bug, halt and surface.
2. Schema conformance: output JSON validates against `wikitext2_ppl_autoregressive/1.0`
3. Reproducibility: bit-identical PPL on second run (deterministic sequence selection + deterministic forward passes)
4. Wall-clock: should be 25-40 min on M4 Pro MPS. >60 min flags performance regression worth investigating.
5. `tokens_scored = N × L = 32,768` (exact, not approximate)

**Output:** R7-B v1.0 conformant baseline JSON at `results/wikitext2_ppl_autoregressive/ppl_ar_tinyllama-1.1b-chat-v1.0_fp16_<TIMESTAMP>.json`. This file becomes the canonical R7-B baseline for TinyLlama-1.1B; R12 references it via `baseline_run_id` for `ppl_headroom` computation.

---

## §9 Cross-architecture extension (post-v1.0 first execution)

Same matrix as R8 v1.1 §9. Each model needs its own R7-B baseline before R12 can run a sweep:

| Model | R7-B baseline status | Notes |
|---|---|---|
| TinyLlama-1.1B | v1.0 first execution (this document §8) | Calibration target; §1.1 invariant gate |
| Llama 3.2 1B | Pending R7-B v1.0 baseline | Independent-family validation |
| Gemma 4 E4B | Pending R7-B v1.0 baseline AND R9.3 (PPL calibration) | Gemma 4 PPL pathology must resolve first per R9.3 backlog |
| Gemma 4 31B | Pending R7-B v1.0 baseline AND R9.3 | Same Gemma 4 dependency |
| Mistral 7B | Pending R7-B v1.0 baseline | Historical reference |
| Phi-4 14B | Pending R7-B v1.0 baseline | Mid-scale; no BOS prepend rule per R7-A §4 also applies in R7-B §4 |

Per-model baselines are NOT optional: R12 cannot legitimately compute `ppl_headroom` against an R7-A baseline (different methodology, see §1.2). Each architecture's R7-B baseline must land before its R12 sweep.

---

## §10 References

- `docs/wikitext2_ppl_methodology.md` (R7-A v1.0) — companion teacher-forcing methodology; shared corpus and tokenisation conventions
- `docs/weight_compression_ppl_headroom_diagnostic.md` (R8 v1.1) — weight-compression diagnostic consuming R7-A v1.0
- `docs/r8_v1.1_disposition_note.md` — disposition record that surfaced R7-B as the necessary KV-cache methodology
- `docs/backlog.md` R12 entry (to-add) — KV-cache diagnostic spec, the primary consumer of R7-B
- `docs/backlog.md` R13 entry — `ppl_headroom` terminology disambiguation (autoscan-internal vs R7-A/R7-B/R8 v1.1 outcome metric)
- `docs/backlog.md` R9.3 entry — Gemma 4 PPL methodology calibration prerequisite for §9 cross-arch extension
- `tools/tern_ppl_bench.py` (commit `5e74307`, PR #20) — R7-A reference implementation; R7-B reference implementation extends this tool with `--methodology autoregressive` flag OR adds sibling `tools/tern_ppl_bench_ar.py` at surgeon's discretion
- `benchmarks/tq_bench_results.json` (2026-03-30) — historical 12 May TinyLlama `b_mse=3` anchor (PPL 7.82→8.14); preserved as documentary record but NOT R7-B-conformant (see §1.2)
- HuggingFace `transformers` documentation on `past_key_values` and `use_cache` for the autoregressive cache API surface

---

*Generated 2026-05-15 — tern-core canonical methodology document. R7-B v1.0 is the autoregressive sibling to R7-A v1.0, enabling R12 KV-cache compression diagnostic. The §1.1 invariant (R7-B baseline ≈ R7-A baseline within 0.5% under no compression) is the methodology's calibration gate.*
