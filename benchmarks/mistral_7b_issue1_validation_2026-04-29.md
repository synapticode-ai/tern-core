# Mistral-7B — Issue #1 ground-truth validation
## 2026-04-29

Confirms Issue #1's "Input X contains infinity at op 295/298"
failure reproduces faithfully against its archived, sha256-
verified original mlpackage. Validates the Session 4 defensive
fix's framing as correctly scoped (forward-only, not
retroactive).

## Method

Palettise the archived Mistral-7B v0.3.0 mlpackage with the
current coremltools (9.0) palettiser using `nbits=2 mode=kmeans`
on `CPU_ONLY`. Same script and parameters as DSR1-7B's clean
validation in Session 4.

## Artefact provenance

Source: KAIST delivery package
  `/Users/syn/synapticode/packages/kaist_delivery/mistral_7b_ternary_v0.3.0.tern-pkg`

Extracted to:
  `/tmp/mistral_pkg_weights/mistral_7b_ternary.mlpackage`

Three-way sha256 match on `weight.bin`
(`24f264857c417d312eeb2c2b5dcbb5085c12a632fa8f6a5ef5f2c599697f417e`):

- `.tern-pkg` extract (verified 2026-04-29)
- WD Passport archived copy (verified 2026-04-29)
- Migration `MANIFEST.md` anchor (recorded 2026-04-18)

Producer: `tern_compiler_version 0.3.0`, exported 2026-03-29.
Predates Session 4 defensive fix (`v0.6.0+`).

## Outcome

Palettise progressed to op **295/298** (three ops from
completion) then crashed at the k-means clustering step:

```
ValueError: Input X contains infinity or a value too large
for dtype('float64').
```

Stack: `palettize_weights` → `blockwise_compress` →
`grouped_channelwise_compress` → `_get_lut_and_indices` →
`compress_kmeans` → `KMeans.fit` → `check_array` →
`_assert_all_finite` → `ValueError`.

Wall-time to failure: ~10:54.

## Interpretation

P-fail with the originally-reported failure mode.
Architecturally clean validation. Five points confirmed:

1. **Issue #1 was accurately described.** Op 295/298 +
   infinity reproduces 32 days later against bit-identical
   input.

2. **v0.3.0 exporter was the producer.** Inf is baked into
   the archived `weight.bin`; current code didn't introduce
   it.

3. **Defensive fix at v0.6.0+ is correctly scoped as
   forward-only.** Future v0.6.0+ exports against Mistral
   source weights would hit `_validate_ternary2_alpha` or
   `_cast_fp16_retain_with_guards` BEFORE the FP16 cast
   site, either raising on the pathological value or
   clamping with operator-visible WARNING. The archived
   v0.3.0 mlpackage stays affected because it's already
   produced; the fix doesn't retroactively scrub baked-in
   Inf.

4. **DSR1-7B Session 4 clean validation contrast holds.**
   Same code path, different model, no Inf produced —
   confirms the bug is Mistral-specific (likely a
   late-layer pathological alpha or out-of-range FP16
   retain weight at one of the last few layers).

5. **No coremltools version skew.** coremltools 9.0 today
   hits the same failure mode as originally reported; not a
   behaviour change between Mar 29 and now.

## Diagnostic surfacing

The progress bar landing on op 295/298 (three ops from
completion) localises the Inf to a small number of
late-layer weights. Plausible candidates if Mistral is ever
re-quantised under v0.6.0+:

- A late-layer `lm_head` projection (FP16 retain) with
  extreme outlier weights — would hit
  `_cast_fp16_retain_with_guards`'s clamp path with WARNING
  naming the specific layer.
- A late-layer `o_proj` or `down_proj` ternary alpha that's
  degenerate — would hit `_validate_ternary2_alpha`'s raise
  path with the specific layer name in the error message.

Either way, a v0.6.0 re-quantisation would surface the
pathological layer rather than silently embedding Inf in the
mlpackage. That's the three-way handling pattern's design
intent (raise-or-clamp with operator-visible signal,
layer-named) working as designed.

## Status

Issue #1 closed in Session 4 by PR #2 (defensive fix landed
at commit `7138b67`, merged as `95f0049`). This validation
confirms the fix's framing without re-opening the issue.

Mistral-7B re-quantisation is not currently scheduled. If it
becomes scheduled, the fix's diagnostic value applies —
expect a clean export with WARNING naming the late-layer
pathological value, OR a ValueError naming the degenerate
ternary2 alpha.
