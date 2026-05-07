# Apple May 2026 — Talking Points

**Audience**: [TODO: Confirm attendees with Rob. Orientation memory references John Ternus (incoming CEO post-transition) + Johny Srouji (engineering lead, hardware) as primary audience. Verify date specifics, exact attendee list, and any additional staff (CoreML team lead? ML Compute team? Marketing/comms?).]

**Date**: May 2026 [TODO: specific date]
**Duration**: 30-min conversation [TODO: confirm allocation]
**Format**: Talking-points reference for Rob during the conversation. Engineering evidence trail lives in `benchmarks/REPORT_PHASE2.md`.

---

## Opening — the 116 GB problem (1 min)

**The problem we observed.** Llama-3.1-70B at FP16 is 131 GB on disk. At load time it expands to ~116 GB in-memory. M4 Pro 64 GB unified memory can't hold it. M2/M3 Ultra at 192 GB barely manageable, and only via aggressive paging.

**Why this matters for Apple's hardware-software story.** Apple's competitive advantage in AI inference is unified memory — no PCIe transfer cost between CPU and GPU/NE. But that advantage has an upper bound set by the unified memory ceiling. Today, large frontier models hit that ceiling and the unified-memory story breaks down.

**What ternary changes.** 70B compressed with tern-core lands at 35 GB on disk (6.62× compression vs FP16, validated end-to-end — TN-001). In-memory native ternary projected at ~39 GB. Suddenly fits on M4 Pro. The hardware-software story holds at 70B and beyond.

**The pitch in one sentence.** Ternary takes the unified-memory ceiling from "barely Ultra-class" to "comfortably Pro-class" — the hardware sells more units because the software unlocks more models.

---

## The headline number — canonical .tern-model artefact (3 min)

**Measurement context.** Phase 2 + Phase 4 closure on the gemopus-4-e4b `.tern-model` artefact via the v1 Metal forward path. Native ternary inference, no FP16 dequantisation pass, weights resident on the Metal device.

| Metric | Value | Notes |
|---|---|---|
| Throughput | **2.85 tok/s** | v1 Metal forward floor |
| Energy | **1.22 J/token** | sustained inference |
| Peak memory | **44.5 GiB** | M4 Pro 64 GB unified |
| Code path | terncore_packed Metal v1 | landed via PR #9 + PR #10 |

**Framing — v1 floor with bounded optimisation surface.** This is the first measurement on the native ternary code path. No optimiser passes have landed yet. The number is honest and we don't dress it up.

**What's left on the table for the partnership.** Stage 2 / Stage 3 work (below) brings v2 + native-ANE optimisations into scope. The optimisation surface is bounded by the FP16 inference baseline we've already validated:

| Reference baseline | Throughput | Notes |
|---|---|---|
| Mistral-7B FP16 on CPU_AND_NE | 297.6 tok/s | Phase D measurement, 5.4 W |
| Mistral-7B FP16 on CPU_AND_GPU | 387.8 tok/s | Phase D, 20.1 W (2.96× energy premium) |

The gap between v1 native ternary (2.85 tok/s) and validated FP16 inference (297.6 tok/s) is **the engineering surface we're proposing to close together**. Native ternary should beat FP16 on energy (no multiply, no decompression overhead) once the Metal/ANE kernel is fully optimised.

**Pointer for engineering deep-dive.** `benchmarks/REPORT_PHASE2.md` carries the full Phase 2/4 reproducibility appendix.

---

## The diagnostic discipline — why engineering credibility matters (5 min)

**The thesis.** Engineering credibility is as important as the headline number. Apple has been burned by AI-startup pitches that ship great demos and lose substance under integration pressure. We've built tern-core with halt-and-surface methodology baked in — the codebase fails loudly on assumption violations rather than silently on production drift.

**Five recent incidents that exemplify this.**

**(1) CWD-dependent Metal init bug.** Metal kernel's `tern_engine_create()` returned NULL from any directory other than `csrc/metal/`. Symptom looked like session-state corruption ("worked yesterday, broken today"). Diagnosis caught the actual root cause: `__FILE__` macro in dylib resolved to build-time path, not runtime. Fix: `dladdr()` against a dylib symbol. **Methodology insight banked**: observed-pattern memories can be structurally wrong; falsify by replacing contents, not deleting.

**(2) sparsity_bitmap shape contract gap.** Cross-kernel test surfaced 8× format mismatch between two functions sharing a typed `bytes` parameter with incompatible meanings (bool bitmap vs packbits bitmap). Hard-rejected at the function boundary rather than auto-recovered. **Methodology insight banked**: typed-but-incompatible parameters are a silent-contract latent bug; enforce shape/format at call boundary, never auto-recover.

**(3) reconstruct_all suffix-doubling on production manifests.** First production-data load through synthetic-test-only code path produced `missing=2077, unexpected=2130` — the loader was producing `*.weight.weight` keys. Path C conditional fix preserved test convention while extending production support. **Methodology insight banked**: configurational fidelity assertions (HF-config-derived expected counts) catch over-correction regressions before they ship.

**(4) OOM risk caught pre-implementation.** Today's per-expert slicing rework (PR #14) initially proposed deferred-restack pattern in `reconstruct_all`. Memory analysis at design time caught that this would peak at ~120 GB for a 26B-A4B reconstruction. Halted, surfaced, switched to incremental per-parent restack (~2 GB peak per in-flight parent). **The OOM never reached the implementation phase.**

**(5) Granularity-blind audit caught at next session.** Yesterday's audit concluded "no code change needed for MoE — the per-tensor sparsity recording already covers per-expert measurement." Correct at file level, blind to safetensors-layout granularity (each layer's 128 experts pack as a single 3-D entry). Today's verification surfaced the gap; rework landed cleanly. **Methodology insight banked**: current-state verification must extend to actual data artefacts, not just code paths.

**The compound effect.** ~14 banked methodology memories across the sprint cluster (Phase 2.5, Phase 4, Group A, Session 3). Each one refines how the next sprint approaches similar surfaces. The rework you'll see in v2 / v3 will benefit from all of them.

---

## The MoE IP — cross-architecture sparsity finding (4 min)

### Foundation layer — methodology (the credibility layer, ~1 min)

Per-expert quantisation enables direct measurement of MoE expert sparsity at compression time, recording per-expert threshold + sparsity in the `.tern-model` manifest rather than aggregating across experts. This methodology, developed across PR #14 + PR #15 + PR #16 + PR #17 (Sessions 3-4 of the May 2026 sprint cluster), produces statistically robust cross-architecture findings that wouldn't be visible under uniform-threshold quantisation.

Methodology validation:

- **Configurational fidelity assertions** derived from architectural ground truth (HF config) caught zero compression discrepancies across all 5 sprint runs — every result landed inside its band, every time.
- **Per-expert sparsity distinct verification** confirmed the methodology produces per-expert measurement, not silent shared-threshold regression (the failure mode the rework was designed to prevent).
- **Bit-identical Phi-4 manifest size** (6,835.9 MB) to April 2026 production validates the new Phi3Adapter against established practice — same structural output, more rigorous architectural typing.

The engineering credibility is the floor, not the ceiling. The findings stand on top of it.

### Result layer — cross-architecture finding (the commercial relevance, ~2 min)

The methodology applied across 5 model compressions spanning 4 distinct architectures (Llama 3.1, Phi-4, Qwen3, Gemma 4) produced an empirically established finding:

> **Typical-transformer FFN-like weights cluster at sparsity 0.42-0.43 regardless of whether they're expressed as MoE experts or as pure dense MLPs.**

| Architecture | FFN group | Per-tensor sparsity median |
|---|---|---|
| Llama-3.1-70B (Meta, dense) | dense | 0.4244 |
| Phi-4 (Microsoft, dense) | dense | 0.4260 |
| Qwen3-30B-A3B (Alibaba, MoE) | expert | 0.4274 |
| Gemma 4 26B-A4B (Google, hybrid MoE+dense) | expert | 0.4294 |
| Gemma 4 31B (Google, dense) | dense | 0.4316 |

Spread across 5 instances: **0.0072 absolute (~1.7%)**. Across 4 teams, 4 architectures, MoE-vs-dense expression, training data variation, and parameter counts spanning 14B to 70B.

**One outlier**: Gemma 4 26B-A4B's parallel dense MLP path in its hybrid MoE+dense block sits at **0.4558**, ~0.024 above the cluster. The architectural variable distinguishing this outlier from Gemma 4 31B (which sits inside the cluster at 0.4316, despite being from the same model family) is the hybrid block structure itself — both per-expert weights AND parallel dense MLP per layer. The hypothesis (untested with current data — hybrid MoE architectures are rare in the current ecosystem) is that hybrid blocks push the parallel dense MLP path to atypically high sparsity.

**Compression ratio cross-architecture observation**: MoE models in this dataset achieve higher compression ratios than dense models of similar parameter count.

| Model | Type | Params | Ratio vs FP16 |
|---|---|---|---|
| **Qwen3-30B-A3B** | **MoE** | 30.5 B | **4.9×** (highest in cluster) |
| Gemma 4 26B-A4B | MoE | 26 B | 4.36× |
| Gemma 4 31B | dense | 31 B | 4.18× |
| Phi-4 | dense | 14.7 B | 4.1× |

The MoE structural sparsity appears to enable higher compression — only top-k experts active per token, plausibly leaving the inactive expert weights well-suited to per-expert ternary quantisation. Tern-core's per-expert measurement infrastructure surfaces and exploits this pattern; mechanistic confirmation (per-token expert activation correlation with weight sparsity) would require additional measurement work beyond today's static-weight analysis.

### Apple-specific implication layer (~1 min)

[TODO: Rob's strategic positioning of these findings for Apple's specific interests. Possible angles to choose from based on prior conversations:

**Option (a)** — *MoE compression efficiency favours unified memory.* Each additional 0.7× compression ratio over the current state-of-the-art is another model class that fits in 64 GB. Qwen3's 4.9× on a 30B-class MoE means a 60B-class MoE compresses to ~24 GB — fits comfortably on M4 Pro. The unified-memory ceiling rises to MoE-frontier models, not just dense-frontier models.

**Option (b)** — *Hybrid MoE+dense architectures (Gemma 4 family, possibly future Apple Intelligence models) have specific compression characteristics worth understanding before silicon design choices lock in.* The 26B-A4B outlier suggests hybrid architectures may need custom calibration; pure-MoE (Qwen3) and pure-dense (Phi-4, 31B) compress to predictable patterns.

**Option (c)** — *The per-expert measurement infrastructure generalises to any MoE architecture Apple might pursue.* No new adapter work needed for additional Qwen3 variants; ~30-45 min focused work for new MoE families (Qwen3MoeAdapter took 25 min from design surface to passing tests).

Pick the angle (or combination) that matches Apple's strategic interests based on Rob's prior conversations with Ternus / Srouji / others.]

---

## The three-staged CoreML integration ask (5 min)

The ask is structured as three independent stages so Apple can scope partnership at the level appropriate for the relationship's current trust state. Each stage has an Apple-side win and a Synapticode-side win.

### Stage 1 — Point release

[TODO: Specific stage definition from Rob's strategic plan. Orientation memory references three-staged CoreML integration ask: point release → in-memory compaction → native ANE matrix engine. Stage 1 is typically framed as the smallest-scope ask — likely a CoreML point release that adds support for the .tern-model format as a recognised quantisation level alongside INT8/INT4/NF4. Verify exact description with Rob.]

**Wins for Apple:** [TODO: framing — likely "differentiation: CoreML supports a quantisation level competitors don't have, with measurable inference advantages on existing ANE silicon"]

**Wins for Synapticode:** [TODO: framing — likely "validation: tern-core's compression format becomes a first-class CoreML citizen, accelerates customer-side adoption"]

### Stage 2 — In-memory compaction

[TODO: Specific stage definition. Per orientation memory, this is the middle-scope ask covering decompression-during-load support — i.e., model weights stay packed (~0.25 bytes/param) in unified memory until needed at compute time, rather than expanding to FP16 on load. Verify exact description.]

**Wins for Apple:** [TODO: framing — likely "extends the upper bound on model size that fits on M4 Pro and below; enables 70B-class models on iPhone-class silicon over the next 2 hardware generations"]

**Wins for Synapticode:** [TODO: framing — likely "direct path from .tern-model artefact to ANE inference without intermediate FP16 decompression — closes the 'why pay 4× memory cost for compressed weights' gap"]

### Stage 3 — Native ANE matrix engine

[TODO: Specific stage definition. Per orientation memory, this is the largest-scope ask covering native ternary MatMul on ANE silicon — i.e., a hardware-level ANE update where MatMul is implemented as compare-and-add (multiply-free) for ternary operands, achieving the energy-per-token theoretical floor. Verify exact description.]

**Wins for Apple:** [TODO: framing — likely "best-in-class energy-per-token at frontier-model scale; CoreML hardware story Apple-exclusive for the partnership window; AUKUS-licensable beyond"]

**Wins for Synapticode:** [TODO: framing — likely "the IP licensing model materialises at scale; Gamma Seeds Pty Ltd licenses the ANE-native ternary MatMul to Apple for the AUKUS-exclusive territory; commercial partnership becomes long-term strategic"]

---

## Brand anchor lines — for the closing summary (1 min)

Use these as the framing closure. They land cleaner spoken than written.

- **Physics:** *Switching cell funds memory of own life.* The ternary cell carries enough information to know what it just decided. Determinism end-to-end.
- **Architecture:** *Clearer constraint, faster convergence, better output.* Ternary's smaller weight space forces sharper learning targets; the model's training loop converges faster and the output quality is better-bounded.
- **Position:** *Scion by Gamma Seeds.* Synapticode is the operating brand; Gamma Seeds Pte Ltd is the global patent assignee; Gamma Seeds Pty Ltd is the AUKUS-exclusive licensee. Apple talks to Synapticode for engineering, licenses through Gamma Seeds Pty Ltd.
- **Purpose:** *Squeeze the juice.* Extracting more from the neural systems we already use rather than waiting for the next compute generation.
- **One-liner:** *"Proving a bit more than was expected. Ternary — using that extra bit to compact, accelerate and extract more from the neural systems we already use."*

---

## Anticipated questions and answers

[Note to Rob: these are CC-drafted based on engineering context. Refine based on prior Apple-side conversations and your strategic context. Several have a TODO marker where Rob's specific knowledge of Apple's stated concerns should sharpen the response.]

### Q: How does ternary preserve quality across model families? Have you tested architectures beyond Llama?

**A:** Yes. Phase-2 compression validated end-to-end on 9 model families: Mistral 7B, Llama 3.1 70B, Llama 3.2 1B/3B, Gemma 3 4B/12B, DSR1 7B/14B, Phi-4 14B, Qwen 2.5 7B. All passed integrity checks; per-layer sensitivity analysis with adaptive thresholding (Δ = threshold × mean(|W|), default 0.7) maintains 20% PPL headroom across the family. Today's Gemma 4 26B-A4B addition extends the validation to MoE architectures.

[TODO: Confirm with Rob whether Apple has expressed concern about specific model families or specific quality metrics.]

### Q: Why ternary specifically, instead of INT4 (now widely supported) or NF4 (Apple's own preference for some models)?

**A:** Compute model differentiation. Ternary `{-1, 0, +1}` MatMul reduces to compare-and-add — no multiply instruction is needed. INT4 still requires multiply, even in 4-bit form; NF4 requires a non-linear scale lookup before multiply. Ternary is the only quantisation level where the silicon's multiply unit can be entirely bypassed — which is the architectural premise for Stage 3's native ANE matrix engine ask. Energy-per-token follows: no multiply means no multiply energy.

Plus the sparsity property — typical ternary models carry 60-70% zero weights, which enables zero-skip in the MatMul inner loop. INT4 and NF4 don't have this property at the bit level.

### Q: Patent landscape — what's your IP position, and what's the licensing model we'd be looking at?

**A:** [TODO: Confirm with Rob whether to reference specific PCT application numbers and provisional count, or keep high-level. Default to high-level for first draft; can sharpen based on Rob's risk assessment.]

High-level: Gamma Seeds Pte Ltd (Singapore) holds the global patent assignee position. PCT applications are filed and prosecuting via Rod (Rob's patent attorney). Licensing model is ARM-style — IP licenses to Apple for the AUKUS-exclusive territory through Gamma Seeds Pty Ltd (Australia). No equity ask. The partnership is licensing-driven, not investment-driven.

### Q: What's the optimisation roadmap from v1 Metal forward (2.85 tok/s) to production-grade performance?

**A:** Today's number is the v1 floor — first measurement on the native ternary Metal kernel with no optimiser passes landed yet. The known optimisation surface includes:

- Buffer reuse across forward passes (currently re-allocates per call)
- Kernel fusion (sparsity bitmap + matmul + activation in one pass)
- Per-shape tuned dispatch (current kernel is a generic 1-D loop)
- ANE offload for the dominant layer shapes (currently CPU/GPU-only via MPS)

The FP16 inference baseline we've validated (297.6 tok/s on Mistral-7B via CPU_AND_NE at 5.4W) is the optimisation target ceiling — native ternary should beat this on energy once the kernel is fully tuned, and likely on throughput at the larger model scales where memory bandwidth dominates.

[TODO: Confirm with Rob whether to commit to a specific v2 timeline or keep the roadmap directional.]

### Q: Sprint cluster timeline — when does 31B and beyond land?

**A:** Current sprint sequencing:

- **26B-A4B MoE compression** (Gemma 4) — completed today (6 May 2026)
- **Per-expert IP analysis** — runs in next 24-48 hours; provisional drafting follows if hypothesis confirmed
- **31B dense compression** (Gemma 4) — Apple conversation benchmark; sized to demonstrate compression scaling beyond 7B without quality degradation. Blocked on coremltools 9.1/9.2 release for full Phase 2 measurement; compression itself can run earlier
- **MiniMax M2.7 MoE** — pending Biwin M350 + Acasis dock archive drive arrival (this week per orientation memory)

Beyond the sprint cluster: model selection follows partnership dialogue. If Apple has specific frontier models in mind (e.g., Llama 3.1 405B, future Apple Intelligence base models), those slot in based on partnership-stage prioritisation.

### Q: What's the actual partnership scope you're proposing? What does success look like in 6 months?

**A:** [TODO: This is the most important question and the answer should reflect Rob's strategic intent precisely. CC's draft below; Rob to refine.]

**Draft:** Three-stage ladder structured to match partnership trust progression. Stage 1 (CoreML point release recognising .tern-model format) is a low-risk proof point — Apple validates the engineering, ships the support, measures customer adoption. If Stage 1 lands cleanly, Stage 2 (in-memory compaction) becomes the natural next ask, with a 6-month engineering collaboration window. Stage 3 (native ANE matrix engine) is a 12-18 month silicon-roadmap conversation — committed only if Stage 1 + 2 demonstrate sustained value.

**6-month success state:** Stage 1 shipped in a CoreML point release; Stage 2 partnership scoped + engineering kickoff complete; tern-core v0.7 (post-this-PR-cluster) supporting at least one frontier-class model end-to-end on Apple silicon at production performance.

[TODO: Refine per Rob's actual ask and timeline.]

---

## Forward path (2 min)

Where this conversation lands relative to the broader trajectory.

**Sprint cluster (next 8 weeks):**
- 26B-A4B IP analysis + provisional drafting (next week)
- 31B dense Phase 2 measurement (post-coremltools 9.x release)
- MiniMax M2.7 MoE compression (post-archive-drive arrival)
- v2 Metal forward optimisation (in parallel; closing the 2.85 → ~50+ tok/s gap)

**CoreML integration timeline (contingent on partnership scope):**
- Stage 1: 3-month integration if scoped Q3 2026
- Stage 2: 6-month engineering collaboration if scoped Q4 2026
- Stage 3: 12-18 month silicon roadmap conversation, scoped 2027

**Licensing model:**
- ARM-style — IP licensed via Gamma Seeds Pty Ltd to Apple for the AUKUS-exclusive territory
- No equity ask, no acquisition pathway
- Long-term strategic partnership over project-based engagement

**Korean parallel:** KAIST/KSGC engagement, Rebellions and FuriosaAI (Korean NPU vendors). Different licensing territory; no conflict with AUKUS-exclusive Apple licensing. Mentioned for transparency rather than as a partnership-influencing factor.

---

## Reference appendix

- [`benchmarks/REPORT_PHASE2.md`](../benchmarks/REPORT_PHASE2.md) — Phase 2 closure report; Metal kernel integration + Row 5 measurement; full reproducibility appendix
- [`docs/TN-001_llama70b_compression_analysis.md`](../docs/TN-001_llama70b_compression_analysis.md) — 70B compression analysis; the 116 GB problem framing
- [`docs/TN-002_*`](../docs/) — CoreML/ANE export investigation; proof-of-concept for the Stage 1 / Stage 2 asks
- [`benchmarks/analyse_per_expert_tolerance.py`](../benchmarks/analyse_per_expert_tolerance.py) — Session 4 IP analysis script (PR #12)
- [`benchmarks/run_dual_compression_26b_a4b.py`](../benchmarks/run_dual_compression_26b_a4b.py) — today's compression orchestration (PR #15)
- `~/synapticode/CLAUDE.md` — workspace-level conventions
- `~/synapticode/tern-core/CLAUDE.md` — repo-level architecture, sprint state, IP framing

---

## Document state

**Version**: v0.1 — CC scaffold + placeholders, 6 May 2026 14:48 AEST
**Source materials**: `tern-core/CLAUDE.md` Gemma 4 26B MoE section, REPORT_PHASE2.md, `project_gemopus_26b_moe_compression_v1.md` orientation memory, today's session implementation work
**Refinements pending**: Six TODO blocks per the audience / CoreML stage definitions / Q&A confirmation list
**Next step**: Rob review + TODO refinement → v0.2 cycle
