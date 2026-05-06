# Chain v3 CoT — design-space deliberation

Companion to:
- planning file — `.cursor/plans/chainv3_cot_branch_*.plan.md`
- literature review — [chainv3_cot_literature_review.md](chainv3_cot_literature_review.md)
- design proposal (forthcoming) — `chainv3_cot_design_proposal.md`
- ablation tracker (forthcoming) — `chainv3_cot_ablation_tracker.md`

This doc captures the design-space exploration that happens *between*
the literature review (what's been done) and the design proposal (what
we are committing to). It is intentionally a deliberation log, not a
spec — the design proposal will distill the locked choices once
hyperparameters and ablations are signed off.

---

## Status

- [x] Literature review (23 references; CoT-focused with positioning context)
- [x] Design-space enumeration (this doc)
- [x] Strategic axis 1 — *what is the intermediate mask?* — **locked: M2 (different-object landmark)**
- [x] Strategic axis 2 — *how is the intermediate mask supervised?* — **locked: W1-pure (no auxiliary mask-level reward; landmark mask shaped only by pretrained class-segmentation prior + architectural gradient via F1)**
- [x] Strategic axis 3 — *where does the mask feed back?* — **locked: F1 (encoder-memory append via `MaskPoolToken`)**
- [x] Tactical axis A — anti-collapse (PixelLM-style margin) — **locked: off in headline**, kept as ablation row if collapse appears
- [x] Tactical axis B — answer-template form (P1 generic / P2 regex-landmark / P3 step-style / **P4 natural first-person**) — **locked: P4**
- [x] Tactical axis C — question-side reasoning prefix — **locked: on at train+eval**
- [x] Tactical axis D — `cot_template_prob` strategy — **locked: regex-hit-gated** (extended by W7 / offline LLM if added)
- [x] Tactical axis E — rationale-text supervision shape — **locked: full LM loss**
- [x] Tactical axis F — training procedure under teacher forcing — **locked: two-pass (R2S-style); see "Training procedure" below**
- [x] Tactical axis G — gradient flow from `M_2` loss back to `M_1` — **locked: stop-gradient on `mask_pool_token`** (option 1b — the pretrained prior shapes `M_1`, no indirect grading via `M_2`'s loss)
- [x] Tactical axis H — LM-loss-pass placement — **locked: pass-2 only** (option 2a — accept the small train-eval mismatch on rationale tokens cross-attending to `mask_pool_token`)
- [x] Phased plan — three-tier CoT supervision strategy: **Tier 1 (regex)** = B1' headline; **Tier 2 (offline LLM task-adaptive rationale)** = planned phase 2 (B9); **Tier 3 (STaR self-generated CoT)** = deferred phase 3 (B11). See "Phased plan" section below.
- [x] Commonsense / human-intention (cs / hi) handling — **chain-v2 fallback in Tier 1** (these queries don't have landmark structure); LLM-rationale handles them naturally in Tier 2.
- [ ] Design proposal write-up (`chainv3_cot_design_proposal.md`)
- [ ] Ablation tracker (`chainv3_cot_ablation_tracker.md`)
- [ ] Implementation kickoff (Tier 1 first)

---

## Locked-in strategic decisions

### M2 — intermediate mask is a *different-object landmark*

Rationale: the chain v3 motivation is **multi-step relational reasoning**
(Surprise3D's `relative_position` and `narrative_perspective` failure
modes). For queries like *"the chair after I enter through the door"*,
the model needs to first locate the *door* (a different object) and
then the *chair*. Same-target coarse-then-fine refinement (M3) does
not address this failure mode; viewpoint embeddings (M4) are too novel
to bet the round on.

This rules out:

- M3 (same-target refinement, SAM-like). Doesn't address relational
  failure.
- M4 (viewpoint / situation token). Too speculative for round 1; revisit
  after we have B-row numbers.
- W3 (re-purpose Reason3D's `[LOC]` token). `[LOC]` was originally
  trained as a coarse *target* mask, not a coarse *anchor*; re-purposing
  it for a different-object landmark would require new GT anyway.
- W4/W5 (pseudo-anchor mined from the final-mask GT). No mechanical way
  to derive a *different-object* landmark from the target's own mask.

### W1-pure — no step-level mask supervision

Rationale: the user's framing is sharper than the original "weak /
consistency" formulation. The two claims:

1. **Architectural feedback is needed** — `[SEG]_2` *must* see `M_1`
   via `MaskPoolToken` (this is why F1 stays locked, not F3).
2. **Step-level mask supervision is not needed** — `M_1` should not
   be directly verified by any auxiliary loss. The pretrained
   chain-v2 ckpt has already learned `"the {class}" + [SEG] →
   {class}-mask` on its single-`[SEG]` regime; that prior is strong
   enough that an `[SEG]_1` produced after a class-name-bearing
   rationale (e.g. *"I need to find the door first."*) decodes a
   correct landmark mask *zero-shot from the prior*, without any
   loss term grading it.

Concretely, the only two losses in the headline are:

```
L = LM_loss(rationale + answer, pass-2 logits)  +  mask_loss(M_2, GT_2)
```

`M_1` has **no direct loss term**. What shapes it:

- **Pretrained class-segmentation prior** — the load-bearing assumption.
  `[SEG]_1`'s hidden state, conditioned on the class name appearing
  immediately before it in the LLM's emitted rationale, decodes to a
  recognizable class mask via the unchanged `mask_decoder` weights.
- **LM loss** on the rationale — teaches the LLM to emit the right
  class name in the rationale position, which is what gives `[SEG]_1`
  its lexical conditioning.
- **Stop-gradient on `mask_pool_token`** (axis G locked to 1b) —
  removes the indirect "grade `M_1` against whatever helps `M_2`'s
  loss" channel. No drift away from the class-mask prior.

This makes the contribution against R2S [8] (ICCV 2025) clean:

> *R2S requires both LLaMA-mined landmark text **and** mined
> intermediate-mask GT. We show that with the same architectural
> feedback (`MaskPoolToken`), the pretrained class-segmentation prior
> alone is sufficient to bootstrap the landmark mask — no mask GT
> mining needed, no auxiliary mask-level reward terms.*

This rules out (vs the original W1 with consistency rewards):

- `L_contain` (containment IoU between `M_1` and dilated `M_2`) — was
  the W1 headline reward; now an ablation row only (B6).
- Cosine reward on `(seg1_hidden, seg2_hidden)` — same status as above.
- W2 (LLaMA-3.1-mined anchor mask GT) — fallback only, if B1' stalls.
- W7 (LLM-mined landmark text only, no mask) — kept as ablation B9
  (see "Offline LLM text labeling" section below).
- All forms of process reward modeling (PRM, "Let's Verify Step by
  Step") — no step-level mask correctness annotations.

### F1 — encoder-output append (vs F2 text-placeholder, F3 no-feedback, F4 cross-attn patch)

Rationale: F1 (the choice from the original planning round) is
mathematically near-equivalent to R2S's F2, but **does not require a
text-template change** at training time, and **does not patch HuggingFace
T5's cross-attention** (F4 risk). The frozen FlanT5-XL keeps working
with one extra encoder-memory row gated by attention mask.

F1 is **not** dropped under the W1-pure pivot — the architectural
feedback is independent of the mask-level loss term. The user's claim
*"text alone is enough condition"* would imply F3, but the actually-locked
position is *"text + the mask-pool feature is the right condition"* —
i.e. the LLM has access to a 1-token summary of what the segmentor
decoded for the landmark when it produces the target tokens. F1 stays.

F3 (no architectural feedback, multi-`[SEG]` decoded independently like
LISA++ / PixelLM) is kept as a *baseline ablation row* (B7) — it tests
whether the architectural feedback matters at all, vs the
text-conditioning alone.

The training-time mechanics of F1 are non-trivial under teacher
forcing because `mask_pool_token` is derived from the model's own
`M_1` decoded mid-sequence. The two-pass forward that handles this
is documented in the "Training procedure" section below (axes F, G, H).

---

## Phased plan: regex → LLM rationale → STaR

The chain v3 CoT effort is deliberately staged. The dataset is
heterogeneous — the actual Surprise3D `surprise_val.json` distribution
is:

| Question type        | Count | %    | Example                                                                          | Has landmark structure? |
|----------------------|-------|------|----------------------------------------------------------------------------------|-------------------------|
| `hi` (human intent)  | 2904  | 28%  | *"I want to relax and read something, what can I use?"*                          | No — affordance         |
| `cs` (commonsense)   | 2898  | 28%  | *"What can be used to sit?"*                                                     | No — class/affordance   |
| `camera_view`        | 1931  | 19%  | *"At the perspective (-2.28, ...), the picture closest to me"*                   | Implicit (camera pose)  |
| `first_view`         | 1185  | 12%  | *"Upon entering with back to the door, the picture closest to me"*               | Yes — *"the door"*      |
| `relative_position`  | 730   | 7%   | *"The seating furniture closest to the door"*                                    | Yes — *"the door"*      |
| `abs` (abs distance) | 550   | 5%   | *"A chair 1.33 meters away from the door"*                                       | Yes — *"the door"*      |

Only ~24% of the dataset (`first_view` + `relative_position` + `abs`)
has the explicit landmark-relational structure that the regex-extraction
plan was designed for. The remaining ~76% — `cs`, `hi`, `camera_view` —
needs different treatment. A one-template-fits-all P4 plan is therefore
*not* the right framing for the whole dataset.

### Why phase rather than commit to one path

Three viable mechanisms exist for "natural CoT", with different cost
envelopes and different generalization stories:

| Tier | Mechanism | Cost | Coverage of dataset | Naturalness |
|------|-----------|------|---------------------|-------------|
| 1    | **Regex landmark extraction** + chain-v2 fallback | Zero extra compute | ~24% gets two-`[SEG]` CoT; ~76% trains as chain-v2 | Template-locked |
| 2    | **Offline LLM task-adaptive rationale** | ~few hours single-GPU LLM inference, one-time | All 100% of samples get a tailored rationale | Per-query-type structure (LLM picks `[SEG]` count) |
| 3    | **STaR self-generated CoT** | Multi-round inference + filter + retrain | All 100%, *self-emitted* | Maximally adaptive (T5 generates own shape) |

We commit to **Tier 1 first**, ship B1', then escalate to Tier 2 only if
B1' is bottlenecked on the ~76% non-landmark majority. Tier 3 is
reserved for a follow-up.

### Tier 1 — Regex landmark CoT (B1' headline)

- **Targets**: `relative_position`, `abs`, `first_view` queries that
  contain a regex-detectable landmark (door, entrance, table, etc.).
  Expected non-fallback rate inside this subset: ~70-80%; across the
  whole dataset: ~15-20%.
- **For all other samples (~80-85%)**: the dataset emits the chain-v2
  single-`[SEG]` template (`"The target object is {name}. [SEG]."`) —
  identical to today's chain-v2 training. The model trains on a *mixed
  diet* of mostly chain-v2 single-`[SEG]` plus a minority of two-`[SEG]`
  P4 chains.
- **Implementation cost**: small. New regex utility in the dataset, new
  `MaskPoolToken` projection module on the model, modified two-pass
  forward when there are ≥ 2 `[SEG]` tokens in the target (one-pass
  otherwise).
- **What it tests**: whether even *partial* CoT signal (on a minority of
  samples) trains the model to use `MaskPoolToken` correctly on the
  relational queries that need it, while leaving the rest of the
  dataset undisturbed.

### Tier 2 — Offline LLM task-adaptive rationale (B9, planned phase 2)

- **Mechanism**: one-time offline pass with an LLM
  (Qwen2.5-7B / Llama-3.1-8B / GPT-4-class for quality) that, per
  sample, generates a *task-appropriate* rationale ending in `[SEG]`,
  with optional intermediate `[SEG]` tokens at points where the LLM
  would commit to a region. The LLM is given the question text and
  optionally the GT object class name; it decides the rationale shape.
- **Per-question-type expected output** (from a single LLM prompt
  template; no manual per-type rules):

  | Type                | Expected rationale shape                                           |
  |---------------------|---------------------------------------------------------------------|
  | `cs` / `hi`         | *"Sitting requires furniture designed for the body. [SEG]."*        |
  | `relative_position` | *"I need to find the door first. [SEG]. Then the closest seat. [SEG]."* |
  | `abs`               | *"I locate the door first. [SEG]. Then the chair within ~1.3 m. [SEG]."* |
  | `first_view`        | *"From the door looking inward, the picture closest to me. [SEG]."* |
  | `camera_view`       | *"Given the camera pose, the closest in-view picture. [SEG]."*      |
- **Implementation cost**: medium. New offline mining script (~1-2 days
  engineering), new annotation field `rationale: str` per sample,
  dataset modification to read `rationale` instead of building from
  templates.
- **What it tests**: whether full-coverage rationale supervision
  (matched to query-type structure) closes any gap left by Tier 1's
  regex coverage. Specifically, B9 vs B1' on the `cs` / `hi` /
  `camera_view` slices is the comparison of interest — does CoT-shaped
  text help on non-relational queries, or is chain-v2 already optimal
  there?
- **Contribution narrative shift**: from *"regex + pretrained prior is
  enough"* (B1') to *"LLM-mined rationale text + pretrained class prior
  is enough; mask-GT mining unnecessary"* (B9). The R2S delta
  (no mined intermediate-mask GT) is preserved either way.

### Tier 3 — STaR self-generated CoT (B11, deferred)

- **Mechanism**: starting from B1' or B9 ckpt, sample N completions per
  training sample with a CoT-trigger prompt; filter chains whose final
  mask IoU ≥ τ vs GT; fine-tune on the kept chains. Repeat for K rounds.
- **Cost**: each round = 1 inference pass over the whole training set
  + 1 fine-tune. K = 2-3. Expensive but well-precedented (STaR [17],
  ReST, RFT).
- **What it tests**: whether T5 *self-emits* better rationales than the
  external LLM gives it (Tier 2). Plausible because T5 has internalized
  the segmentor's class prior; an external LLM has not.
- **Status**: deferred to a follow-up paper / v2; documented for
  completeness only.

### Commonsense / human-intention (cs / hi) handling

`cs` and `hi` together are 56% of the dataset. They have no spatial
landmark structure — they are class/affordance queries (*"What can be
used to sit?"*). The plan per tier:

| Tier | cs / hi treatment                                                    | Rationale                                                                 |
|------|-----------------------------------------------------------------------|---------------------------------------------------------------------------|
| 1    | Chain-v2 single-`[SEG]` template (`"The target object is {name}. [SEG]."`) | These queries don't decompose into spatial steps. Chain-v2's category-legible output is the right shape. |
| 2    | LLM-generated rationale; typically single `[SEG]` with affordance-reasoning prefix (e.g. *"Sitting requires furniture designed for sitting. [SEG]."*) | The LLM produces affordance-style reasoning that may or may not improve over plain chain-v2. B9 vs B1' on the cs/hi slice answers this. |
| 3    | T5 self-emits; STaR rejection sampling keeps the rationale shapes that yield correct masks. | If the model has converged on a good cs/hi rationale shape from Tier 2, STaR refines it; if not, Tier 1 chain-v2 templates are used as the initial prompt structure. |

Critically: **CoT is not assumed to help cs / hi.** The B1' headline
intentionally trains them as chain-v2; a non-zero gap on cs / hi
between B0 (chain-v2) and B1' would actually be a *regression*, not a
win. If B1' regresses on cs / hi, we know the multi-`[SEG]` training
diet is destabilizing the chain-v2 prior on non-relational queries —
mitigation is to lower the proportion of two-`[SEG]` samples in the
training mix, or to escalate to Tier 2 where every sample has a
tailored rationale.

---

## Strategic axes — what was decided and why

The full enumeration is reproduced here so the design proposal can
reference it without re-deriving the space.

### Axis 1 — what *is* the intermediate mask?

| ID | Mask role | Status |
|----|-----------|--------|
| M1 | Coarse anchor / context, *larger* than the target | Live, but folded into M2 (a landmark *is* a coarse anchor by another name) |
| M2 | Different-object landmark | **Locked** |
| M3 | Coarse version of the same target (SAM-like) | Dead; doesn't address relational failure |
| M4 | Viewpoint / observer-frame token | Dead for now; v2 candidate for `absolute_distance` queries |

### Axis 2 — how is the intermediate mask supervised?

| ID | Supervision | Status |
|----|-------------|--------|
| W1 | Weak / consistency-only (containment + cosine similarity rewards) | Demoted to ablation (B6); was the originally locked headline |
| **W1-pure** | **No auxiliary mask-level reward; only LM loss + final mask GT.** Pretrained class-segmentation prior + architectural F1 gradient (or stop-gradient, axis G) shape `M_1`. | **Locked** |
| W2 | LLaMA-3.1-mined intermediate-mask GT (R2S-style) | Reserved as fallback if B1' stalls |
| W3 | Reason3D `[LOC]` re-purposed | Dead given M2 |
| W4 | Self-supervised pseudo-anchor from final-mask GT (dilation / NN) | Dead given M2 |
| W5 | Hybrid pseudo-anchor + consistency | Dead given M2 |
| W7 | LLM-mined **text-only** landmark / target phrase (no mask GT) | Open ablation (B9); see "Offline LLM text labeling" section |

### Axis 3 — where does the mask feed back?

| ID | Mechanism | Status |
|----|-----------|--------|
| F1 | Append `MaskPoolToken` to encoder memory | **Locked** |
| F2 | Inject `MaskPoolToken` into instruction text as placeholder (R2S-exact) | Dead for headline; possible if F1 implementation hits unforeseen issues |
| F3 | No architectural feedback; consistency rewards link the two `[SEG]`s | Ablation row only |
| F4 | Patch T5 cross-attention with extra K/V | Dead; engineering risk |

### Tactical axis A — anti-collapse term

PixelLM [3] uses a cosine *push-apart* loss on token embeddings to
prevent two distinct `[SEG]` outputs collapsing to the same mask.

Locked: **off in the headline (B1')**. Reasoning:

- The pretrained class-segmentation prior is what gives `M_1` its
  shape; the LLM emits *"the door"* before `[SEG]_1` and *"the chair"*
  before `[SEG]_2` (under P4), so the lexical context already
  differentiates the two `[SEG]` hidden states.
- Stop-gradient on `mask_pool_token` (axis G = 1b) means `M_2`'s loss
  cannot pull `M_1` toward `M_2`'s shape via the F1 bridge, removing
  the dominant collapse pathway.
- Adding the cosine push-apart in B1' would *also* be the only
  auxiliary loss term operating on `[SEG]` hidden states; that
  contradicts the W1-pure stance (no step-level mask supervision)
  even though the cosine is on token embeddings rather than masks.

If `B4`-style ablations show collapse (intermediate IoU ≈ final IoU),
the cosine term is added back. Cheap to implement, well-cited.

### Tactical axis F — training procedure under teacher forcing

Chain-v2's training is one T5 forward pass with the encoder memory
fixed for the whole pass (see [reason3d_t5.py:200-216](../lavis/models/reason3d_models/reason3d_t5.py)).
Under F1, `[SEG]_2`'s computation depends on `mask_pool_token`, which
depends on `M_1`, which depends on `[SEG]_1`'s hidden state — a
circular dependency that cannot be resolved in a single teacher-forced
pass.

R2S [8] §3.2 resolves this with a **two-pass** forward; we adopt the
same:

```
Pass 1  (encoder memory = [Qformer | text]):
        ├── full T5 forward, all decoder positions in parallel
        ├── extract seg1_hidden at first [SEG] position
        └── M_1 = mask_decoder(text_hidden_fcs(seg1_hidden))
                ⤷ mask_pool_token = Linear( MassPool(M_1[.detach()], sp_feats) )

Pass 2  (encoder memory = [Qformer | text | mask_pool_token]):
        ├── full T5 forward (encoder rerun cheap; cross-attn re-projects K/V)
        ├── extract seg2_hidden at second [SEG] position
        ├── M_2 = mask_decoder(text_hidden_fcs(seg2_hidden))
        └── LM loss on pass-2 logits

Losses (B1' headline):
    L = LM_loss_pass2(targets) + mask_loss(M_2, GT_2)
```

Compute cost: ~2× T5 forwards. Encoder activations on the 32 Qformer
+ ~25-tok text inputs are tiny relative to the point encoder, so
end-to-end the slowdown is roughly 1.4-1.6× per training step vs
chain-v2.

Inference is one fewer pass — the model generates autoregressively, so
when `[SEG]_1` is emitted we pause, decode `M_1`, compute
`mask_pool_token`, append it to encoder memory, and resume generation.
The decoder self-attention KV cache is preserved (the augmentation is
encoder-side); cross-attention re-projects encoder tokens lazily, so
the new column slots in cleanly. This is one of F1's wins over F2.

### Tactical axis G — gradient flow from `M_2` loss back to `M_1`

Two implementation paths under axis F's two-pass scheme:

- **1a. Connected gradient.** `mask_pool_token = Linear(MassPool(M_1, sp_feats))`
  with no detach. `L_M2` flows back through `seg2_hidden →
  cross_attn(mask_pool_token) → MassPool → M_1 → seg1_hidden`. `M_1`
  receives an indirect gradient: *"adjust to maximize `M_2`'s
  performance"*.
- **1b. Stop-gradient.** `M_1.detach()` before the mass-pool. `M_1`
  receives **no gradient** from this round of training; its shape is
  determined entirely by the pretrained class-segmentation prior +
  the LM loss shaping `seg1_hidden` via the rationale text.

Locked: **1b (stop-gradient)** in the B1' headline. This is the
strictest expression of the W1-pure stance — *"don't directly verify
the landmark, trust the pretrained prior"*. It also removes the
drift channel where `M_1` could be pulled away from being a
recognizable class mask if some other shape happened to give better
mass-pool features for `M_2`.

`B6+` ablation row tests 1a (gradient on) — directly answers whether
indirect supervision helps or just regularizes.

### Tactical axis H — LM-loss-pass placement

In pass 2, every decoder position cross-attends to `mask_pool_token`,
including the rationale tokens *before* `[SEG]_1` — which at inference
are generated before `mask_pool_token` exists. Three options:

- **2a. LM loss on pass 2 only**, all positions. Mild train-eval
  mismatch on rationale tokens. Simplest. R2S accepts this.
- **2b. Split LM loss**: pass-1 loss on rationale-up-to-`[SEG]_1`,
  pass-2 loss on tokens after `[SEG]_1`. Fully matches inference but
  doubles bookkeeping.
- **2c. LM loss on pass 1 only**. The LLM never learns to use the
  mass-pooled landmark feature for emitting target text — defeats
  half of F1's purpose.

Locked: **2a**. If rationale-text quality at inference visibly degrades
(LLM emits weird rationales when `mask_pool_token` is absent), fall
back to 2b.

### Tactical axis B — `cot_template_prob` strategy

Two options:

- **Fixed probability** (initial plan: 0.5). Simple; treats the regex
  hit as a quality bonus only.
- **Regex-hit-gated** — `cot_template_prob = 1.0` for samples where
  the regex extracts a landmark, `0.0` otherwise. Cleaner: every two-step
  template carries a real landmark; one-step samples are the regex
  failures. Maximizes lexical signal for the LLM.

Proposed: **regex-hit-gated**. The fallback "Looking around the scene."
template adds noise without information; better to drop those samples
back to the single-`[SEG]` chain v2 template entirely.

### Tactical axis C — question-side reasoning prefix

Optional prepend to `text_input`:

```
Reason step by step. First identify the relevant landmark in the scene,
then locate the target. Question: {original_question}
```

Cost: ~15 extra encoder tokens. Used at both train and eval (training-time
inclusion teaches the LLM what the prefix means; inference-only use
breaks the train-eval covariate match). Tested vs off in `B3`.

### Tactical axis D — rationale-text LM-loss shape

Three options for the tokens between question-end and `[SEG]_1` (the
"rationale" / landmark text):

- **Full LM loss** — standard CE on every token. The LLM learns
  *exactly* the regex-extracted landmark name.
- **Masked-out** — `labels[rationale_tokens] = -100`. The rationale is
  lexically free; only `[SEG]_1` and the consistency rewards shape it.
- **Weighted** — soft middle (e.g. 0.3-0.5×).

Proposed: **full LM loss**. The LLM needs lexical signal to learn what
landmark text looks like; the regex output is usually defensible on
Surprise3D's structured queries. Masked-out is interesting but
introduces a new training pathology (LLM emits gibberish rationale)
that we cannot debug from the consistency rewards alone.

---

## Direction status (post-locks)

The five coherent directions:

| Direction | Composition | Status |
|-----------|-------------|--------|
| **A — Pretrained-Prior CoT (B1')** | M2 + W1-pure + F1 + P4 + two-pass training (G=1b, H=2a) | **Live (this is the round)** |
| A' — R2S-Lite-Weak (original headline) | M2 + W1 (containment + cosine rewards) + F1 | Demoted to ablation B6 |
| B — R2S-on-Surprise3D | M2 + W2 + F2 | Reserved as fallback / strong-baseline run if A stalls |
| C — Hierarchical Reason3D revival | M2 + W3 + F1 | Dead given W3 ruled out |
| D — Iterative same-target refinement | M3 + W1 + F1 | Dead given M3 ruled out |
| E — LLM-text-augmented CoT (W7) | M2 + W7 + F1 + two-pass | Open ablation B9 (text-only LLM mining; no mask GT) |

---

## How chain-of-thought is trained in the literature (filtered to W1-pure)

Five families, with applicability to our W1-pure constraint (no
auxiliary mask-level rewards; LM loss + final mask GT only):

| Family | Mechanism | Compatible with W1-pure? |
|--------|-----------|---------------------|
| **C1 — Supervised rationale finetuning** | Train on `(question, rationale, answer)` triples; rationale is GT text. Flan-T5 instruction tuning, multi-modal CoT (Zhang 2023, AAAI 2024), R2S [8]. | **Partially** — we *do* supervise the rationale text via LM loss (axis E = full LM loss). The rationale "GT" is the regex-extracted landmark stuffed into the P4 template, not human-mined. So we're a degenerate C1: rationale GT comes from a regex (or in W7, an offline LLM), not from human annotation. |
| **C2 — Process reward models (PRMs)** | Train a separate model to score per-step correctness; use it via PPO. *Let's Verify Step by Step* (Lightman et al. 2023). | **No** — needs step-level labels |
| **C3 — STaR-style self-improvement** | Sample many CoT chains, filter those whose *final* answer is correct, fine-tune on kept chains. Repeat. STaR [17], ReST, RFT. | Yes (but expensive — multi-round, requires sampling). Reserved as Stage C. |
| **C4 — Process-free consistency rewards** | Use intra-sample agreement between intermediate and final outputs as the training signal. | Was the original W1 mechanism; now demoted to **B6 ablation** under the W1-pure pivot. |
| **C5 — Prompt engineering** | Bias the LLM toward step-by-step output via the prompt format itself. Zero-shot CoT trigger phrases (Kojima 2022); structured rationale templates; multi-turn chat. | **Yes — primary lever for our round** (P4 template + question-side prefix). |

Within C5, the literature converges on four patterns of increasing
reasoning-bias:

1. **Trigger phrases** — "Let's think step by step." prepended to the
   question. Kojima 2022; +10-30 GSM8K pp on PaLM. Cheap.
2. **Role priming** — "You are a 3D scene reasoning assistant. First
   identify the landmark, then locate the target." Guides the *type*
   of reasoning.
3. **Structured rationale template** — the *answer* template lexically
   requires step structure: e.g. `"Step 1: <landmark>. [SEG]. Step 2:
   <target>. [SEG]."`. Forces the LLM to emit reasoning structure.
4. **Multi-turn chat structure** — split rationale and answer across
   two assistant turns (R2S [8]'s mechanism). Strongest bias; most
   engineering for our existing chain templates.

We use a combination of (1) trigger phrase via the question-side prefix
(axis C above) and (3) structured rationale template via the answer
template options below.

---

## Answer-template menu (axis B detail)

### P1 — Generic anchor (initial plan)

```
Looking around the scene. [SEG]. The answer is the {target}. [SEG].
```

- Intermediate `[SEG]` has no lexical anchor; the LLM emits a fixed
  placeholder.
- Under the original W1, all landmark-shaping work fell on consistency
  rewards. Under W1-pure, P1 has *no* signal shaping `M_1` (no class
  name preceding `[SEG]_1` to invoke the pretrained prior, no rewards).
  → **incompatible with W1-pure**; only meaningful as a row when
  paired with W1's auxiliary rewards. Dead in this round.

### P2 — Regex-extracted landmark (originally proposed; demoted to B2 ablation)

```
First, the {landmark}. [SEG]. The answer is the {target}. [SEG].
```

`{landmark}` is regex-extracted from the question. Pattern set
targeting Surprise3D's six observed query types (see "Phased plan"
above for the full distribution):

- `near X`, `next to X`, `beside X`, `closest to X`, `farthest from X` → X
- `behind X`, `in front of X`, `to the (left|right) of X` → X
- `after I enter (through|from) X`, *"upon entering with back to X"*,
  `from the X (side|entrance)` → X
- `\d+ ?(m|meters?) (to the (left|right)|in front|behind|away from) X` → X
- `from your position`, `from the entrance` → *"entrance"* (viewpoint cue)
- Fallback (no regex hit): drop to chain-v2 single-`[SEG]` template
  (regex-hit-gated `cot_template_prob = 0` for that sample).

Empirical coverage on `surprise_val.json` (10198 rows; rough estimate
based on the question-type breakdown above):

| Question type        | Count | Regex-applicable? | Expected non-fallback rate |
|----------------------|-------|-------------------|----------------------------|
| `cs`                 | 2898  | No                | 0% (always falls back)     |
| `hi`                 | 2904  | No                | 0% (always falls back)     |
| `camera_view`        | 1931  | No (numeric pose) | 0% (always falls back)     |
| `first_view`         | 1185  | Yes               | ~70-80%                    |
| `relative_position`  | 730   | Yes               | ~80-90%                    |
| `abs`                | 550   | Yes               | ~80-90%                    |

Overall ~15-20% of training samples carry a meaningful landmark under
P4; the remaining ~80-85% train as the single-`[SEG]` chain-v2 baseline.

This is **not enough samples to fully bootstrap the F1 architectural
feedback** on its own — but it is enough to test whether the
mechanism works on the relational subset without disturbing the
non-relational majority. If the relational-slice numbers improve and
the non-relational slice doesn't regress, Tier 1 is a clean win on
its own merits; Tier 2 is then the path to extending CoT-style
supervision to the rest.

### P3 — Step-style template (`B5` ablation)

```
Step 1: I look for the {landmark}. [SEG]. Step 2: The {target} is {relation} the {landmark}. [SEG].
```

`{relation}` is also regex-extracted ("near", "behind", "to the left
of"). More explicit CoT-shape. Risk: surface-form overfitting; the LLM
emits "Step 1 / Step 2" even on simple queries, hurting backward-compat
on chain v2 queries.

### P4 — Natural first-person template (**locked headline**)

```
I need to find the {landmark} first. [SEG]. Then {target_phrase}. [SEG].
```

Where `{target_phrase}` is composed from the question + landmark, e.g.
*"the chair to the right of the door"* or *"the desk near the door"*.

Why P4 is the headline (vs P2):

- **Natural-language phrasing.** Reads like text the frozen FlanT5-XL
  would emit naturally — less train-eval format mismatch than P2's
  stilted *"First, the door."*.
- **Class-name lexical anchor before each `[SEG]`.** P4 places
  *"the {landmark}"* immediately before `[SEG]_1` and *"the {target}"*
  inside `{target_phrase}` immediately before `[SEG]_2`. Both `[SEG]`s
  thus inherit the class-segmentation prior the chain-v2 ckpt has
  learned, which is the load-bearing assumption of W1-pure.
- **Backward-compatible with chain-v2.** When the regex misses
  (no landmark detectable), `cot_template_prob` is gated to zero and
  the sample drops to a chain-v2 single-`[SEG]` template — without
  the LLM having ever been forced to emit *"Step 1 / Step 2"*-style
  surface forms it would now have to suppress.
- **First-person reasoning marker.** The *"I need to ..."* prefix
  signals reasoning intent without requiring the heavier *"Step 1"*
  scaffold.

Concrete example:

> Q: *"the item to your right when you enter through the door"*
> A: *"I need to find the door first. [SEG]. Then the item to the right of the door. [SEG]."*

`B2 = cot_p2_stilted` tests P4 vs P2; `B5 = cot_p3_steps` tests P4 vs P3.

### Question-side reasoning prefix

Independent of P1/P2/P3. Prepended to `text_input`:

```
Reason step by step. First identify the relevant landmark in the scene,
then locate the target. Question: {original_question}
```

Knob: `cot_prompt_prefix: bool = true`. Tested in `B3`.

---

## Proposed headline recipe (B1')

All locked per the Status block above; reproduced here for the
implementation hand-off.

- **Architecture**: F1 — encoder-memory append + `MaskPoolToken`
  (`Reason3DT5ChainV3CoT`); two-pass forward at training (axis F).
- **Answer template**: P4 — *"I need to find the {landmark} first.
  [SEG]. Then {target_phrase}. [SEG]."* with chain-v2 fallback for
  regex misses.
- **Question prefix**: on at train and eval (*"Reason step by step.
  First identify the relevant landmark in the scene, then locate the
  target. Question: ..."*).
- **`cot_template_prob`**: regex-hit-gated (1.0 when landmark
  extracted, 0.0 otherwise → chain-v2 single-`[SEG]` template).
- **Rationale-text supervision**: full LM loss on pass-2 logits
  (axes E + H).
- **Mask-level supervision on `M_1`**: **none** (W1-pure). No
  `L_contain`, no cosine reward. `M_1` shaped by pretrained
  class-segmentation prior + LM loss on rationale text only.
- **Gradient flow on `mask_pool_token`**: stop-gradient (axis G = 1b).
- **Anti-collapse (PixelLM cosine)**: off (axis A locked off).
- **Curriculum**: flat (init from loss-only chain-v2 ckpt; constant
  `lambda_cot = 0` since there are no auxiliary CoT-loss terms in B1';
  the *"`lambda_cot`"* knob is reserved for `B6` only).

The B1' total loss is exactly:

```
L = LM_loss_pass2(targets) + mask_loss(M_2, GT_2)
```

— same loss shape as chain-v2, with the only training-time additions
being (i) the second `[SEG]` index extraction, (ii) the `MaskPoolToken`
projection module, and (iii) the second teacher-forced T5 forward pass
with augmented encoder memory.

## Proposed ablation rows

| #     | Name                  | What changes vs B1'                                                                  | Tests                                                                                  |
|-------|-----------------------|--------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| B0    | `cot_off`             | Single-`[SEG]` chain-v2 (loss-only)                                                  | Sanity floor; should reproduce chain-v2 numbers                                         |
| B1'   | `cot_pure` (headline) | —                                                                                    | All claims together (W1-pure + F1 + P4 + two-pass + stop-grad + pass-2 LM loss)         |
| B2    | `cot_p2_stilted`      | P2 (`"First, the {landmark}. [SEG]."`) instead of P4                                 | Whether the natural-language P4 phrasing matters                                        |
| B3    | `cot_no_prefix`       | Question-side reasoning prefix off                                                   | Whether the Kojima-2022 zero-shot trigger helps                                         |
| B5    | `cot_p3_steps`        | P3 (`"Step 1 / Step 2"`) instead of P4                                               | Whether more verbose CoT structure helps or hurts backward-compat                       |
| B6    | `cot_with_rewards`    | Add containment (`L_contain`) + cosine similarity rewards                            | Whether explicit consistency rewards add anything beyond architectural F1               |
| B6+   | `cot_grad_on_M1`      | Axis G flipped to 1a (no stop-gradient on `mask_pool_token`)                         | Whether the indirect "grade `M_1` via `M_2`'s loss" channel helps or drifts             |
| B7    | `cot_no_arch_feedback`| F3 instead of F1 (no `MaskPoolToken`, no two-pass; just two `[SEG]`s in one forward) | Whether the architectural feedback matters at all                                       |
| B7+   | `cot_full_old_headline` | F1 + W1 rewards + anti-collapse + grad-on (the *previously proposed* B1)           | Documents what the old W1 headline scored; serves as upper-bound on auxiliary losses    |
| B8    | `cot_p4_fixed_prob`   | `cot_template_prob = 0.5` instead of regex-hit-gated                                 | Whether gating helps vs adds complexity                                                 |
| B9    | `cot_llm_rationale` (Tier 2 / W7) | Replace per-sample answer construction with offline-LLM-generated task-adaptive rationale (variable `[SEG]` count) | Whether full-coverage LLM rationale supervision closes the gap on the ~80% non-landmark majority (cs / hi / camera_view) |
| B11   | `cot_star` (Tier 3, deferred) | Self-generated CoT via inference sampling + final-mask-IoU filter + fine-tune (K=2-3 rounds), starting from B1' or B9 ckpt | Whether T5's *self-emitted* rationale outperforms external LLM mining on relational + non-relational subsets |

Key comparisons:

- **B1' vs B6** — directly tests the W1 → W1-pure pivot. If B1' ≥ B6,
  the auxiliary rewards are unnecessary; the contribution is the
  pretrained-prior story.
- **B1' vs B7** — tests F1 vs F3. The user's claim is F1 is needed
  (text alone is *not* enough condition). Should show F1 wins.
- **B1' vs B7+** — direct comparison against the *previous* proposed
  headline (W1 with both rewards and gradient on). Tests whether the
  whole "auxiliary supervision" stack is redundant.
- **B1' vs B9** — tests whether the ~80% non-landmark coverage is the
  bottleneck. **Required to slice by question type**: `cs` / `hi` vs
  relational. If B9 wins on cs / hi but ties on relational: Tier 2 is
  the natural extension. If B9 ties everywhere: Tier 1 is sufficient.
- **B9 vs B11** (deferred) — tests whether self-generated CoT (STaR)
  beats external-LLM-mined CoT.
- **Per-question-type slicing for *all* runs** — given the
  heterogeneous dataset, aggregate metrics hide regression on
  cs / hi when CoT helps relational. The
  [`chainv3_ablation_tracker.md`](chainv3_ablation_tracker.md)
  Table-C breakdown is mandatory, not optional.

`B0` and `B1'` are required runs (Tier 1 ships first); `B6`, `B7`,
`B7+`, `B9` are the high-value ablations; `B2 / B3 / B5 / B6+ / B8`
are tactical tuning; `B11` is deferred to phase 3.

---

## Tier 2 — Offline LLM task-adaptive rationale (B9, planned phase 2)

**Phase 2 of the phased plan: replace regex landmark extraction with an
offline LLM that generates a task-appropriate rationale per sample.**
Status: planned, not implemented yet; B9 is the ablation row that
records this.

### Why this is needed beyond Tier 1

Per the empirical breakdown above, ~80-85% of Surprise3D training
samples don't have a regex-detectable landmark. Under Tier 1 (B1'),
those samples train as chain-v2 single-`[SEG]` — undisturbed but also
not benefiting from CoT. The question Tier 2 answers is: *can a CoT
rationale (of variable shape) help on the non-relational majority?*

### How text decoding currently works

The chain-v2 / chain-v3 datasets construct the LM target from a
template + the GT `object_name` field (see [threedrefer_datasets_chain.py](../lavis/datasets/datasets/threedrefer_datasets_chain.py)):

- `description` → goes into the question template (`text_input`):
  *"Please segment the object according to the given 3D scene and the
  description: {description}."*
- `object_name` → fills a randomly-chosen `CHAIN_ANSWER_LIST` template
  for the answer:
  *"The target object is {name}. [SEG]."* — single `[SEG]`, target
  class only.

There is **no landmark text and no rationale text in the answer
today.** The *landmark* (when one exists) is mentioned only inside the
natural-language `description` on the question side. Tier 1 (regex)
extracts a landmark string from the description and drops it into a
two-`[SEG]` template; Tier 2 (LLM) replaces this entire pathway with
an LLM-generated `rationale` field per sample.

### W7 — supervision axis spec

W7 is a *third* point on the supervision axis between W1-pure and W2:

| What's mined | W1-pure (B1' / Tier 1) | **W7 (B9 / Tier 2)** | W2 (R2S [8]) |
|---|---|---|---|
| Rationale text | template-filled with regex landmark (only when applicable) | **offline LLM, per-sample, per-query-type adaptive** | LLM (LLaMA-3.1) for landmark text only |
| Number of `[SEG]`s in answer | 2 if regex hits, 1 otherwise | **LLM decides per sample** (typically 1 for cs/hi/camera_view, 2 for relational) | 2 (fixed) |
| Landmark `object_id` mapping | n/a | optional (LLM filters to in-scene classes) | yes (used to derive mask) |
| Landmark mask GT | **no** | **no** | yes |
| Used in training as | LM target text | LM target text | LM target text + mask GT for landmark |

The key distinction: **W7 mines text labels only; no mask GT.** The
mask-supervision side stays exactly W1-pure. The contribution against
R2S becomes:

> *R2S requires LLM-mined landmark text **and** mined intermediate-mask
> GT. We show that LLM-mined rationale text alone (without mask GT) is
> sufficient when combined with the pretrained class-segmentation
> prior and architectural feedback.*

### LLM prompt sketch (Tier 2 mining)

A single prompt template, per sample:

```
You are generating reasoning chains for a 3D segmentation model.
Given a question about a 3D scene, write a short reasoning chain
that ends in [SEG]. Use additional [SEG] tokens at points where you
would commit to an intermediate spatial region. Keep it natural and
concise; not every query needs intermediate steps. The model already
knows how to segment named object classes when [SEG] is preceded by
a class name in the text.

Question: {description}
GT target class: {object_name}
Available classes in scene (optional): {scene_classes}

Reasoning chain:
```

Expected outputs by question type:

| Type                | Expected rationale shape                                                          | `[SEG]` count |
|---------------------|------------------------------------------------------------------------------------|---------------|
| `cs`                | *"Sitting suggests furniture like a chair or sofa. The target is a chair. [SEG]."* | 1             |
| `hi`                | *"Reading and relaxing suggests a comfortable chair or sofa. [SEG]."*              | 1             |
| `camera_view`       | *"From the given camera pose, the closest in-view picture is the target. [SEG]."*  | 1             |
| `first_view`        | *"Standing at the door, looking inward, the picture closest to me. [SEG]."*        | 1 (or 2 if landmark named) |
| `relative_position` | *"I locate the door first. [SEG]. Then the seating closest to it. [SEG]."*         | 2             |
| `abs`               | *"I find the door. [SEG]. Then the chair within ~1.3 m of it. [SEG]."*             | 2             |

The LLM is given freedom to vary the rationale across samples within a
type — diversity in surface form is desirable so T5 doesn't overfit to
one phrasing.

### Implementation plan (Tier 2)

1. **Mining script** — new file `scripts/mine_cot_rationale.py`:
   - Read annotation JSON.
   - Per sample, call LLM with the prompt above.
   - Optionally filter / sanity-check outputs (e.g. ensure GT
     `object_name` appears somewhere in the rationale).
   - Write a new annotation JSON with an additional `rationale` field
     per row.
2. **Dataset modification** — extend `ThreeDReferDatasetChain` (or a
   v3 sibling) to use `rationale` from the annotation JSON when
   present; fall back to Tier 1 regex-or-chain-v2 when absent. Knob:
   `use_llm_rationale: bool`.
3. **Forward path** — unchanged. The two-pass logic already activates
   per-sample based on `[SEG]` count in `targets` (1 → one pass,
   2+ → two-pass). LLM-generated single-`[SEG]` rationales fall through
   to the chain-v2 path; multi-`[SEG]` rationales trigger F1.
4. **Eval logging** — new fields in `predictions.jsonl`:
   `rationale_text` (LLM-emitted at training; T5-emitted at eval),
   `seg_count`, `regex_hit` (if Tier 1 fallback engaged).

### Cost estimate

A one-time offline pass with Qwen2.5-7B / Llama-3.1-8B over the
~10k-30k-sample Surprise3D training set is roughly a few hours on a
single A100 — ~free relative to a training run. GPT-4-class API calls
would be more accurate but cost ~$50-200 per pass; doable but worth
trying open-weights first.

### Risks specific to Tier 2

- **LLM hallucinates landmarks not in scene.** Mitigation: pass the
  scene's ScanNet class set as `scene_classes` in the prompt; filter
  outputs to in-scene classes post-hoc.
- **Train-eval mismatch on rationale style.** T5 emits rationales it
  was trained on (LLM-style); but a future eval-time CoT trigger might
  expect different phrasing. Mitigation: keep the LLM prompt's
  rationale style consistent with what we expect T5 to emit at eval.
- **Contribution narrative shift.** Tier 1 (B1') claims *"regex +
  pretrained prior is enough"*; Tier 2 (B9) claims *"LLM-text +
  pretrained prior is enough; no mask-GT mining"*. The latter is
  closer to R2S; the R2S delta (no mask GT mining) is preserved but
  the *"no LLM mining at all"* purity is lost.
- **`cs` / `hi` regression risk.** If the LLM-generated rationale for
  cs / hi is *worse* than chain-v2's plain `"The target object is
  {name}. [SEG]."` (e.g. the LLM emits noisy affordance prose that
  confuses the segmentor), B9 could regress on cs / hi. **Sanity
  check**: per-question-type breakdown of B9 vs B1' is required, not
  just aggregate.

### Decision

Tier 2 is **planned for phase 2**, not implemented in this round.
Resolution: run B1' first, observe per-question-type breakdown.

- If `cs` / `hi` / `camera_view` performance is unchanged at B1' (no
  regression vs B0) and the relational subset improves: B1' is a clean
  win on the relational subset alone.
- If the relational subset doesn't improve at B1': either the
  pretrained-prior assumption is wrong (uncertainty #1) or the
  ~15-20% coverage isn't enough to bootstrap F1. In the latter case,
  promote Tier 2 (B9) to the headline.
- If `cs` / `hi` regress at B1': the multi-`[SEG]` training diet is
  destabilizing the chain-v2 prior on non-relational queries — fix
  via lower CoT mix proportion *or* escalate to Tier 2 so every sample
  has a tailored rationale.

---

## Tier 3 — STaR self-generated CoT (B11, deferred)

**Phase 3 of the phased plan: T5 itself generates rationales via
inference-time sampling, filtered by final-mask correctness, then
fine-tuned on the kept rationales.** Status: deferred to a follow-up.

### Mechanism (STaR [17] applied to chain v3 CoT)

Starting from a B1' or B9 ckpt:

1. For each training sample, run T5 inference with a CoT-trigger
   prompt; sample `N` completions (typically 5-20) with
   temperature > 0.
2. Decode the final mask from each completion's last `[SEG]`. Compute
   IoU vs GT.
3. Keep completions whose final-mask IoU ≥ τ (e.g. τ = 0.5).
4. Fine-tune T5 on those kept completions as new LM targets (with the
   same final-mask GT loss).
5. Repeat for `K` rounds (K = 2-3) until kept-rate plateaus.

The intuition (STaR's): if the final answer is correct, the rationale
that led to it is *probably* coherent. Filter on terminal correctness;
get rationale supervision for free.

### Why this is the most "natural" path

Tier 2 supervises with rationales an *external* LLM emits — but that
LLM doesn't know the segmentor's inductive biases. T5 + segmentor as a
unit, after Tier 2 finetuning, has internalized the class prior; its
self-emitted rationales are likely better-aligned with what makes the
mask decoder produce good outputs than what an external LLM would
write.

### Cost

Each round = 1 inference pass over the entire training set (with `N`
samples per row → `N×` slowdown vs single-greedy) + 1 fine-tune run.
With `N = 10` and `K = 2-3`, total compute is roughly 3-5× a single
fine-tune. Manageable but not cheap.

### What it would test

- Whether T5's *self-emitted* rationale shape outperforms LLM-mined
  Tier 2 rationale shape on the relational subset.
- Whether, on `cs` / `hi`, T5 learns to skip the rationale entirely
  (i.e. emits chain-v2-like single-`[SEG]` answers) when reasoning
  doesn't help — a stronger expression of *"natural CoT"* than Tier 2's
  prompt-template-determined shape.

### Status: deferred

Tier 3 is documented for completeness; not part of this round. If
Tier 2 hits a clear ceiling and we have compute budget, this is the
natural next step.

---

## Things still uncertain (and what would resolve them)

1. **Whether the W1-pure premise holds — does the pretrained
   chain-v2 ckpt produce correct landmark masks zero-shot when
   conditioned on a class-name-bearing rationale?** This is the
   load-bearing assumption. **Resolution: a 30-minute probe** — feed
   the chain-v2 ckpt a synthetic two-`[SEG]` answer like *"I need to
   find the door first. [SEG]. Then the chair. [SEG]."* at inference
   (no training), and inspect M_1 / M_2 IoU vs each class's GT mask
   in the scene. If yes on both: B1' is well-motivated. If no: at
   minimum we need consistency rewards back (→ B6 promoted), or
   possibly W7 / W2 text mining.
2. **Whether F1 vs F3 actually matters under W1-pure.** R2S [8]
   argues yes; PixelLM [3] / LISA++ [2] don't architect-feedback.
   The user's intuition is F1 is needed because the second `[SEG]`
   needs to *see* the landmark mask features beyond the LLM's text.
   **Resolution: B7 vs B1'.**
3. **Whether the auxiliary mask-level rewards (`L_contain` + cosine)
   add anything beyond architectural F1.** This is the W1 → W1-pure
   pivot's key empirical question. **Resolution: B6 vs B1' (and B7+
   for the full old-headline upper bound).**
4. **Whether stop-gradient (1b) on `mask_pool_token` is too strict.**
   1a (gradient on) gives `M_1` an indirect supervision signal that
   could either help (small adaptation around the pretrained prior)
   or drift (M_1 stops being a clean class mask). **Resolution: B6+
   vs B1'.**
5. **Whether P4's regex coverage is enough or W7 / LLM-text labeling
   is needed.** ~50-60% non-fallback rate is the plan; if performance
   is bottlenecked on the fallback samples, **B9 vs B1'** tells us
   whether LLM text labeling (no mask GT) closes the gap.
6. **Whether collapse actually happens with axis A off.** Stop-grad
   plus distinct text class names should prevent it, but we have no
   direct training signal pushing them apart. **Resolution: monitor
   intermediate IoU vs final IoU on B1'; if they converge, add the
   cosine push-apart back as `B4` ablation row.**
7. **Whether the P4 question-side prefix should be at training time
   or inference only.** Standard zero-shot CoT puts it at inference
   only. **Resolution: B3 vs B1'.**
8. **Whether the rationale-text full LM loss vs masked-out works
   better.** Not currently in the ablation grid; can be added as `B10`
   (`cot_rationale_masked`) if B1' numbers are weak.

---

## Cross-references

- **Lit review precedents**:
  - R2S [8] for the architecture (mass-pool + extend memory).
  - PixelLM [3] for the anti-collapse formulation.
  - STaR [17] + Self-Consistency CoT [16] for the soundness of
    final-only supervision + consistency rewards.
  - Kojima 2022 (zero-shot CoT) for the question-side prefix.
- **Loss branch parallels**:
  - `enable_best_of_set`, `enable_scale_aware`, `enable_boundary`,
    `enable_point_aux` in [seg_loss_v3.py](../lavis/models/reason3d_models/seg_loss_v3.py)
    — same flag-gated style we will use for `enable_cot`,
    `enable_anticollapse`, `cot_prompt_prefix`.
  - Per-instance GT collation in
    [threedrefer_datasets_chainv3.py](../lavis/datasets/datasets/threedrefer_datasets_chainv3.py)
    — same per-sample-extras pattern we will reuse for the regex
    landmark + answer template.
  - Per-row JSONL extension in [refer_seg_task_v3.py](../lavis/tasks/refer_seg_task_v3.py)
    — same place we will add `intermediate_iou`, `seg_count`,
    `landmark_text`, `regex_hit`.

---

## Pending user input

All strategic + tactical axes are locked; the phased plan is locked
(Tier 1 → Tier 2 → Tier 3, in that order). Open decisions:

- [ ] **Pretrained-prior probe**: run a 30-minute zero-shot test of the
  chain-v2 ckpt on synthetic two-`[SEG]` answers (uncertainty #1) before
  committing engineering to Tier 1, or skip and commit on the prior
  alone.
- [ ] **Tier 2 timing**: kick off Tier 2 LLM-rationale mining script
  development *in parallel* with Tier 1 training (so B9 is ready to
  run as soon as B1' completes), or sequence strictly (Tier 1 numbers
  first, then decide whether Tier 2 is needed).
- [ ] **LLM choice for Tier 2**: open-weights (Qwen2.5-7B / Llama-3.1-8B)
  vs frontier API (GPT-4-class). Recommendation: start with
  open-weights for cost; revisit if rationale quality is visibly poor.
- [ ] **Confirm the ablation grid** (B0, B1', B2, B3, B5, B6, B6+, B7,
  B7+, B8, B9, B11) covers the questions of interest. Optional adds:
  `B4` (anti-collapse on) if collapse appears at B1', `B10`
  (`cot_rationale_masked`).

Once these decisions land, the next deliverable is
`chainv3_cot_design_proposal.md` (implementation-grade spec for Tier 1)
followed by code changes per the plan file. Tier 2 spec follows after
B1' numbers; Tier 3 spec is deferred.
