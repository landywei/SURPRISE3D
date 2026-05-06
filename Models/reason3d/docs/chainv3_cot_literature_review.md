# Chain v3 CoT — literature review

Companion to:
- design proposal (forthcoming) — `chainv3_cot_design_proposal.md`
- ablation tracker (forthcoming) — `chainv3_cot_ablation_tracker.md`
- loss-branch sibling — `chainv3_ablation_tracker.md`

This doc is a CoT-focused live-research sweep written **before** the
design doc, so the design choices in `chainv3_cot_design_proposal.md` can
cite the precedents directly. Scope (per the planning round): deep on the
CoT axis (multi-step / interleaved `[SEG]` / mask-feedback / weak
self-consistency), with short positioning paragraphs on multi-target 3D
referring and 3D situated reasoning so the doc can stand alone as
motivation for the CoT branch.

> **TL;DR for the design doc.** Two-stage *locate-then-refine* segmentation
> driven by an LLM is well-established in 2D ([1] LISA, [3] PixelLM,
> [5] VistaLLM) and has *one* close 3D analogue: Ning et al. ICCV 2025 R2S
> [8], which is architecturally near-identical to what we are proposing —
> mask-pool relevant-object features, inject them into the instruction,
> rerun the LLM, decode the final `[SEG]`. The crucial gap **R2S leaves
> open** is that its intermediate "relevant objects" are *fully supervised*
> by LLaMA-mined GT annotations on top of ScanRefer / ScanQA. Our chain v3
> CoT branch fills that gap by training the same architecture *weakly*
> (consistency rewards only — containment + mask-feature similarity) on
> Surprise3D, where intermediate-mask GT is not available and would be
> expensive to mine. The supporting literature on chain-of-thought
> self-consistency ([16] Wang et al., [17] STaR) backs the soundness of
> consistency rewards as a substitute for intermediate annotations; the
> mask-pool re-injection mechanism is borrowed nearly directly from
> R2S [8] / ScanReason CoG [7] / PixelLM [3].

---

## Section 1 — Multi-step / interleaved `[SEG]` in 2D vision-language segmentation

### [1] LISA: Reasoning Segmentation via Large Language Model

- Lai et al., **CVPR 2024**, arXiv:2308.00692.
- Adds a special `[SEG]` token to a Vicuna-based LMM. When the model emits
  `[SEG]`, its last-layer hidden embedding is fed (via a linear projector)
  to a frozen SAM-style mask decoder as the prompt. Single-`[SEG]`,
  single-target.
- Loss: standard token-CE on text + BCE + Dice on the SAM-derived mask.
- **Mechanism we borrow.** The "embedding-as-mask" paradigm — an
  LLM-emitted token whose hidden state is consumed by a separate mask
  decoder — is the foundation Reason3D itself was built on, and the
  starting point we extend with mask feedback. LISA does not feed the
  mask back into the LLM; that is the gap we close in chain v3 CoT.

### [2] LISA++: An Improved Baseline for Reasoning Segmentation

- Yang et al., 2024, arXiv:2312.17240.
- Extends [1] to **instance segmentation** (one `[SEG]` per instance,
  multiple `[SEG]`s in one response) and "Segmentation in Dialogue" —
  inline `[SEG]` tokens within natural chat. No architecture change vs
  LISA; only data curation (COCO-derived multi-`[SEG]` QA).
- **Mechanism we borrow.** Confirms that producing **multiple `[SEG]`s in
  a single response** is feasible without architecture changes — the LLM
  emits a sequence containing multiple `[SEG]`s and each one is decoded
  independently. Our chain v3 CoT path also emits two `[SEG]`s in one
  response, but ours feeds the *first* mask back to the LLM; LISA++
  decodes them independently.

### [3] PixelLM: Pixel Reasoning with Large Multimodal Model

- Ren et al., **CVPR 2024**, arXiv:2312.02228.
- Replaces SAM with a lightweight **pixel decoder** + a learnable
  **segmentation codebook** (multiple tokens per scale). The LMM emits
  interleaved text and codebook tokens; each codebook-token hidden is
  decoded by the pixel decoder into a per-target mask. Introduces
  **target refinement loss** that explicitly pushes different targets
  apart in embedding space when they would otherwise collapse to the
  same mask.
- Constructs MUSE (246k QA pairs, 0.9M instances) for multi-target
  reasoning segmentation supervision.
- **Mechanism we borrow.** Two ideas: (i) the multi-token-per-mask
  precedent supports our two-`[SEG]` design at no architectural cost;
  (ii) the **target refinement loss** is conceptually adjacent to our
  *anti-collapse margin* (deferred to v2 in the risks section of our
  design doc) — both punish the model for collapsing distinct outputs
  to the same mask.

### [4] GLaMM: Pixel Grounding Large Multimodal Model

- Rasheed et al., **CVPR 2024**, arXiv:2311.03356.
- Introduces *Grounded Conversation Generation* (GCG): the model produces
  natural-language responses with embedded `<phrase>...</phrase>` tags;
  each tag is grounded to a SAM mask. Uses GranD (7.5M concepts, 810M
  regions) for training.
- **Mechanism we borrow.** GCG is the canonical precedent for treating
  the LLM's text stream as the carrier for *both* reasoning and grounding
  decisions. Our chain v3 CoT branch borrows the same philosophy — the
  LLM decides *when* to emit a coarse-anchor `[SEG]_1` purely via the
  text stream, no separate gating head.

### [5] VistaLLM: Coarse-to-Fine Vision-Language Model

- Pramanick et al., **CVPR 2024** (project: shramanpramanick.github.io/VistaLLM/).
- Unifies coarse (image-level) and fine (mask-level) tasks. Represents
  binary masks as point sequences sampled with **gradient-aware adaptive
  contour sampling** (3-4 mIoU over uniform sampling). Single LLM-decoder
  pipeline; no explicit mask feedback into the LLM.
- **Mechanism we borrow.** Less directly relevant than [1]-[3]; included
  as evidence that mask outputs from an LMM admit many decoder formats
  (token-as-embedding, contour points, segmentation codebook) — so the
  R2S/our own design choice (single `[SEG]` token decoded by SPFormer
  mask decoder) is a reasonable design point.

### [6] NExT-Chat: An LMM for Chat, Detection, and Segmentation

- Zhang et al., **ICML 2024**, arXiv:2311.04498.
- Introduces the **pix2emb** paradigm: special trigger token `□` emitted
  by the LLM, followed by a placeholder `◇` whose hidden state is
  decoded by either a Box decoder (regressing GIoU/L1 loss) or a Mask
  decoder. Same `[SEG]`-token philosophy as LISA but with separate decoders
  for boxes vs masks.
- **Mechanism we borrow.** Confirms that the *same* hidden-as-prompt
  paradigm scales to multiple geometric outputs; relevant when we
  later consider extending chain v3 CoT to predict an intermediate
  *bounding box* anchor before the final `[SEG]` (out of scope for the
  current round but a v2 candidate).

---

## Section 2 — Iterative / chain-of-grounding 3D mask methods

### [7] ScanReason + ReGround3D — 3D Reasoning Grounding with Chain-of-Grounding

- Zhu, Wang, Zhang, Chen, Liu, **ECCV 2024**, arXiv:2407.01525,
  github.com/ZCMax/ScanReason.
- Introduces a 3D reasoning grounding task (10K+ QA-location pairs,
  five reasoning types). The ReGround3D method couples a *visual-centric
  reasoning module* (an MLLM) with a *3D grounding module* (geometry-
  aware decoder). Their **Chain-of-Grounding (CoG)** mechanism
  *interleaves* reasoning and grounding steps at inference: the MLLM
  first reasons about the question, the grounding module produces a
  coarse location, the MLLM consumes the coarse location and produces
  a refined answer, and so on.
- **Mechanism we borrow.** CoG is the closest *naming* precedent; it
  motivates the term "chain of thought" for our 3D mask pipeline.
  ScanReason still treats grounding as bounding-box prediction (not
  per-point masks), and CoG is largely an inference-time scaffold —
  whereas our chain v3 CoT trains the locate-then-refine loop end-to-end
  with consistency rewards on segmentation masks.

### [8] R2S: Relevant Reasoning Segmentation for 3D Point Clouds (closest precedent)

- Ning, Tian, Shi, Lu, He, Pei, Jiang, **ICCV 2025**, pp. 7851-7860,
  arXiv: see ICCV 2025 OpenAccess.
- **Architecturally identical** to what we are proposing for chain v3 CoT,
  modulo the supervision signal. Two stages:
  1. *Reasoning Prior Learning* — instruction `"Given the 3D scene,
     [QUESTION]. Please segment the question-related objects that may
     help answer the question."` The model emits one or more `[SEG]`s
     for **target-relevant** objects; their hidden states are decoded by
     the mask decoder into masks `M_r`. The super-point features `f_p`
     are then **mass-pooled** by `M_r` to give per-relevant-object
     features `f_r` (this is **exactly** our `MaskPoolToken` operation).
  2. *Prior-guided Refinement* — instruction
     `"Given the 3D scene, the question is [QUESTION]. The
     question-related objects are {f_r^1, ..., f_r^n}, and you may need
     to pay attention to them"`. The pooled features `f_r` are inserted
     **as text-embedding placeholders** into the second instruction and
     fed back to the Q-Former + LLM. The refined `[SEG]` hidden is then
     decoded into the **final** mask `M'`.
- **Loss / supervision.** Cross-entropy on text + BCE/Dice on each mask,
  including **fully supervised intermediate masks** `M_r`. They derive
  intermediate-mask GT by extending ScanRefer / ScanQA / their own
  3D ReasonSeg with **LLaMA-3.1-mined target-relevant object
  annotations** (e.g. for `"What is used for sitting near the desk?"`,
  LLaMA labels `desk` as the relevant anchor and the dataset provides
  the desk's GT mask as `M_r`).
- **Relevant Objects Augmentation.** They randomly omit / add relevant
  objects during training to simulate inference-time noise.
- **Architecture details.** Visual backbone = OneFormer3D; LLM = OPT-1.3B
  (frozen); Q-Former = 32 learnable queries; mask decoder = OneFormer3D's
  decoder. Compare with our setup: SPFormer + Q-Former + FlanT5-XL
  (frozen) + Reason3D mask decoder. Architecturally aligned.
- **What we borrow.** *Almost everything*: the two-stage flow, the
  mask-pool aggregation operator, the "extend instruction with pooled
  features" mechanism. This paper is the citation we anchor the design
  doc on.
- **What we add (versus R2S, the explicit gap).** R2S requires
  intermediate-mask GT (LLaMA-mined relevant-object annotations); we do
  **not** have these for Surprise3D and would have to mine them too. Our
  contribution is to show that the **same architecture** can be trained
  with **only the final-mask GT plus consistency rewards** on the
  intermediate mask (containment + mask-feature similarity), avoiding
  the LLaMA-mining cost. R2S's "Relevant Objects Augmentation" is a
  precedent for our `cot_template_prob` randomization (different
  motivation — they fight overfitting on the priors; we ramp the
  intermediate-`[SEG]` prevalence — but the dropout-style mechanism
  is similar).

### [9] Reason3D: Searching and Reasoning 3D Segmentation via LLM (our base)

- Huang, Wang et al., **3DV 2025**, arXiv:2405.17427,
  github.com/kuanchihhuang/reason3d.
- Hierarchical `[LOC]` + `[SEG]` decoding: the LLM first emits `[LOC]`
  for coarse-location, then `[SEG]` for the refined mask. Uses SPFormer
  encoder, Q-Former visual prefix, FlanT5-XL LLM, mask decoder over
  super-points.
- **Mechanism we borrow.** Reason3D **is** the model we extend. Note
  that the original `[LOC]` -> `[SEG]` design is itself a two-step
  decoding scheme — but the second-pass `[SEG]` is conditioned on
  `[LOC]`'s text-level output only; **the `[LOC]` mask never feeds back
  into the LLM**. Our chain v3 CoT branch closes that loop by adding
  `MaskPoolToken` re-injection between `[SEG]_1` and `[SEG]_2`. (In our
  base config we ignore `[LOC]`; the chain-v2 / chain-v3 datasets emit
  text answers ending in a single `[SEG]` and the LLM never produces a
  `[LOC]` token. Our CoT extension is therefore *parallel* to Reason3D's
  hierarchical decoding rather than a replacement of it.)

### [10] LL3DA: Visual Interactive Instruction Tuning for Omni-3D

- Chen et al., **CVPR 2024**, arXiv:2311.18651.
- Direct point-cloud input (no multi-view 2D projection). Captioning,
  3D QA, dense captioning, embodied dialogue. No segmentation output.
- **Mechanism we borrow.** Positions our work — chain v3 CoT extends
  the *segmentation* axis that LL3DA leaves untouched. Cite for context.

### [11] Chat-3D / Chat-3D-v2

- Wang et al., 2023-2024, arXiv:2308.08769 / arXiv:2312.08168.
- 3D-scene dialogue with object-identifier tokens. Multi-object queries
  via per-object identifiers in the conversation. No mask output.
- **Mechanism we borrow.** Cite for context: object-identifier-as-token
  is yet another flavor of pix2emb; we use plain `[SEG]` tokens.

### [12] 3D-LLM: Injecting the 3D World into LLMs

- Hong et al., **NeurIPS 2023**, arXiv:2307.12981.
- Renders 2D multi-view, projects features back to 3D via point-cloud
  features. Adds **3D location tokens** as anchors. 300k 3D-language
  training pairs. Not mask-output; bounding-box-style grounding.
- **Mechanism we borrow.** The "3D location anchor" precedent is the
  conceptual ancestor of our `MaskPoolToken`. Cite for context.

### [13] 3D-VisTA: Pre-trained Transformer for 3D Vision and Text

- Zhu et al., **ICCV 2023**, arXiv:2308.04352.
- Plain Transformer over object-token sequences; pre-trained with masked
  language/object modeling and scene-text matching on ScanScribe (278k
  pairs). Single-stage, no chain.
- **Mechanism we borrow.** Cite as the canonical *single-stage* 3D
  alignment baseline our chain v3 CoT improves upon (only as a
  contrast — we do not adopt anything from it).

---

## Section 3 — Mask-feedback / mask-conditioned re-injection mechanisms

This section explicitly enumerates the *three families* of "feed the
predicted mask back into the language stream" implementations, so the
design doc can justify the option we picked.

### [14] SAM — Iterative Click Prompting

- Kirillov et al., **ICCV 2023**, arXiv:2304.02643.
- Promptable segmentation. The user can give a click, then another click
  if the first mask is wrong; the prompt encoder embeds the new click +
  the **previous mask** (low-resolution) as inputs to the mask decoder.
  This is the canonical **mask-as-prompt** mechanism: mask is fed back
  *into the decoder*, not back to a language model.
- **Why this matters for us.** SAM's mask-feedback target is the *mask
  decoder*, which sees the prior mask via a learned embedding. R2S [8]
  and our design instead feed the mask back into the *language model* via
  mass-pooled features. Our `MaskPoolToken` is to our LLM what SAM's
  prior-mask embedding is to its decoder, but the consumer is different.

### [15] Three options for mask -> LLM feedback (taxonomy)

The literature splits into three concrete patterns. Our chain v3 CoT
picks **(a)**.

- **(a) Append mass-pooled feature to encoder memory; rerun decoder.**
  Used by R2S [8] (insert at *text-embedding* level via a placeholder
  token that the LLM unfolds into the pooled feature) and by our chain
  v3 CoT (append directly to encoder memory; attention mask gates which
  samples get the anchor). Cleanest mathematically; one extra encoder
  token; no second encoder forward needed if hidden states are cached.
  Compatible with frozen LLM.
- **(b) Inject pooled feature as a key/value in cross-attention only.**
  Equivalent to *augmenting the decoder cross-attn keys*, not the
  encoder output. Fewer parameters than (a); harder to implement under
  HuggingFace `T5ForConditionalGeneration` because it requires patching
  the decoder layer. Not adopted in any of [1]-[13].
- **(c) Single decoder pass with multiple `[SEG]`s, no architectural
  feedback.** Used by LISA++ [2] and PixelLM [3]. Cheapest; the LLM has
  no way to revise its second `[SEG]` based on the first mask. Useful
  for *multi-target* prediction (different objects) but not for
  *coarse-then-refine* (same object, two granularities), which is what
  we need.

We picked **(a)** because it (i) matches R2S [8], the closest 3D
precedent, (ii) is implementable with a single new `nn.Linear` and one
encoder-memory concatenation in the existing T5 graph, and (iii) keeps
the LLM frozen (no patching required). Detailed design rationale in
`chainv3_cot_design_proposal.md` §3.

---

## Section 4 — Weak / self-consistency supervision for chain-of-thought reasoning

This is the axis where chain v3 CoT differs most from R2S [8]. Citations
here are the foundation for why we believe consistency rewards are a
viable substitute for intermediate-mask GT.

### [16] Self-Consistency Improves Chain-of-Thought Reasoning

- Wang, Wei, Schuurmans, Le, Chi, Narang, Chowdhery, Zhou,
  **ICLR 2023**, arXiv:2203.11171.
- The foundational result: complex reasoning admits many CoT paths,
  marginalizing over them via plurality vote yields large gains
  (GSM8K 56.5% -> 74.4% with PaLM-540B). Inference-time only;
  no training signal.
- **What we borrow.** The conceptual move — different reasoning paths
  to the same answer should agree — is the basis for our consistency
  rewards. We adapt it from "plurality vote across samples" (their
  use) to "consistency *between* the intermediate and final mask of the
  same sample" (our use). This is the same idea applied at the
  intra-sample, intra-modality level.

### [17] STaR: Self-Taught Reasoner

- Zelikman, Wu, Mu, Goodman, **NeurIPS 2022**, arXiv: see OpenReview.
- Bootstrap loop: sample CoT rationales, fine-tune on those that lead
  to correct answers, repeat. For incorrect answers, *condition on the
  correct answer* to generate a rationale and include those too
  (rationalization).
- **What we borrow.** STaR validates that **terminal correctness** is a
  sufficient training signal to bootstrap intermediate reasoning steps,
  without intermediate-step labels. Our chain v3 CoT is similar in
  spirit: the **final-mask GT** is the only direct supervision, and the
  intermediate `[SEG]_1` is shaped indirectly via consistency rewards
  on the final mask. Stage C (deferred) — REINFORCE-style terminal
  reward on `[SEG]` emission decisions — would close the loop fully.

### [18] Visual CoT: Multi-turn Visual Reasoning with Bounding-Box Anchors

- Shao, Qian, Xiao, Song, Zong, Wang, Liu, Li, **NeurIPS 2024 Spotlight**,
  arXiv:2403.16999.
- Dataset of 438k QA pairs annotated with **intermediate bounding boxes**
  highlighting key regions. Multi-turn pipeline: model first emits a
  bounding box, then crops/zooms, then answers. Uses **fully supervised**
  intermediate boxes (similar to R2S [8]'s fully supervised intermediate
  masks).
- **What we borrow.** Confirms that intermediate-region supervision
  helps, but also that scaling it requires expensive annotation. Our
  contribution: train the same pattern *without* intermediate-region
  supervision, using consistency rewards as the substitute.

### Adjacent work not explicitly cited but worth flagging

- **Self-rewarding LMs / DPO-style preference learning** — not directly
  applicable since we have *only one* output per sample (no preference
  pairs); listed as candidate for Stage C / future work.
- **Quiet-STaR** (Zelikman et al., 2024) — internal reasoning tokens
  trained without explicit supervision via reinforcement-style rewards;
  philosophically aligned with our deferred Stage C, but not used here.

---

## Section 5 — Multi-target referring (positioning context, short)

Brief context for why chain v3 cares about multi-target queries — the
**loss-only branch** ([chainv3_design_proposal.md](chainv3_design_proposal.md))
already addresses this; chain v3 CoT inherits the best-of-set machinery
unchanged.

### [19] Multi3DRefer: Grounding Text to Multiple 3D Objects

- Zhang, Gong, Chang, **ICCV 2023**, arXiv:2309.05251,
  github.com/3dlg-hcvc/multi3drefer.
- Extends ScanRefer to allow zero, one, or many targets per description
  (61,926 descriptions of 11,609 objects across 800 scenes). Annotation
  attributes: spatial / color / texture / shape; eval splits include
  "zero target", "multiple targets", "single with distractors", "single
  without distractors". Introduces the **M3DRef-CLIP** baseline (CLIP +
  online rendering + contrastive learning).
- **Mechanism we borrow.** None directly for chain v3 CoT — but their
  multi-target eval taxonomy is the framework chain v3's loss branch
  (`hit@τ` / per-instance IoU) is built on.

### [20] Mask3D / Mask2Former-style Hungarian set prediction

- Schult et al., 2023, arXiv:2210.03105 (ICRA 2023).
- Hungarian matching between a fixed set of K query masks and the GT
  instance set. Per-instance dice + focal + objectness terms.
- **Mechanism we borrow.** None directly for chain v3 CoT (chain v3
  loss branch's `enable_best_of_set` is a *single-query* min-over-GT
  approximation, not full Hungarian — by design, since Reason3D's mask
  decoder emits one query per `[SEG]`). Cited as the "right way to do
  it" if we ever scale to N parallel `[SEG]`s.

---

## Section 6 — 3D spatial / situated reasoning benchmarks (positioning context, short)

Context for the *failure mode 3* (multi-step relational reasoning) the
CoT branch is designed to address. Not all of these are evaluated on,
but they collectively justify why a multi-step head matters.

### [21] SQA3D: Situated Question Answering in 3D Scenes

- Ma, Yong, Zheng, Li, Liang, Zhu, Huang, **ICLR 2023**, see OpenReview.
- 6.8k unique situations, 20.4k textual situation descriptions, 33.4k
  reasoning questions across 650 ScanNet scenes. Best SOTA 47.20% vs
  human 90.06%. Confirms that situated reasoning is hard for current
  3D MLLMs.

### [22] MSQA: Multi-modal Situated Reasoning in 3D Scenes

- Linghu et al., **NeurIPS 2024 Datasets & Benchmarks**, arXiv:2409.02389,
  msr3d.github.io.
- 251k QA pairs across 9 categories, *interleaved* multi-modal inputs
  (text + images + point clouds for both situation and question), and a
  Multi-modal Situated Next-step Navigation (MSNN) extension. Larger
  successor to SQA3D [21].

### [23] Surprise3D (our benchmark)

- Huang et al., **NeurIPS 2025 Datasets & Benchmarks**, arXiv: see
  OpenReview, mbzuai-liziwen.github.io/Surprise3D/.
- 200k+ vision-language pairs over 900+ ScanNet++ v2 scenes, **89k+
  human-annotated spatial queries** spanning four reasoning skills:
  *relative position*, *narrative perspective*, *parametric perspective*,
  *absolute distance*. Crucially, queries **deliberately exclude object
  names** to eliminate semantic shortcuts; SOTA models drop to near-zero
  when names are removed. This is the dataset chain v3 (and the CoT
  branch) finetunes / evaluates on.
- **Why CoT for this benchmark.** Surprise3D's *relative position* and
  *narrative perspective* questions explicitly demand *first locate
  anchor, then locate target* reasoning — the failure mode chain v3 CoT
  is built around. Cite this paper as the empirical motivation in the
  design doc's §1 (problem framing).

### Adjacent benchmarks worth flagging

- **3DSRBench** (3D spatial VQA, 2024) — relational queries; not used.
- **Anywhere3D-Bench** (multi-level referring, 2024) — open-vocab; not
  used.

---

## Section 7 — Cross-link table (design choice -> precedents)

For each design choice in the forthcoming
[chainv3_cot_design_proposal.md](chainv3_cot_design_proposal.md), the
precedents we lean on:

- **Two-stage encoder-extend-then-decode loop** — R2S [8] (architecture
  identical), ScanReason CoG [7] (3D chain-of-grounding precedent),
  PixelLM [3] (multi-`[SEG]` in 2D), LISA [1] (the foundational
  embedding-as-mask paradigm).
- **`MaskPoolToken` (linear over mass-pooled `sp_feats`)** — R2S [8] §3.2
  Eq. (7) (`mask-pooling to aggregate the super-point features f_p with
  masks M_r`). Drop-in adoption.
- **Insertion of pooled feature into encoder memory (vs into instruction
  text)** — R2S [8] inserts at the *instruction-text* level
  (`"...the question-related objects are {f_r}..."`); we insert at the
  *encoder-output* level. Mathematically near-equivalent; ours avoids a
  text-template change.
- **Per-sample template randomization (`cot_template_prob`)** — R2S [8]
  *Relevant Objects Augmentation* §3.4 (random omission/addition of
  relevant objects). Same dropout-style mechanism, different motivation
  (we ramp the intermediate-`[SEG]` prevalence; they fight overfitting
  on the priors).
- **Containment consistency reward** — adapted from chain-of-thought
  self-consistency [16] applied at the intra-sample, intra-modality
  level; no direct precedent in 3D segmentation, our novel formulation.
- **Mask-feature similarity reward** — adapted from PixelLM [3]'s
  *target refinement loss* (which uses cosine on token embeddings to
  push different targets apart); we use cosine on **`sp_feats`-pooled
  vectors** to push anchor and target *together* (opposite sign,
  different feature space).
- **Final-mask-only supervision (no intermediate-mask GT)** — STaR [17]
  validates that terminal correctness suffices to bootstrap intermediate
  steps; self-consistency [16] validates that intra-sample agreement
  carries useful gradient.
- **Deferred Stage C (REINFORCE on `[SEG]` emission decisions)** — STaR
  [17] (rationalization step), Quiet-STaR (2024), and the broader
  RL-from-LM literature. Out of scope for this round.
- **Per-instance hit@τ metric (loss branch, inherited)** — Multi3DRefer
  [19] eval taxonomy (zero / single / multiple targets).

---

## Open questions surfaced by this lit review

1. **Anti-collapse margin.** PixelLM [3]'s *target refinement loss*
   shows that an explicit push-apart term is needed when distinct
   tokens would otherwise collapse to the same mask. Our containment +
   similarity rewards admit `mask_1 = mask_final` as a trivial solution.
   The plan defers an anti-collapse margin to v2; once we have B1 / B6
   numbers showing collapse (`mean intermediate_iou ≈ mean final_iou`),
   we should adopt PixelLM's formulation directly.
2. **Anchor mining.** R2S [8] uses LLaMA-3.1 to generate intermediate
   anchor annotations on top of ScanRefer / ScanQA. Our plan deliberately
   skips anchor mining (`cot_anchor_template = "Looking around the
   scene."` is a fixed generic anchor). Once chain v3 CoT works in the
   weakly-supervised regime, a follow-up axis is to *replace* the
   generic anchor with mined anchors and see if the supervised-anchor
   version (effectively R2S on Surprise3D) gives further gains. That
   ablation row is out of scope for this round.
3. **`[LOC]` vs `[SEG]_1` overlap.** Reason3D [9] already has a `[LOC]`
   token (hierarchical decoder). We currently ignore `[LOC]` in chain
   v3. A v2 question is whether the intermediate anchor in chain v3
   CoT *is* `[LOC]` re-purposed — i.e. should the LLM emit
   `[LOC] -> mask_loc -> MaskPoolToken -> [SEG] -> mask_final`? This
   would unify chain v3 CoT with Reason3D's original design.

---

## References

Where arXiv ids are unknown, the venue + author + year is cited. Direct
URLs to project pages are kept inline within each annotation.

- [1] Lai et al., "LISA: Reasoning Segmentation via Large Language Model",
  CVPR 2024. arXiv:2308.00692.
- [2] Yang et al., "LISA++: An Improved Baseline for Reasoning
  Segmentation with Large Language Model", 2024. arXiv:2312.17240.
- [3] Ren et al., "PixelLM: Pixel Reasoning with Large Multimodal Model",
  CVPR 2024. arXiv:2312.02228.
- [4] Rasheed et al., "GLaMM: Pixel Grounding Large Multimodal Model",
  CVPR 2024. arXiv:2311.03356.
- [5] Pramanick et al., "Jack of All Tasks Master of Many: Designing
  General-Purpose Coarse-to-Fine Vision-Language Model" (VistaLLM),
  CVPR 2024.
- [6] Zhang et al., "NExT-Chat: An LMM for Chat, Detection, and
  Segmentation", ICML 2024. arXiv:2311.04498.
- [7] Zhu, Wang, Zhang, Chen, Liu, "ScanReason: Empowering 3D Visual
  Grounding with Reasoning Capabilities" (ReGround3D + Chain-of-Grounding),
  ECCV 2024. arXiv:2407.01525.
- [8] Ning, Tian, Shi, Lu, He, Pei, Jiang, "Enhancing Spatial Reasoning
  in Multimodal Large Language Models through Reasoning-based
  Segmentation" (R2S + 3D ReasonSeg), ICCV 2025, pp. 7851-7860.
- [9] Huang, Wang et al., "Reason3D: Searching and Reasoning 3D
  Segmentation via Large Language Model", 3DV 2025. arXiv:2405.17427.
- [10] Chen et al., "LL3DA: Visual Interactive Instruction Tuning for
  Omni-3D Understanding, Reasoning, and Planning", CVPR 2024.
  arXiv:2311.18651.
- [11] Wang et al., "Chat-3D" / "Chat-3D-v2", 2023-2024. arXiv:2308.08769
  / arXiv:2312.08168.
- [12] Hong et al., "3D-LLM: Injecting the 3D World into Large Language
  Models", NeurIPS 2023. arXiv:2307.12981.
- [13] Zhu et al., "3D-VisTA: Pre-trained Transformer for 3D Vision and
  Text Alignment", ICCV 2023. arXiv:2308.04352.
- [14] Kirillov et al., "Segment Anything" (SAM), ICCV 2023.
  arXiv:2304.02643.
- [15] (Taxonomy section of this document; no external citation.)
- [16] Wang, Wei, Schuurmans, Le, Chi, Narang, Chowdhery, Zhou,
  "Self-Consistency Improves Chain of Thought Reasoning in Language
  Models", ICLR 2023. arXiv:2203.11171.
- [17] Zelikman, Wu, Mu, Goodman, "STaR: Bootstrapping Reasoning with
  Reasoning", NeurIPS 2022.
- [18] Shao, Qian, Xiao, Song, Zong, Wang, Liu, Li, "Visual CoT:
  Advancing Multi-Modal Language Models with a Comprehensive Dataset
  and Benchmark for Chain-of-Thought Reasoning", NeurIPS 2024 Spotlight.
  arXiv:2403.16999.
- [19] Zhang, Gong, Chang, "Multi3DRefer: Grounding Text Description to
  Multiple 3D Objects", ICCV 2023. arXiv:2309.05251.
- [20] Schult et al., "Mask3D: Mask Transformer for 3D Semantic Instance
  Segmentation", ICRA 2023. arXiv:2210.03105.
- [21] Ma, Yong, Zheng, Li, Liang, Zhu, Huang, "SQA3D: Situated Question
  Answering in 3D Scenes", ICLR 2023.
- [22] Linghu et al., "Multi-modal Situated Reasoning in 3D Scenes"
  (MSQA), NeurIPS 2024 Datasets & Benchmarks. arXiv:2409.02389.
- [23] Huang et al., "Surprise3D: A Dataset for Spatial Understanding
  and Reasoning in Complex 3D Scenes", NeurIPS 2025 Datasets &
  Benchmarks.

---

## Cross-references to chain v3 docs

- **Loss branch** (already implemented): `chainv3_ablation_tracker.md`,
  the existing `lavis/models/reason3d_models/seg_loss_v3.py`, and
  `lavis/datasets/datasets/threedrefer_datasets_chainv3.py`.
- **CoT branch** (this round; planning):
  `chainv3_cot_design_proposal.md` (forthcoming) and
  `chainv3_cot_ablation_tracker.md` (forthcoming). The plan in
  `.cursor/plans/chainv3_cot_branch_*.plan.md` is the parent.
