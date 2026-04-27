# Presentation Outline: Chain-of-Thought Conditioning for 3D Spatial Reasoning Segmentation

---

## 1. Motivation

**Big problems this work contributes to:**

- In real settings, people rarely give instructions as pure geometry (“the object 1.2 m left of the north wall”). They often describe **goals, habits, and everyday situations** in natural language: *“I’m cold—what should I use?”*, *“I need to throw away trash—where should I put it?”*, *“the room is too dark—what can I turn on?”* Surprise3D explicitly includes **common sense** and **human intention** query families alongside spatial categories; these require the model to connect language to **function and intent**, not only relative layout.
- State-of-the-art 3D vision–language models are often trained and evaluated on **named-object** referring expressions (“the *chair* near the window”). That lets models **shortcut** via category matching. Name-free, situation-driven queries force the model to decide **which objects in the scene** satisfy the description—without an explicit category label in the prompt.
- Even when the model produces a mask, **standard Reason3D-style decoding** often collapses to filler text (e.g. *“Sure, [SEG].”*), so failures are a **black box**: you cannot see *what category* the model committed to before grounding. For debugging, safety, and future constraints (e.g. re-ranking by text), we need the model to **externalize** a short verbal commitment before segmentation.

**Applications once these problems are better solved:**

- Assistive and home agents that follow **intent- and habit-level** instructions in 3D (“warm up”, “dispose of waste”, “improve lighting”) without the user naming products
- Interfaces for non-expert users who describe needs in **common language** rather than object catalogs
- Any pipeline where **auditable decisions** matter: knowing *what* the model thought it was segmenting before trusting the mask

---

## 2. SOTA

**3D vision–language grounding:**
- ScanRefer, ReferIt3D — 3D localization from language; queries typically name or strongly cue the object
- 3D-LLM, LL3DA, Chat-Scene — richer 3D–language interaction; still often tied to named or identifier-based references

**3D reasoning segmentation / grounding with reasoning:**
- Reason3D — LLM + `[SEG]` + hierarchical mask decoder; strong on classic referring data
- ScanReason — reasoning-oriented 3D grounding (bbox + QA scale smaller than Surprise3D)
- 3D ReasonSeg (ICCV 2025) — reasoning-style segmentation on point clouds (parallel line of work)

**Benchmark positioning:**
- Surprise3D — large-scale **name-free** queries on ScanNet++, **segmentation** output, includes **common sense** and **human intention** alongside spatial reasoning types

---

## 3. Challenges

1. **Semantic shortcuts** on named-reference training vs. **name-free** evaluation
2. **Faithful supervision / evaluation** — instance-label vocabulary and text preprocessing must match the full query distribution (including functional wording and numeric distances)
3. **Grounding bottleneck** — compressing “what object + where” into a single `[SEG]` state
4. **Opaque decoding** — filler outputs hide category errors; hard to diagnose or constrain
5. **Ambiguity under common sense / intention** — many objects can *plausibly* satisfy a query; the model must align with annotator intent

---

## 4. Project goals

| Challenge | What we address |
|---|---|
| Name-free + intent/common-sense queries | Fine-tune Reason3D on Surprise3D with corrected preprocessing |
| Opaque `[SEG]` only | **Reason3D+Chain** — supervise short **object-name** (or phrase) text before `[SEG]`; inference exposes **decoded text** for diagnosis |
| Faithful GT / queries | Full-vocabulary instance labels; fix text preprocessing for distance queries (see repo docs) |

---

## 5. Approach

**Base:** Reason3D (SPFormer → Q-Former → Flan-T5-XL → `[SEG]` → mask decoder), LAVIS-style training.

**Chain:** Same architecture; training targets become templates like *“The answer is the {name}. [SEG].”* from dataset `object_name` (oracle at train time; model generates at test time).

**Inference:** Optional repetition penalty + no-repeat n-gram to reduce degenerate loops in longer chain outputs.

---

## 6. Technical details

*(Same structure as report: encoder, Q-Former, LM, mask head; preprocessing flags; chain dataset wrapper; decode constraints.)*

---

## 7. Experimental setup

- **Benchmark:** Surprise3D (includes common sense, human intention, and spatial types)
- **Metrics:** mIoU, Acc@50, Acc@25
- **Subset vs full val:** document `n` and preprocessing; full val for direct comparison to published numbers when available

---

## 8. Main results

### Quantitative (evaluation subset `n=531`; published rows are full val — not directly comparable)

| Model | Setting | mIoU | Acc@50 | Acc@25 |
|---|---|---|---|---|
| Reason3D† | Zero-shot (published) | 6.08 | 4.57 | 9.09 |
| Reason3D† | Fine-tuned (published) | 11.00 | 9.06 | 16.14 |
| Reason3D | Zero-shot (ours) | 4.93 | 4.90 | 7.53 |
| Reason3D | Fine-tuned, ckpt@1 | **13.76** | **13.37** | **22.79** |
| Reason3D+Chain | Zero-shot (ours) | 4.69 | 4.52 | 7.16 |
| Reason3D+Chain | Fine-tuned, ckpt@1 | 13.46 | 12.43 | 22.22 |

### Qualitative (aligned with decoded text)

- **Baseline** often decodes *“Sure, [SEG].”* — no category signal.
- **Chain** decodes explicit categories, e.g. *“The target object is monitor, whiteboard. [SEG].”* for *“which objects can display information?”* while GT may be **poster** — failure is **legible** (category disambiguation), not hidden.

---

## 9. Ablations

### Training curve (`n=531`)

| Model | Checkpoint | mIoU | Acc@50 | Acc@25 |
|---|---|---|---|---|
| Reason3D | 2k iter | 9.38 | 6.78 | 16.76 |
| Reason3D | ckpt@1 (best) | **13.76** | **13.37** | **22.79** |
| Reason3D | ckpt@2 | 13.11 | 10.73 | 22.03 |
| Reason3D+Chain | 1k iter | 6.99 | 6.03 | 10.92 |
| Reason3D+Chain | ckpt@1 (best) | **13.46** | **12.43** | **22.22** |

### Decode constraints (chain, 10k iter)

| Greedy | + rep. penalty 1.2 + no-repeat 3-gram |
|---|---|
| 10.80 mIoU | 10.86 mIoU |

---

## 10. Conclusions and future work

- Chain competitive on subset; main win = **interpretability** + path to constraints / rationales
- **Future:** human (and LM) **rationales** especially where perspective + intent compound (e.g. narrative perspective); full-val eval; optional geometry module once stable
