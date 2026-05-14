# Harness Engineering for Spatial Reasoning

> One-line pitch: treat spatial reasoning the way **coding-agent harnesses** treat software engineering — a model-driven **agent** acting in a spatial **environment** with a set of **tools** and a **persistent memory**. Stop training new spatial models; engineer the agent harness around an off-the-shelf base model.

> **Modality-agnostic.** The harness consumes whatever the benchmark provides — RGB only, RGB+depth, multi-view, video, or point cloud — and exposes the appropriate tools to the agent. The harness is not tied to any single input modality and is independent of the multimodal-CoT-for-point-clouds idea (`IDEA_multimodal_cot_3d.md`).

---

## 1. Motivation

- **The agentic-harness trend is hot and works**: coding agents (Claude Code, Cursor agent, SWE-agent, OpenHands, Codex) showed that a strong **harness** around a generic LLM beats specialized fine-tunes. Harness engineering — designing the environment, tool API, agent loop, and memory — is now a central research and product surface.
- **Spatial reasoning has not had this treatment**. Most spatial-reasoning papers focus on **model-side fixes**: bigger data, new architectures, RL fine-tuning. Inference-time agentic harnesses are underdeveloped relative to the coding-agent ecosystem.
- **Spatial reasoning is unusually well-suited to harnesses**: it has rich, **callable structure** (detectors, depth, segmentation, scene-graph builders, geometric kernels, qualitative-relation checkers), and **verifiable intermediate state** (boxes, depth orderings, constraint consistency) — perfect material for tools and memory.
- **Hypothesis**: a well-engineered agent harness around a strong base model can match or beat custom-trained spatial models on benchmarks like 3DSRBench / CV-Bench / Omni3D-Bench / SQA3D, regardless of input modality.

---

## 2. The Harness, Formally

Mirroring the structure of coding-agent harnesses:

- **Environment** — the spatial scene provided by the benchmark. Modality is whatever is available (RGB / RGB-D / multi-view / video / point cloud). The environment exposes:
  - a question or task,
  - the raw modality input,
  - any metadata (intrinsics, image size, camera poses).
- **Policy (agent)** — an off-the-shelf base model (e.g. Qwen2.5-VL, GPT-4o, or a 3D-LLM) acting as a **model-driven** decision maker. It selects which tool to call next, when to stop, and what to answer. Loop is **agent-driven**, not controller-driven.
- **Action space (tools)** — a curated set of callable tools. The agent emits structured tool calls; the harness executes them and returns observations.
- **Memory** — a persistent store that survives across turns within a single question (and optionally across questions). Form is TBD; candidates include:
  - structured scene graph,
  - tool-output log (what was called, what came back),
  - free-form scratchpad,
  - hybrid.
- **Agent loop** — observe → plan → call tool → observe result → update memory → continue or answer. Standard ReAct / agent-loop pattern.
- **Termination & answer formatting** — agent emits a final answer in benchmark-specified format.

### 2.1 Candidate tool set

The same tool catalog as before, but now framed as **actions in the environment**, dispatched per modality:

- **2D perception**: detector, segmenter, OCR, region crop, region zoom.
- **2.5D / 3D from images**: monocular depth, multi-view stereo, plane fit, partial reconstruction.
- **Native 3D**: point cloud sub-cloud crop, voxelize, mesh slicer, point-cloud renderer.
- **Symbolic**: scene-graph builder, qualitative relation extractor (RCC / Allen-style), CSP/SMT/QCN consistency checker.
- **Numeric kernels**: distance / angle / size / IoU / depth-ordering.
- **Memory ops** (often considered tools too): write to scene graph, read from scene graph, mark a region as known.

### 2.2 Memory — open design question

We don't yet commit to a memory format. Two leading options:

- **Structured (scene-graph-centric)**: agent maintains a typed graph of objects, attributes, 3D positions, and relations. Tools read/write to it. Easy to verify and inspect.
- **Hybrid (graph + scratchpad)**: structured graph for grounded facts, free-form text for hypotheses and partial reasoning.

Decide empirically based on which form the base model uses correctly.

---

## 3. Why Now — Connection to the Agentic-Harness Wave

Recent emphasis in agentic system design (mostly in coding / software) gives us proven recipes we can port:

- Strong tool API design beats stronger base models.
- Explicit memory + scratchpad outperforms long context.
- Sub-agents / role specialization help when the action space is large.
- Replay / retry / self-critique loops are cheap wins.
- Lightweight verifiers as tools (linters, test runners) supervise the agent without retraining it.

Each of those maps cleanly onto spatial reasoning:

| Coding harness pattern | Spatial harness analog |
| --- | --- |
| Run tests / linter | Run QCN consistency checker / geometric verifier |
| Read file / grep | Detect objects / crop region / query scene graph |
| Edit file | Update scene graph / add object / refine relation |
| Compiler errors | Constraint violations from QCN / depth-ordering checks |
| Sub-agents (planner / executor) | Spatial planner / geometric specialist / symbolic specialist |
| Persistent workspace | Persistent scene graph across turns |

This analogy is the **core narrative** of the paper.

---

## 4. Gap Analysis vs Existing Agentic Work

| Paper | Harness slot it covers | What is still missing |
| --- | --- | --- |
| SpatiO (arXiv:2604.21190) | Multi-agent role orchestration | Not a true tool-using agent loop; no persistent memory; no symbolic tools |
| GCA (arXiv:2511.22659) | Geometric-tool task solver + constraint formalization | No persistent memory; agent loop is shallow |
| IR3D-Bench (arXiv:2506.23329) | Programmatic rendering via Blender + MCP | Eval framework, not an inference harness |
| SpatialReasoner (arXiv:2504.20024) | Internal 3D representation (trained) | No external tools or agent loop |
| SpatiaLQA recursive SG (arXiv:2602.20901) | Scene-graph decomposition | No tool use; controller-driven not agent-driven |

**Unclaimed land**: a **true agent harness** in the coding-agent sense — model-driven loop, structured action space, persistent memory, optional sub-agents — applied to spatial reasoning, with a first-class **qualitative spatial-temporal reasoning module** (QCN-style consistency checking, composition tables, constraint propagation) plugged in as both a tool and a verifier.

---

## 5. Concrete Proposal Variants

Listed from most-agentic (preferred) to least-agentic.

### (A1) Full Spatial Agent Harness (preferred MVP)
Single base model in a ReAct-style loop. Action space: detector, depth, scene-graph read/write, QCN checker, numeric kernels. Persistent scene-graph memory across turns. Terminates by emitting final answer.

### (A2) Decomposition Agent
Agent first decomposes the question into atomic spatial primitives (`depth(A) > depth(B)?`, `left_of(A,B)?`, `size(A) > size(B)?`), then dispatches each to the appropriate tool, then recombines.

### (A3) Multi-Role Agent Harness
Planner agent issues sub-goals; specialist agents (perception, geometry, symbolic) execute. Memory is shared. Closer to SpatiO but with real tool use, not just VLM ensembling.

### (V) Verifier / Best-of-N as a Subordinate Tool *(not MVP)*
Best-of-N + structural reward is *one possible tool* the agent can invoke (e.g., "ask for a confidence-checked second opinion"). It is *not* the centerpiece anymore.

---

## 6. Minimum Viable Experiment

- Base model: strongest accessible model for the chosen modality (e.g. Qwen2.5-VL-7B for image benchmarks; 3D-LLM / PointLLM-V2 for point-cloud benchmarks). The harness is base-model-agnostic.
- Benchmark: pick one (3DSRBench for RGB, SQA3D for point cloud).
- Pipeline: variant **A1** end-to-end.
- Minimum tool set:
  - detector / segmenter,
  - depth (or point-cloud query),
  - scene-graph read/write,
  - QCN consistency checker,
  - numeric kernel (distance / direction / size).
- Memory: structured scene graph for the MVP. Iterate later.
- Agent loop: ReAct-style with explicit `tool_call` + `observation` turns, capped at e.g. 10 steps.
- Compare: zero-shot base model, base + single-tool calls (no memory), full harness.
- Ablations:
  - drop persistent memory (per-turn only),
  - drop QCN checker,
  - drop scene-graph writes (read-only),
  - vary max steps.

Expected outcome: full harness > base by a clean margin; ablations show **memory + symbolic verifier** are the two most important non-perception components.

---

## 7. Design Tradeoffs to Decide Early

- Action space size — small (3–4 tools) for stable agent behavior, or large (10+) for expressivity?
- Memory representation — scene graph only, or hybrid scratchpad?
- Single agent vs multi-role (planner + specialists)?
- Training allowed (e.g., SFT on harness traces) or strictly inference-time?
- Compute budget per question (tools + multi-step looping is expensive).
- Single-image, multi-view, video, or point cloud first?

---

## 8. Risks / Failure Modes

- Agent gets stuck in tool-call loops or repeated wrong calls.
- Tools too noisy on benchmark inputs (e.g., monodepth wrong) → propagates errors through memory.
- Memory pollution: false facts in scene graph corrupt later reasoning.
- Latency explosion under long agent loops.
- Reward hacking if any reward signal is used.

Mitigations:
- Cap step count; require monotonic memory growth (no overwrite without justification).
- Per-tool confidence and abstention.
- Verifier tool (QCN) flags inconsistencies, agent must resolve before answering.
- Calibration set separate from eval.

---

## 9. Why This Could Be a Paper

- **Narrative**: "harness engineering for spatial reasoning" — a clean port of the agentic-harness wave from coding into spatial AI. The framing alone is novel for this domain.
- **Concrete contribution**: a tool API + memory schema for spatial reasoning, with a working agent loop and ablations showing memory and symbolic tools matter.
- **Strong baselines** (SpatiO, GCA, SpatialReasoner) to compare against directly, plus zero-shot base.
- **Modality-agnostic**: same harness across RGB / RGB-D / point-cloud benchmarks.
- **Reproducible**: no model retraining required.

---

## 10. Related Work

- SpatialThinker (arXiv:2511.07403) — dense spatial rewards in RL; reward-hacking story relevant to verifier design.
- SpatialReasoner (arXiv:2504.20024) — explicit-3D intermediate representation baseline.
- SpatiO (arXiv:2604.21190) — multi-agent test-time orchestration.
- GCA (arXiv:2511.22659) — geometrically constrained agent.
- IR3D-Bench (arXiv:2506.23329) — inverse rendering as agentic eval.
- SpatiaLQA (arXiv:2602.20901) — recursive scene-graph reasoning.
- QTSR survey — *Qualitative Spatial and Temporal Reasoning: Current Status and Future Challenges* (for QCN tool design).
- Agentic-harness inspiration (coding domain): SWE-agent, OpenHands, Claude Code / Cursor agent design notes, ReAct, Reflexion. To be cited as the *methodological precedent* we're porting to spatial reasoning.

---

## 11. Next Steps

1. Lock the action space (initial 5 tools) and memory schema (scene graph for MVP).
2. Pick benchmark + base model for the first run.
3. Implement A1 end-to-end with a minimal ReAct loop.
4. Run ablations on memory and on the QCN/symbolic tool specifically.
5. Compare against SpatiO / GCA / zero-shot base.
6. If gains hold, extend to A2 (decomposition) and A3 (multi-role).
