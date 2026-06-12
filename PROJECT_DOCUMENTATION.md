# Project Documentation — Week 2 RAG Application
### LumenLeaf Customer Support KB with Hybrid Search
*Mohan Gujral · The Gen Academy, Mastering Agentic AI Bootcamp · Week 2*

---

## 1. Project overview

I built a customer-support RAG bot (suggested Use Case 4) for a fictional smart-home company, "LumenLeaf." A customer asks product, policy, or troubleshooting questions in a web chat; the system retrieves from an 11-document knowledge base using **hybrid search** (dense embeddings + sparse BM25 fused in one Pinecone index), generates an answer with **Llama-3.3-70B on Nebius Token Factory** that must cite its source documents, and — critically — **escalates to a human instead of answering** when retrieval confidence is low or the model judges the context insufficient. The refusal path was designed before the happy path.

**One-liner (per the framework):** My RAG app helps support agents and customers answer product and troubleshooting questions from a customer-support KB (11 docs) in a web chat UI with 100% measured faithfulness, <5s latency, and confidence-based human escalation.

**The framework, filled in:**

| Field | Decision |
|---|---|
| Use case | Customers/agents ask product, policy & troubleshooting questions in a chat widget |
| Corpus | 11 authored markdown docs: 5 policy docs, 2 product manuals, troubleshooting guide w/ error codes, FAQ, resolved tickets (~14k words, English, owned by "support ops") |
| Ingestion + cleaning | Local md files → whitespace/blank-line normalization (deliberately light; punctuation preserved for BM25 tokens like "LL-E45") |
| Ingestion + freshness | One-shot ingest script; re-run on corpus change (deterministic chunk IDs make it idempotent); freshness SLA n/a for static demo corpus |
| Chunking + embedding | RecursiveCharacterTextSplitter, heading-aware separators, 1000 chars/150 overlap → 27 chunks; OpenAI text-embedding-3-small (1536-dim) — chunk size and model capacity matched |
| Retrieve | Pinecone serverless (dotproduct), hybrid α·dense + (1−α)·sparse with α=0.6, top-k=5 |
| Generation | Llama-3.3-70B-Instruct via Nebius Token Factory, temp 0.1, mandatory inline citations, ESCALATE token |
| Eval | 20 labeled queries, 7 categories; hit@1/hit@5 across 3 retrieval modes; LLM-as-judge faithfulness; first-contact resolution + escalation recall |

## 2. Datasets used

The corpus is an **authored, realistic knowledge base** (11 markdown documents, ~14,000 words) rather than scraped documents. This was a deliberate decision: a controlled corpus let me (a) plant exact-match content — error codes (LL-E45, PAY-204), policy IDs (POL-RET-02), SKUs (LL-B200) — that makes the dense-vs-sparse comparison *demonstrable*, (b) design eval questions with known ground truth, including questions the KB genuinely cannot answer, and (c) publish everything to GitHub without licensing concerns. I evaluated the public Bitext customer-support dataset (26.8k Q&A pairs) first, but its templated placeholder entities (`{{Order Number}}`) make it training data, not a knowledge base.

The evaluation dataset is `data/eval_queries.json`: 20 questions across 7 categories (direct, exact-code, paraphrase, multi-doc, ambiguous, unanswerable, implicit-negative), each labeled with expected source documents and expected escalation behavior.

## 3. How I used AI coding tools (Claude Cowork) and key prompts

I built this working conversationally with Claude in Cowork mode — Claude wrote the code, I ran everything locally, configured keys, tested, and made the product decisions. Representative prompts from the workflow:

- *"Review the project handout and come up with a plan... I'm not a strong developer but can understand concepts, debug, and configure keys. I need to finish in 2 days."* → produced the project plan and division of labor.
- *"Which keys do you need on priority?"* → Pinecone → OpenAI → Nebius, and the .env scaffold.
- *"What chunking strategy are we adopting and what are the pros and cons?"* → led to documenting the structure-aware recursive splitting decision.
- *"Create a system design markdown file and capture the design decisions... we will continue to add more."* → SYSTEM_DESIGN.md with the DD-01…DD-07 decision log.
- *"Add inline comments and detailed explanations... I'm most interested in where you make key decisions or set parameters that influence outcomes."* → fully annotated codebase, used as my code-review/learning pass.
- *"Explain a) what is top score and how is it calculated b) what is faithfulness and why is it blank for many rows?"* → diagnosed the first eval run **before** tuning anything.
- *"Let's set the confidence threshold to 0.22 and re-run the eval."* → the calibration iteration.
- *"When I asked whether you ship to Australia... I was expecting it to escalate."* → surfaced the inferred-negative design question; I decided on a grounded "no" over escalation.

## 4. Iterations tried (and what each one taught)

**Iteration 1 — chunk size.** Started at 2000 chars: only 13 chunks, too coarse (whole documents in 1–2 chunks). Reduced to 1000/150 → 27 section-aligned chunks. *Lesson: chunk size is a retrieval-granularity decision, not a storage decision.*

**Iteration 2 — the confidence gate (the big one).** Initial threshold 0.45 was an intuition-based guess. Eval run #1: retrieval was perfect (100% hit@5 in all modes) yet first-contact resolution was **29%** — the gate refused 12 of 17 answerable questions before generation. Score-distribution analysis showed answerable questions scored 0.229–0.65 and clearly-unanswerable 0.09–0.21 (fused dotproduct scores are not normalized — 0.45 is not "45% confident"). Recalibrated to 0.22 → final run: **100% resolution, 4.83/5 avg faithfulness, 100% citations, 100% escalation recall, 2.7s latency.** *Lesson: every threshold must be set from measured distributions, not intuition.*

**Iteration 3 — the eval itself needed fixing.** (a) hit@5 saturated at 100% everywhere on a 27-chunk corpus — added strict **hit@1** ("did the right doc *win*, not just make the shortlist") to differentiate retrieval modes. (b) Run #2 exposed a labeling bug: "Do you ship to Australia?" was labeled *unanswerable*, but the bot correctly inferred a grounded "no" from the policy's closed list (US/CA/UK) — re-labeled to a new *implicit-negative* category. *Lesson: evals test your assumptions, not just your system.*

**Iteration 4 — answer tone.** The grounded "no" came out as reasoning narration ("There is no mention of shipping to Australia, so the answer is no"). Added a prompt rule: answer closed-list negatives naturally, never narrate ("We currently ship only to the US, Canada, and the UK"). I explicitly chose first-contact resolution over forced escalation for inferred negatives — a product decision, documented in the design log.

**Iteration 5 — environment debugging.** `uvicorn` failed with ModuleNotFoundError despite a working venv — bare `uvicorn` resolved to Anaconda's copy instead of the venv's. Fix: always `python -m uvicorn`. *Lesson: run module commands through the interpreter you mean.*

## 5. Learnings and observations

1. **The model was never the problem.** Llama-3.3-70B was faithful (100% ≥4/5) whenever retrieval and gating did their jobs. All failures in two eval runs came from one miscalibrated threshold and one mislabeled test question — exactly the handout's claim that RAG fails at chunking, retrieval, and evaluation.
2. **Hybrid retrieval is insurance, and the eval proved you can't predict which pure mode fails.** Textbook says sparse collapses on paraphrases; on my corpus it was pure *dense* that slipped (50% hit@1 on multi-doc queries) while sparse held — because an authored support corpus is BM25's best case (distinctive vocabulary everywhere). Hybrid was the only mode at 100% hit@1. The honest claim isn't "hybrid is always better at X"; it's that each pure mode has an unpredictable failure mode and fusion hedges both.
3. **Design the refusal first.** The two-stage escalation (score gate before generation; model ESCALATE token after reading context) catches different failure types — the gate catches *retrieval* misses cheaply, the model catches *semantic* insufficiency the gate can't see. The "Australia" case proved a third category exists: questions answerable only by inference, which are a product decision, not an engineering one.
4. **Un-normalized scores are a trap.** Dotproduct similarity has no intuitive scale; treating 0.45 as "45%" silently destroyed the system's usefulness while every component worked perfectly.
5. **LLM-as-judge works but needs honesty about bias.** Judge = generator model (self-preference risk); strict rubric at temperature 0 plus manual review of flagged rows is the mitigation. A different judge model is the production fix.
6. **A decision log beats a perfect memory.** Capturing every choice (DD-01…DD-07) with rationale, trade-offs, and calibration history in SYSTEM_DESIGN.md as we worked meant the documentation, demo script, and code comments never drifted apart.

## 6. Deliverables

- **Code:** GitHub repository (link in submission form) — fully commented for review
- **Evaluation:** EVALUATION_REPORT.md (auto-generated) + eval_results.json (raw)
- **Design:** SYSTEM_DESIGN.md (architecture diagram + decision log DD-01…DD-07)
- **Bonus:** FastAPI + single-page chat UI showing citations, retrieval score, latency, and visually-distinct escalations
- **Demo video:** ≤5 min, script in DEMO_TALK_TRACK.md
