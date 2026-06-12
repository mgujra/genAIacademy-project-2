# Demo Talk Track — 5-Minute Video

> Timings assume ~140 words/min. Practice once with a timer; cut from the
> architecture section first if running long — the evaluation story is the differentiator.

---

## 0:00–0:30 · The one-liner and the problem

**Show:** title slide or SYSTEM_DESIGN.md header.

> "My RAG app helps support agents and customers answer product and troubleshooting questions from a customer-support knowledge base — 11 documents covering policies, product manuals, error codes, and past tickets — in a chat interface, with a hard rule: never hallucinate. If the knowledge base can't answer, the bot must say so and escalate to a human. I built this with LangChain and LangGraph, OpenAI embeddings, Pinecone, and Llama-3.3-70B running on Nebius Token Factory."

## 0:30–1:15 · Architecture in 45 seconds

**Show:** the Mermaid diagram in SYSTEM_DESIGN.md.

> "Documents are chunked along their section headings — about a thousand characters each — and every chunk is stored in Pinecone twice over: a dense embedding that captures *meaning*, and a sparse BM25 vector that captures *exact keywords*. That's hybrid search. Why both? A customer who types 'error LL-E45' needs exact keyword matching — embeddings blur error codes together. A customer who types 'my light keeps blinking' needs semantic matching — there's no keyword overlap with the manual's word, 'flicker.' At query time I fuse the two scores, sixty-forty in favor of dense.
>
> Before any answer is generated, a confidence gate checks the best retrieval score. Too low means the KB probably doesn't cover it — we escalate to a human *instead of* generating. And even past the gate, the model itself can return an ESCALATE token if the retrieved context doesn't truly answer. Refusal was designed before the happy path."

## 1:15–2:15 · Live demo

**Show:** terminal or chat UI. Run three queries live:

1. `What does error LL-E45 mean?` — *point out the citation* "[troubleshooting-guide.md]"
2. `My light keeps blinking on and off` — "no shared keywords with the docs, semantic search finds 'dimmer interference' anyway"
3. `What is your CEO's salary?` — "score 0.09, gate catches it, instant honest handoff — no hallucination"

## 2:15–4:15 · The evaluation story (the heart of the demo)

**Show:** EVALUATION_REPORT.md.

> "Instead of eyeballing a few queries, I built a 20-question evaluation set where every question is labeled with the document that should answer it, and whether the bot should escalate. The questions are organized into categories that each probe a *different failure mode* of RAG: **direct** questions as a baseline; **exact-code** questions like error numbers, where keyword search shines and embeddings blur; **paraphrase** questions where the customer's words share no vocabulary with the docs — that's embedding territory, and where keyword search collapses; **multi-doc** questions whose answer spans two documents; **ambiguous** one-liners like 'my device isn't working'; and **unanswerable** questions, where the only correct answer is refusing to answer.
>
> I measure retrieval two ways. **hit@5 asks: did the right document make the shortlist? hit@1 asks: did it win?** Think of a search engine — if the page you wanted shows up fifth, the engine technically found it, but you wouldn't call that a great result. On a small corpus almost everything passes hit@5, so hit@1 is where the three retrieval modes — hybrid, pure dense, pure sparse — actually separate, category by category. Sparse drops on paraphrases, dense slips on exact codes, hybrid holds up on both. That table is the empirical argument for hybrid search.
>
> And the evaluation caught a real bug. My first run scored 100% on retrieval but only **29% first-contact resolution** — the bot escalated twelve answerable questions. The culprit was one number: the confidence threshold. I'd guessed 0.45, but fused dot-product scores aren't percentages — they're not normalized — and real answerable questions scored as low as 0.23. The gate was slicing right through the middle of the good-question range. I recalibrated the threshold to 0.22 using the measured score distributions — answerable questions clustered at 0.23 to 0.65, garbage questions at 0.09 to 0.21 — and re-ran: **100% first-contact resolution, 100% of answers rated faithful by an LLM judge, 100% citation rate, 3.4-second average latency.** That threshold is the single most important number in the system: too high and you refuse customers you could help; too low and you generate from weak context — which is exactly how RAG hallucinates. You can only set it from measured data, not intuition.
>
> The re-run also exposed a labeling bug in my own test set: 'Do you ship to Australia?' — I'd marked it unanswerable, but the bot answered it correctly by *inference*: the KB says we ship to the US, Canada, and the UK, so a grounded 'no' is better than a handoff. The eval didn't just test the bot; it tested my assumptions."

## 4:15–5:00 · Close

**Show:** SYSTEM_DESIGN.md decision log, then the repo.

> "Every design decision — chunking strategy, embedding model, the hybrid weighting, the gate, the eval methodology — is logged in the system design doc with its rationale, trade-offs, and when to revisit it, including the full calibration history of the threshold. What I'd do next: a cross-encoder reranker if the corpus grows, and refitting BM25 automatically on corpus changes. The big lesson from this week: **the model was never the problem — retrieval quality and evaluation discipline were where all the actual work happened.**"

---

## Cheat sheet — numbers to have ready

| Stat | Value |
|---|---|
| Corpus | 11 docs → 27 chunks |
| Embeddings | text-embedding-3-small, 1536-dim |
| Fusion | α = 0.6 dense / 0.4 sparse |
| Gate | 0.45 → **0.22** (calibrated from run #1 data) |
| Run #1 → #2 first-contact resolution | 29% → **100%** |
| Faithfulness (LLM judge) | 4.82/5 avg, 100% ≥ 4 |
| Citation rate | 100% |
| Avg latency | 3.4s (budget: <5s) |
| Eval categories | direct · exact-code · paraphrase · multi-doc · ambiguous · unanswerable · implicit-negative |

## Likely instructor questions

- *Why is the judge the same model as the generator?* Known limitation (self-preference bias) — mitigated by strict rubric at temp 0 and manual review of flagged rows; a different judge model is the production fix.
- *Why not cosine similarity?* Hybrid dense+sparse in Pinecone requires dotproduct; that's also why scores aren't normalized and the threshold had to be calibrated empirically.
- *Why α=0.6?* Most real support queries are paraphrases (favor dense), but error codes need a strong sparse channel. The eval's per-category hit@1 validates the choice.
- *What breaks at scale?* BM25 params are corpus-frozen (refit on every corpus change), and a flat threshold may need per-category tuning as score distributions shift.
