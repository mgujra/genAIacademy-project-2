# Week 2 RAG Project Plan — Customer Support KB with Hybrid Search

## One-liner
My RAG app helps support agents and customers answer product and troubleshooting
questions from a public product-support corpus (FAQs + manuals, ~50–100 docs) in a
web chat UI with ≥90% faithfulness, <5s latency, and confidence-based human escalation.

## Decisions (the Framework)
| Layer | Decision |
|---|---|
| Use case | Project 4: Customer Support KB. Customer asks product/troubleshooting questions in a chat UI. |
| Build track | Track 2: LangChain + LangGraph (Python). |
| Corpus | Public support dataset (FAQs, product manuals, support tickets) — sourced Day 1, committed to repo. |
| Ingestion + cleaning | Local files → loader → strip markup/boilerplate → normalize text. One-shot ingest script; refresh = re-run. |
| Chunking + embedding | RecursiveCharacterTextSplitter ~500 tokens / 50 overlap; OpenAI `text-embedding-3-small` (1536-dim) — matched capacity. |
| Retrieve | Pinecone, **hybrid**: dense (OpenAI embeddings) + sparse (BM25 encoder), top-k=5, weighted fusion. |
| Generate | **Nebius Token Factory** LLM (e.g., Llama-3.3-70B) for answers — satisfies the mandatory Nebius requirement. Cited answers only from retrieved context. |
| Refusal path | LangGraph confidence gate: low retrieval score or low answer confidence → "escalate to human" response, never hallucinate. Designed first. |
| Eval | 20 realistic support queries (incl. ambiguous, multi-doc, unanswerable) → first-contact resolution rate, faithfulness, failure analysis. |

## Architecture
docs → clean → chunk → embed (dense + sparse) → Pinecone hybrid index
→ retriever (top-k fusion) → confidence gate → LLM w/ citations (Nebius) → answer **or** escalate
→ FastAPI backend + single-page chat UI (bonus)

## Day 1 (build the pipeline)
1. **You:** create/confirm API keys — OpenAI, Pinecone, Nebius Token Factory. Put them in `.env`.
2. **Claude:** project scaffold, requirements, corpus sourcing + cleaning.
3. **Claude:** ingest script (chunk → embed → upsert to Pinecone hybrid index). **You:** run it.
4. **Claude:** hybrid retriever + LangGraph flow (retrieve → confidence gate → generate-with-citations / escalate).
5. **Together:** smoke-test with 5 queries; tune top-k / threshold.

## Day 2 (eval, UI, deliverables)
1. **Claude:** 20-query eval harness + metrics (resolution rate, faithfulness scores, failure analysis) → evaluation report doc.
2. **Claude:** chatbot UI (FastAPI + single HTML page) wired to the pipeline. **You:** run + test.
3. **Claude:** README + project documentation doc (overview, dataset, prompts used, iterations, learnings).
4. **You:** push to GitHub, record ≤5-min demo video, submit form.

## Deliverables checklist
- [x] Working hybrid-search support bot with escalation logic (smoke-tested + evaluated)
- [x] 20-query evaluation + resolution metrics document (EVALUATION_REPORT.md; final rerun with hit@1 pending)
- [x] System design doc with decision log (SYSTEM_DESIGN.md — bonus deliverable)
- [x] Demo talk track drafted (DEMO_TALK_TRACK.md — revisit at the end)
- [ ] Project documentation (Google Doc)
- [ ] GitHub repo
- [ ] ≤5-min demo video
- [ ] Bonus: chatbot UI (in progress)
- [x] Nebius Token Factory used for generation (Llama-3.3-70B, verified in smoke test + eval)

## Your action items to unblock Day 1
1. OpenAI API key (embeddings)
2. Pinecone API key (free tier is fine — serverless index)
3. Nebius Token Factory account + API key (https://tokenfactory.nebius.com) — **mandatory per handout**
4. Paste them into `.env` when I scaffold the project
