# LumenLeaf Support RAG — Customer Support KB with Hybrid Search

A customer-support Q&A bot over a product knowledge base, built for The Gen Academy Week 2 project (Use Case 4). It combines **dense semantic search and sparse keyword search** in a single Pinecone hybrid index, generates **cited answers** with Llama-3.3-70B on **Nebius Token Factory**, and **escalates to a human** instead of hallucinating when it isn't confident.

> **One-liner:** My RAG app helps support agents and customers answer product and troubleshooting questions from a customer-support knowledge base (11 docs: policies, manuals, error codes, tickets) in a web chat UI with 100% measured faithfulness, <5s latency, and confidence-based human escalation.

## Results (20-query evaluation)

| Metric | Value |
|---|---|
| First-contact resolution | 100% |
| Faithfulness (LLM judge, % scoring ≥4/5) | 100% (avg 4.82) |
| Citation rate | 100% |
| Escalation recall on unanswerable questions | 100% (after test-set re-label) |
| Avg end-to-end latency | 3.4s |

The first eval run scored only **29% first-contact resolution** — the confidence gate was miscalibrated. See [EVALUATION_REPORT.md](EVALUATION_REPORT.md) and the calibration story in [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) (DD-04).

## Architecture

```
docs → clean → chunk (heading-aware, ~1000 chars) → dense (OpenAI) + sparse (BM25)
     → Pinecone hybrid index (dotproduct)
query → hybrid retrieval (α·dense + (1−α)·sparse, α=0.6, top-k 5)
      → confidence gate (≥0.22 or escalate)
      → Llama-3.3-70B via Nebius — cited answer, or ESCALATE token → human handoff
```

Full design rationale with a decision log (DD-01…DD-07): [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md)

## Stack

LangChain (splitters) · LangGraph (stateful flow with conditional escalation) · OpenAI `text-embedding-3-small` · Pinecone serverless (hybrid) · `pinecone-text` BM25 · **Nebius Token Factory** (`meta-llama/Llama-3.3-70B-Instruct`, generation + LLM-as-judge) · FastAPI + vanilla-JS chat UI

## Run it

See [QUICKSTART.md](QUICKSTART.md). Short version:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env             # fill in your three API keys
python -m src.ingest             # build the hybrid index (one-time)
python -m src.rag_graph "What does error LL-E45 mean?"   # CLI smoke test
python -m src.evaluate           # full evaluation -> EVALUATION_REPORT.md
python -m uvicorn src.app:app --reload    # chat UI at http://127.0.0.1:8000
```

## Repo map

| Path | What it is |
|---|---|
| `src/config.py` | Every tunable (α, top-k, gate threshold, chunk size) with calibration history |
| `src/ingest.py` | Corpus → chunks → dense+sparse vectors → Pinecone |
| `src/retriever.py` | Hybrid retrieval with the α fusion knob |
| `src/rag_graph.py` | LangGraph flow: retrieve → gate → generate/escalate |
| `src/evaluate.py` | 20-query harness: hit@1/hit@5 × 3 modes, faithfulness judge, report generator |
| `src/app.py` + `static/` | FastAPI + chat UI (bonus deliverable) |
| `data/corpus/` | 11-doc knowledge base (authored, realistic) |
| `data/eval_queries.json` | Labeled eval set, 7 categories |
| `SYSTEM_DESIGN.md` | Decision log — the "why" behind every layer |
| `EVALUATION_REPORT.md` | Auto-generated metrics + failure analysis |
| `PROJECT_DOCUMENTATION.md` | Submission write-up (overview, prompts, iterations, learnings) |

## Honest limitations

Corpus is small and authored (controlled vocabulary made the hybrid comparison demonstrable; real scraped docs would stress cleaning more). Judge model = generator model (self-preference risk; mitigated by rubric at temp 0 + manual review). BM25 stats are corpus-frozen — adding docs means re-running ingest. No multi-turn memory in the chat UI.
