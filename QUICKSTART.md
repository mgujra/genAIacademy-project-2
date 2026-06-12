# Quickstart — (Mac)

Open Terminal, then:

```bash
cd ~/claude/Projects/"Gen AI Academy - Project 2"

# 1. One-time setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Ingest: chunk -> embed -> build Pinecone hybrid index (run once)
python -m src.ingest

# 3. Smoke-test the full RAG flow
python -m src.rag_graph "How do I reset my LL-B200 smart bulb?"
python -m src.rag_graph "What does error LL-E45 mean?"
python -m src.rag_graph "Can I get a refund after 45 days?"
python -m src.rag_graph "Do you ship to Australia?"          # should answer from policy (no) or escalate
python -m src.rag_graph "What is the CEO's salary?"          # should ESCALATE
```

Expected ingest output: `Loaded 11 documents` → `Created 27 chunks` → `Upserted 27 vectors to 'support-kb-hybrid'`.

If anything errors, paste the full output back into our chat and I'll fix it.

```bash
# 4. Evaluation: 20 queries, 3 retrieval modes, LLM-judged faithfulness (~2-3 min)
python -m src.evaluate                    # writes EVALUATION_REPORT.md + data/eval_results.json
python -m src.evaluate --retrieval-only   # fast variant: hit@1/hit@5 only, no generation

# 5. Chatbot UI (bonus) — then open http://127.0.0.1:8000
# (python -m ensures uvicorn runs inside the venv, not a global/conda copy)
python -m uvicorn src.app:app --reload
```

Notes:
- First run downloads small NLTK data files automatically (needed by the BM25 sparse encoder).
- Step 2 creates the Pinecone index `support-kb-hybrid` (serverless, aws us-east-1) if missing.
- Re-running ingest is safe; it overwrites the same chunk IDs.

## Status log
- ✅ Ingestion: 11 docs → 27 chunks → Pinecone hybrid index (verified)
- ✅ Smoke test: cited answers + both escalation paths working
- ✅ Eval run #1 (gate=0.45): caught over-escalation — 29% first-contact resolution
- ✅ Gate recalibrated to 0.22 from measured score distributions (see DD-04)
- ✅ Eval run #2: 100% resolution, 4.82/5 faithfulness, 3.4s avg latency
- ✅ Harness upgraded: hit@1 metric added, Q18 re-labeled implicit-negative
- ⬜ Final eval run with hit@1 (rerun `python -m src.evaluate`)
- ⬜ Chatbot UI · README · project documentation · demo recording
