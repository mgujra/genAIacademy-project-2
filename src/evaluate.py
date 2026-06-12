"""Evaluation harness: 20 queries -> retrieval comparison + end-to-end metrics + report.

Three measurements (DD-06):
  1. RETRIEVAL QUALITY, hit@5 per mode: the same 20 questions are retrieved
     three ways -- hybrid (alpha=0.6), pure dense (1.0), pure sparse (0.0) --
     and we check whether an expected source file appears in the top-5.
     This isolates WHERE hybrid search wins (exact codes vs paraphrases).
  2. END-TO-END BEHAVIOR: full RAG flow per question. Measures first-contact
     resolution (answered when answerable), escalation precision/recall
     (escalated exactly when it should), and citation presence.
  3. FAITHFULNESS, LLM-as-judge: a second Nebius call re-reads each generated
     answer NEXT TO the retrieved context and scores grounding 1-5. We use a
     judge because faithfulness is about meaning, not string overlap. Judge
     uses temperature 0 and a rubric to keep scores stable.

Run:  python -m src.evaluate          (full: retrieval + e2e + judge + report)
      python -m src.evaluate --retrieval-only   (fast, no generation calls)
"""
import json
import sys
import time

from openai import OpenAI

from src import config
from src.rag_graph import answer_question
from src.retriever import retrieve

EVAL_PATH = config.DATA_DIR / "eval_queries.json"
RESULTS_PATH = config.DATA_DIR / "eval_results.json"
REPORT_PATH = config.ROOT / "EVALUATION_REPORT.md"

# The three retrieval modes we compare. Keys become report column names.
MODES = {"hybrid (α=0.6)": 0.6, "dense (α=1.0)": 1.0, "sparse (α=0.0)": 0.0}

JUDGE_PROMPT = """You are a strict evaluator. Given a customer support ANSWER and the CONTEXT chunks it was generated from, score how faithful the answer is to the context.

Rubric:
5 = every claim is directly supported by the context
4 = all claims supported; minor paraphrase liberties
3 = mostly supported; one unsupported but harmless claim
2 = contains a claim that contradicts or is absent from the context
1 = substantially fabricated

Reply with ONLY a JSON object: {"score": <1-5>, "issue": "<one short sentence, or 'none'>"}"""


def eval_retrieval(queries: list[dict]) -> dict:
    """Measurement 1: hit@5 for each retrieval mode.

    'Hit' = at least one expected source file appears among the top-5 chunks.
    Unanswerable queries are skipped here (they have no expected source).
    """
    rows = []
    for q in queries:
        if not q["expected_sources"]:
            continue
        row = {"id": q["id"], "category": q["category"], "question": q["question"]}
        for mode, alpha in MODES.items():
            docs = retrieve(q["question"], alpha=alpha)
            # hit@5: expected doc anywhere in top-5 (lenient recall proxy).
            hit = any(d["source"] in q["expected_sources"] for d in docs)
            # hit@1: the #1 ranked chunk comes from an expected doc. Much
            # stricter -- this is what differentiates retrieval modes on a
            # small corpus where everything lands somewhere in the top-5.
            # Rank matters: the generator reads chunks in order, and a wrong
            # #1 means the best context arrives diluted.
            hit1 = bool(docs) and docs[0]["source"] in q["expected_sources"]
            row[mode] = {"hit": hit, "hit1": hit1,
                         "top_score": docs[0]["score"] if docs else 0.0,
                         "top_source": docs[0]["source"] if docs else None}
        rows.append(row)
        print(f"  retrieval q{q['id']:>2} | " + " | ".join(
            f"{m.split()[0]}: {'@1' if row[m]['hit1'] else ('@5' if row[m]['hit'] else 'MISS')}" for m in MODES))
    return {"rows": rows}


def judge_faithfulness(client: OpenAI, question: str, answer: str, docs: list[dict]) -> dict:
    """Measurement 3: LLM-as-judge grounding score for one answer."""
    context = "\n\n---\n\n".join(f"[{d['source']}]\n{d['text']}" for d in docs)
    resp = client.chat.completions.create(
        model=config.GEN_MODEL,
        temperature=0.0,  # judging must be repeatable, not creative
        max_tokens=120,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:\n{answer}"},
        ],
    )
    raw = resp.choices[0].message.content.strip()
    try:
        # Models sometimes wrap JSON in ```json fences -- strip before parsing.
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"score": None, "issue": f"judge returned unparseable: {raw[:80]}"}


def eval_end_to_end(queries: list[dict]) -> dict:
    """Measurement 2 (+3): full RAG flow per query, then judge the answers."""
    judge_client = OpenAI(api_key=config.NEBIUS_API_KEY, base_url=config.NEBIUS_BASE_URL)
    rows = []
    for q in queries:
        t0 = time.time()
        result = answer_question(q["question"])
        latency = time.time() - t0

        row = {
            "id": q["id"], "category": q["category"], "question": q["question"],
            "should_escalate": q["should_escalate"],
            "escalated": result["escalated"],
            "top_score": round(result.get("top_score", 0.0), 3),
            "latency_s": round(latency, 2),
            "answer": result["answer"],
            "sources": result.get("sources", []),
            # Escalation correctness: did behavior match expectation?
            "escalation_correct": result["escalated"] == q["should_escalate"],
            # Citation check only applies to actual generated answers.
            "has_citation": ("[" in result["answer"] and ".md]" in result["answer"]) if not result["escalated"] else None,
        }
        # Judge only generated answers -- an escalation has no claims to verify.
        if not result["escalated"]:
            row["faithfulness"] = judge_faithfulness(judge_client, q["question"], result["answer"], result["docs"])
        rows.append(row)
        print(f"  e2e q{q['id']:>2} | escalated={result['escalated']!s:<5} "
              f"expected={q['should_escalate']!s:<5} | {latency:.1f}s "
              f"| faith={row.get('faithfulness', {}).get('score', '—')}")
    return {"rows": rows}


def summarize(retrieval: dict, e2e: dict) -> dict:
    """Aggregate raw rows into the headline metrics for the report."""
    s = {}
    # hit@5 per mode, overall and per category -- the hybrid-vs-pure story.
    rrows = retrieval["rows"]
    for mode in MODES:
        s[f"hit@5 {mode}"] = sum(r[mode]["hit"] for r in rrows) / len(rrows)
        s[f"hit@1 {mode}"] = sum(r[mode]["hit1"] for r in rrows) / len(rrows)
    cats = sorted({r["category"] for r in rrows})
    # Per-category cells hold BOTH metrics: {"hit5": x, "hit1": y}.
    s["by_category"] = {
        c: {m: {"hit5": sum(r[m]["hit"] for r in rrows if r["category"] == c)
                  / sum(1 for r in rrows if r["category"] == c),
                "hit1": sum(r[m]["hit1"] for r in rrows if r["category"] == c)
                  / sum(1 for r in rrows if r["category"] == c)} for m in MODES}
        for c in cats
    }
    erows = e2e["rows"]
    answerable = [r for r in erows if not r["should_escalate"]]
    unanswerable = [r for r in erows if r["should_escalate"]]
    answered = [r for r in answerable if not r["escalated"]]
    # First-contact resolution: answerable questions actually answered
    # (the support-industry metric named in the project handout).
    s["first_contact_resolution"] = len(answered) / len(answerable)
    # Escalation recall: unanswerable questions correctly refused.
    s["escalation_recall"] = sum(r["escalated"] for r in unanswerable) / len(unanswerable)
    s["citation_rate"] = (sum(bool(r["has_citation"]) for r in answered) / len(answered)) if answered else 0.0
    scores = [r["faithfulness"]["score"] for r in answered
              if r.get("faithfulness", {}).get("score") is not None]
    s["faithfulness_avg"] = sum(scores) / len(scores) if scores else None
    # % of judged answers scoring >= 4 -- maps to the one-liner's "% faithfulness".
    s["faithfulness_pct_4plus"] = (sum(sc >= 4 for sc in scores) / len(scores)) if scores else None
    s["avg_latency_s"] = sum(r["latency_s"] for r in erows) / len(erows)
    return s


def write_report(summary: dict, retrieval: dict, e2e: dict):
    """Render EVALUATION_REPORT.md -- the submission deliverable."""
    pct = lambda v: f"{v*100:.0f}%"
    lines = [
        "# Evaluation Report — LumenLeaf Support RAG",
        "",
        f"20 queries across 6 categories · hybrid α={config.HYBRID_ALPHA} · top-k={config.TOP_K} · gate={config.CONFIDENCE_THRESHOLD}",
        "",
        "## Headline metrics",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| First-contact resolution (answerable Qs answered) | {pct(summary['first_contact_resolution'])} |",
        f"| Escalation recall (unanswerable Qs refused) | {pct(summary['escalation_recall'])} |",
        f"| Citation rate in generated answers | {pct(summary['citation_rate'])} |",
        f"| Faithfulness (avg judge score, 1–5) | {summary['faithfulness_avg']:.2f} |" if summary["faithfulness_avg"] else "| Faithfulness | n/a |",
        f"| Faithfulness ≥4 (\"% faithful\") | {pct(summary['faithfulness_pct_4plus'])} |" if summary["faithfulness_pct_4plus"] is not None else "",
        f"| Avg end-to-end latency | {summary['avg_latency_s']:.1f}s |",
        "",
        "## Retrieval: hybrid vs pure dense vs pure sparse",
        "",
        "_Each cell: hit@1 / hit@5. hit@1 = best-ranked chunk is from the expected doc (strict); hit@5 = expected doc anywhere in top-5 (lenient)._",
        "",
        "| Mode | Overall |" + "".join(f" {c} |" for c in summary["by_category"]),
        "|---|---|" + "---|" * len(summary["by_category"]),
    ]
    for mode in MODES:
        row = f"| {mode} | {pct(summary[f'hit@1 {mode}'])} / {pct(summary[f'hit@5 {mode}'])} |"
        for c in summary["by_category"]:
            cell = summary["by_category"][c][mode]
            row += f" {pct(cell['hit1'])} / {pct(cell['hit5'])} |"
        lines.append(row)
    lines += [
        "",
        "## Per-query results (end-to-end)",
        "",
        "| # | Category | Escalated (expected) | Top score | Faith. | Latency |",
        "|---|---|---|---|---|---|",
    ]
    for r in e2e["rows"]:
        faith = r.get("faithfulness", {}).get("score", "—")
        ok = "✅" if r["escalation_correct"] else "❌"
        lines.append(f"| {r['id']} | {r['category']} | {r['escalated']} ({r['should_escalate']}) {ok} "
                     f"| {r['top_score']} | {faith} | {r['latency_s']}s |")
    lines += [
        "",
        "## Failure analysis",
        "",
        "_Auto-flagged rows (escalation mismatch, faithfulness <4, or judge issue). Add manual commentary below._",
        "",
    ]
    for r in e2e["rows"]:
        f = r.get("faithfulness", {})
        if (not r["escalation_correct"]) or (f.get("score") is not None and f["score"] < 4):
            lines.append(f"- **Q{r['id']}** ({r['category']}): \"{r['question']}\" — "
                         f"escalated={r['escalated']} expected={r['should_escalate']}, "
                         f"faith={f.get('score', '—')}, issue: {f.get('issue', 'escalation mismatch')}")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {REPORT_PATH.name}")


def main():
    queries = json.loads(EVAL_PATH.read_text())
    print(f"Loaded {len(queries)} eval queries\n[1/3] Retrieval comparison (3 modes x {len(queries)} queries)...")
    retrieval = eval_retrieval(queries)

    if "--retrieval-only" in sys.argv:
        print(json.dumps({m: sum(r[m]["hit"] for r in retrieval["rows"]) for m in MODES}, indent=2))
        return

    print("\n[2/3] End-to-end RAG + faithfulness judging...")
    e2e = eval_end_to_end(queries)

    print("\n[3/3] Summarizing...")
    summary = summarize(retrieval, e2e)
    RESULTS_PATH.write_text(json.dumps(
        {"summary": {k: v for k, v in summary.items() if k != "by_category"},
         "by_category": summary["by_category"],
         "retrieval": retrieval, "end_to_end": e2e}, indent=2), encoding="utf-8")
    print(f"Raw results -> {RESULTS_PATH.name}")
    write_report(summary, retrieval, e2e)


if __name__ == "__main__":
    main()
