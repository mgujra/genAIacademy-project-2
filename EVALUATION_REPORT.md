# Evaluation Report — LumenLeaf Support RAG

20 queries across 6 categories · hybrid α=0.6 · top-k=5 · gate=0.22

## Headline metrics

| Metric | Value |
|---|---|
| First-contact resolution (answerable Qs answered) | 100% |
| Escalation recall (unanswerable Qs refused) | 100% |
| Citation rate in generated answers | 100% |
| Faithfulness (avg judge score, 1–5) | 4.83 |
| Faithfulness ≥4 ("% faithful") | 100% |
| Avg end-to-end latency | 2.7s |

## Retrieval: hybrid vs pure dense vs pure sparse

_Each cell: hit@1 / hit@5. hit@1 = best-ranked chunk is from the expected doc (strict); hit@5 = expected doc anywhere in top-5 (lenient)._

| Mode | Overall | ambiguous | direct | exact-code | implicit-negative | multi-doc | paraphrase |
|---|---|---|---|---|---|---|---|
| hybrid (α=0.6) | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% |
| dense (α=1.0) | 94% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 50% / 100% | 100% / 100% |
| sparse (α=0.0) | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% | 100% / 100% |

## Per-query results (end-to-end)

| # | Category | Escalated (expected) | Top score | Faith. | Latency |
|---|---|---|---|---|---|
| 1 | direct | False (False) ✅ | 0.426 | 5 | 3.68s |
| 2 | direct | False (False) ✅ | 0.627 | 5 | 1.54s |
| 3 | direct | False (False) ✅ | 0.385 | 5 | 2.4s |
| 4 | direct | False (False) ✅ | 0.65 | 5 | 2.07s |
| 5 | direct | False (False) ✅ | 0.483 | 4 | 2.87s |
| 6 | exact-code | False (False) ✅ | 0.367 | 5 | 2.38s |
| 7 | exact-code | False (False) ✅ | 0.43 | 5 | 2.06s |
| 8 | exact-code | False (False) ✅ | 0.625 | 5 | 2.63s |
| 9 | exact-code | False (False) ✅ | 0.455 | 5 | 2.37s |
| 10 | paraphrase | False (False) ✅ | 0.293 | 4 | 4.6s |
| 11 | paraphrase | False (False) ✅ | 0.383 | 4 | 3.19s |
| 12 | paraphrase | False (False) ✅ | 0.229 | 5 | 2.64s |
| 13 | paraphrase | False (False) ✅ | 0.344 | 5 | 8.17s |
| 14 | multi-doc | False (False) ✅ | 0.372 | 5 | 4.27s |
| 15 | multi-doc | False (False) ✅ | 0.374 | 5 | 2.29s |
| 16 | ambiguous | False (False) ✅ | 0.377 | 5 | 3.8s |
| 17 | ambiguous | False (False) ✅ | 0.386 | 5 | 1.63s |
| 18 | implicit-negative | False (False) ✅ | 0.331 | 5 | 1.72s |
| 19 | unanswerable | True (True) ✅ | 0.114 | — | 0.2s |
| 20 | unanswerable | True (True) ✅ | 0.209 | — | 0.33s |

## Failure analysis

_Auto-flagged rows (escalation mismatch, faithfulness <4, or judge issue). Add manual commentary below._
