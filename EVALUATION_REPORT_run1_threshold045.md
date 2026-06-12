# Evaluation Report — LumenLeaf Support RAG

20 queries across 6 categories · hybrid α=0.6 · top-k=5 · gate=0.45

## Headline metrics

| Metric | Value |
|---|---|
| First-contact resolution (answerable Qs answered) | 29% |
| Escalation recall (unanswerable Qs refused) | 100% |
| Citation rate in generated answers | 100% |
| Faithfulness (avg judge score, 1–5) | 5.00 |
| Faithfulness ≥4 ("% faithful") | 100% |
| Avg end-to-end latency | 1.5s |

## Retrieval: hybrid vs pure dense vs pure sparse (hit@5)

| Mode | Overall | ambiguous | direct | exact-code | multi-doc | paraphrase |
|---|---|---|---|---|---|---|
| hybrid (α=0.6) | 100% | 100% | 100% | 100% | 100% | 100% |
| dense (α=1.0) | 100% | 100% | 100% | 100% | 100% | 100% |
| sparse (α=0.0) | 100% | 100% | 100% | 100% | 100% | 100% |

## Per-query results (end-to-end)

| # | Category | Escalated (expected) | Top score | Faith. | Latency |
|---|---|---|---|---|---|
| 1 | direct | True (False) ❌ | 0.426 | — | 0.18s |
| 2 | direct | False (False) ✅ | 0.627 | 5 | 2.98s |
| 3 | direct | True (False) ❌ | 0.385 | — | 0.23s |
| 4 | direct | False (False) ✅ | 0.65 | 5 | 5.47s |
| 5 | direct | False (False) ✅ | 0.483 | 5 | 7.07s |
| 6 | exact-code | True (False) ❌ | 0.367 | — | 0.25s |
| 7 | exact-code | True (False) ❌ | 0.43 | — | 0.24s |
| 8 | exact-code | False (False) ✅ | 0.625 | 5 | 5.57s |
| 9 | exact-code | False (False) ✅ | 0.455 | 5 | 4.2s |
| 10 | paraphrase | True (False) ❌ | 0.293 | — | 0.37s |
| 11 | paraphrase | True (False) ❌ | 0.383 | — | 0.2s |
| 12 | paraphrase | True (False) ❌ | 0.229 | — | 0.16s |
| 13 | paraphrase | True (False) ❌ | 0.344 | — | 0.51s |
| 14 | multi-doc | True (False) ❌ | 0.372 | — | 0.25s |
| 15 | multi-doc | True (False) ❌ | 0.374 | — | 0.19s |
| 16 | ambiguous | True (False) ❌ | 0.377 | — | 0.29s |
| 17 | ambiguous | True (False) ❌ | 0.386 | — | 0.21s |
| 18 | unanswerable | True (True) ✅ | 0.331 | — | 0.3s |
| 19 | unanswerable | True (True) ✅ | 0.114 | — | 0.23s |
| 20 | unanswerable | True (True) ✅ | 0.21 | — | 0.29s |

## Failure analysis

_Auto-flagged rows (escalation mismatch, faithfulness <4, or judge issue). Add manual commentary below._

- **Q1** (direct): "How long do refunds take to process?" — escalated=True expected=False, faith=—, issue: escalation mismatch
- **Q3** (direct): "How do I enable two-factor authentication?" — escalated=True expected=False, faith=—, issue: escalation mismatch
- **Q6** (exact-code): "What does error LL-E45 mean?" — escalated=True expected=False, faith=—, issue: escalation mismatch
- **Q7** (exact-code): "I'm getting error PAY-204 at checkout" — escalated=True expected=False, faith=—, issue: escalation mismatch
- **Q10** (paraphrase): "My light keeps blinking on and off, what is wrong with it?" — escalated=True expected=False, faith=—, issue: escalation mismatch
- **Q11** (paraphrase): "The app cannot find my device even though it is plugged in and powered" — escalated=True expected=False, faith=—, issue: escalation mismatch
- **Q12** (paraphrase): "Tracking says my package arrived but I never got it" — escalated=True expected=False, faith=—, issue: escalation mismatch
- **Q13** (paraphrase): "Can I stop my kids from turning the smart plug on and off?" — escalated=True expected=False, faith=—, issue: escalation mismatch
- **Q14** (multi-doc): "I want to return a defective bulb. Do I pay for the return label, and can I get a replacement instead of a refund?" — escalated=True expected=False, faith=—, issue: escalation mismatch
- **Q15** (multi-doc): "If my internet goes down, will my bulbs still work and will schedules still run?" — escalated=True expected=False, faith=—, issue: escalation mismatch
- **Q16** (ambiguous): "My device isn't working" — escalated=True expected=False, faith=—, issue: escalation mismatch
- **Q17** (ambiguous): "How much does shipping cost?" — escalated=True expected=False, faith=—, issue: escalation mismatch