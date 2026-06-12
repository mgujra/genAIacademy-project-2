"""LangGraph RAG flow: retrieve -> confidence gate -> generate (cited) | escalate.

This file is the "answer a question" half of RAG, wired as a small state
machine (graph) instead of a straight function call. Why a graph? Because the
flow BRANCHES: weak retrieval must take a different path (escalate) than
strong retrieval (generate). LangGraph makes that branching explicit,
inspectable, and easy to extend (e.g., add a reranker node later).

Generation runs on Nebius Token Factory (mandatory per project handout). [DD-05]

Run a single query:  python -m src.rag_graph "How do I reset my smart bulb?"
"""
import sys
from typing import TypedDict

from langgraph.graph import END, StateGraph
from openai import OpenAI

from src import config
from src.retriever import retrieve

# The system prompt is where FAITHFULNESS is enforced -- the project's primary
# metric. Three load-bearing choices:
#   1. "Answer ONLY using the provided context" -> forbids the model's own
#      world knowledge, the #1 source of plausible-sounding hallucinations.
#   2. Mandatory inline [source.md] citations -> every claim is checkable,
#      and the eval can verify answers against the cited chunk.
#   3. The ESCALATE token -> gives the model a sanctioned way OUT. Without an
#      explicit escape hatch, LLMs answer anyway; with one, they use it.
SYSTEM_PROMPT = """You are LumenLeaf's customer support assistant.
Answer ONLY using the provided context chunks. Rules:
1. Every factual claim must come from the context. Never invent policies, numbers, or steps.
2. Cite sources inline using [source_filename] after each claim, e.g. [returns-and-refunds.md].
3. If the context does not contain the answer, reply exactly: ESCALATE
4. Be concise and friendly. Use steps for procedures.
5. When the context lists a closed set (countries, models, payment methods) and the
   customer asks about something outside it, give a natural, direct "no" — e.g.
   "We currently ship only to the US, Canada, and the UK, so unfortunately not
   Australia." Never narrate your reasoning or say "the context does not mention".
"""

# What the customer sees on either escalation path. Mirrors the handout's
# advice: "I could not find this" beats a confident hallucination.
ESCALATION_MESSAGE = (
    "I'm not confident I can answer that accurately from our knowledge base, "
    "so I'm connecting you with a human support agent. You can also reach us "
    "24/7 via live chat or support@lumenleaf.example."
)


class RAGState(TypedDict, total=False):
    """The shared state dict that flows through the graph.

    Each node receives the current state and returns only the keys it updates;
    LangGraph merges them. total=False = all keys optional, since they're
    filled in progressively (question -> docs -> answer).
    """
    question: str       # input
    docs: list[dict]    # retrieved chunks (id, score, text, source)
    top_score: float    # best fused score -- the confidence signal
    answer: str         # final text shown to the user
    escalated: bool     # did we hand off to a human?
    sources: list[str]  # deduped source files, for the UI/eval


def node_retrieve(state: RAGState) -> RAGState:
    """Node 1: hybrid retrieval (see retriever.py / DD-03)."""
    docs = retrieve(state["question"])
    # Results arrive sorted by fused score, so docs[0] is the best match.
    # Its score becomes our retrieval-confidence proxy for the gate.
    return {"docs": docs, "top_score": docs[0]["score"] if docs else 0.0}


def gate(state: RAGState) -> str:
    """[DD-04] Conditional edge -- escalation path #1 (retrieval failure).

    Returns the NAME of the next node. If even the best chunk scored below
    the threshold, the KB likely doesn't cover this topic: generating from
    weak context is exactly how RAG hallucinates, and we'd pay for a Nebius
    call just to produce something untrustworthy. Refuse early instead.
    """
    if not state["docs"] or state["top_score"] < config.CONFIDENCE_THRESHOLD:
        return "escalate"
    return "generate"


def node_generate(state: RAGState) -> RAGState:
    """Node 2: grounded answer generation via Nebius. [DD-05]"""
    # Nebius speaks the OpenAI wire protocol, so the standard OpenAI client
    # works -- only api_key and base_url change. This one call satisfies the
    # course's "must use Nebius Token Factory" requirement.
    client = OpenAI(api_key=config.NEBIUS_API_KEY, base_url=config.NEBIUS_BASE_URL)

    # Pack retrieved chunks into a labeled context block. Each chunk is
    # prefixed with its [source] so the model can cite, plus its score for
    # transparency. "---" separators keep chunk boundaries unambiguous.
    context = "\n\n---\n\n".join(
        f"[{d['source']}] (score={d['score']:.3f})\n{d['text']}" for d in state["docs"]
    )
    resp = client.chat.completions.create(
        model=config.GEN_MODEL,
        # temperature=0.1: near-deterministic. Support answers need precision
        # and repeatability, not creativity. (Not 0.0 -- a little smoothing
        # avoids degenerate repetition on some models.)
        temperature=0.1,
        # Hard cap on answer length: controls cost/latency and keeps answers
        # support-ticket-sized. 600 tokens =~ 450 words, ample for a procedure.
        max_tokens=600,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nCustomer question: {state['question']}"},
        ],
    )
    answer = resp.choices[0].message.content.strip()

    # Escalation path #2 (generation-time): retrieval scored OK, but the model
    # judged the chunks don't actually answer the question (e.g., question
    # about "Australia shipping" retrieves the shipping doc, which only lists
    # US/CA/UK). The gate can't catch this case -- only the model can.
    if answer.startswith("ESCALATE"):
        return {"answer": ESCALATION_MESSAGE, "escalated": True, "sources": []}

    return {
        "answer": answer,
        "escalated": False,
        # set-comprehension dedupes (5 chunks often span 2-3 files); sorted for
        # stable display order.
        "sources": sorted({d["source"] for d in state["docs"]}),
    }


def node_escalate(state: RAGState) -> RAGState:
    """Node 3: the refusal path -- a fixed, honest handoff message."""
    return {"answer": ESCALATION_MESSAGE, "escalated": True, "sources": []}


def build_graph():
    """Wire the nodes into a graph:

        retrieve --(gate: score >= 0.45)--> generate --> END
                 \\--(gate: score <  0.45)--> escalate --> END
    """
    g = StateGraph(RAGState)
    g.add_node("retrieve", node_retrieve)
    g.add_node("generate", node_generate)
    g.add_node("escalate", node_escalate)
    g.set_entry_point("retrieve")
    # The branch: after retrieve, gate() picks which node runs next.
    g.add_conditional_edges("retrieve", gate, {"generate": "generate", "escalate": "escalate"})
    g.add_edge("generate", END)
    g.add_edge("escalate", END)
    return g.compile()


# Compile the graph once per process, lazily, and reuse it -- same singleton
# idea as _clients() in retriever.py.
_graph = None

def answer_question(question: str) -> RAGState:
    """Public entry point used by the CLI below, the eval, and the chat UI."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    # invoke() runs the graph: state flows retrieve -> (gate) -> generate|escalate.
    return _graph.invoke({"question": question})


if __name__ == "__main__":
    # CLI for smoke-testing: everything after the script name is the question.
    q = " ".join(sys.argv[1:]) or "How do I reset my smart bulb?"
    result = answer_question(q)
    print(f"\nQ: {q}")
    print(f"Escalated: {result['escalated']} | top score: {result.get('top_score', 0):.3f}")
    print(f"\n{result['answer']}")
