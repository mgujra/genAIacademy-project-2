"""Hybrid retriever: dense (OpenAI) + sparse (BM25) over Pinecone. [DD-03]

Query-time mirror of ingest.py: the question is encoded BOTH ways
(same embedding model, same BM25 statistics), then Pinecone scores it
against the stored chunks in a single fused query.
"""
from functools import lru_cache

from openai import OpenAI
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder

from src import config


@lru_cache(maxsize=1)
def _clients():
    """Build API clients once and reuse them for every query.

    lru_cache(maxsize=1) on a no-arg function = a simple singleton: the first
    call constructs the clients, every later call returns the same objects.
    Recreating clients (and re-loading BM25 params from disk) per query would
    add latency for zero benefit.
    """
    openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    index = Pinecone(api_key=config.PINECONE_API_KEY).Index(config.INDEX_NAME)
    bm25 = BM25Encoder()
    # CRITICAL: load the SAME corpus statistics that ingest.py fitted.
    # A BM25 encoder fitted on different text would produce incompatible
    # sparse vectors and silently break keyword matching.
    bm25.load(str(config.BM25_PARAMS_PATH))
    return openai_client, index, bm25


def _scale(dense: list[float], sparse: dict, alpha: float):
    """Convex-combination weighting -- the standard Pinecone hybrid pattern.

    Pinecone computes: score = dot(q_dense, d_dense) + dot(q_sparse, d_sparse).
    It has no per-part weight parameter, so we bake the weights into the QUERY
    vectors themselves before sending:
        q_dense  * alpha      -> dense part contributes alpha * (its score)
        q_sparse * (1-alpha)  -> sparse part contributes (1-alpha) * (its score)
    Document vectors stay untouched; only the query is scaled.
    """
    scaled_dense = [v * alpha for v in dense]
    scaled_sparse = {
        "indices": sparse["indices"],
        "values": [v * (1 - alpha) for v in sparse["values"]],
    }
    return scaled_dense, scaled_sparse


def retrieve(query: str, top_k: int = config.TOP_K, alpha: float = config.HYBRID_ALPHA) -> list[dict]:
    """Return top-k chunks with fused relevance scores.

    alpha is exposed as a parameter (not hard-coded) so the evaluation can
    compare retrieval modes on identical questions:
        alpha=1.0 -> pure dense (semantic)
        alpha=0.0 -> pure sparse (keyword/BM25)
        alpha=0.6 -> our production hybrid default
    """
    openai_client, index, bm25 = _clients()

    # Encode the question both ways -- with the same models used at ingest.
    dense = openai_client.embeddings.create(model=config.EMBED_MODEL, input=query).data[0].embedding
    # encode_queries (not encode_documents): BM25 weights query terms
    # differently from document terms -- asymmetry is part of the algorithm.
    sparse = bm25.encode_queries(query)

    if alpha >= 1.0:
        # Pure dense: omit the sparse vector entirely.
        kwargs = {"vector": dense}
    elif alpha <= 0.0:
        # Pure sparse: Pinecone's API *requires* a dense vector in every query,
        # so we send all-zeros -- dot(0, anything) = 0 -> only BM25 contributes.
        kwargs = {"vector": [0.0] * len(dense), "sparse_vector": sparse}
    else:
        # True hybrid: scale both sides per the convex combination.
        d, s = _scale(dense, sparse, alpha)
        kwargs = {"vector": d, "sparse_vector": s}

    # include_metadata=True returns the chunk text + source stored at ingest --
    # this is what the generator will read as context.
    res = index.query(top_k=top_k, include_metadata=True, **kwargs)

    # Flatten Pinecone's response into plain dicts the rest of the app consumes.
    # `score` is the fused dotproduct -- NOT normalized to [0,1] -- which is why
    # the confidence threshold (DD-04) had to be calibrated empirically.
    return [
        {
            "id": m["id"],
            "score": m["score"],
            "text": m["metadata"]["text"],
            "source": m["metadata"]["source"],
        }
        for m in res["matches"]
    ]
