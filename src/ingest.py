"""Ingest pipeline: load corpus -> clean -> chunk -> embed (dense + sparse) -> Pinecone.

This is the "build the library" half of RAG. It runs once (or whenever the
corpus changes). The query-time half lives in retriever.py / rag_graph.py.

Run:  python -m src.ingest
"""
import re
import time

from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

from src import config


def load_and_clean() -> list[dict]:
    """Load markdown docs and apply light cleaning.

    Cleaning is deliberately minimal because the corpus is already clean
    markdown. With scraped HTML/PDFs this step would do much more
    (strip nav boilerplate, decode entities, fix encoding).
    Over-cleaning is a real risk too: lowercasing or stripping punctuation
    here would hurt BM25, which relies on exact tokens like "LL-E45".
    """
    docs = []
    # sorted() makes ingestion order deterministic -> stable chunk IDs across runs.
    for path in sorted(config.DATA_DIR.glob("corpus/*.md")):
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"\n{3,}", "\n\n", text)   # 3+ blank lines -> one blank line
        text = re.sub(r"[ \t]+", " ", text)      # runs of spaces/tabs -> single space
        # NOTE: we preserve newlines -- the chunker splits on them (headings, paragraphs).
        docs.append({"source": path.name, "text": text.strip()})
    print(f"Loaded {len(docs)} documents")
    return docs


def chunk(docs: list[dict]) -> list[dict]:
    """[DD-01] Split docs into retrieval units.

    "Recursive" = try the first separator; any piece still too big gets
    re-split with the next separator, and so on. Our priority order means:
    prefer breaking at section headings (a chunk == one policy/procedure),
    fall back to paragraphs, then lines, then words only if forced.
    This is what makes the chunking "structure-aware" rather than blind
    fixed-size slicing.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,        # max size; real chunks are often smaller
        chunk_overlap=config.CHUNK_OVERLAP,  # shared chars between neighbors
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],  # best -> worst break points
    )
    chunks = []
    for doc in docs:
        for i, piece in enumerate(splitter.split_text(doc["text"])):
            chunks.append({
                # Human-readable, deterministic ID, e.g. "warranty-policy.md#2".
                # Re-running ingest produces the SAME IDs, so upserts overwrite
                # instead of duplicating (idempotent ingestion).
                "id": f"{doc['source']}#{i}",
                "text": piece,
                # Source filename rides along as metadata -> enables citations
                # in answers, and lets a chunk "know" which doc it came from.
                "source": doc["source"],
            })
    print(f"Created {len(chunks)} chunks")
    return chunks


def embed_dense(texts: list[str]) -> list[list[float]]:
    """[DD-02] Dense vectors: one 1536-dim embedding per chunk via OpenAI.

    Dense embeddings encode MEANING -- "bulb keeps blinking" lands near
    "dimmer interference flicker" even with zero shared words.
    """
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    out = []
    # Batch 100 texts per API call: fewer round-trips, and comfortably under
    # the API's per-request input limits. (27 chunks = a single call here,
    # but the loop makes this corpus-size-proof.)
    for i in range(0, len(texts), 100):
        batch = texts[i : i + 100]
        resp = client.embeddings.create(model=config.EMBED_MODEL, input=batch)
        out.extend([d.embedding for d in resp.data])
    return out


def get_index(pc: Pinecone):
    """Create the Pinecone index on first run, or connect to the existing one."""
    existing = [i.name for i in pc.list_indexes()]
    if config.INDEX_NAME not in existing:
        print(f"Creating index '{config.INDEX_NAME}' ...")
        pc.create_index(
            name=config.INDEX_NAME,
            dimension=config.EMBED_DIM,
            # [DD-03] KEY CONSTRAINT: hybrid (dense+sparse in one query) only
            # works with the dotproduct metric. Cosine would normalize away
            # the magnitude information that the sparse fusion relies on.
            metric="dotproduct",
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )
        # Index creation is async -- poll until it's ready to accept vectors.
        while not pc.describe_index(config.INDEX_NAME).status["ready"]:
            time.sleep(2)
    return pc.Index(config.INDEX_NAME)


def main():
    docs = load_and_clean()
    chunks = chunk(docs)
    texts = [c["text"] for c in chunks]

    # ---- Sparse vectors [DD-03] ----
    # BM25 scores exact-word matches, weighted by how rare a word is in the
    # corpus ("LL-E45" is rare -> huge weight; "the" is everywhere -> ~zero).
    # fit() learns those corpus statistics; they MUST be identical at query
    # time, so we dump() them to disk and retriever.py load()s the same file.
    bm25 = BM25Encoder()
    bm25.fit(texts)
    bm25.dump(str(config.BM25_PARAMS_PATH))
    # A sparse vector is just {word_id: weight} for the words actually present --
    # thousands of dims conceptually, but only ~dozens of non-zeros per chunk.
    sparse_vecs = bm25.encode_documents(texts)
    print("BM25 sparse vectors encoded")

    # ---- Dense vectors [DD-02] ----
    dense_vecs = embed_dense(texts)
    print("Dense embeddings created")

    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    index = get_index(pc)

    # Each record carries BOTH representations of the same chunk. That's the
    # whole hybrid trick: one stored chunk, two ways to match it.
    vectors = [
        {
            "id": c["id"],
            "values": dense,              # 1536 floats (semantic)
            "sparse_values": sparse,      # {indices, values} (keyword)
            # The chunk TEXT itself is stored as metadata -- Pinecone returns it
            # with each match, so answering needs no second lookup elsewhere.
            "metadata": {"text": c["text"], "source": c["source"]},
        }
        for c, dense, sparse in zip(chunks, dense_vecs, sparse_vecs)
    ]
    # Upsert in batches of 50 to stay under Pinecone's per-request size cap.
    # "Upsert" = insert or overwrite by ID -> safe to re-run ingestion anytime.
    for i in range(0, len(vectors), 50):
        index.upsert(vectors=vectors[i : i + 50])
    print(f"Upserted {len(vectors)} vectors to '{config.INDEX_NAME}'")
    print(index.describe_index_stats())


if __name__ == "__main__":
    main()
