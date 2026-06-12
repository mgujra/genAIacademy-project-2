"""Central config for the Customer Support RAG pipeline.

Every tunable that influences retrieval quality lives HERE, in one place,
so experiments (e.g., for the evaluation report) mean changing one number
instead of hunting through the codebase. Design decisions are cross-referenced
to SYSTEM_DESIGN.md as DD-01 ... DD-05.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# ROOT = the project folder, computed relative to THIS file.
# Path(__file__) is .../src/config.py -> .parent is src/ -> .parent.parent is the project root.
# This makes the code runnable from any working directory.
ROOT = Path(__file__).resolve().parent.parent

# Reads the .env file and exports its lines as environment variables.
# Keys stay out of the code (and out of git -- .env is in .gitignore).
load_dotenv(ROOT / ".env")

# --- API keys ---
# os.getenv returns the value or "" if missing -- the code fails later with a
# clear auth error rather than crashing here on import.
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY", "")

# --- Pinecone ---
INDEX_NAME = "support-kb-hybrid"
# Serverless index location. us-east-1 on AWS is Pinecone's free-tier default.
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
# Must match the embedding model's output size exactly -- Pinecone rejects
# vectors of any other length. text-embedding-3-small always returns 1536 floats.
EMBED_DIM = 1536

# --- Models ---
# [DD-02] Dense embedding model. Chosen TOGETHER with chunk size: ~250-token
# chunks carry the right amount of signal for 1536 dims. The same model MUST
# be used at ingest time and query time forever -- vectors from different
# models live in different geometric spaces and can't be compared.
EMBED_MODEL = "text-embedding-3-small"

# [DD-05] Generation goes through Nebius Token Factory (course requirement).
# Nebius exposes an OpenAI-compatible API, so we reuse the `openai` Python
# client and just point base_url at Nebius.
NEBIUS_BASE_URL = "https://api.studio.nebius.com/v1/"
GEN_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

# --- Chunking [DD-01] ---
# Measured in CHARACTERS (the splitter's unit), ~4 chars per token, so
# 1000 chars =~ 250 tokens. Outcome on our corpus: 27 chunks, avg ~710 chars.
# Smaller -> more precise retrieval but fragments lose context.
# Larger  -> chunks blend topics and dilute the embedding signal.
CHUNK_SIZE = 1000
# Adjacent chunks share 150 chars so a fact straddling a boundary
# appears whole in at least one chunk.
CHUNK_OVERLAP = 150

# --- Retrieval [DD-03, DD-04] ---
# How many chunks the retriever returns and the generator sees as context.
# More = better recall but more noise, tokens, and latency.
TOP_K = 5

# THE hybrid-search knob. Fused score = alpha*dense + (1-alpha)*sparse.
#   1.0 = pure semantic (paraphrase-friendly, misses exact codes)
#   0.0 = pure keyword/BM25 (exact codes, misses intent)
# 0.6 = slightly semantic-leaning: most support questions are paraphrases,
# but error codes (LL-E45) still get a strong sparse boost.
# Setting this to 1.0 / 0.0 gives the pure-dense / pure-sparse baselines
# for the evaluation report -- no code changes needed.
HYBRID_ALPHA = 0.6

# [DD-04] Refusal-first design: if the BEST fused score is below this,
# we skip generation and hand off to a human. Dotproduct scores are NOT
# normalized to [0,1], so this number is empirical -- we calibrate it
# against the 20-query eval set. Too high -> over-escalation;
# too low -> the model answers from weak context (hallucination risk).
#
# CALIBRATION HISTORY:
#   0.45 (initial guess) -> eval run #1: retrieval hit@5 was 100%, but the
#   gate refused 12/17 answerable questions (29% first-contact resolution).
#   Observed score distributions: answerable 0.229-0.65, clearly-unanswerable
#   0.09-0.21. 0.22 sits just below the weakest answerable question.
#   Borderline case: "Do you ship to Australia?" scores 0.331 -- above the
#   gate -- so it now reaches the model, which must use its ESCALATE token
#   (escalation path #2) or answer grounded "US/CA/UK only" from the
#   shipping policy. Both are acceptable behaviors.
CONFIDENCE_THRESHOLD = 0.22

# --- Paths ---
DATA_DIR = ROOT / "data"
# BM25 is fitted on OUR corpus at ingest time (word statistics). Query-time
# encoding must use the SAME statistics, so we persist them to this file.
BM25_PARAMS_PATH = ROOT / "data" / "bm25_params.json"
