# zoho_embed_runner.py
import os
import sys
import json
import time
from typing import List, Dict, Any

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.services.zoho_prepare import prepare_and_embed_zoho_rows

# -------- load env --------
load_dotenv()
print("[INFO] Environment variables loaded.")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")  # Gemini embeddings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_REGION = os.getenv("PINECONE_ENV", "us-east-1")  # using as region for serverless
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "nisaa-knowledge")
NAMESPACE = "zoho-knowledge"

MODEL_DIMS = {"models/embedding-001": 768}
EMBED_DIM = MODEL_DIMS.get(EMBEDDING_MODEL, 768)

# -------- checks --------
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is missing.")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is missing.")
if not PINECONE_INDEX:
    raise RuntimeError("PINECONE_INDEX is missing.")

print(f"[INFO] Using embedding model: {EMBEDDING_MODEL} (dim={EMBED_DIM})")
print(f"[INFO] Pinecone index: {PINECONE_INDEX}, namespace: {NAMESPACE}")

# -------- init clients --------
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, api_key=GOOGLE_API_KEY)
print("[INFO] GoogleGenerativeAIEmbeddings client initialized.")

pc = Pinecone(api_key=PINECONE_API_KEY)
print("[INFO] Pinecone client initialized.")

existing = {idx.name for idx in pc.list_indexes()}
if PINECONE_INDEX not in existing:
    print(f"[INFO] Creating Pinecone index: {PINECONE_INDEX}")
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud=os.getenv("CLOUD_STORAGE", "aws"), region=PINECONE_REGION),
    )
else:
    print(f"[INFO] Pinecone index '{PINECONE_INDEX}' already exists.")

index = pc.Index(PINECONE_INDEX)
print("[INFO] Pinecone index client ready.")

# -------- embed_fn with batching + retries --------
def embed_fn(texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
    assert len(texts) == len(metadatas), "texts and metadatas length mismatch"
    print(f"[INFO] Embedding {len(texts)} texts...")

    vectors = embeddings.embed_documents(texts)
    if any(len(v) != EMBED_DIM for v in vectors):
        raise RuntimeError(f"Embedding dimension mismatch. Expected {EMBED_DIM}.")
    print("[INFO] Texts embedded successfully.")

    upserts = [
        {"id": m["vector_id"], "values": vec, "metadata": m}
        for vec, m in zip(vectors, metadatas)
    ]
    print(f"[INFO] Prepared {len(upserts)} vectors for upsert.")

    max_retries = 3
    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            BATCH = 100
            for i in range(0, len(upserts), BATCH):
                index.upsert(vectors=upserts[i:i+BATCH], namespace=NAMESPACE)
                print(f"[INFO] Upserted batch {i // BATCH + 1} ({len(upserts[i:i+BATCH])} vectors).")
            print("[INFO] All vectors upserted successfully.")
            return
        except Exception as e:
            print(f"[WARNING] Upsert attempt {attempt} failed: {e}")
            if attempt == max_retries:
                raise
            time.sleep(backoff)
            backoff *= 2
            print(f"[INFO] Retrying upsert (attempt {attempt + 1}) after {backoff}s backoff...")

# -------- helpers --------
def run_from_json_file(job_id: str, payload: dict):
    print("[INFO] Running embedding from JSON file payload...")
    summary = prepare_and_embed_zoho_rows(
        payload,
        embed_fn=embed_fn,
        namespace=NAMESPACE,
        id_prefix="zoho",
        chunk_size=1000,
        chunk_overlap=120,
        embed_batch=100,
        job_id=job_id
    )
    print("[INFO] Embedding completed.")
    print("Summary:", summary)

# def run_from_stdin():
#     print("[INFO] Running embedding from STDIN...")
#     payload = json.load(sys.stdin)
#     summary = prepare_and_embed_zoho_rows(
#         payload,
#         embed_fn=embed_fn,
#         namespace=NAMESPACE,
#         id_prefix="zoho",
#         chunk_size=1000,
#         chunk_overlap=120,
#         embed_batch=100,
#     )
#     print("[INFO] Embedding completed.")
#     print("Summary:", summary)
