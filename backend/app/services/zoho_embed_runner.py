import os
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Callable

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain.embeddings.openai import OpenAIEmbeddings

# IMPORTANT: keep the local import name the same (no try/except fallback).
from app.services.zoho_prepare import prepare_and_embed_zoho_rows


# -------------------- Environment --------------------
def load_env() -> Dict[str, str]:
    """Load all environment variables (names/defaults preserved)."""
    load_dotenv()
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY", ""),
        "PINECONE_REGION": os.getenv("PINECONE_ENV", "us-east-1"),
        "PINECONE_INDEX": os.getenv("PINECONE_INDEX", "nisaa-knowledge"),
        "CLOUD_STORAGE": os.getenv("CLOUD_STORAGE", "aws"),
    }


MODEL_DIMS = {"text-embedding-3-small": 1536}  # correct dim for OpenAI text-embedding-3-small


def validate_env(cfg: Dict[str, str]) -> Tuple[str, str, str, str, str, str, str]:
    """Validate the minimum required env vars and return the tuple used downstream."""
    openai_key = cfg["OPENAI_API_KEY"]
    pinecone_key = cfg["PINECONE_API_KEY"]
    pinecone_index = cfg["PINECONE_INDEX"]
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY is missing.")
    if not pinecone_key:
        raise RuntimeError("PINECONE_API_KEY is missing.")
    if not pinecone_index:
        raise RuntimeError("PINECONE_INDEX is missing.")
    return (
        openai_key,
        cfg["OPENAI_MODEL"],
        cfg["EMBEDDING_MODEL"],
        pinecone_key,
        cfg["PINECONE_REGION"],
        pinecone_index,
        cfg["CLOUD_STORAGE"],
    )


# -------------------- Clients --------------------
def init_embeddings(embedding_model: str, openai_key: str) -> Tuple[OpenAIEmbeddings, int]:
    """Initialize the embedding client and return (client, expected_dim)."""
    dim = MODEL_DIMS.get(embedding_model, 1536)
    print(f"[INFO] Embedding model: {embedding_model} (dim={dim})")
    client = OpenAIEmbeddings(model=embedding_model, openai_api_key=openai_key)
    return client, dim


def init_pinecone(pinecone_api_key: str) -> Pinecone:
    pc = Pinecone(api_key=pinecone_api_key)
    print("[INFO] Pinecone client initialized.")
    return pc


def ensure_index(pc: Pinecone, index_name: str, dim: int, region: str, cloud: str = "aws") -> None:
    existing = {idx.name for idx in pc.list_indexes()}
    if index_name not in existing:
        print(f"[INFO] Creating Pinecone index '{index_name}' (dim={dim}, region={region}, cloud={cloud})...")
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
    else:
        print(f"[INFO] Pinecone index '{index_name}' already exists.")


def get_index(pc: Pinecone, index_name: str):
    idx = pc.Index(index_name)
    print("[INFO] Pinecone index client ready.")
    return idx


# -------------------- Upsert helpers --------------------
def build_upserts(vectors: List[List[float]], metas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pair embeddings with metadatas in Pinecone's expected format."""
    return [{"id": m["vector_id"], "values": v, "metadata": m} for v, m in zip(vectors, metas)]


def upsert_with_retries(
    index,
    upserts: List[Dict[str, Any]],
    namespace: str,
    *,
    batch_size: int = 100,
    max_retries: int = 3,
) -> None:
    """Robust upsert with basic exponential backoff."""
    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            for i in range(0, len(upserts), batch_size):
                batch = upserts[i : i + batch_size]
                index.upsert(vectors=batch, namespace=namespace)
                print(f"[INFO] Upserted batch {i // batch_size + 1} ({len(batch)} vectors) into '{namespace}'.")
            print("[INFO] All vectors upserted successfully.")
            return
        except Exception as e:
            print(f"[WARN] Upsert attempt {attempt} failed: {e}")
            if attempt == max_retries:
                raise
            time.sleep(backoff)
            print(f"[INFO] Retrying in {backoff:.1f}s...")
            backoff *= 2.0


def make_embed_fn(
    embeddings: OpenAIEmbeddings,
    index,
    expected_dim: int,
    upsert_namespace: str,
) -> Callable[[List[str], List[Dict[str, Any]]], None]:
    """
    Factory for the `embed_fn(texts, metadatas)` expected by zoho_prepare.prepare_and_embed_zoho_rows.
    Ensures dim matches and writes to the dynamic namespace.
    """
    def embed_fn(texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        assert len(texts) == len(metadatas), "texts and metadatas length mismatch"
        print(f"[INFO] Embedding {len(texts)} text chunks...")
        vectors = embeddings.embed_documents(texts)
        if any(len(v) != expected_dim for v in vectors):
            raise RuntimeError(f"Embedding dimension mismatch. Expected {expected_dim}.")
        upserts = build_upserts(vectors, metadatas)
        upsert_with_retries(index, upserts, namespace=upsert_namespace, batch_size=100, max_retries=3)

    return embed_fn


# -------------------- Orchestrator --------------------
@dataclass
class OrchestratorConfig:
    openai_api_key: str
    openai_model: str
    embedding_model: str
    pinecone_api_key: str
    pinecone_region: str
    pinecone_index: str
    cloud_storage: str


class EmbedOrchestrator:
    """Coordinates env, clients, and the full run without changing public APIs."""

    def __init__(self, cfg: OrchestratorConfig) -> None:
        self.cfg = cfg
        self.embeddings = None
        self.embed_dim = None
        self.pc = None
        self.index = None

    def init_clients(self) -> None:
        self.embeddings, self.embed_dim = init_embeddings(self.cfg.embedding_model, self.cfg.openai_api_key)
        self.pc = init_pinecone(self.cfg.pinecone_api_key)
        ensure_index(self.pc, self.cfg.pinecone_index, self.embed_dim, self.cfg.pinecone_region, cloud=self.cfg.cloud_storage)
        self.index = get_index(self.pc, self.cfg.pinecone_index)

    def run(
        self,
        *,
        job_id: str,
        payload: dict,
        company_name: str,
    ) -> Dict[str, Any]:
        """Main flow: prepare rows -> embed -> upsert (namespace=company_name)."""
        if self.embeddings is None or self.index is None or self.embed_dim is None:
            self.init_clients()

        embed_fn = make_embed_fn(
            embeddings=self.embeddings,
            index=self.index,
            expected_dim=self.embed_dim,
            upsert_namespace=company_name,
        )

        print(f"[INFO] Model: {self.cfg.embedding_model}")
        print(f"[INFO] Index: {self.cfg.pinecone_index}")
        print(f"[INFO] Namespace (upsert): {company_name}")
        print("[INFO] Starting prepare_and_embed_zoho_rows...")

        summary = prepare_and_embed_zoho_rows(
            payload,
            embed_fn=embed_fn,
            namespace=company_name,
            id_prefix="zoho",
            chunk_size=1000,
            chunk_overlap=120,
            embed_batch=16,
            job_id=job_id,
        )
        print("[INFO] Embedding completed.")
        print("Summary:", summary)
        return summary


# -------------------- Public entrypoint (unchanged) --------------------
def run_from_json_file(job_id: str, payload: dict, company_name: str):
    cfg_raw = load_env()
    cfg = OrchestratorConfig(*validate_env(cfg_raw))
    orchestrator = EmbedOrchestrator(cfg)
    return orchestrator.run(job_id=job_id, payload=payload, company_name=company_name)
