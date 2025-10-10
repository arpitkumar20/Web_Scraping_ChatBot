import os
import re
import json
import hashlib
import logging
from typing import Any, Dict, List, Tuple, Iterable, Optional

from pinecone import Pinecone, ServerlessSpec

from app.helper.utils import COMMON

# --------- Defaults ---------
# Known embedding dimensions for convenience (pick up in runner)
MODEL_DIMS = {
    "models/embedding-001": 768,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}

# ---------- Logger Setup ----------
logger = logging.getLogger("pg_embed_logic")

# ---------- Small Utilities ----------
def _slug(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
    return s.lower()

def _clean_val(v: Any) -> Any:
    return " ".join(v.split()) if isinstance(v, str) else v

def _to_str(v: Any) -> str:
    v = _clean_val(v)
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(v)

def _flatten(d: Dict[str, Any], parent: str = "") -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        key = f"{parent}.{k}" if parent else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out

def _stable_row_id(row: Dict[str, Any], primary_key_col: Optional[str] = None) -> str:
    if primary_key_col and primary_key_col in row and row[primary_key_col]:
        return str(row[primary_key_col])
    for k in ("id", "ID", "uuid", "UUID", "pk", "row_id", "RowID"):
        if k in row and row[k]:
            return str(row[k])
    payload = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

def _content_hash(row: Dict[str, Any]) -> str:
    payload = json.dumps(
        {k: _clean_val(v) for k, v in sorted(row.items())},
        ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()

def _row_to_text(row: Dict[str, Any]) -> str:
    parts = []
    for k in sorted(row.keys()):
        sv = _to_str(row[k])
        if sv:
            parts.append(f"{k}: {sv}")
    return "\n".join(parts)

def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks, start = [], 0
    L = len(text)
    while start < L:
        end = min(L, start + chunk_size)
        chunks.append(text[start:end])
        if end >= L:
            break
        start = max(0, end - chunk_overlap)
    return chunks

# ---------- Dynamic JSON Normalization ----------
def _iter_row_sets(payload: Any) -> Iterable[Tuple[str, List[Dict[str, Any]], Dict[str, Any]]]:
    """Yield (table_name, rows, ctx) from flexible JSON shapes."""
    if isinstance(payload, dict) and isinstance(payload.get("result"), list):
        rows = payload.get("result", [])
        schema = payload.get("schema_name", "")
        table_name = payload.get("table_and_columns", [{}])[0].get("table_name", "unknown_table")
        yield table_name, rows, {"schema_name": schema, "primary_column": payload.get("primary_column")}
        return

    for key in ("rows", "data", "records"):
        if isinstance(payload, dict) and isinstance(payload.get(key), list):
            yield "unknown_table", payload.get(key), {}
            return

    if isinstance(payload, dict):
        for k, v in payload.items():
            if isinstance(v, list) and (not v or isinstance(v[0], dict)):
                yield k, v, {}
        return

    if isinstance(payload, list) and (not payload or isinstance(payload[0], dict)):
        yield "unknown_table", payload, {}
        return

    if isinstance(payload, list):
        for item in payload:
            yield from _iter_row_sets(item)
        return

# ---------- Main Embedder ----------
def embed_postgres_like_json(
    payload: Any,
    embeddings,
    pc_client: Pinecone,
    *,
    namespace: str = None,
    id_prefix: str = "pg",
    chunk_size: int = 1000,
    chunk_overlap: int = 120,
    embed_batch: int = 100,
    embed_dim: int = 1536,
    index_name: str = "nisaa-knowledge",
    env_region: str = "us-east-1",
):
    """
    payload: JSON-like object (dict/list) representing rows/tables
    embeddings: object with .embed_documents(List[str]) -> List[List[float]]
    pc_client: Pinecone client
    """
    try:
        resp = pc_client.list_indexes()
        try:
            names = {x.name for x in resp}
        except AttributeError:
            names = {x["name"] for x in resp}

        if index_name not in names:
            logger.info(f"🆕 Creating Pinecone index: {index_name}")
            pc_client.create_index(
                name=index_name,
                dimension=embed_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=env_region),
            )
        index = pc_client.Index(index_name)
        logger.info(f"Connected to Pinecone index: {index_name}")

        total_rows, total_chunks = 0, 0
        buffer_texts, buffer_metas = [], []

        def flush():
            nonlocal buffer_texts, buffer_metas
            if not buffer_texts:
                return
            vectors = embeddings.embed_documents(buffer_texts)
            if any(len(v) != embed_dim for v in vectors):
                raise RuntimeError(f"Embedding dimension mismatch. Expected {embed_dim}.")
            batch = [
                {"id": m["vector_id"], "values": vec, "metadata": m}
                for vec, m in zip(vectors, buffer_metas)
            ]
            for i in range(0, len(batch), 100):
                index.upsert(vectors=batch[i:i+100], namespace=namespace)
            logger.info(f"Upserted {len(batch)} vectors to Pinecone namespace '{namespace}'.")
            buffer_texts, buffer_metas = [], []

        for table_name, rows, ctx in _iter_row_sets(payload):
            logger.info(f"Processing table '{table_name}' with {len(rows)} rows.")
            table_s = _slug(table_name) or "unknown-table"
            schema_s = _slug(ctx.get("schema_name") or "")
            pkey = ctx.get("primary_column")

            for row in rows or []:
                if not isinstance(row, dict):
                    continue
                flat_row = _flatten(row)
                total_rows += 1
                row_pk = _stable_row_id(flat_row, primary_key_col=pkey)
                text = _row_to_text(flat_row)
                chunks = _split_text(text, chunk_size, chunk_overlap)
                total_chunks += len(chunks)

                group_id = f"{id_prefix}:{schema_s or 'public'}:{table_s}:{row_pk}"
                row_hash = _content_hash(flat_row)
                preview = _to_str(flat_row.get("text") or flat_row.get("message") or flat_row.get("name") or row_pk)[:200]

                for i, ch in enumerate(chunks, 1):
                    vector_id = f"{group_id}:attributes:{i}"
                    meta = {
                        "source": "postgresql",
                        "namespace": namespace,
                        "schema": ctx.get("schema_name") or "",
                        "table": table_name,
                        "row_pk": str(row_pk),
                        "group_id": group_id,
                        "field": "attributes",
                        "chunk_no": i,
                        "chunk_total": len(chunks),
                        "text_content": ch,
                        "preview": preview,
                        "content_hash": row_hash,
                        "vector_id": vector_id,
                    }
                    buffer_texts.append(ch)
                    buffer_metas.append(meta)
                    if len(buffer_texts) >= embed_batch:
                        flush()

        flush()
        logger.info(f"✅ Done. Rows: {total_rows}, Chunks: {total_chunks}, Namespace: {namespace}")
        COMMON.save_name(namespace=namespace,folder_path="web_info",filename="web_info.json")
    except Exception as e:
        logger.exception(f"❌ Embedding pipeline failed: {e}")
        raise
