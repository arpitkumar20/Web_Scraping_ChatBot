import os
import json
import time
import re
import hashlib
from typing import Any, Dict, List, Iterable, Tuple, Callable, Optional

from pinecone import Pinecone, ServerlessSpec
from langchain.embeddings.openai import OpenAIEmbeddings

from app.helper.utils import COMMON

# -------------------- Utilities --------------------
def _slug(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
    return s.lower()

def _clean_val(v: Any) -> Any:
    if isinstance(v, str):
        return " ".join(v.split())
    return v

def _to_str(v: Any) -> str:
    v = _clean_val(v)
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(v)

def _prefer_display_value(val: Any) -> Any:
    if isinstance(val, dict) and "display_value" in val:
        return val.get("display_value")
    return val

def _stable_row_id(row: Dict[str, Any]) -> str:
    for k in ("ID", "id", "row_id", "uuid", "pk", "Zoho_ID", "zoho_id"):
        if k in row and row[k]:
            return str(row[k])
    payload = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

def _content_hash(row: Dict[str, Any], include: Optional[List[str]] = None, exclude: Optional[List[str]] = None) -> str:
    if include is not None:
        keys = [k for k in include if k in row]
    else:
        keys = list(row.keys())
    if exclude:
        keys = [k for k in keys if k not in exclude]
    keys = sorted(keys)
    norm = {k: _clean_val(_prefer_display_value(row.get(k))) for k in keys}
    payload = json.dumps(norm, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()

def _row_to_text(row: Dict[str, Any]) -> str:
    lines: List[str] = []
    for key in sorted(row.keys()):
        val = _prefer_display_value(row[key])
        s = _to_str(val)
        if s != "":
            lines.append(f"{key}: {s}")
    return "\n".join(lines)

def _split_text(text: str, csize: int, overlap: int) -> List[str]:
    if len(text) <= csize:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + csize
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks

def _iter_reports_and_rows(payload: Any) -> Iterable[Tuple[str, Dict[str, Any], Dict[str, str]]]:
    """Yield (report_name, row_dict, context_dict) across multiple common Zoho payload shapes."""
    if isinstance(payload, dict) and isinstance(payload.get("applications"), list):
        for app in payload["applications"]:
            if not isinstance(app, dict):
                continue
            owner = _to_str(app.get("owner_name"))
            app_link = _to_str(app.get("app_link_name"))
            reports = app.get("reports") or []
            for rep in reports:
                if not isinstance(rep, dict):
                    continue
                if rep.get("report_status") not in (200, "200", None):
                    continue
                md = rep.get("report_metadata") or {}
                report_name = _to_str(md.get("report_link_name") or md.get("report_display_name") or "")
                data = (((rep.get("report_data") or {}).get("records") or {}).get("data"))
                if isinstance(data, list):
                    for r in data:
                        if isinstance(r, dict):
                            yield (report_name, r, {"owner_name": owner, "app_link_name": app_link})
        return
    if isinstance(payload, dict) and "records" in payload:
        report_name = _to_str(payload.get("report_link_name") or payload.get("report_name") or "")
        data = (payload.get("records") or {}).get("data")
        if isinstance(data, list):
            for r in data:
                if isinstance(r, dict):
                    yield (report_name, r, {})
        return
    if isinstance(payload, dict) and "records" not in payload:
        for k, v in payload.items():
            if isinstance(v, list) and (len(v) == 0 or (isinstance(v[0], dict) if v else True)):
                for r in v:
                    if isinstance(r, dict):
                        yield (_to_str(k), r, {})
        return
    if isinstance(payload, list) and (len(payload) == 0 or isinstance(payload[0], dict)):
        for r in payload:
            if isinstance(r, dict):
                yield ("", r, {})
        return
    if isinstance(payload, list):
        for item in payload:
            yield from _iter_reports_and_rows(item)
        return
    return

# -------------------- Pinecone + OpenAI --------------------
MODEL_DIMS = {"text-embedding-3-small": 1536}

def init_embeddings(openai_key: str, embedding_model: str) -> Tuple[OpenAIEmbeddings, int]:
    dim = MODEL_DIMS.get(embedding_model, 1536)
    client = OpenAIEmbeddings(model=embedding_model, openai_api_key=openai_key)
    print(f"[INFO] OpenAI embedding model: {embedding_model} (dim={dim})")
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

def build_upserts(vectors: List[List[float]], metas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"id": m["vector_id"], "values": v, "metadata": m} for v, m in zip(vectors, metas)]

def upsert_with_retries(index, upserts: List[Dict[str, Any]], namespace: str, batch_size: int = 100, max_retries: int = 3):
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
    def embed_fn(texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        assert len(texts) == len(metadatas), "texts and metadatas length mismatch"
        print(f"[INFO] Embedding {len(texts)} text chunks...")
        vectors = embeddings.embed_documents(texts)
        if any(len(v) != expected_dim for v in vectors):
            raise RuntimeError(f"Embedding dimension mismatch. Expected {expected_dim}.")
        upserts = build_upserts(vectors, metadatas)
        upsert_with_retries(index, upserts, namespace=upsert_namespace, batch_size=100, max_retries=3)
    return embed_fn

# -------------------- Main API --------------------
def prepare_and_embed_zoho_rows(
    zoho_data: Any,
    embed_fn: Callable[[List[str], List[Dict[str, Any]]], None],
    *,
    namespace: str = "zoho-knowledge",
    id_prefix: str = "zoho",
    chunk_size: int = 1000,
    chunk_overlap: int = 120,
    embed_batch: int = 100,
    job_id: str = None
) -> Dict[str, int]:
    print("[INFO] Starting prepare_and_embed_zoho_rows...")

    report_count = 0
    row_count = 0
    chunk_count = 0

    buffer_texts: List[str] = []
    buffer_metas: List[Dict[str, Any]] = []

    def _flush():
        nonlocal buffer_texts, buffer_metas
        if not buffer_texts:
            return
        print(f"[INFO] Flushing {len(buffer_texts)} chunks to embed_fn...")
        embed_fn(buffer_texts, buffer_metas)
        buffer_texts, buffer_metas = [], []
        print("[INFO] Flush completed.")

    seen_reports = set()
    for report_name, row, ctx in _iter_reports_and_rows(zoho_data):
        if report_name not in seen_reports:
            seen_reports.add(report_name)
            report_count += 1
            print(f"[INFO] Processing report: '{report_name}'")
        row_count += 1
        print(f"[INFO] Processing row {row_count} (report: '{report_name}')")

        row_pk = _stable_row_id(row)
        owner = ctx.get("owner_name", "")
        app = ctx.get("app_link_name", "")

        owner_s = _slug(owner)
        app_s = _slug(app)
        report_s = _slug(report_name) if report_name else "unknown-report"

        preview = _to_str(row.get("Full_Name") or row.get("Name") or row.get("Title") or row_pk)[:120]

        text = _row_to_text(row)
        chunks = _split_text(text, chunk_size, chunk_overlap)
        chunk_total = len(chunks)
        chunk_count += chunk_total
        print(f"[INFO] Row {row_count} split into {chunk_total} chunk(s)")

        group_id = f"{id_prefix}:{owner_s}:{app_s}:{report_s}:{row_pk}"
        row_hash = _content_hash(row)

        for i, ch in enumerate(chunks, 1):
            vector_id = f"{group_id}:attributes:{i}"
            meta = {
                "source": "zoho",
                "namespace": namespace,
                "owner_name": owner,
                "app_link_name": app,
                "report": report_name,
                "row_pk": row_pk,
                "group_id": group_id,
                "field": "attributes",
                "chunk_no": i,
                "chunk_total": chunk_total,
                "text_content": ch,
                "preview": preview,
                "content_hash": row_hash,
                "vector_id": vector_id,
            }
            buffer_texts.append(ch)
            buffer_metas.append(meta)
            if len(buffer_texts) >= embed_batch:
                _flush()

    _flush()
    summary = {"reports": report_count, "rows": row_count, "chunks": chunk_count, "namespace": namespace, "job_id": job_id}
    print(f"[INFO] Finished embedding: {report_count} reports, {row_count} rows, {chunk_count} chunks.")
    COMMON.save_name(namespace=namespace,folder_path="web_info",filename="web_info.json")
    return summary
