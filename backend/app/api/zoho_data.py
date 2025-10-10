"""
zoho_embedding_pipeline.py

- Fixes Pinecone upsert to try common SDK signatures (new/old).
- Clean text and preserve tabular structure (columns/rows) for reports.
- Saves embedding-ready JSON to embeddings/<namespace>/embedding_ready.json.
- Keeps function names unchanged.
- Does NOT save any error details in JSON (only valid fetched data).
"""

import os
import re
import json
import uuid
import time
import logging
import threading
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from flask import Blueprint, request, jsonify
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# External clients
# ---------------------------------------------------------------------------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from pinecone import Pinecone
except Exception:
    Pinecone = None
    try:
        import pinecone as pinecone_client
    except Exception:
        pinecone_client = None

# Placeholder (must be replaced by actual Zoho class at runtime)
from app.models.zoho_connectors import Zoho

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

DEFAULT_CHUNK_SIZE = int(os.getenv("EMBED_CHUNK_SIZE", 1500))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("EMBED_CHUNK_OVERLAP", 200))
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", 50))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

zoho_bp = Blueprint("zoho", __name__)

job_status: Dict[str, Dict[str, Any]] = {}
lock = threading.Lock()

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    return s[:240]


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json_file(path: str, obj: Any) -> None:
    """Save JSON safely without any error data."""
    safe_mkdir(os.path.dirname(path))
    # Remove any error keys recursively before saving
    def clean_errors(o):
        if isinstance(o, dict):
            return {k: clean_errors(v) for k, v in o.items() if "error" not in k.lower()}
        elif isinstance(o, list):
            return [clean_errors(v) for v in o]
        return o

    cleaned_obj = clean_errors(obj)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cleaned_obj, f, ensure_ascii=False, indent=2)


def load_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_text(s: Any, max_len: Optional[int] = None) -> str:
    """Clean strings: remove HTML tags, control chars, collapse whitespace."""
    if s is None:
        return ""
    if not isinstance(s, str):
        try:
            s = json.dumps(s, ensure_ascii=False)
        except Exception:
            s = str(s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("\xa0", " ")
    s = re.sub(r"[\r\n\t]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if max_len and len(s) > max_len:
        return s[:max_len]
    return s


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    if not text:
        return []
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if start >= text_len:
            break
    return chunks


def _make_doc_id(namespace: str, owner: str, app: str, report: str, chunk_idx: int) -> str:
    base = f"{namespace}~{owner}~{app}~{report}~{chunk_idx}"
    return _slug(base) + f"~{chunk_idx}"

# ---------------------------------------------------------------------------
# Prepare embedding docs (preserve table structure)
# ---------------------------------------------------------------------------
def prepare_embedding_docs_from_zoho(data: Dict[str, Any], namespace: str,
                                     chunk_size: int = DEFAULT_CHUNK_SIZE,
                                     overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """
    Prepares embedding documents from Zoho data.
    Preserves table structure and ensures metadata is Pinecone-compatible.
    """
    apps = data.get("applications") or []
    prepared: List[Dict[str, Any]] = []

    for app in apps:
        owner_name = app.get("owner_name") or "unknown_owner"
        app_link_name = app.get("app_link_name") or app.get("app_name") or "unknown_app"
        reports = app.get("reports") or []

        for r in reports:
            report_meta = r.get("report_metadata") or {}
            report_data = r.get("report_data") or r.get("data") or r.get("rows") or ""
            report_link_name = report_meta.get("report_link_name") or report_meta.get("name") or "report"

            # Extract rows if possible
            rows = None
            if isinstance(report_data, list) and all(isinstance(it, dict) for it in report_data):
                rows = report_data
            elif isinstance(report_data, dict) and isinstance(report_data.get("rows"), list):
                if all(isinstance(it, dict) for it in report_data["rows"]):
                    rows = report_data["rows"]

            # Build text parts for embedding
            text_parts = []
            meta_parts = [str(report_meta.get(k)) for k in ("report_link_name", "report_name", "description") if report_meta.get(k)]
            if meta_parts:
                text_parts.append(" | ".join(meta_parts))

            structured_table = None
            if rows:
                # Collect all unique columns
                columns = list(rows[0].keys())
                for row in rows:
                    for k in row.keys():
                        if k not in columns:
                            columns.append(k)

                # Clean rows
                cleaned_rows = [[clean_text(row.get(c, ""), max_len=10000) for c in columns] for row in rows]
                structured_table = {"columns": columns, "rows": cleaned_rows, "rows_count": len(cleaned_rows)}

                # Include first 50 rows as text for embedding
                sample_rows = [" | ".join(columns)]
                for rdata in cleaned_rows[:50]:
                    sample_rows.append(" | ".join(rdata))
                text_parts.append("\n".join(sample_rows))
            else:
                # Fallback: stringify report_data
                if isinstance(report_data, (list, dict)):
                    report_data_str = json.dumps(report_data, ensure_ascii=False)
                else:
                    report_data_str = str(report_data)
                text_parts.append(clean_text(report_data_str[:200_000]))

            combined = "\n\n".join([clean_text(t) for t in text_parts if t])
            if not combined:
                continue

            # Chunk text and prepare metadata
            for idx, c in enumerate(chunk_text(combined, chunk_size, overlap)):
                # Ensure metadata is always Pinecone-compatible
                metadata_dict = {
                    "owner_name": owner_name,
                    "app_link_name": app_link_name,
                    "report_link_name": report_link_name,
                }

                if structured_table:
                    metadata_dict["table_columns_count"] = len(structured_table["columns"])
                    metadata_dict["table_rows_count"] = structured_table["rows_count"]
                else:
                    metadata_dict["table_columns_count"] = 0
                    metadata_dict["table_rows_count"] = 0

                prepared.append({
                    "id": _make_doc_id(namespace, owner_name, app_link_name, report_link_name, idx),
                    "namespace": namespace,
                    "owner_name": owner_name,
                    "app_link_name": app_link_name,
                    "report_link_name": report_link_name,
                    "chunk_index": idx,
                    "text": c,
                    "metadata": metadata_dict,
                    "extra_data": structured_table  # this will NOT go to Pinecone
                })

    return prepared


# ---------------------------------------------------------------------------
# Embedding + Pinecone helpers
# ---------------------------------------------------------------------------
@dataclass
class OpenAIEmbedder:
    client: Any
    model: str

    def embed_documents(self, texts: List[str], retry: int = 3) -> List[List[float]]:
        for attempt in range(retry):
            try:
                resp = self.client.embeddings.create(model=self.model, input=texts)
                return [d.embedding for d in resp.data]
            except Exception as e:
                logger.warning("Embedding attempt %d failed: %s", attempt + 1, e)
                time.sleep(1 + attempt)
        raise RuntimeError("Embedding failed after retries")

def _get_pinecone_index(pc_client: Any, index_name: str):
    if hasattr(pc_client, "Index"):
        return pc_client.Index(index_name)
    if 'pinecone_client' in globals() and globals().get('pinecone_client'):
        try:
            return globals()['pinecone_client'].Index(index_name)
        except Exception:
            pass
    if hasattr(pc_client, "upsert"):
        return pc_client
    raise RuntimeError("Unsupported Pinecone client version")


def upsert_vectors_to_pinecone(index_obj: Any, vectors: List[Tuple[str, List[float], Dict[str, Any]]], namespace: str):
    """
    Upsert vectors to Pinecone using the correct SDK signature.
    Only attempts supported method, avoids raising detailed error info.
    vectors: list of (id, vector, metadata)
    """
    if not vectors:
        return

    # Convert to dict format for Pinecone
    items = [{"id": vid, "values": vec, "metadata": (meta or {})} for vid, vec, meta in vectors]

    # Modern SDK expects `vectors=` keyword
    try:
        return index_obj.upsert(vectors=items, namespace=namespace)
    except TypeError:
        # fallback for older SDK that uses `items=` keyword
        return index_obj.upsert(items=items, namespace=namespace)


# def embed_and_upsert(prepared_docs, openai_client, pc_client, index_name, namespace, batch_size=BATCH_SIZE):
#     embedder = OpenAIEmbedder(openai_client, EMBEDDING_MODEL)
#     index_obj = _get_pinecone_index(pc_client, index_name)
#     total = len(prepared_docs)
#     for i in range(0, total, batch_size):
#         batch = prepared_docs[i:i+batch_size]
#         texts = [b["text"] for b in batch]
#         embeddings = embedder.embed_documents(texts)
#         vectors = [(b["id"], emb, b["metadata"]) for b, emb in zip(batch, embeddings)]
#         upsert_vectors_to_pinecone(index_obj, vectors, namespace)
#         logger.info("Upserted %d/%d vectors", min(i+batch_size, total), total)

def embed_and_upsert(prepared_docs, openai_client, pc_client, index_name, namespace, batch_size=BATCH_SIZE):
    """
    Embeds only the 'text' from prepared_docs and upserts to Pinecone.
    Ignores extra_data or structured table.
    """
    embedder = OpenAIEmbedder(openai_client, EMBEDDING_MODEL)
    index_obj = _get_pinecone_index(pc_client, index_name)
    total = len(prepared_docs)

    for i in range(0, total, batch_size):
        batch = prepared_docs[i:i+batch_size]

        # Only extract text for embedding
        texts = [b["text"] for b in batch]

        embeddings = embedder.embed_documents(texts)

        # Minimal metadata only (without extra_data)
        vectors = [
            (
                b["id"], 
                emb, 
                {
                    "owner_name": b.get("owner_name", ""),
                    "app_link_name": b.get("app_link_name", ""),
                    "report_link_name": b.get("report_link_name", ""),
                    "chunk_index": b.get("chunk_index", 0)
                }
            )
            for b, emb in zip(batch, embeddings)
        ]

        upsert_vectors_to_pinecone(index_obj, vectors, namespace)
        logger.info("Upserted %d/%d vectors (text only)", min(i + batch_size, total), total)


# ---------------------------------------------------------------------------
# Flask routes + background workers
# ---------------------------------------------------------------------------
@zoho_bp.route("/fetch-zoho-details", methods=["POST"])
def fetch_all_zoho_data():
    try:
        data = request.json or {}
        company_name = data.get("company_name")
        if not company_name:
            return jsonify({"error": "'company_name' is required"}), 400

        if Zoho is None:
            return jsonify({"success": False, "message": "Zoho client not available."}), 500

        zoho = Zoho(data.get("client_id"), data.get("client_secret"), data.get("refresh_token"))
        status, code = zoho.connection_test()
        if code != 200:
            return jsonify({"success": False, "message": "Zoho connection failed"}), 401

        job_id = str(uuid.uuid4())
        with lock:
            job_status[job_id] = {"job_id": job_id, "embedding_status": "queued", "step": "queued", "company_name": company_name}

        threading.Thread(target=_fetch_and_embed_zoho_data, args=(data, company_name, job_id), daemon=True).start()

        return jsonify({"success": True, "message": "Zoho connection successful. Data fetching started.", "job_id": job_id}), 200

    except Exception as e:
        logger.exception("Error in Zoho connection", exc_info=e)
        return jsonify({"success": False, "error": str(e)}), 500


def _fetch_and_embed_zoho_data(data, company_name, job_id):
    try:
        with lock:
            job_status[job_id]["step"] = "fetching_started"

        zoho = Zoho(data.get("client_id"), data.get("client_secret"), data.get("refresh_token"))
        applications = zoho.get_all_applications()
        if not applications:
            job_status[job_id]["embedding_status"] = "failed"
            job_status[job_id]["step"] = "no_applications"
            return

        final_result = []
        for app in applications:
            owner = app.get("owner_name")
            link = app.get("app_link_name")
            reports_resp, _ = zoho.fetch_reports_list(owner_name=owner, app_link_name=link)
            if not reports_resp.get("success"):
                continue
            report_details = []
            for report in reports_resp.get("reports", []):
                link_name = report.get("report_link_name")
                report_data, status = zoho.fetch_report_deatils(owner, link, link_name)
                if status == 200:
                    # Clean the report data for text embedding
                    if isinstance(report_data, list):
                        cleaned_rows = []
                        for row in report_data:
                            if isinstance(row, dict):
                                cleaned_row = {k: clean_text(v, max_len=10000) for k, v in row.items()}
                                cleaned_rows.append(cleaned_row)
                        report_data = cleaned_rows
                    elif isinstance(report_data, dict):
                        report_data = {k: clean_text(v, max_len=10000) for k, v in report_data.items()}
                    else:
                        report_data = clean_text(report_data, max_len=200_000)

                    report_details.append({"report_metadata": report, "report_data": report_data})
            final_result.append({"owner_name": owner, "app_link_name": link, "reports": report_details})

        payload = {"applications": final_result}

        namespace = _slug(company_name)
        prepared_docs = prepare_embedding_docs_from_zoho(payload, namespace)

        # Remove extra_data from each doc before saving JSON
        for doc in prepared_docs:
            if "extra_data" in doc:
                del doc["extra_data"]

        embedding_folder = os.path.join("embeddings", namespace)
        embedding_ready_path = os.path.join(embedding_folder, "embedding_ready.json")

        save_json_file(embedding_ready_path, {
            "company_name": company_name,
            "namespace": namespace,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "total_chunks": len(prepared_docs),
            "prepared_docs": prepared_docs
        })

        with lock:
            job_status[job_id]["step"] = "prepared_on_disk"

        # Start embedding in background (text only)
        threading.Thread(target=run_from_json_file, args=(job_id, {"embedding_ready_path": embedding_ready_path}, company_name), daemon=True).start()

    except Exception as e:
        logger.exception("Error fetching Zoho data", exc_info=e)
        with lock:
            job_status[job_id]["embedding_status"] = "failed"
            job_status[job_id]["step"] = "error"

@zoho_bp.route('/job-status/<job_id>', methods=['GET'])
def embedding_status(job_id):
    with lock:
        if job_id not in job_status:
            return jsonify({"error": "Invalid job_id"}), 404
        return jsonify(job_status[job_id])

def run_from_json_file(job_id, data, company_name):
    try:
        namespace = _slug(company_name)
        prepared_docs = None

        if data.get("embedding_ready_path"):
            stored = load_json_file(data["embedding_ready_path"])
            prepared_docs = stored.get("prepared_docs")

        if not prepared_docs:
            job_status[job_id]["embedding_status"] = "failed"
            return

        pc_client = None
        if Pinecone:
            pc_client = Pinecone(api_key=PINECONE_API_KEY)
        elif globals().get('pinecone_client'):
            pc = globals()['pinecone_client']
            pc.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
            pc_client = pc

        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        embed_and_upsert(prepared_docs, openai_client, pc_client, PINECONE_INDEX, namespace)

        with lock:
            job_status[job_id]["embedding_status"] = "completed"
            job_status[job_id]["step"] = "finished"

    except Exception as e:
        logger.exception("Embedding job failed", exc_info=e)
        with lock:
            job_status[job_id]["embedding_status"] = "failed"
            job_status[job_id]["step"] = "error"
