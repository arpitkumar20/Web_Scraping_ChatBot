# import os
# import re
# import json
# import uuid
# import time
# import logging
# import threading
# from typing import List, Dict, Any, Tuple, Optional
# from dataclasses import dataclass
# from datetime import datetime
# from flask import Flask, Blueprint, request, jsonify
# from dotenv import load_dotenv

# # ------------------ Load .env ------------------
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_ENV = os.getenv("PINECONE_ENV")
# PINECONE_INDEX = os.getenv("PINECONE_INDEX")
# EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
# DEFAULT_CHUNK_SIZE = int(os.getenv("EMBED_CHUNK_SIZE", 1500))
# DEFAULT_CHUNK_OVERLAP = int(os.getenv("EMBED_CHUNK_OVERLAP", 200))
# BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", 50))

# # ------------------ Logging ------------------
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

# # ------------------ External Clients ------------------
# try:
#     from openai import OpenAI
# except Exception:
#     OpenAI = None

# try:
#     from pinecone import Pinecone
# except Exception:
#     Pinecone = None
#     try:
#         import pinecone as pinecone_client
#     except Exception:
#         pinecone_client = None

# # Placeholder for Zoho connector
# from app.models.zoho_connectors import Zoho

# # ------------------ Flask Setup ------------------
# zoho_bp = Blueprint("zoho", __name__)
# job_status: Dict[str, Dict[str, Any]] = {}
# lock = threading.Lock()

# # ------------------ Utilities ------------------
# def _slug(s: str) -> str:
#     s = (s or "").strip().lower()
#     s = re.sub(r"\s+", "-", s)
#     s = re.sub(r"[^a-z0-9._-]+", "-", s)
#     return s[:240]

# def safe_mkdir(path: str) -> None:
#     os.makedirs(path, exist_ok=True)

# def save_json_file(path: str, obj: Any) -> None:
#     safe_mkdir(os.path.dirname(path))
#     def clean_errors(o):
#         if isinstance(o, dict):
#             return {k: clean_errors(v) for k, v in o.items() if "error" not in k.lower()}
#         elif isinstance(o, list):
#             return [clean_errors(v) for v in o]
#         return o
#     cleaned_obj = clean_errors(obj)
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(cleaned_obj, f, ensure_ascii=False, indent=2)

# def load_json_file(path: str) -> Any:
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)

# def clean_text(s: Any, max_len: Optional[int] = None) -> str:
#     if s is None:
#         return ""
#     if not isinstance(s, str):
#         try:
#             s = json.dumps(s, ensure_ascii=False)
#         except Exception:
#             s = str(s)
#     s = re.sub(r"<[^>]+>", " ", s)
#     s = s.replace("\xa0", " ")
#     s = re.sub(r"[\r\n\t]+", " ", s)
#     s = re.sub(r"\s+", " ", s).strip()
#     if max_len and len(s) > max_len:
#         return s[:max_len]
#     return s

# def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
#     if not text:
#         return []
#     text = text.strip()
#     if len(text) <= chunk_size:
#         return [text]
#     chunks = []
#     start = 0
#     while start < len(text):
#         end = start + chunk_size
#         chunks.append(text[start:end])
#         start = end - overlap
#         if start < 0: start = 0
#         if start >= len(text): break
#     return chunks

# def _make_doc_id(namespace: str, owner: str, app: str, report: str, chunk_idx: int) -> str:
#     base = f"{namespace}~{owner}~{app}~{report}~{chunk_idx}"
#     return _slug(base) + f"~{chunk_idx}"

# # ------------------ Embedding Doc Preparation ------------------
# def prepare_embedding_docs_from_zoho(data: Dict[str, Any], namespace: str,
#                                      chunk_size: int = DEFAULT_CHUNK_SIZE,
#                                      overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[Dict[str, Any]]:
#     prepared: List[Dict[str, Any]] = []
#     apps = data.get("applications") or []

#     for app in apps:
#         owner_name = app.get("owner_name") or "unknown_owner"
#         app_link_name = app.get("app_link_name") or app.get("app_name") or "unknown_app"
#         reports = app.get("reports") or []

#         for r in reports:
#             report_meta = r.get("report_metadata") or {}
#             report_data = r.get("report_data") or r.get("data") or r.get("rows") or ""
#             report_link_name = report_meta.get("report_link_name") or report_meta.get("name") or "report"

#             rows = None
#             if isinstance(report_data, list) and all(isinstance(it, dict) for it in report_data):
#                 rows = report_data
#             elif isinstance(report_data, dict) and isinstance(report_data.get("rows"), list):
#                 if all(isinstance(it, dict) for it in report_data["rows"]):
#                     rows = report_data["rows"]

#             text_parts = []
#             meta_parts = [str(report_meta.get(k)) for k in ("report_link_name", "report_name", "description") if report_meta.get(k)]
#             if meta_parts: text_parts.append(" | ".join(meta_parts))

#             if rows:
#                 columns = list(rows[0].keys())
#                 for row in rows:
#                     for k in row.keys():
#                         if k not in columns:
#                             columns.append(k)
#                 cleaned_rows = [[clean_text(row.get(c, ""), max_len=10000) for c in columns] for row in rows]
#                 sample_rows = [" | ".join(columns)]
#                 for rdata in cleaned_rows[:50]:
#                     sample_rows.append(" | ".join(rdata))
#                 text_parts.append("\n".join(sample_rows))
#             else:
#                 if isinstance(report_data, (list, dict)):
#                     report_data_str = json.dumps(report_data, ensure_ascii=False)
#                 else:
#                     report_data_str = str(report_data)
#                 text_parts.append(clean_text(report_data_str[:200_000]))

#             combined = "\n\n".join([clean_text(t) for t in text_parts if t])
#             if not combined: continue

#             for idx, c in enumerate(chunk_text(combined, chunk_size, overlap)):
#                 cleaned_chunk = c
#                 for seq in ["\\n", "\n", "\\r", "\r", "\\t", "\t", "\\\\", "\\"]:
#                     cleaned_chunk = cleaned_chunk.replace(seq, " ")
#                 cleaned_chunk = cleaned_chunk.replace('\\"', '"').replace("\\'", "'")
#                 cleaned_chunk = re.sub(r'\s{2,}', ' ', cleaned_chunk).strip()
#                 cleaned_chunk = re.sub(r'"\s*:\s*"', ' | ', cleaned_chunk)
#                 cleaned_chunk = re.sub(r'"\s*,\s*"', ' | ', cleaned_chunk)
#                 cleaned_chunk = cleaned_chunk.replace('{', '').replace('}', '')
#                 cleaned_chunk = re.sub(r'\s*\|\s*', ' | ', cleaned_chunk)
#                 cleaned_chunk = re.sub(r'\s*\n\s*', '\n', cleaned_chunk)
#                 cleaned_chunk = cleaned_chunk.encode("utf-8", "ignore").decode("utf-8")

#                 prepared.append({
#                     "id": _make_doc_id(namespace, owner_name, app_link_name, report_link_name, idx),
#                     "report_link_name": report_link_name,
#                     "chunk_index": idx,
#                     "text": cleaned_chunk
#                 })
#     return prepared

# # ------------------ OpenAI Embedder ------------------
# @dataclass
# class OpenAIEmbedder:
#     client: Any
#     model: str

#     def embed_documents(self, texts: List[str], retry: int = 3) -> List[List[float]]:
#         for attempt in range(retry):
#             try:
#                 resp = self.client.embeddings.create(model=self.model, input=texts)
#                 return [d.embedding for d in resp.data]
#             except Exception as e:
#                 logger.warning("Embedding attempt %d failed: %s", attempt + 1, e)
#                 time.sleep(1 + attempt)
#         raise RuntimeError("Embedding failed after retries")

# def _get_pinecone_index(pc_client: Any, index_name: str):
#     if hasattr(pc_client, "Index"): return pc_client.Index(index_name)
#     if 'pinecone_client' in globals() and globals().get('pinecone_client'):
#         return globals()['pinecone_client'].Index(index_name)
#     if hasattr(pc_client, "upsert"): return pc_client
#     raise RuntimeError("Unsupported Pinecone client version")

# def upsert_vectors_to_pinecone(index_obj: Any, vectors: List[Tuple[str, List[float], Dict[str, Any]]], namespace: str):
#     if not vectors: return
#     items = [{"id": vid, "values": vec, "metadata": (meta or {})} for vid, vec, meta in vectors]
#     try:
#         return index_obj.upsert(vectors=items, namespace=namespace)
#     except TypeError:
#         return index_obj.upsert(items=items, namespace=namespace)

# def embed_and_upsert(prepared_docs, openai_client, pc_client, index_name, namespace, batch_size=BATCH_SIZE):
#     embedder = OpenAIEmbedder(openai_client, EMBEDDING_MODEL)
#     index_obj = _get_pinecone_index(pc_client, index_name)
#     total = len(prepared_docs)

#     for i in range(0, total, batch_size):
#         batch = prepared_docs[i:i+batch_size]
#         texts = [b["text"] for b in batch]
#         embeddings = embedder.embed_documents(texts)
#         vectors = [
#             (
#                 b["id"],
#                 emb,
#                 {
#                     "owner_name": b.get("owner_name", ""),
#                     "app_link_name": b.get("app_link_name", ""),
#                     "report_link_name": b.get("report_link_name", ""),
#                     "chunk_index": b.get("chunk_index", 0)
#                 }
#             )
#             for b, emb in zip(batch, embeddings)
#         ]
#         upsert_vectors_to_pinecone(index_obj, vectors, namespace)
#         logger.info("Upserted %d/%d vectors (text only)", min(i + batch_size, total), total)

# # ------------------ Fetch & Embed Zoho ------------------
# def _fetch_and_embed_zoho_data(data, company_name, job_id):
#     try:
#         with lock:
#             job_status[job_id]["step"] = "fetching_started"

#         zoho = Zoho(data.get("client_id"), data.get("client_secret"), data.get("refresh_token"))
#         applications = zoho.get_all_applications()
#         if not applications:
#             with lock:
#                 job_status[job_id]["embedding_status"] = "failed"
#                 job_status[job_id]["step"] = "no_applications"
#             return

#         final_result = []
#         for app in applications:
#             owner = app.get("owner_name")
#             link = app.get("app_link_name")
#             reports_resp, _ = zoho.fetch_reports_list(owner_name=owner, app_link_name=link)
#             if not reports_resp.get("success"): continue

#             report_details = []
#             for report in reports_resp.get("reports", []):
#                 link_name = report.get("report_link_name")
#                 report_data, status = zoho.fetch_report_deatils(owner, link, link_name)
#                 if status != 200: continue

#                 if isinstance(report_data, list):
#                     cleaned_rows = []
#                     for row in report_data:
#                         if isinstance(row, dict):
#                             cleaned_row = {k: clean_text(v) for k, v in row.items()}
#                             cleaned_rows.append(cleaned_row)
#                     report_data = cleaned_rows
#                 elif isinstance(report_data, dict):
#                     report_data = {k: clean_text(v) for k, v in report_data.items()}
#                 else:
#                     report_data = clean_text(report_data)

#                 report_details.append({"report_metadata": report, "report_data": report_data})
#             final_result.append({"owner_name": owner, "app_link_name": link, "reports": report_details})

#         payload = {"applications": final_result}
#         namespace = _slug(company_name)
#         prepared_docs = prepare_embedding_docs_from_zoho(payload, namespace)

#         cleaned_docs = [
#             {"id": doc["id"], "report_link_name": doc["report_link_name"], "chunk_index": doc["chunk_index"], "text": doc["text"]}
#             for doc in prepared_docs
#         ]
#         embedding_folder = os.path.join("embeddings", namespace)
#         os.makedirs(embedding_folder, exist_ok=True)
#         embedding_ready_path = os.path.join(embedding_folder, "embedding_ready.json")
#         save_json_file(embedding_ready_path, {
#             "company_name": company_name,
#             "generated_at": datetime.utcnow().isoformat() + "Z",
#             "total_chunks": len(cleaned_docs),
#             "prepared_docs": cleaned_docs
#         })

#         with lock: job_status[job_id]["step"] = "prepared_on_disk"

#         pc_client = Pinecone(api_key=PINECONE_API_KEY)
#         openai_client = OpenAI(api_key=OPENAI_API_KEY)
#         embed_and_upsert(cleaned_docs, openai_client, pc_client, PINECONE_INDEX, namespace)

#         with lock:
#             job_status[job_id]["embedding_status"] = "completed"
#             job_status[job_id]["step"] = "finished"

#     except Exception as e:
#         logger.exception("Error fetching Zoho data", exc_info=e)
#         with lock:
#             job_status[job_id]["embedding_status"] = "failed"
#             job_status[job_id]["step"] = "error"

# # ------------------ Flask Route ------------------
# @zoho_bp.route("/fetch-zoho-details", methods=["POST"])
# def fetch_all_zoho_data():
#     try:
#         data = request.json or {}
#         company_name = data.get("company_name")
#         if not company_name:
#             return jsonify({"error": "'company_name' is required"}), 400
#         if Zoho is None:
#             return jsonify({"success": False, "message": "Zoho client not available."}), 500

#         zoho = Zoho(data.get("client_id"), data.get("client_secret"), data.get("refresh_token"))
#         status, code = zoho.connection_test()
#         if code != 200:
#             return jsonify({"success": False, "message": "Zoho connection failed"}), 401

#         job_id = str(uuid.uuid4())
#         with lock:
#             job_status[job_id] = {"job_id": job_id, "embedding_status": "queued", "step": "queued", "company_name": company_name}

#         threading.Thread(target=_fetch_and_embed_zoho_data, args=(data, company_name, job_id), daemon=True).start()
#         return jsonify({"success": True, "message": "Zoho connection successful. Data fetching started.", "job_id": job_id}), 200

#     except Exception as e:
#         logger.exception("Error in Zoho connection", exc_info=e)
#         return jsonify({"success": False, "error": str(e)}), 500










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
from flask import Flask, Blueprint, request, jsonify
from dotenv import load_dotenv
from app.helper.utils import COMMON

# ------------------ Load .env ------------------
load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_ENV = os.getenv("PINECONE_ENV")
# PINECONE_INDEX = os.getenv("PINECONE_INDEX")
# CLOUD_STORAGE = os.getenv("PINECONE_CLOUD", "aws")
# EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
# DEFAULT_CHUNK_SIZE = int(os.getenv("EMBED_CHUNK_SIZE", 1500))
# DEFAULT_CHUNK_OVERLAP = int(os.getenv("EMBED_CHUNK_OVERLAP", 200))
# BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", 50))


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
DEFAULT_CHUNK_SIZE = int(os.getenv("EMBED_CHUNK_SIZE", 1500))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("EMBED_CHUNK_OVERLAP", 200))
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", 50))
CLOUD_STORAGE = os.getenv("PINECONE_CLOUD", "aws")

# ------------------ Logging ------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ------------------ External Clients ------------------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from pinecone import Pinecone, ServerlessSpec
except Exception:
    Pinecone = None
    try:
        import pinecone as pinecone_client
    except Exception:
        pinecone_client = None

# Placeholder for Zoho connector
from app.models.zoho_connectors import Zoho

# ------------------ Flask Setup ------------------
zoho_bp = Blueprint("zoho", __name__)
job_status: Dict[str, Dict[str, Any]] = {}
lock = threading.Lock()

# ------------------ Utilities ------------------
def _slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    return s[:240]

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json_file(path: str, obj: Any) -> None:
    safe_mkdir(os.path.dirname(path))
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
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0: start = 0
        if start >= len(text): break
    return chunks

def _make_doc_id(namespace: str, owner: str, app: str, report: str, chunk_idx: int) -> str:
    base = f"{namespace}~{owner}~{app}~{report}~{chunk_idx}"
    return _slug(base) + f"~{chunk_idx}"

# ------------------ Embedding Doc Preparation ------------------
def prepare_embedding_docs_from_zoho(data: Dict[str, Any], namespace: str,
                                     chunk_size: int = DEFAULT_CHUNK_SIZE,
                                     overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    apps = data.get("applications") or []

    for app in apps:
        owner_name = app.get("owner_name") or "unknown_owner"
        app_link_name = app.get("app_link_name") or app.get("app_name") or "unknown_app"
        reports = app.get("reports") or []

        for r in reports:
            report_meta = r.get("report_metadata") or {}
            report_data = r.get("report_data") or r.get("data") or r.get("rows") or ""
            report_link_name = report_meta.get("report_link_name") or report_meta.get("name") or "report"

            rows = None
            if isinstance(report_data, list) and all(isinstance(it, dict) for it in report_data):
                rows = report_data
            elif isinstance(report_data, dict) and isinstance(report_data.get("rows"), list):
                if all(isinstance(it, dict) for it in report_data["rows"]):
                    rows = report_data["rows"]

            text_parts = []
            meta_parts = [str(report_meta.get(k)) for k in ("report_link_name", "report_name", "description") if report_meta.get(k)]
            if meta_parts: text_parts.append(" | ".join(meta_parts))

            if rows:
                columns = list(rows[0].keys())
                for row in rows:
                    for k in row.keys():
                        if k not in columns:
                            columns.append(k)
                cleaned_rows = [[clean_text(row.get(c, ""), max_len=10000) for c in columns] for row in rows]
                sample_rows = [" | ".join(columns)]
                for rdata in cleaned_rows[:50]:
                    sample_rows.append(" | ".join(rdata))
                text_parts.append("\n".join(sample_rows))
            else:
                if isinstance(report_data, (list, dict)):
                    report_data_str = json.dumps(report_data, ensure_ascii=False)
                else:
                    report_data_str = str(report_data)
                text_parts.append(clean_text(report_data_str[:200_000]))

            combined = "\n\n".join([clean_text(t) for t in text_parts if t])
            if not combined: continue

            for idx, c in enumerate(chunk_text(combined, chunk_size, overlap)):
                cleaned_chunk = c
                for seq in ["\\n", "\n", "\\r", "\r", "\\t", "\t", "\\\\", "\\"]:
                    cleaned_chunk = cleaned_chunk.replace(seq, " ")
                cleaned_chunk = cleaned_chunk.replace('\\"', '"').replace("\\'", "'")
                cleaned_chunk = re.sub(r'\s{2,}', ' ', cleaned_chunk).strip()
                cleaned_chunk = re.sub(r'"\s*:\s*"', ' | ', cleaned_chunk)
                cleaned_chunk = re.sub(r'"\s*,\s*"', ' | ', cleaned_chunk)
                cleaned_chunk = cleaned_chunk.replace('{', '').replace('}', '')
                cleaned_chunk = re.sub(r'\s*\|\s*', ' | ', cleaned_chunk)
                cleaned_chunk = re.sub(r'\s*\n\s*', '\n', cleaned_chunk)
                cleaned_chunk = cleaned_chunk.encode("utf-8", "ignore").decode("utf-8")

                prepared.append({
                    "id": _make_doc_id(namespace, owner_name, app_link_name, report_link_name, idx),
                    "report_link_name": report_link_name,
                    "chunk_index": idx,
                    "text": cleaned_chunk
                })
    return prepared

# ------------------ Existing OpenAI Embedder (kept intact) ------------------
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

# ------------------ Background Embedding Code Integration ------------------
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings

def clean_text_background(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\\", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_json(file_path: str) -> dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def create_documents_from_prepared_docs(prepared_docs: list) -> list:
    docs = []
    for doc in prepared_docs:
        text = clean_text_background(doc.get("text", ""))
        metadata = {
            "id": doc.get("id"),
            "report_link_name": doc.get("report_link_name"),
            "chunk_index": doc.get("chunk_index")
        }
        docs.append(Document(page_content=text, metadata=metadata))
    return docs

def split_text_safely(documents: list, max_chars: int = 1000) -> list:
    chunks = []
    for doc in documents:
        rows = [r.strip() for r in doc.page_content.split("|") if r.strip()]
        temp_chunk = ""
        chunk_index = 0
        for row in rows:
            if len(temp_chunk) + len(row) + 3 > max_chars:
                metadata = doc.metadata.copy()
                metadata["sub_chunk_index"] = chunk_index
                chunks.append(Document(page_content=temp_chunk.strip(), metadata=metadata))
                temp_chunk = row
                chunk_index += 1
            else:
                temp_chunk += " | " + row if temp_chunk else row
        if temp_chunk:
            metadata = doc.metadata.copy()
            metadata["sub_chunk_index"] = chunk_index
            chunks.append(Document(page_content=temp_chunk.strip(), metadata=metadata))
    return chunks

def generate_embeddings(chunks: list, model: str, api_key: str) -> list:
    embeddings = OpenAIEmbeddings(model=model, openai_api_key=api_key)
    texts = [chunk.page_content for chunk in chunks]
    return embeddings.embed_documents(texts)

def init_pinecone_index(api_key: str, index_name: str, dimension: int, cloud: str, region: str):
    pc = Pinecone(api_key=api_key)
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(name=index_name, dimension=dimension, metric="cosine",
                        spec=ServerlessSpec(cloud=cloud, region=region))
    return pc.Index(name=index_name)

def upsert_vectors(index, chunks: list, vectors: list, namespace: str, batch_size=100):
    to_upsert = []
    for vector, chunk in zip(vectors, chunks):
        unique_id = str(uuid.uuid4())
        metadata = chunk.metadata.copy()
        metadata.update({"source_text": chunk.page_content})
        to_upsert.append({"id": unique_id, "values": vector, "metadata": metadata})
    for i in range(0, len(to_upsert), batch_size):
        batch = to_upsert[i:i + batch_size]
        index.upsert(vectors=batch, namespace=namespace)

def store_embeddings_from_zoho_json(namespace: str, json_file_path: str) -> str:
    logger.info(f"Using namespace: '{namespace}'")
    data = load_json(json_file_path)
    prepared_docs = data.get("prepared_docs", [])
    if not prepared_docs:
        raise ValueError("No prepared_docs found in JSON.")

    documents = create_documents_from_prepared_docs(prepared_docs)
    chunks = split_text_safely(documents)
    vectors = generate_embeddings(chunks, EMBEDDING_MODEL, OPENAI_API_KEY)
    index = init_pinecone_index(PINECONE_API_KEY, PINECONE_INDEX, len(vectors[0]), CLOUD_STORAGE, PINECONE_ENV)
    upsert_vectors(index, chunks, vectors, namespace)
    COMMON.save_name(namespace=namespace,folder_path="web_info",filename="web_info.json")
    logger.info(f"✅ Background embedding completed for namespace '{namespace}'")
    return namespace

# ------------------ Modified Background Integration ------------------
def _fetch_and_embed_zoho_data(data, company_name, job_id):
    try:
        with lock:
            job_status[job_id]["step"] = "fetching_started"

        zoho = Zoho(data.get("client_id"), data.get("client_secret"), data.get("refresh_token"))
        applications = zoho.get_all_applications()
        if not applications:
            with lock:
                job_status[job_id]["embedding_status"] = "failed"
                job_status[job_id]["step"] = "no_applications"
            return

        final_result = []
        for app in applications:
            owner = app.get("owner_name")
            link = app.get("app_link_name")
            reports_resp, _ = zoho.fetch_reports_list(owner_name=owner, app_link_name=link)
            if not reports_resp.get("success"): continue

            report_details = []
            for report in reports_resp.get("reports", []):
                link_name = report.get("report_link_name")
                report_data, status = zoho.fetch_report_deatils(owner, link, link_name)
                if status != 200: continue
                if isinstance(report_data, list):
                    report_data = [{k: clean_text(v) for k, v in r.items()} for r in report_data if isinstance(r, dict)]
                elif isinstance(report_data, dict):
                    report_data = {k: clean_text(v) for k, v in report_data.items()}
                else:
                    report_data = clean_text(report_data)
                report_details.append({"report_metadata": report, "report_data": report_data})
            final_result.append({"owner_name": owner, "app_link_name": link, "reports": report_details})

        payload = {"applications": final_result}
        namespace = _slug(company_name)
        prepared_docs = prepare_embedding_docs_from_zoho(payload, namespace)
        cleaned_docs = [
            {"id": d["id"], "report_link_name": d["report_link_name"], "chunk_index": d["chunk_index"], "text": d["text"]}
            for d in prepared_docs
        ]
        embedding_folder = os.path.join("embeddings", namespace)
        os.makedirs(embedding_folder, exist_ok=True)
        embedding_ready_path = os.path.join(embedding_folder, "embedding_ready.json")
        save_json_file(embedding_ready_path, {
            "company_name": company_name,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "total_chunks": len(cleaned_docs),
            "prepared_docs": cleaned_docs
        })

        with lock:
            job_status[job_id]["step"] = "prepared_on_disk"

        # ✅ Run your background embedding after JSON ready
        store_embeddings_from_zoho_json(namespace, embedding_ready_path)

        with lock:
            job_status[job_id]["embedding_status"] = "completed"
            job_status[job_id]["step"] = "finished"

    except Exception as e:
        logger.exception("Error fetching Zoho data", exc_info=e)
        with lock:
            job_status[job_id]["embedding_status"] = "failed"
            job_status[job_id]["step"] = "error"

# ------------------ Flask Route ------------------
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

@zoho_bp.route("/job-status/<job_id>", methods=["GET"])
def zoho_job_status(job_id: str):
    """
    Fetch the current status of a Zoho data fetch & embedding job.
    """
    try:
        with lock:
            job_info = job_status.get(job_id)

        if not job_info:
            return jsonify({"success": False, "message": f"Invalid or unknown job_id: {job_id}"}), 404

        return jsonify({
            "success": True,
            "job_id": job_id,
            "company_name": job_info.get("company_name"),
            "embedding_status": job_info.get("embedding_status"),
            "step": job_info.get("step"),
            "details": job_info
        }), 200

    except Exception as e:
        logger.exception("Error fetching Zoho job status", exc_info=e)
        return jsonify({"success": False, "error": str(e)}), 500

