import os
import re
import uuid
from dotenv import load_dotenv
from threading import Thread, Lock
from langchain.schema import Document
from flask import Blueprint, request, jsonify

from app.helper.utils import COMMON
from app.core.logging import get_logger
from app.services.file_content import process_file
from app.services.embeddings_store_v2 import generate_embeddings, init_pinecone_index, upsert_vectors

load_dotenv()

# ----------------- Config & Logger -----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
CLOUD_STORAGE = os.getenv("CLOUD_STORAGE")
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')

extract_documents = Blueprint("documents", __name__)
logger = get_logger(__name__)

lock = Lock()
job_status = {}

# # ----------------- API Routes -----------------
@extract_documents.route('/extract-content', methods=['POST'])
def file_content():
    """
    Extract text from uploaded files and start background embedding.
    Returns a list of job_ids for status tracking.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No files part'}), 400

        files = request.files.getlist('file')
        if not files or any(f.filename == '' for f in files):
            return jsonify({'error': 'No selected files'}), 400

        company_name = request.form.get("company_name")
        if not company_name:
            return jsonify({'error': "'company_name' is required"}), 400

        jobs_info = []

        for file in files:
            job_id = str(uuid.uuid4())
            result = process_file(file)
            raw_text = result['extracted_text']

            with lock:
                job_status[job_id] = {
                    "job_id": job_id,
                    "embedding_status": "queued",
                    "step": "queued",
                    "namespace": None,
                    "error": None,
                    "filename": file.filename
                }

            Thread(
                target=store_embeddings_from_text,
                args=(company_name, raw_text, job_id),
                daemon=True
            ).start()

            jobs_info.append({"filename": file.filename, "job_id": job_id})

        return jsonify({
            "status": "embedding_started",
            "jobs": jobs_info
        })

    except Exception as e:
        logger.exception(f"Error in /extract-content: {e}")
        return jsonify({'error': str(e)}), 500

@extract_documents.route('/job-status/<job_id>', methods=['GET'])
def embedding_status(job_id):
    """
    Fetch the current status of a file embedding job.
    """
    with lock:
        if job_id not in job_status:
            return jsonify({"error": "Invalid job_id"}), 404
        return jsonify(job_status[job_id])

def split_documents(documents: list, chunk_size=1000, chunk_overlap=150) -> list:
    from langchain.text_splitter import CharacterTextSplitter
    try:
        from langchain.schema import Document
    except Exception:
        from langchain.docstore.document import Document  # older LC

    def to_doc(d):
        return d if isinstance(d, Document) else Document(
            page_content=d.get("page_content", d.get("text", "")),
            metadata=d.get("metadata", {}) if isinstance(d.get("metadata", {}), dict) else {}
        )

    def detect_pk_field(docs):
        common = {"pk", "id", "doc_id", "unique_id", "primary_key"}
        for d in docs:
            md = getattr(d, "metadata", None) or (d.get("metadata", {}) if isinstance(d, dict) else {})
            if isinstance(md, dict):
                for k in md:
                    if k.lower() in common:
                        return k
        return None

    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    pk_field = detect_pk_field(documents)

    if not pk_field:
        # original behavior
        return splitter.split_documents([to_doc(d) for d in documents])

    # group by pk, but DO NOT concatenate; split rowwise within each group
    groups = {}
    for d in documents:
        doc = to_doc(d)
        pk_val = doc.metadata.get(pk_field)
        groups.setdefault(pk_val, []).append(doc)

    chunks = []
    for _, group in groups.items():
        # split each doc in the group individually to keep it rowwise
        pieces = splitter.split_documents(group)
        chunks.extend(pieces)

    return chunks

def _row_docs_from_text(raw_text: str):
    # split on blank lines OR long separators (---, ***, ===), fallback to single lines
    parts = [p.strip() for p in re.split(r'(?:\n{2,}|^[\-=*]{3,}$)', raw_text, flags=re.MULTILINE) if p and p.strip()]
    if len(parts) <= 1:
        # fallback: split per non-empty line
        parts = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]

    return [Document(page_content=part, metadata={"pk": str(i)}) for i, part in enumerate(parts)]

# ----------------- Main Function -----------------
def store_embeddings_from_text(company_name: str, raw_text: str, job_id: str) -> str:
    """
    Process raw text input (no file), generate embeddings using Gemini,
    and store them in Pinecone for retrieval.
    Updates job_status using job_id.
    """
    try:
        if not raw_text or not raw_text.strip():
            raise ValueError("No valid text provided for embedding.")

        namespace = company_name
        logger.info(f"Using company_name as namespace: {namespace}")

        # --- Update job status: processing started ---
        with lock:
            if job_id in job_status:
                job_status[job_id].update({"embedding_status": "processing", "step": "creating_documents"})

        # Step 1: Create a single Document with basic metadata
        documents = _row_docs_from_text(raw_text)

        logger.info(f"Created {len(documents)} Document object(s).")

        # --- Update job status: splitting documents ---
        with lock:
            if job_id in job_status:
                job_status[job_id].update({"step": "splitting_documents"})

        # Step 2: Split into chunks for higher embedding precision
        chunks = split_documents(documents, chunk_size=1000, chunk_overlap=150)
        logger.info(f"Split text into {len(chunks)} chunks for embedding.")

        # --- Update job status: generating embeddings ---
        with lock:
            if job_id in job_status:
                job_status[job_id].update({"step": "generating_embeddings"})

        # Step 3: Generate embeddings using Gemini 1.5 Pro
        vectors = generate_embeddings(chunks, EMBEDDING_MODEL, OPENAI_API_KEY)
        if not vectors:
            raise RuntimeError("Embedding generation failed or returned empty results.")
        logger.info(f"Generated {len(vectors)} embedding vectors.")

        # --- Update job status: initializing Pinecone ---
        with lock:
            if job_id in job_status:
                job_status[job_id].update({"step": "initializing_pinecone"})

        # Step 4: Initialize Pinecone index (create if missing)
        dimension = len(vectors[0]) if vectors else 768
        index = init_pinecone_index(
            api_key=PINECONE_API_KEY,
            index_name=PINECONE_INDEX,
            dimension=dimension,
            cloud=CLOUD_STORAGE,
            region=PINECONE_ENV
        )

        # --- Update job status: upserting vectors ---
        with lock:
            if job_id in job_status:
                job_status[job_id].update({"step": "upserting_vectors"})
        

         # Step 5: Prepare minimal metadata for each chunk to avoid 40 KB limit
        safe_metadata_list = []
        for i, chunk in enumerate(chunks):
            # Only store chunk index and source; avoid full text
            meta = {
                "source": str(chunk),
                "chunk_id": i
            }
            safe_metadata_list.append(meta)

        # Step 5: Upsert vectors into Pinecone
        upsert_vectors(index, chunks, vectors, namespace, safe_metadata_list)
        logger.info(f"✅ Successfully embedded and stored data in namespace '{namespace}'.")

        # --- Update job status: completed ---
        with lock:
            if job_id in job_status:
                job_status[job_id].update({"embedding_status": "completed", "step": "done", "namespace": namespace})

        COMMON.save_name(namespace=namespace,folder_path="web_info",filename="web_info.json")
        return namespace

    except Exception as e:
        logger.exception(f"Error embedding text for namespace '{company_name}': {e}")
        # --- Update job status: error ---
        with lock:
            if job_id in job_status:
                job_status[job_id].update({"embedding_status": "failed", "step": "error", "error": str(e)})
        return None