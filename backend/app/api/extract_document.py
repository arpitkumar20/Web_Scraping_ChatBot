import os
import tempfile
import uuid
from dotenv import load_dotenv
from threading import Thread, Lock
from app.core.logging import get_logger
from flask import Blueprint, request, jsonify
from app.services.file_content import process_file
from app.services.embeddings_store_v2 import create_documents, split_documents, generate_embeddings, init_pinecone_index, upsert_vectors
from app.services.zoho_embeded_funtions import load_csv, csv_to_string, split_text_into_chunks_data
from app.helper.utils import COMMON

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

# ----------------- API Routes -----------------
@extract_documents.route('/extract-content', methods=['POST'])
def file_content():
    """
    Extract text from uploaded file and start background embedding.
    Returns a job_id for status tracking.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if not file.filename:
            return jsonify({'error': 'No selected file'}), 400

        company_name = request.form.get("company_name")
        if not company_name:
            return jsonify({'error': "'company_name' is required"}), 400

        job_id = str(uuid.uuid4())
        if file.filename.lower().endswith('.csv'):
            # Save uploaded CSV to a temporary file before thread starts
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, file.filename)
            file.save(temp_path)

            with lock:
                job_status[job_id] = {
                    "job_id": job_id,
                    "embedding_status": "queued",
                    "step": "queued",
                    "namespace": None,
                    "error": None,
                    "filename": file.filename
                }

            # Pass the safe file path, not the stream
            Thread(
                target=process_csv_to_embedding_with_job,
                args=(company_name, temp_path, job_id),
                daemon=True
            ).start()

            return jsonify({
                "status": "embedding_started",
                "job_id": job_id,
                "filename": file.filename
            })
        else:
            result = process_file(file)
            raw_text = result['extracted_text']
            # return raw_text
            # --- Create job_id for background embedding ---
            
            with lock:
                job_status[job_id] = {
                    "job_id": job_id,
                    "embedding_status": "queued",
                    "step": "queued",
                    "namespace": None,
                    "error": None,
                    "filename": file.filename
                }

            # --- Start embedding in background ---
            Thread(target=store_embeddings_from_text, args=(company_name, raw_text, job_id), daemon=True).start()

            return jsonify({
                "status": "embedding_started",
                "job_id": job_id,
                "filename": file.filename
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
        metadata = {"source": "raw_text_input"}
        documents = create_documents([raw_text], [metadata])
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



def process_csv_to_embedding_with_job(
    namespace: str,
    file_path: str,
    job_id: str
):
    """
    Process a CSV file: load, convert to string, chunk, embed, and upsert to Pinecone.
    Updates job_status using job_id.
    """
    try:
        # --- Update job status: started ---
        with lock:
            if job_id in job_status:
                job_status[job_id].update({"embedding_status": "processing", "step": "loading_csv"})

        # Step 1: Load CSV
        df = load_csv(file_path)
        logger.info(f"✅ CSV loaded with {len(df)} rows and {len(df.columns)} columns.")

        # --- Update job status: converting CSV to text ---
        with lock:
            if job_id in job_status:
                job_status[job_id].update({"step": "converting_csv_to_text"})

        # Step 2: Convert CSV to string
        csv_text = csv_to_string(df)

        # --- Update job status: splitting text ---
        with lock:
            if job_id in job_status:
                job_status[job_id].update({"step": "splitting_text"})

        # Step 3: Split into chunks
        chunks = split_text_into_chunks_data(csv_text)
        logger.info(f"Split into {len(chunks)} safe chunks for embeddings.")

        # --- Update job status: generating embeddings ---
        with lock:
            if job_id in job_status:
                job_status[job_id].update({"step": "generating_embeddings"})

        # Step 4: Generate embeddings
        vectors = generate_embeddings(chunks, EMBEDDING_MODEL, OPENAI_API_KEY)
        if not vectors:
            raise RuntimeError("Embedding generation failed or returned empty results.")
        logger.info(f"Generated {len(vectors)} embeddings.")

        # --- Update job status: initializing Pinecone ---
        with lock:
            if job_id in job_status:
                job_status[job_id].update({"step": "initializing_pinecone"})

        # Step 5: Initialize Pinecone index
        vector_dim = len(vectors[0])
        index = init_pinecone_index(PINECONE_API_KEY, PINECONE_INDEX, vector_dim, CLOUD_STORAGE, PINECONE_ENV)

        # --- Update job status: upserting vectors ---
        with lock:
            if job_id in job_status:
                job_status[job_id].update({"step": "upserting_vectors"})

        # Step 6: Prepare minimal metadata to avoid Pinecone limits
        safe_metadata_list = []
        for i, chunk in enumerate(chunks):
            meta = {"source": str(chunk), "chunk_id": i}
            safe_metadata_list.append(meta)

        # Upsert vectors into Pinecone
        upsert_vectors(index, chunks, vectors, namespace, safe_metadata_list)
        logger.info(f"✅ All documents successfully added to namespace '{namespace}'.")

        # --- Update job status: completed ---
        with lock:
            if job_id in job_status:
                job_status[job_id].update({"embedding_status": "completed", "step": "done", "namespace": namespace})

        COMMON.save_name(namespace=namespace, folder_path="web_info", filename="web_info.json")
        return index, chunks, vectors

    except Exception as e:
        logger.exception(f"Error embedding CSV for namespace '{namespace}': {e}")
        # --- Update job status: failed ---
        with lock:
            if job_id in job_status:
                job_status[job_id].update({"embedding_status": "failed", "step": "error", "error": str(e)})
        return None, None, None