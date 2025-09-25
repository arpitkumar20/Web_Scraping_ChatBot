import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

from pinecone import Pinecone, exceptions as pinecone_exceptions
from langchain.schema import Document


# ---------------- Metadata Sanitizer ---------------- #
def sanitize_metadata(metadata: dict, max_size: int = 40960) -> dict:
    """Ensure Pinecone metadata is valid (string/number/bool/list[str]) and within size limits."""
    if not isinstance(metadata, dict):
        return {}

    def fix_value(value):
        if value is None:
            return ""  # replace null
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, list):
            return [str(v) for v in value]  # force all strings
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)  # dict → JSON string
        return str(value)  # fallback to string

    clean_meta = {k: fix_value(v) for k, v in metadata.items()}

    # Enforce Pinecone metadata size limit
    meta_str = json.dumps(clean_meta, ensure_ascii=False)
    if len(meta_str.encode("utf-8")) > max_size:
        logging.warning(
            f"⚠ Metadata too large ({len(meta_str)} bytes). Truncating to {max_size}."
        )
        reduced = {}
        size = 0
        for k, v in clean_meta.items():
            v_str = json.dumps(v, ensure_ascii=False)
            v_size = len(v_str.encode("utf-8"))
            if size + v_size < max_size:
                reduced[k] = v
                size += v_size
            else:
                logging.debug(f"Dropped metadata field '{k}' to fit size limit.")
        return reduced

    return clean_meta


# ---------------- Embedding Store Function ---------------- #
def store_embeddings_from_folder(folder_path: str, index_name: str = "nisaa-knowledge"):
    """
    Load embeddings from embedding_ready.json and store them into Pinecone.
    """
    try:
        # Load Pinecone
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not set in environment.")

        pc = Pinecone(api_key=api_key)

        # Ensure index exists
        if index_name not in [idx["name"] for idx in pc.list_indexes()]:
            raise ValueError(f"Pinecone index '{index_name}' does not exist.")

        index = pc.Index(index_name)

        # Load embedding_ready.json
        embedding_file = Path(folder_path) / "chunks" / "embedding_ready.json"
        if not embedding_file.exists():
            raise FileNotFoundError(f"embedding_ready.json not found at {embedding_file}")

        logging.info(f"📂 Loading embedding data from {embedding_file} ...")
        with open(embedding_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        texts = data.get("texts", [])
        metadatas = data.get("metadatas", [])
        embeddings = data.get("embeddings", [])

        if not (texts and metadatas and embeddings):
            raise ValueError("Embedding file missing required fields (texts/metadatas/embeddings).")

        if not (len(texts) == len(metadatas) == len(embeddings)):
            raise ValueError("Mismatch between texts, metadatas, and embeddings lengths.")

        logging.info(f"✅ Loaded {len(texts)} embeddings from file.")

        # Prepare batches
        batch_size = 100
        total = len(embeddings)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = []
            for i in range(start, end):
                vector = embeddings[i]
                metadata = sanitize_metadata(metadatas[i])
                batch.append((str(i), vector, metadata))

            try:
                index.upsert(vectors=batch, namespace=folder_path.replace("\\", "_").replace("/", "_"))
                logging.info(
                    f"⬆️ Upserted batch {start // batch_size + 1} "
                    f"({len(batch)} vectors) into namespace '{folder_path}'."
                )
            except pinecone_exceptions.PineconeApiException as e:
                logging.error(f"❌ Pinecone API Error during upsert: {e}")
                continue

        logging.info(f"🎉 Finished storing {total} embeddings into Pinecone index '{index_name}'.")
        return True

    except Exception as e:
        logging.error(f"❌ Error in store_embeddings_from_folder: {e}", exc_info=True)
        return False
