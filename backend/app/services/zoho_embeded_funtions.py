import os
import logging
import json
import re
import uuid
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

# ------------------ Load .env ------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
CLOUD_STORAGE = os.getenv("CLOUD_STORAGE")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------ Utility Functions ------------------

def clean_text(text: str) -> str:
    """
    Clean text to remove unwanted characters and preserve table structure.
    """
    if not text:
        return ""
    text = text.replace("\\", " ")  # remove backslashes
    text = re.sub(r"\s+", " ", text)  # collapse multiple spaces/newlines
    text = text.strip()
    return text

def load_json(file_path: str) -> dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_documents_from_prepared_docs(prepared_docs: list) -> list:
    """
    Only extract 'text' field from prepared_docs and create Document objects.
    """
    documents = []
    for doc in prepared_docs:
        text = clean_text(doc.get("text", ""))
        metadata = {
            "id": doc.get("id"),
            "report_link_name": doc.get("report_link_name"),
            "chunk_index": doc.get("chunk_index")
        }
        documents.append(Document(page_content=text, metadata=metadata))
    return documents

def split_text_safely(documents: list, max_chars: int = 1000) -> list:
    """
    Split each document by rows ('|') and make sure no chunk exceeds max_chars.
    """
    chunks = []

    for doc in documents:
        rows = [r.strip() for r in doc.page_content.split("|") if r.strip()]
        temp_chunk = ""
        chunk_index = 0

        for row in rows:
            # If adding this row exceeds max_chars, finalize current chunk
            if len(temp_chunk) + len(row) + 3 > max_chars:
                metadata = doc.metadata.copy()
                metadata["sub_chunk_index"] = chunk_index
                chunks.append(Document(page_content=temp_chunk.strip(), metadata=metadata))
                temp_chunk = row
                chunk_index += 1
            else:
                temp_chunk += " | " + row if temp_chunk else row

        # Add remaining text
        if temp_chunk:
            metadata = doc.metadata.copy()
            metadata["sub_chunk_index"] = chunk_index
            chunks.append(Document(page_content=temp_chunk.strip(), metadata=metadata))

    return chunks


def generate_embeddings(chunks: list, model: str, api_key: str) -> list:
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY.")
    logger.info(f"Using OpenAI model '{model}' for embeddings.")
    embeddings = OpenAIEmbeddings(model=model, openai_api_key=api_key)
    texts = [chunk.page_content for chunk in chunks]
    return embeddings.embed_documents(texts)

def init_pinecone_index(api_key: str, index_name: str, dimension: int, cloud: str, region: str):
    pc = Pinecone(api_key=api_key)
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        logger.info(f"Creating Pinecone index '{index_name}' with dim={dimension} ...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region)
        )
    else:
        logger.info(f"Pinecone index '{index_name}' already exists.")
    return pc.Index(name=index_name)

def upsert_vectors(index, chunks: list, vectors: list, namespace: str, batch_size=100):
    vectors_to_upsert = []
    for i, (vector, chunk) in enumerate(zip(vectors, chunks)):
        unique_id = str(uuid.uuid4())
        metadata = chunk.metadata.copy()
        metadata.update({"source_text": chunk.page_content})
        vectors_to_upsert.append({
            "id": unique_id,
            "values": vector,
            "metadata": metadata
        })

    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i + batch_size]
        index.upsert(vectors=batch, namespace=namespace)
        logger.info(f"Upserted batch {i // batch_size + 1} into namespace '{namespace}'.")

# ------------------ Main Function ------------------

def store_embeddings_from_zoho_json(namespace: str, json_file_path: str) -> str:
    """
    Generate embeddings from Zoho JSON and store in Pinecone.
    """
    try:
        logger.info(f"Using namespace: '{namespace}'")
        data = load_json(json_file_path)

        prepared_docs = data.get("prepared_docs", [])
        if not prepared_docs:
            raise ValueError("No prepared_docs found in JSON.")

        # Create Document objects from 'text'
        documents = create_documents_from_prepared_docs(prepared_docs)
        logger.info(f"Created {len(documents)} Document objects.")

        # Split long documents row-wise to avoid token limit
        chunks = split_text_safely(documents, max_chars=1000)
        logger.info(f"Split into {len(chunks)} safe chunks for embeddings.")


        # Generate embeddings
        vectors = generate_embeddings(chunks, EMBEDDING_MODEL, OPENAI_API_KEY)
        logger.info(f"Generated {len(vectors)} embeddings.")

        # Initialize Pinecone
        vector_dim = len(vectors[0])
        index = init_pinecone_index(PINECONE_API_KEY, PINECONE_INDEX, vector_dim, CLOUD_STORAGE, PINECONE_ENV)

        # Upsert embeddings
        upsert_vectors(index, chunks, vectors, namespace)
        logger.info(f"✅ All documents successfully added to namespace '{namespace}'.")
        return namespace

    except Exception as e:
        logger.error(f"Error embedding documents: {e}")
        return None

# ------------------ Example Usage ------------------
# if __name__ == "__main__":
#     namespace = "zoho_reports"
#     json_file_path = "backend/embeddings/zoho_2/embedding_ready.json"  # <-- Update path
#     store_embeddings_from_zoho_json(namespace, json_file_path)