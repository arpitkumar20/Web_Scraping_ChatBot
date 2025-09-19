import os
import json
import re
import uuid
from dotenv import load_dotenv
from langchain.schema import Document

from app.core.logger import logger
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv('PINECONE_ENV')
PINECONE_INDEX = os.getenv('PINECONE_INDEX')
CLOUD_STORAGE = os.getenv('CLOUD_STORAGE')
EMBEDDING_MODEL= os.getenv('EMBEDDING_MODEL')


def store_embeddings_from_folder(folder_path: str):
    # Extract site name from folder path
    # try:
    #     site_name = folder_path.split("/www.")[1].split("/")[0].replace('.', '_')
    # except IndexError:
    #     logger.error("Invalid folder path format. Expected to find '/www.<site_name>/' in path.")
    #     return None

    # Regex pattern to extract domain-like folder name (e.g., www.example.com, example.in, my-site.org)
    domain_pattern = re.compile(r'([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})')

    # Search for domain in the folder path
    match = domain_pattern.search(folder_path)
    if not match:
        logger.error("No valid domain found in the folder path.")
        return None

    site_name = match.group(1).replace('.', '_')

    unique_namespace = f"{site_name}_{uuid.uuid4().hex}"
    logger.info(f"Generated unique namespace: {unique_namespace}")

    json_file_path = os.path.join(folder_path, 'embedding_ready.json')

    if not os.path.exists(json_file_path):
        logger.error(f"Embedding file not found at {json_file_path}")
        return None

    logger.info(f"Loading data from JSON file at {json_file_path}...")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts = data.get("texts", [])
    metadatas = data.get("metadatas", [])

    logger.info(f"Loaded {len(texts)} texts and {len(metadatas)} metadatas.")

    if not texts or not metadatas:
        logger.warning("No texts or metadatas found in embedding_ready.json")
        return None

    # Step 1: Create Document objects
    documents = [
        Document(page_content=text, metadata=metadata)
        for text, metadata in zip(texts, metadatas)
    ]
    logger.info(f"Created {len(documents)} Document objects.")

    # Step 2: Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split documents into {len(chunks)} chunks.")

    # Step 3: Generate Embeddings
    logger.info("Generating embeddings using GoogleGenerativeAIEmbeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, api_key=GOOGLE_API_KEY)
    texts_for_embedding = [chunk.page_content for chunk in chunks]
    vector_data = embeddings.embed_documents(texts_for_embedding)
    logger.info(f"Generated {len(vector_data)} embedding vectors.")

    # Step 4: Initialize Pinecone Client
    logger.info("Initializing Pinecone client...")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if PINECONE_INDEX not in [idx.name for idx in pc.list_indexes()]:
        logger.info(f"Creating Pinecone index: {PINECONE_INDEX}")
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud=CLOUD_STORAGE, region=PINECONE_ENV)
        )
    else:
        logger.info(f"Pinecone index '{PINECONE_INDEX}' already exists.")

    index = pc.Index(name=PINECONE_INDEX)

    # Step 5: Prepare and Upsert Vectors
    vectors_to_upsert = []
    for i, (vector, chunk) in enumerate(zip(vector_data, chunks)):
        unique_id = f"{site_name}_{uuid.uuid4().hex}"
        metadata = chunk.metadata.copy()
        metadata.update({"source_text": chunk.page_content})

        vectors_to_upsert.append({
            'id': unique_id,
            'values': vector,
            'metadata': metadata
        })

    # Upsert in batches (if needed, here we use 100 per batch as example)
    batch_size = 100
    for j in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[j:j + batch_size]
        index.upsert(vectors=batch, namespace=unique_namespace)
        logger.info(f"Upserted batch {j // batch_size + 1} into namespace '{unique_namespace}'.")

    logger.info(f"✅ All documents successfully added to namespace '{unique_namespace}'.")
    return unique_namespace
