# # import os
# # import logging
# # import json
# # import re
# # import uuid
# # from dotenv import load_dotenv
# # from langchain.schema import Document
# # from langchain_text_splitters import CharacterTextSplitter
# # from langchain_google_genai import GoogleGenerativeAIEmbeddings
# # from pinecone import Pinecone, ServerlessSpec

# # load_dotenv()

# # GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# # PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# # PINECONE_ENV = os.getenv('PINECONE_ENV')
# # PINECONE_INDEX = os.getenv('PINECONE_INDEX')
# # CLOUD_STORAGE = os.getenv('CLOUD_STORAGE')
# # EMBEDDING_MODEL= os.getenv('EMBEDDING_MODEL')

# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)


# # def store_embeddings_from_folder(folder_path: str):
# #     # Extract site name from folder path
# #     # try:
# #     #     site_name = folder_path.split("/www.")[1].split("/")[0].replace('.', '_')
# #     # except IndexError:
# #     #     logger.error("Invalid folder path format. Expected to find '/www.<site_name>/' in path.")
# #     #     return None

# #     # Regex pattern to extract domain-like folder name (e.g., www.example.com, example.in, my-site.org)
# #     domain_pattern = re.compile(r'([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})')

# #     # Search for domain in the folder path
# #     match = domain_pattern.search(folder_path)
# #     if not match:
# #         logger.error("No valid domain found in the folder path.")
# #         return None

# #     site_name = match.group(1).replace('.', '_')

# #     unique_namespace = f"{site_name}_{uuid.uuid4().hex}"
# #     logger.info(f"Generated unique namespace: {unique_namespace}")

# #     json_file_path = os.path.join(folder_path, 'embedding_ready.json')

# #     if not os.path.exists(json_file_path):
# #         logger.error(f"Embedding file not found at {json_file_path}")
# #         return None

# #     logger.info(f"Loading data from JSON file at {json_file_path}...")
# #     with open(json_file_path, 'r', encoding='utf-8') as f:
# #         data = json.load(f)

# #     texts = data.get("texts", [])
# #     metadatas = data.get("metadatas", [])

# #     logger.info(f"Loaded {len(texts)} texts and {len(metadatas)} metadatas.")

# #     if not texts or not metadatas:
# #         logger.warning("No texts or metadatas found in embedding_ready.json")
# #         return None

# #     # Step 1: Create Document objects
# #     documents = [
# #         Document(page_content=text, metadata=metadata)
# #         for text, metadata in zip(texts, metadatas)
# #     ]
# #     logger.info(f"Created {len(documents)} Document objects.")

# #     # Step 2: Split documents into chunks
# #     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
# #     chunks = text_splitter.split_documents(documents)
# #     logger.info(f"Split documents into {len(chunks)} chunks.")

# #     # Step 3: Generate Embeddings
# #     logger.info("Generating embeddings using GoogleGenerativeAIEmbeddings...")
# #     embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, api_key=GOOGLE_API_KEY)
# #     texts_for_embedding = [chunk.page_content for chunk in chunks]
# #     vector_data = embeddings.embed_documents(texts_for_embedding)
# #     logger.info(f"Generated {len(vector_data)} embedding vectors.")

# #     # Step 4: Initialize Pinecone Client
# #     logger.info("Initializing Pinecone client...")
# #     pc = Pinecone(api_key=PINECONE_API_KEY)

# #     if PINECONE_INDEX not in [idx.name for idx in pc.list_indexes()]:
# #         logger.info(f"Creating Pinecone index: {PINECONE_INDEX}")
# #         pc.create_index(
# #             name=PINECONE_INDEX,
# #             dimension=768,
# #             metric="cosine",
# #             spec=ServerlessSpec(cloud=CLOUD_STORAGE, region=PINECONE_ENV)
# #         )
# #     else:
# #         logger.info(f"Pinecone index '{PINECONE_INDEX}' already exists.")

# #     index = pc.Index(name=PINECONE_INDEX)

# #     # Step 5: Prepare and Upsert Vectors
# #     vectors_to_upsert = []
# #     for i, (vector, chunk) in enumerate(zip(vector_data, chunks)):
# #         unique_id = f"{site_name}_{uuid.uuid4().hex}"
# #         metadata = chunk.metadata.copy()
# #         metadata.update({"source_text": chunk.page_content})

# #         vectors_to_upsert.append({
# #             'id': unique_id,
# #             'values': vector,
# #             'metadata': metadata
# #         })

# #     # Upsert in batches (if needed, here we use 100 per batch as example)
# #     batch_size = 100
# #     for j in range(0, len(vectors_to_upsert), batch_size):
# #         batch = vectors_to_upsert[j:j + batch_size]
# #         index.upsert(vectors=batch, namespace=unique_namespace)
# #         logger.info(f"Upserted batch {j // batch_size + 1} into namespace '{unique_namespace}'.")

# #     logger.info(f"✅ All documents successfully added to namespace '{unique_namespace}'.")
# #     return unique_namespace




# import os
# import logging
# import json
# import re
# import uuid
# from dotenv import load_dotenv
# from langchain.schema import Document
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from pinecone import Pinecone, ServerlessSpec

# load_dotenv()

# # ----------------- Config & Logger -----------------
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_ENV = os.getenv("PINECONE_ENV")
# PINECONE_INDEX = os.getenv("PINECONE_INDEX")
# CLOUD_STORAGE = os.getenv("CLOUD_STORAGE")
# EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # ----------------- Utility Functions -----------------
# def extract_site_name(folder_path: str) -> str:
#     """Extract a domain-like site name from the folder path."""
#     domain_pattern = re.compile(r'([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})')
#     match = domain_pattern.search(folder_path)
#     if not match:
#         raise ValueError("No valid domain found in the folder path.")
#     return match.group(1).replace('.', '_')

# def load_json(file_path: str) -> dict:
#     """Load JSON data from a file."""
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"File not found at {file_path}")
#     with open(file_path, 'r', encoding='utf-8') as f:
#         return json.load(f)

# def create_documents(texts: list, metadatas: list) -> list:
#     """Convert texts and metadatas to Document objects."""
#     if not texts or not metadatas:
#         raise ValueError("No texts or metadatas found to create documents.")
#     return [Document(page_content=text, metadata=metadata) for text, metadata in zip(texts, metadatas)]

# def split_documents(documents: list, chunk_size=1000, chunk_overlap=150) -> list:
#     """Split documents into smaller chunks."""
#     splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     return splitter.split_documents(documents)

# def generate_embeddings(chunks: list, model: str, api_key: str) -> list:
#     """Generate embeddings for each chunk."""
#     embeddings = GoogleGenerativeAIEmbeddings(model=model, api_key=api_key)
#     texts = [chunk.page_content for chunk in chunks]
#     return embeddings.embed_documents(texts)

# def init_pinecone_index(api_key: str, index_name: str, dimension: int, cloud: str, region: str):
#     """Initialize Pinecone client and create index if it doesn't exist."""
#     pc = Pinecone(api_key=api_key)
#     if index_name not in [idx.name for idx in pc.list_indexes()]:
#         logger.info(f"Creating Pinecone index '{index_name}'...")
#         pc.create_index(name=index_name, dimension=dimension, metric="cosine",
#                         spec=ServerlessSpec(cloud=cloud, region=region))
#     else:
#         logger.info(f"Pinecone index '{index_name}' already exists.")
#     return pc.Index(name=index_name)

# # def upsert_vectors(index, chunks: list, vectors: list, namespace: str, site_name: str, batch_size=100):
# #     """Prepare and upsert vectors into Pinecone."""
# #     vectors_to_upsert = []
# #     for vector, chunk in zip(vectors, chunks):
# #         unique_id = f"{site_name}_{uuid.uuid4().hex}"
# #         metadata = chunk.metadata.copy()
# #         metadata.update({"source_text": chunk.page_content})
# #         vectors_to_upsert.append({'id': unique_id, 'values': vector, 'metadata': metadata})

# #     for j in range(0, len(vectors_to_upsert), batch_size):
# #         batch = vectors_to_upsert[j:j + batch_size]
# #         index.upsert(vectors=batch, namespace=namespace)
# #         logger.info(f"Upserted batch {j // batch_size + 1} into namespace '{namespace}'.")

# # # ----------------- Main Function -----------------
# # def store_embeddings_from_folder(folder_path: str) -> str:
# #     """Main function to process folder, generate embeddings, and store in Pinecone."""
# #     try:
# #         site_name = extract_site_name(folder_path)
# #         namespace = f"{site_name}_{uuid.uuid4().hex}"
# #         logger.info(f"Generated unique namespace: {namespace}")

# #         data = load_json(os.path.join(folder_path, 'embedding_ready.json'))
# #         documents = create_documents(data.get("texts", []), data.get("metadatas", []))
# #         logger.info(f"Created {len(documents)} Document objects.")

# #         chunks = split_documents(documents)
# #         logger.info(f"Split documents into {len(chunks)} chunks.")

# #         vectors = generate_embeddings(chunks, EMBEDDING_MODEL, GOOGLE_API_KEY)
# #         logger.info(f"Generated {len(vectors)} embedding vectors.")

# #         index = init_pinecone_index(PINECONE_API_KEY, PINECONE_INDEX, dimension=768, cloud=CLOUD_STORAGE, region=PINECONE_ENV)
# #         upsert_vectors(index, chunks, vectors, namespace, site_name)

# #         logger.info(f"✅ All documents successfully added to namespace '{namespace}'.")
# #         return namespace

# #     except Exception as e:
# #         logger.error(f"Error processing folder '{folder_path}': {e}")
# #         return None


# def upsert_vectors(index, chunks: list, vectors: list, namespace: str, batch_size=100):
#     """Prepare and upsert vectors into Pinecone using company_name as namespace."""
#     vectors_to_upsert = []
#     for vector, chunk in zip(vectors, chunks):
#         unique_id = f"{uuid.uuid4().hex}"
#         metadata = chunk.metadata.copy()
#         metadata.update({"source_text": chunk.page_content})
#         vectors_to_upsert.append({'id': unique_id, 'values': vector, 'metadata': metadata})

#     for j in range(0, len(vectors_to_upsert), batch_size):
#         batch = vectors_to_upsert[j:j + batch_size]
#         index.upsert(vectors=batch, namespace=namespace)
#         logger.info(f"Upserted batch {j // batch_size + 1} into namespace '{namespace}'.")

# # ----------------- Main Function -----------------
# def store_embeddings_from_folder(company_name: str, folder_path: str) -> str:
#     """Main function to process folder, generate embeddings, and store in Pinecone."""
#     try:
#         namespace = company_name
#         logger.info(f"Using company_name as namespace: {namespace}")

#         data = load_json(os.path.join(folder_path, 'embedding_ready.json'))
#         documents = create_documents(data.get("texts", []), data.get("metadatas", []))
#         logger.info(f"Created {len(documents)} Document objects.")

#         chunks = split_documents(documents)
#         logger.info(f"Split documents into {len(chunks)} chunks.")

#         vectors = generate_embeddings(chunks, EMBEDDING_MODEL, GOOGLE_API_KEY)
#         logger.info(f"Generated {len(vectors)} embedding vectors.")

#         index = init_pinecone_index(PINECONE_API_KEY, PINECONE_INDEX, dimension=768, cloud=CLOUD_STORAGE, region=PINECONE_ENV)
#         upsert_vectors(index, chunks, vectors, namespace)

#         logger.info(f"✅ All documents successfully added to namespace '{namespace}'.")
#         return namespace

#     except Exception as e:
#         logger.error(f"Error processing folder '{folder_path}': {e}")
#         return None





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

def extract_site_name(folder_path: str) -> str:
    domain_pattern = re.compile(r'([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})')
    match = domain_pattern.search(folder_path)
    if not match:
        raise ValueError("No valid domain found in the folder path.")
    return match.group(1).replace('.', '_')

def load_json(file_path: str) -> dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_documents(texts: list, metadatas: list) -> list:
    if not texts or not metadatas:
        raise ValueError("No texts or metadatas found to create documents.")
    return [Document(page_content=text, metadata=metadata) for text, metadata in zip(texts, metadatas)]

def split_documents(documents: list, chunk_size=1000, chunk_overlap=150) -> list:
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

def generate_embeddings(chunks: list, model: str, api_key: str) -> list:
    """Generate embeddings using OpenAI model."""
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY.")
    
    logger.info(f"Using OpenAI model '{model}' for embeddings.")
    embeddings = OpenAIEmbeddings(model=model, openai_api_key=api_key)
    texts = [chunk.page_content for chunk in chunks]
    return embeddings.embed_documents(texts)

def init_pinecone_index(api_key: str, index_name: str, dimension: int, cloud: str, region: str):
    """Initialize Pinecone client and create index if not exists."""
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

def upsert_vectors(index, chunks: list, vectors: list, namespace: str, metadata_list: list = None, batch_size=100):
    """Upsert vectors to Pinecone safely with optional metadata_list."""
    vectors_to_upsert = []

    for i, (vector, chunk) in enumerate(zip(vectors, chunks)):
        unique_id = str(uuid.uuid4())

        # Use provided metadata_list if given, else default to chunk metadata
        if metadata_list:
            metadata = metadata_list[i]
        else:
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

def store_embeddings_from_folder(company_name: str, folder_path: str) -> str:
    """Main function: generate and upload embeddings to Pinecone."""
    try:
        namespace = company_name
        logger.info(f"Using company_name as namespace: {namespace}")

        data_path = os.path.join(folder_path, "embedding_ready.json")
        data = load_json(data_path)
        documents = create_documents(data.get("texts", []), data.get("metadatas", []))
        logger.info(f"Created {len(documents)} Document objects.")

        chunks = split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks.")

        vectors = generate_embeddings(chunks, EMBEDDING_MODEL, OPENAI_API_KEY)
        logger.info(f"Generated {len(vectors)} embeddings.")

        vector_dim = len(vectors[0])
        index = init_pinecone_index(PINECONE_API_KEY, PINECONE_INDEX, vector_dim, CLOUD_STORAGE, PINECONE_ENV)
        upsert_vectors(index, chunks, vectors, namespace)

        logger.info(f"✅ All documents successfully added to namespace '{namespace}'.")
        return namespace

    except Exception as e:
        logger.error(f"Error embedding documents: {e}")
        return None
