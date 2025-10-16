# import os
# import logging
# import json
# import re
# import uuid
# from dotenv import load_dotenv
# from langchain.schema import Document
# from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from pinecone import Pinecone, ServerlessSpec
# from sentence_transformers import SentenceTransformer

# # ------------------ Load .env ------------------
# load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_ENV = os.getenv("PINECONE_ENV")
# PINECONE_INDEX = os.getenv("PINECONE_INDEX")
# CLOUD_STORAGE = os.getenv("CLOUD_STORAGE")
# EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # ------------------ Utility Functions ------------------

# def clean_text(text: str) -> str:
#     """
#     Clean text to remove unwanted characters and preserve table structure.
#     """
#     if not text:
#         return ""
#     text = text.replace("\\", " ")  # remove backslashes
#     text = re.sub(r"\s+", " ", text)  # collapse multiple spaces/newlines
#     text = text.strip()
#     return text

# def load_json(file_path: str) -> dict:
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"File not found at {file_path}")
#     with open(file_path, 'r', encoding='utf-8') as f:
#         return json.load(f)

# def create_documents_from_prepared_docs(prepared_docs: list) -> list:
#     """
#     Only extract 'text' field from prepared_docs and create Document objects.
#     """
#     documents = []
#     for doc in prepared_docs:
#         text = clean_text(doc.get("text", ""))
#         metadata = {
#             "id": doc.get("id"),
#             "report_link_name": doc.get("report_link_name"),
#             "chunk_index": doc.get("chunk_index")
#         }
#         documents.append(Document(page_content=text, metadata=metadata))
#     return documents

# def split_text_safely(documents: list, max_chars: int = 1000) -> list:
#     """
#     Split each document by rows ('|') and make sure no chunk exceeds max_chars.
#     """
#     chunks = []

#     for doc in documents:
#         rows = [r.strip() for r in doc.page_content.split("|") if r.strip()]
#         temp_chunk = ""
#         chunk_index = 0

#         for row in rows:
#             # If adding this row exceeds max_chars, finalize current chunk
#             if len(temp_chunk) + len(row) + 3 > max_chars:
#                 metadata = doc.metadata.copy()
#                 metadata["sub_chunk_index"] = chunk_index
#                 chunks.append(Document(page_content=temp_chunk.strip(), metadata=metadata))
#                 temp_chunk = row
#                 chunk_index += 1
#             else:
#                 temp_chunk += " | " + row if temp_chunk else row

#         # Add remaining text
#         if temp_chunk:
#             metadata = doc.metadata.copy()
#             metadata["sub_chunk_index"] = chunk_index
#             chunks.append(Document(page_content=temp_chunk.strip(), metadata=metadata))

#     return chunks

# def split_text_safely(documents: list, max_chars: int = 1000) -> list:
#     """
#     Split each document (Document or raw string) by rows ('|') and make sure no chunk exceeds max_chars.
#     """
#     chunks = []

#     for doc in documents:
#         # If doc is a string, wrap it into a temporary Document with empty metadata
#         if isinstance(doc, str):
#             text = clean_text(doc)
#             metadata = {}
#         elif isinstance(doc, Document):
#             text = clean_text(doc.page_content)
#             metadata = doc.metadata.copy()
#         else:
#             raise TypeError(f"Expected str or Document, got {type(doc)}")

#         rows = [r.strip() for r in text.split("|") if r.strip()]
#         temp_chunk = ""
#         chunk_index = 0

#         for row in rows:
#             if len(temp_chunk) + len(row) + 3 > max_chars:
#                 chunk_metadata = metadata.copy()
#                 chunk_metadata["sub_chunk_index"] = chunk_index
#                 chunks.append(Document(page_content=temp_chunk.strip(), metadata=chunk_metadata))
#                 temp_chunk = row
#                 chunk_index += 1
#             else:
#                 temp_chunk += " | " + row if temp_chunk else row

#         if temp_chunk:
#             chunk_metadata = metadata.copy()
#             chunk_metadata["sub_chunk_index"] = chunk_index
#             chunks.append(Document(page_content=temp_chunk.strip(), metadata=chunk_metadata))

#     return chunks



# def generate_embeddings(chunks: list, model: str, api_key: str) -> list:
#     if not api_key:
#         raise ValueError("Missing OPENAI_API_KEY.")
#     logger.info(f"Using OpenAI model '{model}' for embeddings.")
#     embeddings = OpenAIEmbeddings(model=model, openai_api_key=api_key)
#     texts = [chunk.page_content for chunk in chunks]
#     return embeddings.embed_documents(texts)

# def generate_embeddings(chunks: list, model: str, api_key: str) -> list:
#     if not api_key:
#         raise ValueError("Missing OPENAI_API_KEY.")
    
#     logger.info(f"Using OpenAI model '{model}' for embeddings.")
#     embeddings = OpenAIEmbeddings(model=model, openai_api_key=api_key)
    
#     texts = []
#     for i, chunk in enumerate(chunks, start=1):
#         print(f"Chunk {i}: {chunk.page_content}")  # Print chunk content
#         texts.append(chunk.page_content)
    
#     return embeddings.embed_documents(texts)

# def init_pinecone_index(api_key: str, index_name: str, dimension: int, cloud: str, region: str):
#     pc = Pinecone(api_key=api_key)
#     existing = [idx.name for idx in pc.list_indexes()]
#     if index_name not in existing:
#         logger.info(f"Creating Pinecone index '{index_name}' with dim={dimension} ...")
#         pc.create_index(
#             name=index_name,
#             dimension=dimension,
#             metric="cosine",
#             spec=ServerlessSpec(cloud=cloud, region=region)
#         )
#     else:
#         logger.info(f"Pinecone index '{index_name}' already exists.")
#     return pc.Index(name=index_name)

# def upsert_vectors(index, chunks: list, vectors: list, namespace: str, batch_size=100):
#     vectors_to_upsert = []
#     for i, (vector, chunk) in enumerate(zip(vectors, chunks)):
#         unique_id = str(uuid.uuid4())
#         metadata = chunk.metadata.copy()
#         metadata.update({"source_text": chunk.page_content})
#         vectors_to_upsert.append({
#             "id": unique_id,
#             "values": vector,
#             "metadata": metadata
#         })

#     for i in range(0, len(vectors_to_upsert), batch_size):
#         batch = vectors_to_upsert[i:i + batch_size]
#         index.upsert(vectors=batch, namespace=namespace)
#         logger.info(f"Upserted batch {i // batch_size + 1} into namespace '{namespace}'.")


# def split_text_into_chunks(text: str):
#     """Split text into overlapping chunks for embeddings."""
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200,
#         separators=["\n\n", "\n", ".", "!", "?", " ", ""],
#     )
#     return splitter.split_text(text)

# ------------------ Main Function ------------------

# def store_embeddings_from_zoho_json(namespace: str, json_file_path: str) -> str:
#     """
#     Generate embeddings from Zoho JSON and store in Pinecone.
#     """
#     try:
#         logger.info(f"Using namespace: '{namespace}'")
#         # data = load_json(json_file_path)

#         prepared_docs = data.get("prepared_docs", [])
#         if not prepared_docs:
#             raise ValueError("No prepared_docs found in JSON.")

#         # Create Document objects from 'text'
#         documents = create_documents_from_prepared_docs(prepared_docs)
#         logger.info(f"Created {len(documents)} Document objects.")

#         # Split long documents row-wise to avoid token limit
#         chunks = split_text_safely(documents, max_chars=1000)
#         logger.info(f"Split into {len(chunks)} safe chunks for embeddings.")


#         # Generate embeddings
#         vectors = generate_embeddings(chunks, EMBEDDING_MODEL, OPENAI_API_KEY)
#         logger.info(f"Generated {len(vectors)} embeddings.")

#         # Initialize Pinecone
#         vector_dim = len(vectors[0])
#         index = init_pinecone_index(PINECONE_API_KEY, PINECONE_INDEX, vector_dim, CLOUD_STORAGE, PINECONE_ENV)

#         # Upsert embeddings
#         upsert_vectors(index, chunks, vectors, namespace)
#         logger.info(f"✅ All documents successfully added to namespace '{namespace}'.")
#         return namespace

#     except Exception as e:
#         logger.error(f"Error embedding documents: {e}")
#         return None

# ------------------ Example Usage ------------------
# if __name__ == "__main__":
#     namespace = "Test_Zoho_1111"
    # json_file_path = "backend/embeddings/zoho_2/embedding_ready.json"  # <-- Update path
    # store_embeddings_from_zoho_json(namespace, json_file_path)


    # json_file_path = "/home/user/Web_Scraping_ChatBot/backend/api_response.json"


    # # Load JSON file
    # with open(json_file_path, "r", encoding="utf-8") as f:
    #     data = json.load(f)

    # all_records = []

    # for app in data.get("applications", []):
    #     reports = app.get("reports", [])
    #     for report in reports:
    #         report_data = report.get("report_data", {})
    #         records_str = report_data.get("records", "")
    #         if not records_str:
    #             continue  # skip if no records

    #         try:
    #             # records_str is a JSON string, parse it
    #             records_json = json.loads(records_str)
    #             records_list = records_json.get("data", [])
    #             all_records.extend(records_list)  # add to combined list
    #         except json.JSONDecodeError as e:
    #             print(f"Error decoding records for app {app.get('app_link_name')}: {e}")
    #             continue

    # records_text = json.dumps(all_records, indent=2)

    # # Call your chunking function   
    # splitext = split_text_into_chunks(text=records_text)
    # print(">>>>>>>>>>splitext>>>>>>>>>>>>>>>>>>>",splitext)
    # # store_embeddings_from_zoho_json(namespace, json_file_path)

    # chunks = split_text_safely(splitext, max_chars=1000)
    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # logger.info(f"Split into {len(chunks)} safe chunks for embeddings.")
    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>")


    # # Generate embeddings
    # vectors = generate_embeddings(chunks, EMBEDDING_MODEL, OPENAI_API_KEY)
    # logger.info(f"Generated {len(vectors)} embeddings.")

    # # Initialize Pinecone
    # vector_dim = len(vectors[0])
    # index = init_pinecone_index(PINECONE_API_KEY, PINECONE_INDEX, vector_dim, CLOUD_STORAGE, PINECONE_ENV)

    # # Upsert embeddings
    # upsert_vectors(index, chunks, vectors, namespace)
    # logger.info(f"✅ All documents successfully added to namespace '{namespace}'.")





from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from io import StringIO
import os
import uuid
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
CLOUD_STORAGE = os.getenv("CLOUD_STORAGE")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

def load_csv(csv_input: str):
    """Smart CSV loader that auto-detects delimiters and handles bad lines."""
    try:
        # Detect if the input is raw CSV text or file path
        if isinstance(csv_input, str) and "\n" in csv_input:
            csv_data = StringIO(csv_input)
        else:
            csv_data = csv_input

        delimiters = [',', ';', '\t']
        for delimiter in delimiters:
            try:
                df = pd.read_csv(csv_data, delimiter=delimiter, engine='python', on_bad_lines='skip', dtype=str)
                if not df.empty and df.shape[1] > 0:
                    return df.fillna("")
            except Exception:
                if isinstance(csv_data, StringIO):
                    csv_data.seek(0)
                continue

        df = pd.read_csv(csv_data, delimiter=',', engine='python', on_bad_lines='skip', header=None, dtype=str)
        return df.fillna("")
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV: {str(e)}")


def csv_to_string(df: pd.DataFrame) -> str:
    """Convert DataFrame to a detailed string representation of rows and columns."""
    rows_as_text = []
    for index, row in df.iterrows():
        row_text = ", ".join([f"{col}: {row[col]}" for col in df.columns])
        rows_as_text.append(f"Row {index + 1}: {row_text}")
    return "\n".join(rows_as_text)

def split_text_into_chunks_data(text: str):
    """Split text into overlapping chunks for embeddings."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )
    return splitter.split_text(text)

def generate_embeddings(chunks: list, model: str, api_key: str) -> list:
    """Generate embeddings using OpenAI model."""
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY.")
    
    print(f"Using OpenAI model '{model}' for embeddings.")
    embeddings = OpenAIEmbeddings(model=model, openai_api_key=api_key)
    
    # If chunks are strings, pass directly
    texts = [str(chunk) for chunk in chunks]
    return embeddings.embed_documents(texts)

def init_pinecone_index(api_key: str, index_name: str, dimension: int, cloud: str, region: str):
    """Initialize Pinecone client and create index if not exists."""
    pc = Pinecone(api_key=api_key)
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        print(f"Creating Pinecone index '{index_name}' with dim={dimension} ...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region)
        )
    else:
        print(f"Pinecone index '{index_name}' already exists.")
    return pc.Index(name=index_name)


def _safe_copy_metadata(m):
    """Return a dict copy for metadata; if not a dict, convert to a dict safely."""
    if isinstance(m, dict):
        return m.copy()
    # if it's None or primitive, place it under a known key
    return {"original_metadata": str(m)}


def upsert_vectors(index, chunks: list, vectors: list, namespace: str, metadata_list: list = None, batch_size=100):
    """Upsert vectors to Pinecone safely with optional metadata_list."""
    vectors_to_upsert = []

    for i, (vector, chunk) in enumerate(zip(vectors, chunks)):
        unique_id = str(uuid.uuid4())

        # Use provided metadata_list if given, else default to chunk metadata
        if metadata_list:
            raw_meta = metadata_list[i]
            metadata = _safe_copy_metadata(raw_meta)
        else:
            # chunk may be a Document or a string
            if hasattr(chunk, "metadata"):
                raw_meta = getattr(chunk, "metadata")
                metadata = _safe_copy_metadata(raw_meta)
            else:
                metadata = {}
            # attach source_text safely (may be large; consider truncating if needed)
            if hasattr(chunk, "page_content"):
                metadata.update({"source_text": chunk.page_content})
            else:
                metadata.update({"source_text": str(chunk)})

        vectors_to_upsert.append({
            "id": unique_id,
            "values": vector,
            "metadata": metadata
        })

    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i + batch_size]
        index.upsert(vectors=batch, namespace=namespace)
        print(f"Upserted batch {i // batch_size + 1} into namespace '{namespace}'.")



# # Example usage:
# if __name__ == "__main__":
#     # --- Case 1: If you have a file path ---
#     # file_path = "/home/user/Videos/Zoho_Details/Pre-Camp Survey Report (1).csv" Patient Registrations Report
#     file_path = "/home/user/Videos/Zoho_Details/Patient Registrations Report.csv"
#     df = load_csv(file_path)
#     csv_text = csv_to_string(df)

#     chunks = split_text_into_chunks(csv_text)
#     print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
#     print(f"Split into {len(chunks)} safe chunks for embeddings.")
#     print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

#     # Generate embeddings
#     vectors = generate_embeddings(chunks, EMBEDDING_MODEL, OPENAI_API_KEY)
#     print(f"Generated {len(vectors)} embeddings.")

#     # Initialize Pinecone
#     vector_dim = len(vectors[0])
#     index = init_pinecone_index(PINECONE_API_KEY, PINECONE_INDEX, vector_dim, CLOUD_STORAGE, PINECONE_ENV)

#     # Upsert embeddings
#     upsert_vectors(index, chunks, vectors, 'ZOHO_DATA')
#     print(f"✅ All documents successfully added to namespace '.")


# def process_csv_to_embedding(
#     file_path: str,
#     namespace: str,
# ):
#     """
#     Process a CSV file: load, convert to string, chunk, embed, and upsert to Pinecone.
#     """
#     # Load CSV
#     df = load_csv(file_path)
#     print(f"✅ CSV loaded with {len(df)} rows and {len(df.columns)} columns.")

#     # Convert to string
#     csv_text = csv_to_string(df)

#     # Split into chunks
#     chunks = split_text_into_chunks(csv_text)
#     print(f"Split into {len(chunks)} safe chunks for embeddings.")

#     # Generate embeddings
#     vectors = generate_embeddings(chunks, EMBEDDING_MODEL, OPENAI_API_KEY)
#     print(f"Generated {len(vectors)} embeddings.")

#     # Initialize Pinecone
#     vector_dim = len(vectors[0])
#     index = init_pinecone_index(PINECONE_API_KEY, PINECONE_INDEX, vector_dim, CLOUD_STORAGE, PINECONE_ENV)

#     # Upsert vectors
#     upsert_vectors(index, chunks, vectors, namespace)
#     print(f"✅ All documents successfully added to namespace '{namespace}'.")

#     return index, chunks, vectors

