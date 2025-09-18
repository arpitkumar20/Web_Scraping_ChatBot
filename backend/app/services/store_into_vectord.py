
import os
import logging
from flask import json
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv('PINECONE_ENV')
PINECONE_INDEX = os.getenv('PINECONE_INDEX')
NAMESPACE=os.getenv('NAMESPACE')

from langchain.schema import Document
from langchain_text_splitters import CharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Step 1: Load JSON file
logger.info("Loading data from JSON file...")
with open('/home/user/Nissa_whatsapp_chat_bot/backend/rag_data/www.apple.com/chunks/embedding_ready.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

texts = data.get("texts", [])
metadatas = data.get("metadatas", [])

logger.info(f"Loaded {len(texts)} texts and {len(metadatas)} metadatas.")

# Step 2: Create Document objects
documents = []
for text, metadata in zip(texts, metadatas):
    documents.append(Document(page_content=text, metadata=metadata))

logger.info(f"Created {len(documents)} Document objects.")

# Step 3: Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = text_splitter.split_documents(documents)

logger.info(f"Split documents into {len(chunks)} chunks.")

# Step 4: Generate Embeddings using Google GenAI
logger.info("Generating embeddings using GoogleGenerativeAIEmbeddings...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=GOOGLE_API_KEY)

texts_for_embedding = [chunk.page_content for chunk in chunks]
vector_data = embeddings.embed_documents(texts_for_embedding)

logger.info(f"Generated {len(vector_data)} embedding vectors.")

# Step 5: Initialize Pinecone Client
logger.info("Initializing Pinecone client...")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it does not exist
if PINECONE_INDEX not in [idx.name for idx in pc.list_indexes()]:
    logger.info(f"Creating Pinecone index: {PINECONE_INDEX}")
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=768,  # Assuming embedding dimension is 768
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Adjust region if needed
    )
else:
    logger.info(f"Pinecone index {PINECONE_INDEX} already exists.")

# Initialize PineconeVectorStore with embeddings
vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX,
    embedding=embeddings,  # Pass embedding here
    namespace=NAMESPACE 
)

# Add documents to Pinecone
logger.info("Adding documents to Pinecone index...")
vectorstore.add_documents(documents=chunks)

logger.info("All documents successfully added to Pinecone.")