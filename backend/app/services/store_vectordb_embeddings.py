# import os
# import logging
# from typing import List, Dict
# from flask import json
# from dotenv import load_dotenv
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_pinecone import PineconeVectorStore
# from pinecone import Pinecone, ServerlessSpec
# from langchain.schema import Document
# from langchain_text_splitters import CharacterTextSplitter

# load_dotenv()

# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_ENV = os.getenv('PINECONE_ENV')
# PINECONE_INDEX = os.getenv('PINECONE_INDEX')
# NAMESPACE = "doctors_all_informations"

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# def process_and_store_data(result_list: List[Dict]):
#     logger.info(f"Processing {len(result_list)} items from result list.")

#     # Step 1: Convert each item into a Document object
#     documents = []
#     for item in result_list:
#         # Build a meaningful text from the structured data
#         text = (
#             f"Dr Name: {item.get('name')}\n"
#             f"Specialty: {item.get('specialty')}\n"
#             f"Qualifications: {item.get('qualifications')}\n"
#             f"Experience: {item.get('experience_years')} years\n"
#             f"Available Days: {item.get('available_days')}\n"
#             f"Available Time: {item.get('available_time')}"
#         )
#         metadata = {"id": item.get('id')}
#         documents.append(Document(page_content=text, metadata=metadata))

#     logger.info(f"Created {len(documents)} Document objects.")

#     # Step 2: Split documents into chunks
#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
#     chunks = text_splitter.split_documents(documents)
#     logger.info(f"Split documents into {len(chunks)} chunks.")

#     # Step 3: Generate Embeddings
#     logger.info("Generating embeddings using GoogleGenerativeAIEmbeddings...")
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=GOOGLE_API_KEY)

#     texts_for_embedding = [chunk.page_content for chunk in chunks]
#     vector_data = embeddings.embed_documents(texts_for_embedding)
#     logger.info(f"Generated {len(vector_data)} embedding vectors.")

#     # Step 4: Initialize Pinecone Client
#     logger.info("Initializing Pinecone client...")
#     pc = Pinecone(api_key=PINECONE_API_KEY)

#     # Create index if not exists
#     if PINECONE_INDEX not in [idx.name for idx in pc.list_indexes()]:
#         logger.info(f"Creating Pinecone index: {PINECONE_INDEX}")
#         pc.create_index(
#             name=PINECONE_INDEX,
#             dimension=768,  # Assuming embedding dimension
#             metric="cosine",
#             spec=ServerlessSpec(cloud="aws", region="us-east-1")
#         )
#     else:
#         logger.info(f"Pinecone index {PINECONE_INDEX} already exists.")

#     # Step 5: Initialize Vector Store
#     vectorstore = PineconeVectorStore(
#         index_name=PINECONE_INDEX,
#         embedding=embeddings,
#         namespace=NAMESPACE
#     )

#     # Step 6: Add documents to Pinecone
#     logger.info("Adding documents to Pinecone index...")
#     vectorstore.add_documents(documents=chunks)
#     logger.info("All documents successfully added to Pinecone.")
