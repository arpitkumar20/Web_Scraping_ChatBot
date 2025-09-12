# """
# Main web scraper using LangChain built-in functions
# This file handles all scraping, cleaning, and chunking
# """
 
# import os
# import json
# import logging
# from pathlib import Path
# from datetime import datetime
# from typing import List, Dict, Optional
 
# # Load environment variables
# # from dotenv import load_dotenv
# # load_dotenv()
 
# # LangChain imports
# from langchain_community.document_loaders import (
#     WebBaseLoader,
#     RecursiveUrlLoader,
#     SitemapLoader
# )
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_transformers import Html2TextTransformer
# from langchain.schema import Document
 
# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
# logger = logging.getLogger(__name__)
 
 
# class WebsiteProcessor:
#     """Main class to process website data for RAG"""
   
#     def __init__(self, website_url: str):
#         """Initialize with website URL"""
#         self.website_url = website_url
#         self.output_dir = Path("rag_data")
       
#         # Create output directories
#         self.output_dir.mkdir(exist_ok=True)
#         (self.output_dir / "raw").mkdir(exist_ok=True)
#         (self.output_dir / "cleaned").mkdir(exist_ok=True)
#         (self.output_dir / "chunks").mkdir(exist_ok=True)

#         self.documents = []
#         self.cleaned_docs = []
#         self.chunks = []
       
#         logger.info(f"Initialized processor for {website_url}")
   
#     def scrape_website(self) -> List[Document]:
#         """Scrape website using LangChain loaders"""
#         logger.info("Starting website scraping...")
       
#         try:
#             # Try sitemap first
#             logger.info("Trying sitemap loader...")
#             sitemap_loader = SitemapLoader(
#                 web_path=f"{self.website_url}/sitemap.xml"
#             )
#             self.documents = sitemap_loader.load()
#             logger.info(f"Loaded {len(self.documents)} pages from sitemap")
           
#         except Exception as e:
#             logger.warning(f"Sitemap failed: {e}")
#             logger.info("Using recursive URL loader...")
           
#             # Fallback to recursive loader
#             loader = RecursiveUrlLoader(
#                 url=self.website_url,
#                 max_depth=2,
#                 prevent_outside=True
#             )
#             self.documents = loader.load()
#             logger.info(f"Loaded {len(self.documents)} pages recursively")
       
#         # Save raw documents
#         self._save_documents(self.documents, "raw/scraped_docs.json")
#         return self.documents
   
#     def clean_documents(self) -> List[Document]:
#         """Clean documents using LangChain transformers"""
#         logger.info("Cleaning documents...")
       
#         # Use Html2TextTransformer for cleaning
#         html_transformer = Html2TextTransformer()
#         self.cleaned_docs = html_transformer.transform_documents(self.documents)
       
#         # Additional cleaning
#         for doc in self.cleaned_docs:
#             # Remove extra whitespace
#             doc.page_content = ' '.join(doc.page_content.split())
#             # Add cleaning metadata
#             doc.metadata['cleaned'] = True
#             doc.metadata['cleaned_at'] = datetime.now().isoformat()
       
#         logger.info(f"Cleaned {len(self.cleaned_docs)} documents")
#         self._save_documents(self.cleaned_docs, "cleaned/cleaned_docs.json")
#         return self.cleaned_docs
   
#     def create_chunks(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
#         """Create chunks using LangChain splitter"""
#         logger.info("Creating chunks...")
       
#         # Use RecursiveCharacterTextSplitter
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap,
#             length_function=len,
#             separators=["\n\n", "\n", ". ", " ", ""]    
#         )
       
#         # Split documents
#         self.chunks = text_splitter.split_documents(self.cleaned_docs)
       
#         # Add chunk metadata
#         for i, chunk in enumerate(self.chunks):
#             chunk.metadata['chunk_id'] = i
#             chunk.metadata['total_chunks'] = len(self.chunks)
       
#         logger.info(f"Created {len(self.chunks)} chunks")
#         self._save_documents(self.chunks, "chunks/document_chunks.json")
       
#         # Save embedding-ready format
#         self._save_embedding_ready()
       
#         return self.chunks
   
#     def _save_documents(self, docs: List[Document], filename: str):
#         """Save documents to JSON file"""
#         filepath = self.output_dir / filename
       
#         data = []
#         for doc in docs:
#             data.append({
#                 'content': doc.page_content,
#                 'metadata': doc.metadata
#             })
       
#         with open(filepath, 'w', encoding='utf-8') as f:
#             json.dump(data, f, indent=2, ensure_ascii=False)
       
#         logger.info(f"Saved to {filepath}")
   
#     def _save_embedding_ready(self):
#         """Save chunks in format ready for embeddings"""
#         texts = [chunk.page_content for chunk in self.chunks]
#         metadatas = [chunk.metadata for chunk in self.chunks]
       
#         embedding_data = {
#             'texts': texts,
#             'metadatas': metadatas,
#             'total_chunks': len(texts),
#             'created_at': datetime.now().isoformat()
#         }
       
#         filepath = self.output_dir / "chunks" / "embedding_ready.json"
#         with open(filepath, 'w', encoding='utf-8') as f:
#             json.dump(embedding_data, f, indent=2, ensure_ascii=False)
       
#         logger.info(f"Saved embedding-ready data to {filepath}")




"""
Main web scraper using LangChain built-in functions
This file handles all scraping, cleaning, and chunking
"""
 
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from urllib.parse import urlparse
import uuid
 
# Load environment variables
from dotenv import load_dotenv
load_dotenv()
 
# LangChain imports
from langchain_community.document_loaders import (
    WebBaseLoader,
    RecursiveUrlLoader,
    SitemapLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_transformers import Html2TextTransformer
from langchain.schema import Document
 
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
 
 
class WebsiteProcessor:
    """Main class to process website data for RAG"""
   
    def __init__(self, website_url: str):
        """Initialize with website URL"""
        self.website_url = website_url
        self.output_dir = Path("rag_data")
        # self.website_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, urlparse(website_url).netloc))
        self.website_folder = self.output_dir / f"{urlparse(website_url).netloc}"
       
        # Create output directories
        self.website_folder.mkdir(parents=True, exist_ok=True)
        (self.website_folder / "raw").mkdir(exist_ok=True)
        (self.website_folder / "cleaned").mkdir(exist_ok=True)
        (self.website_folder / "chunks").mkdir(exist_ok=True)
       
        self.documents = []
        self.cleaned_docs = []
        self.chunks = []
       
        logger.info(f"Initialized processor for {website_url}")
   
    def scrape_website(self) -> List[Document]:
        """Scrape website using LangChain loaders"""
        logger.info("Starting website scraping...")
       
        try:
            # Try sitemap first
            logger.info("Trying sitemap loader...")
            sitemap_loader = SitemapLoader(
                web_path=f"{self.website_url}/sitemap.xml"
            )
            self.documents = sitemap_loader.load()
            logger.info(f"Loaded {len(self.documents)} pages from sitemap")
           
        except Exception as e:
            logger.warning(f"Sitemap failed: {e}")
            logger.info("Using recursive URL loader...")
           
            # Fallback to recursive loader
            loader = RecursiveUrlLoader(
                url=self.website_url,
                max_depth=2,
                prevent_outside=True
            )
            self.documents = loader.load()
            logger.info(f"Loaded {len(self.documents)} pages recursively")
       
        # Save raw documents
        self._save_documents(self.documents, "raw/scraped_docs.json")
        return self.documents
   
    def clean_documents(self) -> List[Document]:
        """Clean documents using LangChain transformers"""
        logger.info("Cleaning documents...")
       
        # Use Html2TextTransformer for cleaning
        html_transformer = Html2TextTransformer()
        self.cleaned_docs = html_transformer.transform_documents(self.documents)
       
        # Additional cleaning
        for doc in self.cleaned_docs:
            # Remove extra whitespace
            doc.page_content = ' '.join(doc.page_content.split())
            # Add cleaning metadata
            doc.metadata['cleaned'] = True
            doc.metadata['cleaned_at'] = datetime.now().isoformat()
       
        logger.info(f"Cleaned {len(self.cleaned_docs)} documents")
        self._save_documents(self.cleaned_docs, "cleaned/cleaned_docs.json")
        return self.cleaned_docs
   
    def create_chunks(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """Create chunks using LangChain splitter"""
        logger.info("Creating chunks...")
       
        # Use RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]    
        )
       
        # Split documents
        self.chunks = text_splitter.split_documents(self.cleaned_docs)
       
        # Add chunk metadata
        for i, chunk in enumerate(self.chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['total_chunks'] = len(self.chunks)
       
        logger.info(f"Created {len(self.chunks)} chunks")
        self._save_documents(self.chunks, "chunks/document_chunks.json")
       
        # Save embedding-ready format
        self._save_embedding_ready()
       
        return self.chunks
   
    def _save_documents(self, docs: List[Document], filename: str):
        """Save documents to JSON file"""
        filepath = self.website_folder / filename
       
        data = []
        for doc in docs:
            data.append({
                'content': doc.page_content,
                'metadata': doc.metadata
            })
       
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
       
        logger.info(f"Saved to {filepath}")
   
    def _save_embedding_ready(self):
        """Save chunks in format ready for embeddings"""
        texts = [chunk.page_content for chunk in self.chunks]
        metadatas = [chunk.metadata for chunk in self.chunks]
       
        embedding_data = {
            'texts': texts,
            'metadatas': metadatas,
            'total_chunks': len(texts),
            'created_at': datetime.now().isoformat()
        }
       
        filepath = self.website_folder / "chunks" / "embedding_ready.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(embedding_data, f, indent=2, ensure_ascii=False)
       
        logger.info(f"Saved embedding-ready data to {filepath}")
 