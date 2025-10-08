import json
import logging
from app.web_scraping.web_scraper import WebsiteProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

def web_scraping(url):
    # Website to scrape

    logging.info("=" * 60)
    logging.info("RAG DATA PREPARATION PIPELINE")
    logging.info("=" * 60)

    logging.info("📦 Initializing processor...")
    processor = WebsiteProcessor(url)

    logging.info("🌐 Scraping website...")
    documents = processor.scrape_website()
    logging.info(f"✅ Scraped {len(documents)} pages")

    logging.info("🧹 Cleaning documents...")
    cleaned_docs = processor.clean_documents()
    logging.info(f"✅ Cleaned {len(cleaned_docs)} documents")

    logging.info("📄 Creating chunks...")
    chunks = processor.create_chunks(chunk_size=1000, chunk_overlap=200)
    logging.info(f"✅ Created {len(chunks)} chunks")

    logging.info("=" * 60)
    logging.info("PROCESSING COMPLETE!")
    logging.info("=" * 60)

    # web_info = processor.get_website_info()
    # logging.info("✅ Website informations fetched sucessfully")

    embedding_path = processor.website_folder / "chunks" / "embedding_ready.json"
    if embedding_path.exists():
        with open(embedding_path, 'r', encoding='utf-8') as f:
            embedding_data = json.load(f)

        logging.info("📊 Statistics:")
        logging.info(f"   • Total chunks ready for embedding: {embedding_data['total_chunks']}")
        logging.info(f"   • Average chunk size: ~1000 characters")
        logging.info(f"   • Ready for vector database: YES ✅")
        logging.info(f"Embedding path of that folder is: {embedding_path}")
        logging.info("✅ Web scraping process completed successfully. Returning embedding path.")
        # return web_info, embedding_path
        return embedding_path
    else:
        logging.warning("⚠️ Embedding data not found. Returning None.")
        return None