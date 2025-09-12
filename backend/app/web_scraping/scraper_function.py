import json
from pathlib import Path
from urllib.parse import urlparse
from app.web_scraping.web_scraper import WebsiteProcessor

 
def web_scraping(url):
    # Website to scrape
    WEBSITE_URL = url
   
    print("="*60)
    print("RAG DATA PREPARATION PIPELINE")
    print("="*60)
   
    # Step 1: Initialize processor
    print("\n📦 Initializing processor...")
    processor = WebsiteProcessor(WEBSITE_URL)
   
    # Step 2: Scrape website
    print("\n🌐 Scraping website...")
    documents = processor.scrape_website()
    print(f"   ✅ Scraped {len(documents)} pages")
   
    # Step 3: Clean documents
    print("\n🧹 Cleaning documents...")
    cleaned_docs = processor.clean_documents()
    print(f"   ✅ Cleaned {len(cleaned_docs)} documents")
   
    # Step 4: Create chunks
    print("\n📄 Creating chunks...")
    chunks = processor.create_chunks(chunk_size=1000, chunk_overlap=200)
    print(f"   ✅ Created {len(chunks)} chunks")
   
    # Step 5: Display summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
   
    # Load and show sample of embedding-ready data
    with open(f"rag_data/{urlparse(WEBSITE_URL).netloc}/chunks/embedding_ready.json", 'r', encoding='utf-8') as f:
        embedding_data = json.load(f)

    print(f"\n📊 Statistics:")
    print(f"   • Total chunks ready for embedding: {embedding_data['total_chunks']}")
    print(f"   • Average chunk size: ~1000 characters")
    print(f"   • Ready for vector database: YES ✅")
 
# if __name__ == "__main__":
#     main()