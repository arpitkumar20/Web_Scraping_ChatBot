import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import List
from urllib.parse import urlparse, urljoin

from dotenv import load_dotenv
import requests
load_dotenv()

from langchain_community.document_loaders import (
    SitemapLoader,
    RecursiveUrlLoader,
    SeleniumURLLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_transformers import Html2TextTransformer
from langchain.schema import Document

from bs4 import BeautifulSoup
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class WebsiteProcessor:
    """Scraper with sitemap-first and SPA fallback strategy"""

    def __init__(self, website_url: str):
        self.website_url = website_url
        self.output_dir = Path("rag_data")
        self.website_folder = self.output_dir / f"{urlparse(website_url).netloc}"

        self.website_folder.mkdir(parents=True, exist_ok=True)
        (self.website_folder / "raw").mkdir(exist_ok=True)
        (self.website_folder / "cleaned").mkdir(exist_ok=True)
        (self.website_folder / "chunks").mkdir(exist_ok=True)

        self.documents = []
        self.cleaned_docs = []
        self.chunks = []

        logger.info(f"Initialized processor for {website_url}")


    def get_website_info(self) -> dict:
        """
        Fetch website details (title, description, keywords, main heading).
        Works for any type of website without hardcoded classification.
        """
        try:
            resp = requests.get(self.website_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(resp.text, "html.parser")

            # Title
            title = soup.title.string.strip() if soup.title else "No Title Found"

            # Meta description
            description = ""
            desc_tag = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
            if desc_tag and desc_tag.get("content"):
                description = desc_tag["content"].strip()

            # Meta keywords
            keywords = ""
            keywords_tag = soup.find("meta", attrs={"name": "keywords"})
            if keywords_tag and keywords_tag.get("content"):
                keywords = keywords_tag["content"].strip()

            # First heading (h1)
            heading = ""
            h1_tag = soup.find("h1")
            if h1_tag:
                heading = h1_tag.get_text(strip=True)

            # Fallback if description is empty → take first paragraph
            if not description:
                first_p = soup.find("p")
                if first_p:
                    description = first_p.get_text(strip=True)[:200]  # limit length

            website_info = {
                "url": self.website_url,
                "title": title,
                "description": description,
                "keywords": keywords,
                "main_heading": heading
            }

            # --- Logging output ---
            logger.info(f"Website Info Extracted → Title: {title}, Heading: {heading}, Description: {description[:80]}...")
            return website_info

        except Exception as e:
            logger.error(f"Failed to fetch website info: {e}")
            return {
                "url": self.website_url,
                "title": "Unknown",
                "description": "Unknown",
                "keywords": "",
                "main_heading": ""
            }

    def _extract_all_links_spa(self, max_pages=200, headless=True) -> List[str]:
        """Extract internal links with dynamic JS crawling using Playwright"""
        if not sync_playwright:
            logger.warning("Playwright not available, SPA crawling skipped.")
            return []

        visited = set()
        queue = [self.website_url]
        all_links = set()
        domain = urlparse(self.website_url).netloc

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=headless)
            context = browser.new_context()
            page = context.new_page()

            while queue and len(visited) < max_pages:
                url = queue.pop(0)
                if url in visited:
                    continue

                try:
                    page.goto(url, timeout=60000, wait_until="domcontentloaded")
                    time.sleep(1)
                    html = page.content()
                    visited.add(url)
                    all_links.add(url)

                    soup = BeautifulSoup(html, "html.parser")
                    for a in soup.select("a[href]"):
                        href = a["href"].strip()
                        if not href or href.startswith(("mailto:", "tel:", "javascript:")):
                            continue
                        abs_url = href if href.startswith("http") else urljoin(url, href)
                        if urlparse(abs_url).netloc != domain:
                            continue
                        abs_url = abs_url.split("#")[0]
                        if abs_url not in visited and abs_url not in queue:
                            queue.append(abs_url)

                except Exception as e:
                    logger.debug(f"Error visiting {url}: {e}")
                    continue

            browser.close()
        return sorted(all_links)

    def _is_spa_site(self) -> bool:
        """Heuristic to check if site is SPA by inspecting main page HTML for SPA markers"""
        try:
            from requests import get
        except ImportError:
            logger.warning("Requests library not installed, cannot detect SPA heuristics.")
            return False

        try:
            resp = get(self.website_url, timeout=10)
            html = resp.text.lower()
            # Heuristics: check if main page strongly depends on JS SPA frameworks or empty content
            spa_signals = ['<div id="root"', 'window.__INITIAL_STATE__', 'ng-app', 'vue', 'react', 'spa']
            if any(signal in html for signal in spa_signals):
                logger.info("Site likely a SPA based on HTML heuristics.")
                return True
        except Exception as e:
            logger.warning(f"SPA detection failed: {e}")
        return False

    def scrape_website(self) -> List[Document]:
        """Main scrape method with sitemap first, then SPA or recursive fallback"""

        logger.info("Starting website scraping with sitemap-first approach...")

        # Step 1: Try sitemap loader
        try:
            logger.info("Trying sitemap loader...")
            sitemap_loader = SitemapLoader(web_path=f"{self.website_url}/sitemap.xml")
            self.documents = sitemap_loader.load()
            logger.info(f"Loaded {len(self.documents)} pages from sitemap")
        except Exception as e:
            logger.warning(f"Sitemap loading failed: {e}")
            self.documents = []

        # Step 2: If no documents loaded from sitemap, check if SPA or fallback recursive
        if len(self.documents) == 0:
            is_spa = self._is_spa_site()

            if is_spa:
                logger.info("Detected SPA site, using dynamic SPA crawling and Selenium loader...")
                spa_urls = self._extract_all_links_spa(max_pages=200, headless=True)
                if spa_urls:
                    try:
                        loader = SeleniumURLLoader(
                            urls=spa_urls,
                            headless=True,
                            browser="chrome",
                            continue_on_failure=True
                        )
                        self.documents = loader.load()
                        logger.info(f"Loaded {len(self.documents)} pages from SPA Selenium loader")
                    except Exception as e:
                        logger.warning(f"SeleniumURLLoader failed on SPA URLs: {e}")
                        self.documents = []
                else:
                    logger.warning("No URLs found from SPA crawler.")
            else:
                logger.info("Not SPA or unable to confirm SPA. Using recursive URL loader fallback...")
                try:
                    loader = RecursiveUrlLoader(
                        url=self.website_url,
                        max_depth=2,
                        prevent_outside=True
                    )
                    self.documents = loader.load()
                    logger.info(f"Loaded {len(self.documents)} pages recursively")
                except Exception as e:
                    logger.error(f"Recursive URL loader failed: {e}")
                    self.documents = []

        self._save_documents(self.documents, "raw/scraped_docs.json")
        return self.documents

    def clean_documents(self) -> List[Document]:
        """Clean scraped documents"""
        logger.info("Cleaning documents...")
        html_transformer = Html2TextTransformer()
        self.cleaned_docs = html_transformer.transform_documents(self.documents)

        for doc in self.cleaned_docs:
            doc.page_content = ' '.join(doc.page_content.split())
            doc.metadata['cleaned'] = True
            doc.metadata['cleaned_at'] = datetime.now().isoformat()

        logger.info(f"Cleaned {len(self.cleaned_docs)} documents")
        self._save_documents(self.cleaned_docs, "cleaned/cleaned_docs.json")
        return self.cleaned_docs

    def create_chunks(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """Chunk cleaned documents for embeddings"""
        logger.info("Creating chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.chunks = text_splitter.split_documents(self.cleaned_docs)

        for i, chunk in enumerate(self.chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['total_chunks'] = len(self.chunks)

        logger.info(f"Created {len(self.chunks)} chunks")
        self._save_documents(self.chunks, "chunks/document_chunks.json")
        self._save_embedding_ready()
        return self.chunks

    def _save_documents(self, docs: List[Document], filename: str):
        filepath = self.website_folder / filename
        data = [{
            'content': doc.page_content,
            'metadata': doc.metadata
        } for doc in docs]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved to {filepath}")

    def _save_embedding_ready(self):
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