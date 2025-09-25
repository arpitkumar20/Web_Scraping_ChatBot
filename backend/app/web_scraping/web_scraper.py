# web_scraper.py
import json
import logging
import time
import random
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any
from urllib.parse import urlparse, urljoin

from dotenv import load_dotenv
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import chardet
from bs4 import BeautifulSoup

load_dotenv()

from langchain_community.document_loaders import (
    SitemapLoader,
    RecursiveUrlLoader,
    SeleniumURLLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_transformers import Html2TextTransformer
from langchain.schema import Document

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------
# Request/session helper
# -----------------------
def get_request_session(proxies: dict = None):
    """Return a requests session with retries, headers, and optional proxy"""
    session = requests.Session()

    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504, 429]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Rotating headers
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) Gecko/20100101 Firefox/118.0",
    ]
    session.headers.update({"User-Agent": random.choice(user_agents)})

    if proxies:
        session.proxies.update(proxies)

    return session

# -----------------------
# JSON-LD & Classification
# -----------------------
def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def _stable_entity_id(domain: str, page_url: str, entity_obj: dict) -> str:
    """
    Prefer @id or url-like fields; fallback to hash(domain+page_url+name_or_desc_snippet)
    """
    if isinstance(entity_obj, dict):
        cand = entity_obj.get('@id') or entity_obj.get('url') or entity_obj.get('sku') or entity_obj.get('identifier')
        if cand:
            # normalize
            return f"{domain}::{str(cand)}"
        # try common name fields
        name = entity_obj.get('name') or entity_obj.get('headline') or entity_obj.get('title')
        if name:
            snippet = (name[:128]).strip()
            return f"{domain}::{page_url}::{_sha256(snippet)}"
    # final fallback
    return f"{domain}::{page_url}::{_sha256(json.dumps(entity_obj, sort_keys=True)[:256])}"

def extract_entities_from_jsonld(html: str) -> List[dict]:
    """
    Parse all <script type="application/ld+json"> blocks in a page.
    - Handles single dicts, arrays, @graph.
    - Flattens nested entities (hasVariant, offers).
    Returns a flat list of dicts.
    """
    entities = []
    if not html:
        return entities

    try:
        soup = BeautifulSoup(html, "html.parser")
        scripts = soup.find_all("script", attrs={"type": "application/ld+json"})
        for s in scripts:
            raw = s.string or "".join(s.contents) if s.contents else ""
            if not raw.strip():
                continue

            try:
                parsed = json.loads(raw)
            except Exception:
                try:
                    parsed = json.loads(raw.strip().replace("\n", ""), strict=False)
                except Exception:
                    continue

            # Recursive flatten
            def flatten(obj):
                out = []
                if isinstance(obj, list):
                    for el in obj:
                        out.extend(flatten(el))
                elif isinstance(obj, dict):
                    if "@graph" in obj and isinstance(obj["@graph"], list):
                        out.extend(flatten(obj["@graph"]))
                    elif "hasVariant" in obj and isinstance(obj["hasVariant"], list):
                        out.append(obj)
                        out.extend(flatten(obj["hasVariant"]))
                    elif "offers" in obj and isinstance(obj["offers"], list):
                        out.append(obj)
                        out.extend(flatten(obj["offers"]))
                    else:
                        out.append(obj)
                return out

            entities.extend(flatten(parsed))

    except Exception as e:
        logger.debug(f"JSON-LD extraction error: {e}")

    return entities


# Heuristic rules for static vs dynamic classification
_DYNAMIC_KEY_PATTERNS = [
    "price", "cost", "currency", "availability", "opening", "closing", "time", "date",
    "start", "end", "availabilityStarts", "availabilityEnds", "rating", "reviewCount",
    "interactionCount", "views", "count", "fee", "amount", "amountDue", "duration"
]

_STATIC_KEY_PATTERNS = [
    "name", "title", "description", "about", "headline", "articleBody", "bio", "summary",
    "spec", "feature", "brand", "manufacturer", "category", "color", "material"
]

def _looks_like_date_or_time(val: Any) -> bool:
    if not isinstance(val, str):
        return False
    # quick ISO-like heuristics
    if "T" in val and ":" in val:
        return True
    if any(tok in val.lower() for tok in ["am", "pm", "mon", "tue", "wed", "thurs", "fri", "-", "–"]) and any(c.isdigit() for c in val):
        return True
    if val.count("-") >= 2 and any(c.isdigit() for c in val):
        return True
    return False

def _is_numeric_like(val: Any) -> bool:
    if isinstance(val, (int, float)):
        return True
    if isinstance(val, str):
        v = val.strip().replace(",", "").replace(" ", "")
        # currency-like
        if any(c in v for c in "$₹£€"):
            return True
        try:
            float(v)
            return True
        except Exception:
            return False
    return False

def _classify_key_value(path: str, key: str, value: Any) -> str:
    """
    Return 'dynamic' or 'static' for one key:value pair using heuristics.
    """
    key_lower = key.lower() if isinstance(key, str) else ""
    path_lower = path.lower() if isinstance(path, str) else ""

    # Explicit dynamic key patterns
    for pat in _DYNAMIC_KEY_PATTERNS:
        if pat in key_lower or pat in path_lower:
            return "dynamic"

    # Explicit static key patterns
    for pat in _STATIC_KEY_PATTERNS:
        if pat in key_lower or pat in path_lower:
            return "static"

    # value based rules
    if _is_numeric_like(value):
        return "dynamic"
    if isinstance(value, str):
        if len(value.split()) >= 8:
            # long descriptive text -> static
            return "static"
        if _looks_like_date_or_time(value):
            return "dynamic"
        if len(value) <= 6:
            # short strings (Yes/No, enums) - treat dynamic
            return "dynamic"
    if isinstance(value, (list, dict)):
        # further analysis required at recursion level; default to static (so data is preserved)
        return "static"

    # default conservative: static
    return "static"

def classify_fields(obj: dict) -> Tuple[Dict[str, Any], Dict[str, Any], List[str], List[str]]:
    """
    Recursively classify fields into static_fields and dynamic_fields.
    Returns:
        static_fields: dict (path -> value)
        dynamic_fields: dict (path -> value)
        static_paths: list of paths
        dynamic_paths: list of paths
    """
    static_fields = {}
    dynamic_fields = {}
    static_paths = []
    dynamic_paths = []

    def walk(current: Any, path_prefix: str = ""):
        if isinstance(current, dict):
            for k, v in current.items():
                path = f"{path_prefix}.{k}" if path_prefix else k
                if isinstance(v, dict):
                    # recursively traverse, but also decide if this key looks dynamic entirely
                    decision = _classify_key_value(path, k, v)
                    if decision == "dynamic":
                        dynamic_fields[path] = v
                        dynamic_paths.append(path)
                    else:
                        # dive deeper
                        walk(v, path)
                elif isinstance(v, list):
                    # lists: classify element-wise; store entire list under same path if mixed
                    # attempt to classify based on element types
                    # if list of primitives, classify based on first element heuristics
                    if len(v) == 0:
                        static_fields[path] = v
                        static_paths.append(path)
                    else:
                        # primitive list?
                        if all(not isinstance(i, dict) and not isinstance(i, list) for i in v):
                            dec = _classify_key_value(path, k, v[0])
                            if dec == "dynamic":
                                dynamic_fields[path] = v
                                dynamic_paths.append(path)
                            else:
                                static_fields[path] = v
                                static_paths.append(path)
                        else:
                            # mixed or dict list -> store whole list as static by default but also walk items
                            static_fields[path] = v
                            static_paths.append(path)
                            # walk each dict in list for deeper fields (use index in path)
                            for idx, item in enumerate(v):
                                walk(item, f"{path}[{idx}]")
                else:
                    dec = _classify_key_value(path, k, v)
                    if dec == "dynamic":
                        dynamic_fields[path] = v
                        dynamic_paths.append(path)
                    else:
                        static_fields[path] = v
                        static_paths.append(path)
        else:
            # primitives at root
            path = path_prefix or "value"
            dec = _classify_key_value(path, path, current)
            if dec == "dynamic":
                dynamic_fields[path] = current
                dynamic_paths.append(path)
            else:
                static_fields[path] = current
                static_paths.append(path)

    walk(obj, "")
    return static_fields, dynamic_fields, static_paths, dynamic_paths

def consolidate_entities(entities: list) -> list:
    """
    Consolidate entities by entity_id, keeping only the richest version.
    Richness is scored by number of static + dynamic fields/paths.
    If multiple have the same score, fields are merged.
    """
    entity_map = {}

    for ent in entities:
        eid = ent["entity_id"]
        score = len(ent.get("static_fields", {})) + len(ent.get("dynamic_fields", {}))
        score += len(ent.get("static_paths", [])) + len(ent.get("dynamic_paths", []))

        if eid not in entity_map or score > entity_map[eid]["score"]:
            entity_map[eid] = {"entity": ent, "score": score}
        elif score == entity_map[eid]["score"]:
            # merge fields if tied
            prev = entity_map[eid]["entity"]
            merged = {
                **prev,
                "static_fields": {**prev.get("static_fields", {}), **ent.get("static_fields", {})},
                "dynamic_fields": {**prev.get("dynamic_fields", {}), **ent.get("dynamic_fields", {})}
            }
            entity_map[eid]["entity"] = merged

    return [data["entity"] for data in entity_map.values()]


# -----------------------
# WebsiteProcessor class
# -----------------------
class WebsiteProcessor:
    """Scraper with sitemap-first and SPA fallback strategy + entity-aware chunking"""

    def __init__(self, website_url: str, proxies: dict = None, max_entities: int = 1000, max_chunks_per_entity: int = 10):
        self.website_url = website_url
        self.output_dir = Path("rag_data")
        self.website_folder = self.output_dir / f"{urlparse(website_url).netloc}"

        self.website_folder.mkdir(parents=True, exist_ok=True)
        (self.website_folder / "raw").mkdir(exist_ok=True)
        (self.website_folder / "cleaned").mkdir(exist_ok=True)
        (self.website_folder / "chunks").mkdir(exist_ok=True)
        (self.website_folder / "entities").mkdir(exist_ok=True)

        self.documents: List[Document] = []
        self.cleaned_docs: List[Document] = []
        self.chunks: List[Document] = []
        self.session = get_request_session(proxies=proxies)

        self.max_entities = max_entities
        self.max_chunks_per_entity = max_chunks_per_entity

        logger.info(f"Initialized processor for {website_url}")

    # -----------------------
    # Website info
    # -----------------------
    def get_website_info(self) -> dict:
        """Fetch website details (title, description, keywords, main heading)."""
        try:
            resp = self.session.get(self.website_url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            title = soup.title.string.strip() if soup.title else "No Title Found"

            desc_tag = soup.find("meta", attrs={"name": "description"}) \
                        or soup.find("meta", attrs={"property": "og:description"})
            description = desc_tag["content"].strip() if desc_tag and desc_tag.get("content") else ""

            keywords_tag = soup.find("meta", attrs={"name": "keywords"})
            keywords = keywords_tag["content"].strip() if keywords_tag and keywords_tag.get("content") else ""

            heading = soup.find("h1").get_text(strip=True) if soup.find("h1") else ""

            if not description:
                first_p = soup.find("p")
                if first_p:
                    description = first_p.get_text(strip=True)[:200]

            website_info = {
                "url": self.website_url,
                "title": title,
                "description": description,
                "keywords": keywords,
                "main_heading": heading
            }

            logger.info(f"Website Info Extracted → Title: {title}, Heading: {heading}, Desc: {description[:80]}...")
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

    # -----------------------
    # Sitemap loader with encoding
    # -----------------------
    def _load_sitemap_with_encoding(self, sitemap_url: str):
        """Custom sitemap loader that handles non-UTF8 encodings gracefully"""
        try:
            resp = self.session.get(sitemap_url, timeout=15)
            resp.raise_for_status()

            # Detect encoding
            detected = chardet.detect(resp.content)
            encoding = detected.get("encoding", "utf-8") or "utf-8"

            try:
                text = resp.content.decode(encoding, errors="strict")
            except UnicodeDecodeError:
                logger.warning(f"Decoding with {encoding} failed, retrying with errors='ignore'")
                text = resp.content.decode(encoding, errors="ignore")

            # Save raw sitemap for debugging
            (self.website_folder / "raw" / "sitemap_raw.xml").write_text(text, encoding="utf-8")

            # Load sitemap with LangChain loader
            sitemap_loader = SitemapLoader(web_path=sitemap_url)
            return sitemap_loader.load()

        except Exception as e:
            logger.warning(f"Sitemap fetch failed: {e}")
            return []

    # -----------------------
    # Playwright SPA crawling
    # -----------------------
    def _extract_all_links_spa(self, max_pages=200, headless=True) -> List[str]:
        """
        Extract internal links with dynamic JS crawling using Playwright.
        Capture JSON-LD if present, but still continue crawling links.
        """
        if not sync_playwright:
            logger.warning("Playwright not available, SPA crawling skipped.")
            return []

        visited, queue, all_links = set(), [self.website_url], set()
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
                    page.goto(url, timeout=20000, wait_until="domcontentloaded")
                    page.wait_for_timeout(5000)
                    html = page.content()

                    visited.add(url)
                    all_links.add(url)

                    # ✅ Capture JSON-LD but don’t short-circuit crawling
                    entities = extract_entities_from_jsonld(html)
                    if entities:
                        logger.info(f"✅ JSON-LD found on {url}, capturing it.")
                        self.documents.append(
                            Document(
                                page_content=json.dumps(entities, ensure_ascii=False),
                                metadata={
                                    "source": url,
                                    "raw_html": html,
                                    "extracted_from": "jsonld-direct"
                                }
                            )
                        )

                    # Continue crawling links
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
                    logger.warning(f"Playwright failed at {url}: {e}")
                    continue

            browser.close()

        return sorted(all_links)

    # -----------------------
    # SPA detection
    # -----------------------
    def _is_spa_site(self) -> bool:
        """Check if site is SPA by looking for common SPA markers"""
        try:
            resp = self.session.get(self.website_url, timeout=10)
            html = resp.text.lower()
            spa_signals = ['<div id="root"', 'window.__INITIAL_STATE__', 'ng-app', 'vue', 'react', 'spa']
            if any(signal in html for signal in spa_signals):
                logger.info("Site likely a SPA based on HTML heuristics.")
                return True
        except Exception as e:
            logger.warning(f"SPA detection failed: {e}")
        return False

    # -----------------------
    # Scrape website (unchanged strategy)
    # -----------------------
    def scrape_website(self) -> List[Document]:
        """Scrape with sitemap first, then SPA (Playwright JSON-LD first), or recursive fallback.
        Ensures raw_html is always preserved by performing a secondary raw fetch for every page.
        """
        logger.info("Starting website scraping...")

        # Step 1: Try sitemap loader with encoding fix
        sitemap_url = f"{self.website_url.rstrip('/')}/sitemap.xml"
        self.documents = self._load_sitemap_with_encoding(sitemap_url)

        if self.documents:
            logger.info(f"Loaded {len(self.documents)} pages from sitemap")
        else:
            # Step 2: If no sitemap, check if SPA
            logger.info("No valid sitemap docs, falling back...")
            if self._is_spa_site():
                logger.info("Detected SPA, crawling with Playwright...")
                spa_urls = self._extract_all_links_spa(max_pages=200, headless=True)

                if self.documents:
                    logger.info(f"✅ Captured {len(self.documents)} JSON-LD documents via Playwright, skipping Selenium.")
                elif spa_urls:
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
                        logger.warning(f"SeleniumURLLoader failed: {e}")
                        self.documents = []
                else:
                    logger.warning("No URLs found from SPA crawler.")
            else:
                # Step 3: Recursive fallback if not SPA
                logger.info("Using recursive URL loader fallback...")
                try:
                    loader = RecursiveUrlLoader(url=self.website_url, max_depth=2, prevent_outside=True)
                    self.documents = loader.load()
                    logger.info(f"Loaded {len(self.documents)} pages recursively")
                except Exception as e:
                    logger.error(f"Recursive loader failed: {e}")
                    self.documents = []

        # ✅ Step 4: Secondary raw fetch to preserve untouched HTML with JSON-LD
        refreshed_docs = []
        for d in self.documents:
            url = d.metadata.get("source") or self.website_url
            try:
                resp = self.session.get(url, timeout=20)
                resp.raise_for_status()
                raw_html = resp.text
                d.metadata["raw_html"] = raw_html
            except Exception as e:
                logger.warning(f"Failed secondary raw fetch for {url}: {e}")
                # fallback: at least keep page_content
                d.metadata["raw_html"] = d.page_content
            refreshed_docs.append(d)

        self.documents = refreshed_docs

        # Save results (if any)
        if self.documents:
            self._save_documents(self.documents, "raw/scraped_docs.json")
        else:
            logger.warning("⚠️ No documents scraped at all.")
        return self.documents
    # -----------------------
    # Cleaning docs: preserve raw_html in metadata then transform
    # -----------------------
    def clean_documents(self) -> List[Document]:
        """Clean scraped documents.
        - JSON-LD docs (extracted_from=jsonld-direct) are kept as-is.
        - All others are passed through Html2TextTransformer for cleaning.
        Always preserves raw_html for JSON-LD parsing later.
        """
        logger.info("Cleaning documents... (preserve raw_html docs)")

        if not self.documents:
            logger.warning("No raw documents to clean.")
            return []

        html_transformer = Html2TextTransformer()
        cleaned_docs = []

        for doc in self.documents:
            # ✅ Preserve raw_html before cleaning
            if "raw_html" not in doc.metadata:
                doc.metadata["raw_html"] = doc.page_content

            # Case 1: JSON-LD direct docs → keep as-is
            if doc.metadata.get("extracted_from") == "jsonld-direct":
                logger.debug(f"Preserving JSON-LD doc from {doc.metadata.get('source')}")
                doc.metadata["cleaned"] = False
                doc.metadata["cleaned_at"] = datetime.now().isoformat()
                cleaned_docs.append(doc)
                continue

            # Case 2: Normal HTML → clean to plain text
            try:
                transformed = html_transformer.transform_documents([doc])[0]
                transformed.page_content = " ".join(transformed.page_content.split())
                transformed.metadata["cleaned"] = True
                transformed.metadata["cleaned_at"] = datetime.now().isoformat()

                # ✅ Always preserve raw_html for JSON-LD
                if "raw_html" not in transformed.metadata:
                    transformed.metadata["raw_html"] = doc.metadata["raw_html"]

                cleaned_docs.append(transformed)
            except Exception as e:
                logger.error(f"Failed to clean doc from {doc.metadata.get('source')}: {e}")
                continue

        self.cleaned_docs = cleaned_docs
        logger.info(f"Cleaned {len(self.cleaned_docs)} documents")
        self._save_documents(self.cleaned_docs, "cleaned/cleaned_docs.json")
        return self.cleaned_docs
    # -----------------------
    # Create chunks (entity-aware, fallback to character splitting)
    # -----------------------

    def create_chunks(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """
        Entity-aware chunking with static/dynamic separation.
        - Consolidates entities to keep only the richest per entity_id.
        - Deduplicates chunks by content + hashes.
        - Fallback to normal text if no JSON-LD is present.
        """
        logger.info("Creating chunks (entity-aware, static/dynamic, deduplicated) ...")
        self.chunks = []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        domain = urlparse(self.website_url).netloc
        entity_map: Dict[str, Dict[str, Any]] = {}
        fallback_docs: List[Document] = []

        # ✅ Step 1: Collect entities and fallback text
        for doc in self.documents:
            source_url = doc.metadata.get("source") or self.website_url
            raw_html = doc.metadata.get("raw_html") or doc.page_content or ""

            entities = extract_entities_from_jsonld(raw_html)
            if entities:
                for entity in entities:
                    try:
                        static_fields, dynamic_fields, static_paths, dynamic_paths = classify_fields(entity)
                        score = len(static_fields) + len(dynamic_fields) + len(static_paths) + len(dynamic_paths)
                        entity_type = entity.get("@type", "Unknown")
                        entity_id = _stable_entity_id(domain, source_url, entity)

                        # store or update richer entity
                        if entity_id in entity_map:
                            if score > entity_map[entity_id]["score"]:
                                entity_map[entity_id] = {
                                    "entity": entity,
                                    "static_fields": static_fields,
                                    "dynamic_fields": dynamic_fields,
                                    "static_paths": static_paths,
                                    "dynamic_paths": dynamic_paths,
                                    "source_url": source_url,
                                    "entity_type": entity_type,
                                    "score": score,
                                }
                        else:
                            entity_map[entity_id] = {
                                "entity": entity,
                                "static_fields": static_fields,
                                "dynamic_fields": dynamic_fields,
                                "static_paths": static_paths,
                                "dynamic_paths": dynamic_paths,
                                "source_url": source_url,
                                "entity_type": entity_type,
                                "score": score,
                            }
                    except Exception as e:
                        logger.error(f"Error processing entity on {source_url}: {e}")
                        continue
            else:
                fallback_docs.append(doc)

        # ✅ Step 2: Consolidate entities before chunking
        consolidated_entities = consolidate_entities([
            {
                "entity_id": eid,
                "entity_type": data["entity_type"],
                "source_url": data["source_url"],
                "static_fields": data["static_fields"],
                "dynamic_fields": data["dynamic_fields"],
                "static_paths": data["static_paths"],
                "dynamic_paths": data["dynamic_paths"],
                "entity": data["entity"]
            }
            for eid, data in entity_map.items()
        ])

        for ent in consolidated_entities:
            static_fields = ent["static_fields"]
            dynamic_fields = ent["dynamic_fields"]
            static_paths = ent["static_paths"]
            dynamic_paths = ent["dynamic_paths"]
            entity = ent["entity"]
            entity_type = ent["entity_type"]
            entity_id = ent["entity_id"]
            source_url = ent["source_url"]

            # Build static summary
            summary_parts = []
            if "name" in static_fields:
                summary_parts.append(str(static_fields["name"]))
            if "description" in static_fields:
                summary_parts.append(str(static_fields["description"]))
            if "brand" in static_fields:
                summary_parts.append(f"Brand: {static_fields['brand']}")
            for k, v in static_fields.items():
                if k not in ["name", "description", "brand"] and isinstance(v, (str, int, float)):
                    summary_parts.append(str(v))

            static_text = ". ".join([p for p in summary_parts if p]) or json.dumps(static_fields, ensure_ascii=False)

            # Hashes
            static_hash = _sha256(static_text)
            dynamic_hash = _sha256(json.dumps(dynamic_fields, sort_keys=True, default=str))

            # Split if too large
            sub_chunks = [static_text]
            if len(static_text) > chunk_size:
                sub_chunks = text_splitter.split_text(static_text)
                sub_chunks = sub_chunks[:self.max_chunks_per_entity]

            for idx, sub in enumerate(sub_chunks):
                cleaned_text = " ".join(
                    Html2TextTransformer().transform_documents([Document(page_content=sub)])[0].page_content.split()
                )
                meta = {
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                    "source_url": source_url,
                    "chunk_index": idx,
                    "chunk_total": len(sub_chunks),
                    "static_hash": static_hash,
                    "dynamic_hash": dynamic_hash,
                    "static_paths": static_paths,
                    "dynamic_paths": dynamic_paths,
                    "static_fields": static_fields,
                    "dynamic_fields": dynamic_fields,
                    "raw_jsonld": entity,
                    "provenance": {"extracted_from": "jsonld", "fetch_ts": datetime.now().isoformat()},
                    "created_at": datetime.now().isoformat()
                }
                self.chunks.append(Document(page_content=cleaned_text, metadata=meta))

        # ✅ Step 3: Fallback chunks (non-JSON-LD)
        for doc in fallback_docs:
            source_url = doc.metadata.get("source") or self.website_url
            text = doc.page_content or ""
            if not text:
                continue

            cleaned_text = " ".join(
                Html2TextTransformer().transform_documents([Document(page_content=text)])[0].page_content.split()
            )
            sub_docs = text_splitter.split_documents(
                [Document(page_content=cleaned_text, metadata={"source_url": source_url})]
            )

            for idx, sd in enumerate(sub_docs):
                meta = {
                    "entity_id": f"{domain}::{source_url}::{_sha256(sd.page_content)}",
                    "entity_type": "fallback",
                    "source_url": source_url,
                    "chunk_index": idx,
                    "chunk_total": len(sub_docs),
                    "fallback": True,
                    "provenance": {"extracted_from": "cleaned_text", "fetch_ts": datetime.now().isoformat()},
                    "created_at": datetime.now().isoformat()
                }
                self.chunks.append(Document(page_content=sd.page_content, metadata=meta))

        logger.info(f"Created {len(self.chunks)} chunks before deduplication")

        # ✅ Step 4: Deduplicate by (content + hashes)
        unique_chunks = []
        seen = set()
        for chunk in self.chunks:
            key = (chunk.page_content.strip(), chunk.metadata.get("static_hash"), chunk.metadata.get("dynamic_hash"))
            if key not in seen:
                seen.add(key)
                unique_chunks.append(chunk)

        dedup_count = len(self.chunks) - len(unique_chunks)
        if dedup_count > 0:
            logger.info(f"🧹 Deduplicated {dedup_count} duplicate chunks")

        self.chunks = unique_chunks

        # ✅ Step 5: Save results
        self._save_documents(self.chunks, "chunks/document_chunks.json")
        self._save_embedding_ready(self.chunks)
        self._save_entities_ready(self.chunks)
        return self.chunks
    # -----------------------
    # Save helpers (preserve and enrich metadata)
    # -----------------------
    def _save_documents(self, docs: List[Document], filename: str):
        filepath = self.website_folder / filename
        data = [{
            'content': doc.page_content,
            'metadata': doc.metadata
        } for doc in docs]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved to {filepath}")

    def _save_embedding_ready(self, chunks: List[Document]):
        """
        Save embedding_ready.json in Pinecone-compatible format.
        Ensures:
          - No null values in metadata
          - Only allowed types (str, int, float, bool, list[str])
          - Metadata < 40KB
        """
        texts = [chunk.page_content for chunk in chunks]
        metadatas = []

        for chunk in chunks:
            raw_meta = chunk.metadata

            clean_meta = {
                "entity_id": str(raw_meta.get("entity_id") or ""),
                "entity_type": str(raw_meta.get("entity_type") or ""),
                "source_url": str(raw_meta.get("source_url") or ""),
                "chunk_index": int(raw_meta.get("chunk_index") or 0),
                "chunk_total": int(raw_meta.get("chunk_total") or 1),
                "static_hash": str(raw_meta.get("static_hash") or ""),
                "dynamic_hash": str(raw_meta.get("dynamic_hash") or ""),
                "created_at": str(raw_meta.get("created_at") or datetime.now().isoformat()),
            }

            # ✅ Truncate overly long strings (safety)
            for k, v in clean_meta.items():
                if isinstance(v, str) and len(v) > 2000:
                    clean_meta[k] = v[:2000] + "..."

            metadatas.append(clean_meta)

        embedding_data = {
            "texts": texts,
            "metadatas": metadatas,
            "total_chunks": len(texts),
            "created_at": datetime.now().isoformat()
        }

        filepath = self.website_folder / "chunks" / "embedding_ready.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(embedding_data, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Saved Pinecone-compatible embedding-ready data to {filepath}")

    def _save_entities_ready(self, chunks: List[Document]):
        """
        Save a nested entity view (entities -> chunks) for debugging & incremental update persistence.
        Also persists per-entity files under entities/ for change detection.
        Uses hashed filenames to avoid Windows path length issues.
        """
        import hashlib

        entities_map: Dict[str, Dict[str, Any]] = {}
        for chunk in chunks:
            meta = chunk.metadata
            eid = meta.get('entity_id') or meta.get('source_url') or f"unknown::{_sha256(chunk.page_content)}"
            if eid not in entities_map:
                entities_map[eid] = {
                    "entity_id": eid,
                    "entity_type": meta.get('entity_type', 'unknown'),
                    "source_url": meta.get('source_url', ''),
                    "chunks": [],
                    "static_hash": meta.get('static_hash'),
                    "dynamic_hash": meta.get('dynamic_hash'),
                    "static_paths": meta.get('static_paths', []),
                    "dynamic_paths": meta.get('dynamic_paths', []),
                    "static_fields": meta.get('static_fields', {}),
                    "dynamic_fields": meta.get('dynamic_fields', {}),
                    "last_seen": meta.get('created_at')
                }
            entities_map[eid]["chunks"].append({
                "chunk_id": f"{eid}::{meta.get('chunk_index')}",
                "text_preview": chunk.page_content[:300],
                "chunk_index": meta.get('chunk_index'),
                "chunk_total": meta.get('chunk_total'),
                "provenance": meta.get('provenance', {})
            })

        # write a single entities_ready.json for the site
        entities_list = list(entities_map.values())
        filepath = self.website_folder / "chunks" / "entities_ready.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({"entities": entities_list, "created_at": datetime.now().isoformat()}, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved entities-ready data to {filepath}")

        # persist per-entity file for change detection/upserts
        for eid, ent in entities_map.items():
            # Use hash for filename to avoid long paths
            safe_hash = hashlib.sha1(eid.encode("utf-8")).hexdigest()[:16]
            per_path = self.website_folder / "entities" / f"{safe_hash}.json"
            with open(per_path, 'w', encoding='utf-8') as f:
                json.dump(ent, f, indent=2, ensure_ascii=False)

        logger.info(f"Persisted {len(entities_map)} entity files for change detection under {self.website_folder / 'entities'}")


#---------------------------------------------------------------------------------------------------------------------------------------

# import json
# import logging
# import time
# from pathlib import Path
# from datetime import datetime
# from typing import List
# from urllib.parse import urlparse, urljoin

# from dotenv import load_dotenv
# import requests
# load_dotenv()

# from langchain_community.document_loaders import (
#     SitemapLoader,
#     RecursiveUrlLoader,
#     SeleniumURLLoader
# )
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_transformers import Html2TextTransformer
# from langchain.schema import Document

# from bs4 import BeautifulSoup
# try:
#     from playwright.sync_api import sync_playwright
# except ImportError:
#     sync_playwright = None

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
# logger = logging.getLogger(__name__)

# class WebsiteProcessor:
#     """Scraper with sitemap-first and SPA fallback strategy"""

#     def __init__(self, website_url: str):
#         self.website_url = website_url
#         self.output_dir = Path("rag_data")
#         self.website_folder = self.output_dir / f"{urlparse(website_url).netloc}"

#         self.website_folder.mkdir(parents=True, exist_ok=True)
#         (self.website_folder / "raw").mkdir(exist_ok=True)
#         (self.website_folder / "cleaned").mkdir(exist_ok=True)
#         (self.website_folder / "chunks").mkdir(exist_ok=True)

#         self.documents = []
#         self.cleaned_docs = []
#         self.chunks = []

#         logger.info(f"Initialized processor for {website_url}")


#     def get_website_info(self) -> dict:
#         """
#         Fetch website details (title, description, keywords, main heading).
#         Works for any type of website without hardcoded classification.
#         """
#         try:
#             resp = requests.get(self.website_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
#             soup = BeautifulSoup(resp.text, "html.parser")

#             # Title
#             title = soup.title.string.strip() if soup.title else "No Title Found"

#             # Meta description
#             description = ""
#             desc_tag = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
#             if desc_tag and desc_tag.get("content"):
#                 description = desc_tag["content"].strip()

#             # Meta keywords
#             keywords = ""
#             keywords_tag = soup.find("meta", attrs={"name": "keywords"})
#             if keywords_tag and keywords_tag.get("content"):
#                 keywords = keywords_tag["content"].strip()

#             # First heading (h1)
#             heading = ""
#             h1_tag = soup.find("h1")
#             if h1_tag:
#                 heading = h1_tag.get_text(strip=True)

#             # Fallback if description is empty → take first paragraph
#             if not description:
#                 first_p = soup.find("p")
#                 if first_p:
#                     description = first_p.get_text(strip=True)[:200]  # limit length

#             website_info = {
#                 "url": self.website_url,
#                 "title": title,
#                 "description": description,
#                 "keywords": keywords,
#                 "main_heading": heading
#             }

#             # --- Logging output ---
#             logger.info(f"Website Info Extracted → Title: {title}, Heading: {heading}, Description: {description[:80]}...")
#             return website_info

#         except Exception as e:
#             logger.error(f"Failed to fetch website info: {e}")
#             return {
#                 "url": self.website_url,
#                 "title": "Unknown",
#                 "description": "Unknown",
#                 "keywords": "",
#                 "main_heading": ""
#             }

#     def _extract_all_links_spa(self, max_pages=200, headless=True) -> List[str]:
#         """Extract internal links with dynamic JS crawling using Playwright"""
#         if not sync_playwright:
#             logger.warning("Playwright not available, SPA crawling skipped.")
#             return []

#         visited = set()
#         queue = [self.website_url]
#         all_links = set()
#         domain = urlparse(self.website_url).netloc

#         with sync_playwright() as pw:
#             browser = pw.chromium.launch(headless=headless)
#             context = browser.new_context()
#             page = context.new_page()

#             while queue and len(visited) < max_pages:
#                 url = queue.pop(0)
#                 if url in visited:
#                     continue

#                 try:
#                     page.goto(url, timeout=60000, wait_until="domcontentloaded")
#                     time.sleep(1)
#                     html = page.content()
#                     visited.add(url)
#                     all_links.add(url)

#                     soup = BeautifulSoup(html, "html.parser")
#                     for a in soup.select("a[href]"):
#                         href = a["href"].strip()
#                         if not href or href.startswith(("mailto:", "tel:", "javascript:")):
#                             continue
#                         abs_url = href if href.startswith("http") else urljoin(url, href)
#                         if urlparse(abs_url).netloc != domain:
#                             continue
#                         abs_url = abs_url.split("#")[0]
#                         if abs_url not in visited and abs_url not in queue:
#                             queue.append(abs_url)

#                 except Exception as e:
#                     logger.debug(f"Error visiting {url}: {e}")
#                     continue

#             browser.close()
#         return sorted(all_links)

#     def _is_spa_site(self) -> bool:
#         """Heuristic to check if site is SPA by inspecting main page HTML for SPA markers"""
#         try:
#             from requests import get
#         except ImportError:
#             logger.warning("Requests library not installed, cannot detect SPA heuristics.")
#             return False

#         try:
#             resp = get(self.website_url, timeout=10)
#             html = resp.text.lower()
#             # Heuristics: check if main page strongly depends on JS SPA frameworks or empty content
#             spa_signals = ['<div id="root"', 'window.__INITIAL_STATE__', 'ng-app', 'vue', 'react', 'spa']
#             if any(signal in html for signal in spa_signals):
#                 logger.info("Site likely a SPA based on HTML heuristics.")
#                 return True
#         except Exception as e:
#             logger.warning(f"SPA detection failed: {e}")
#         return False

#     def scrape_website(self) -> List[Document]:
#         """Main scrape method with sitemap first, then SPA or recursive fallback"""

#         logger.info("Starting website scraping with sitemap-first approach...")

#         # Step 1: Try sitemap loader
#         try:
#             logger.info("Trying sitemap loader...")
#             sitemap_loader = SitemapLoader(web_path=f"{self.website_url}/sitemap.xml")
#             self.documents = sitemap_loader.load()
#             logger.info(f"Loaded {len(self.documents)} pages from sitemap")
#         except Exception as e:
#             logger.warning(f"Sitemap loading failed: {e}")
#             self.documents = []

#         # Step 2: If no documents loaded from sitemap, check if SPA or fallback recursive
#         if len(self.documents) == 0:
#             is_spa = self._is_spa_site()

#             if is_spa:
#                 logger.info("Detected SPA site, using dynamic SPA crawling and Selenium loader...")
#                 spa_urls = self._extract_all_links_spa(max_pages=200, headless=True)
#                 if spa_urls:
#                     try:
#                         loader = SeleniumURLLoader(
#                             urls=spa_urls,
#                             headless=True,
#                             browser="chrome",
#                             continue_on_failure=True
#                         )
#                         self.documents = loader.load()
#                         logger.info(f"Loaded {len(self.documents)} pages from SPA Selenium loader")
#                     except Exception as e:
#                         logger.warning(f"SeleniumURLLoader failed on SPA URLs: {e}")
#                         self.documents = []
#                 else:
#                     logger.warning("No URLs found from SPA crawler.")
#             else:
#                 logger.info("Not SPA or unable to confirm SPA. Using recursive URL loader fallback...")
#                 try:
#                     loader = RecursiveUrlLoader(
#                         url=self.website_url,
#                         max_depth=2,
#                         prevent_outside=True
#                     )
#                     self.documents = loader.load()
#                     logger.info(f"Loaded {len(self.documents)} pages recursively")
#                 except Exception as e:
#                     logger.error(f"Recursive URL loader failed: {e}")
#                     self.documents = []

#         self._save_documents(self.documents, "raw/scraped_docs.json")
#         return self.documents

#     def clean_documents(self) -> List[Document]:
#         """Clean scraped documents"""
#         logger.info("Cleaning documents...")
#         html_transformer = Html2TextTransformer()
#         self.cleaned_docs = html_transformer.transform_documents(self.documents)

#         for doc in self.cleaned_docs:
#             doc.page_content = ' '.join(doc.page_content.split())
#             doc.metadata['cleaned'] = True
#             doc.metadata['cleaned_at'] = datetime.now().isoformat()

#         logger.info(f"Cleaned {len(self.cleaned_docs)} documents")
#         self._save_documents(self.cleaned_docs, "cleaned/cleaned_docs.json")
#         return self.cleaned_docs

#     def create_chunks(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
#         """Chunk cleaned documents for embeddings"""
#         logger.info("Creating chunks...")
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=chunk_size,
#             chunk_overlap=chunk_overlap,
#             length_function=len,
#             separators=["\n\n", "\n", ". ", " ", ""]
#         )
#         self.chunks = text_splitter.split_documents(self.cleaned_docs)

#         for i, chunk in enumerate(self.chunks):
#             chunk.metadata['chunk_id'] = i
#             chunk.metadata['total_chunks'] = len(self.chunks)

#         logger.info(f"Created {len(self.chunks)} chunks")
#         self._save_documents(self.chunks, "chunks/document_chunks.json")
#         self._save_embedding_ready()
#         return self.chunks

#     def _save_documents(self, docs: List[Document], filename: str):
#         filepath = self.website_folder / filename
#         data = [{
#             'content': doc.page_content,
#             'metadata': doc.metadata
#         } for doc in docs]
#         with open(filepath, 'w', encoding='utf-8') as f:
#             json.dump(data, f, indent=2, ensure_ascii=False)
#         logger.info(f"Saved to {filepath}")

#     def _save_embedding_ready(self):
#         texts = [chunk.page_content for chunk in self.chunks]
#         metadatas = [chunk.metadata for chunk in self.chunks]

#         embedding_data = {
#             'texts': texts,
#             'metadatas': metadatas,
#             'total_chunks': len(texts),
#             'created_at': datetime.now().isoformat()
#         }

#         filepath = self.website_folder / "chunks" / "embedding_ready.json"
#         with open(filepath, 'w', encoding='utf-8') as f:
#             json.dump(embedding_data, f, indent=2, ensure_ascii=False)

#         logger.info(f"Saved embedding-ready data to {filepath}")


#------------------------------------------------------------------------------------------------------------------------------


# import json
# import logging
# import time
# import random
# from pathlib import Path
# from datetime import datetime
# from typing import List
# from urllib.parse import urlparse, urljoin

# from dotenv import load_dotenv
# import requests
# from requests.adapters import HTTPAdapter
# from urllib3.util.retry import Retry

# import chardet
# from bs4 import BeautifulSoup

# load_dotenv()

# from langchain_community.document_loaders import (
#     SitemapLoader,
#     RecursiveUrlLoader,
#     SeleniumURLLoader
# )
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_transformers import Html2TextTransformer
# from langchain.schema import Document

# try:
#     from playwright.sync_api import sync_playwright
# except ImportError:
#     sync_playwright = None

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
# logger = logging.getLogger(__name__)


# # --- GLOBAL REQUEST SESSION WITH RETRIES, HEADERS, PROXY ---
# def get_request_session(proxies: dict = None):
#     """Return a requests session with retries, headers, and optional proxy"""
#     session = requests.Session()

#     retries = Retry(
#         total=5,
#         backoff_factor=1,
#         status_forcelist=[500, 502, 503, 504, 429]
#     )
#     adapter = HTTPAdapter(max_retries=retries)
#     session.mount("http://", adapter)
#     session.mount("https://", adapter)

#     # Rotating headers
#     user_agents = [
#         "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119 Safari/537.36",
#         "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16 Safari/605.1.15",
#         "Mozilla/5.0 (X11; Linux x86_64) Gecko/20100101 Firefox/118.0",
#     ]
#     session.headers.update({"User-Agent": random.choice(user_agents)})

#     if proxies:
#         session.proxies.update(proxies)

#     return session


# class WebsiteProcessor:
#     """Scraper with sitemap-first and SPA fallback strategy"""

#     def __init__(self, website_url: str, proxies: dict = None):
#         self.website_url = website_url
#         self.output_dir = Path("rag_data")
#         self.website_folder = self.output_dir / f"{urlparse(website_url).netloc}"

#         self.website_folder.mkdir(parents=True, exist_ok=True)
#         (self.website_folder / "raw").mkdir(exist_ok=True)
#         (self.website_folder / "cleaned").mkdir(exist_ok=True)
#         (self.website_folder / "chunks").mkdir(exist_ok=True)

#         self.documents = []
#         self.cleaned_docs = []
#         self.chunks = []
#         self.session = get_request_session(proxies=proxies)

#         logger.info(f"Initialized processor for {website_url}")

#     def get_website_info(self) -> dict:
#         """Fetch website details (title, description, keywords, main heading)."""
#         try:
#             resp = self.session.get(self.website_url, timeout=15)
#             resp.raise_for_status()
#             soup = BeautifulSoup(resp.text, "html.parser")

#             title = soup.title.string.strip() if soup.title else "No Title Found"

#             desc_tag = soup.find("meta", attrs={"name": "description"}) \
#                         or soup.find("meta", attrs={"property": "og:description"})
#             description = desc_tag["content"].strip() if desc_tag and desc_tag.get("content") else ""

#             keywords_tag = soup.find("meta", attrs={"name": "keywords"})
#             keywords = keywords_tag["content"].strip() if keywords_tag and keywords_tag.get("content") else ""

#             heading = soup.find("h1").get_text(strip=True) if soup.find("h1") else ""

#             if not description:
#                 first_p = soup.find("p")
#                 if first_p:
#                     description = first_p.get_text(strip=True)[:200]

#             website_info = {
#                 "url": self.website_url,
#                 "title": title,
#                 "description": description,
#                 "keywords": keywords,
#                 "main_heading": heading
#             }

#             logger.info(f"Website Info Extracted → Title: {title}, Heading: {heading}, Desc: {description[:80]}...")
#             return website_info

#         except Exception as e:
#             logger.error(f"Failed to fetch website info: {e}")
#             return {
#                 "url": self.website_url,
#                 "title": "Unknown",
#                 "description": "Unknown",
#                 "keywords": "",
#                 "main_heading": ""
#             }

#     def _load_sitemap_with_encoding(self, sitemap_url: str):
#         """Custom sitemap loader that handles non-UTF8 encodings gracefully"""
#         try:
#             resp = self.session.get(sitemap_url, timeout=15)
#             resp.raise_for_status()

#             # Detect encoding
#             detected = chardet.detect(resp.content)
#             encoding = detected.get("encoding", "utf-8") or "utf-8"

#             try:
#                 text = resp.content.decode(encoding, errors="strict")
#             except UnicodeDecodeError:
#                 logger.warning(f"Decoding with {encoding} failed, retrying with errors='ignore'")
#                 text = resp.content.decode(encoding, errors="ignore")

#             # Save raw sitemap for debugging
#             (self.website_folder / "raw" / "sitemap_raw.xml").write_text(text, encoding="utf-8")

#             # Load sitemap with LangChain loader
#             sitemap_loader = SitemapLoader(web_path=sitemap_url)
#             return sitemap_loader.load()

#         except Exception as e:
#             logger.warning(f"Sitemap fetch failed: {e}")
#             return []

#     def _extract_all_links_spa(self, max_pages=200, headless=True) -> List[str]:
#         """Extract internal links with dynamic JS crawling using Playwright"""
#         if not sync_playwright:
#             logger.warning("Playwright not available, SPA crawling skipped.")
#             return []

#         visited, queue, all_links = set(), [self.website_url], set()
#         domain = urlparse(self.website_url).netloc

#         with sync_playwright() as pw:
#             browser = pw.chromium.launch(headless=headless)
#             context = browser.new_context()
#             page = context.new_page()

#             while queue and len(visited) < max_pages:
#                 url = queue.pop(0)
#                 if url in visited:
#                     continue
#                 try:
#                     page.goto(url, timeout=60000, wait_until="networkidle")
#                     time.sleep(1.5)
#                     html = page.content()
#                     visited.add(url)
#                     all_links.add(url)

#                     soup = BeautifulSoup(html, "html.parser")
#                     for a in soup.select("a[href]"):
#                         href = a["href"].strip()
#                         if not href or href.startswith(("mailto:", "tel:", "javascript:")):
#                             continue
#                         abs_url = href if href.startswith("http") else urljoin(url, href)
#                         if urlparse(abs_url).netloc != domain:
#                             continue
#                         abs_url = abs_url.split("#")[0]
#                         if abs_url not in visited and abs_url not in queue:
#                             queue.append(abs_url)

#                 except Exception as e:
#                     logger.debug(f"Error visiting {url}: {e}")
#                     continue

#             browser.close()
#         return sorted(all_links)

#     def _is_spa_site(self) -> bool:
#         """Check if site is SPA by looking for common SPA markers"""
#         try:
#             resp = self.session.get(self.website_url, timeout=10)
#             html = resp.text.lower()
#             spa_signals = ['<div id="root"', 'window.__INITIAL_STATE__', 'ng-app', 'vue', 'react', 'spa']
#             if any(signal in html for signal in spa_signals):
#                 logger.info("Site likely a SPA based on HTML heuristics.")
#                 return True
#         except Exception as e:
#             logger.warning(f"SPA detection failed: {e}")
#         return False

#     def scrape_website(self) -> List[Document]:
#         """Scrape with sitemap first, then SPA or recursive fallback"""
#         logger.info("Starting website scraping...")

#         # Step 1: Try sitemap loader with encoding fix
#         sitemap_url = f"{self.website_url.rstrip('/')}/sitemap.xml"
#         self.documents = self._load_sitemap_with_encoding(sitemap_url)

#         if not self.documents:
#             logger.info("No valid sitemap docs, falling back...")
#             if self._is_spa_site():
#                 logger.info("Detected SPA, crawling with Playwright + Selenium...")
#                 spa_urls = self._extract_all_links_spa(max_pages=200, headless=True)
#                 if spa_urls:
#                     try:
#                         loader = SeleniumURLLoader(
#                             urls=spa_urls, headless=True, browser="chrome", continue_on_failure=True
#                         )
#                         self.documents = loader.load()
#                         logger.info(f"Loaded {len(self.documents)} pages from SPA Selenium loader")
#                     except Exception as e:
#                         logger.warning(f"SeleniumURLLoader failed: {e}")
#                         self.documents = []
#             else:
#                 logger.info("Using recursive URL loader fallback...")
#                 try:
#                     loader = RecursiveUrlLoader(url=self.website_url, max_depth=2, prevent_outside=True)
#                     self.documents = loader.load()
#                     logger.info(f"Loaded {len(self.documents)} pages recursively")
#                 except Exception as e:
#                     logger.error(f"Recursive loader failed: {e}")
#                     self.documents = []

#         self._save_documents(self.documents, "raw/scraped_docs.json")
#         return self.documents

#     def clean_documents(self) -> List[Document]:
#         """Clean scraped documents: remove scripts, styles, compress whitespace"""
#         logger.info("Cleaning documents...")
#         html_transformer = Html2TextTransformer()
#         self.cleaned_docs = html_transformer.transform_documents(self.documents)

#         for doc in self.cleaned_docs:
#             doc.page_content = ' '.join(doc.page_content.split())
#             doc.metadata['cleaned'] = True
#             doc.metadata['cleaned_at'] = datetime.now().isoformat()

#         logger.info(f"Cleaned {len(self.cleaned_docs)} documents")
#         self._save_documents(self.cleaned_docs, "cleaned/cleaned_docs.json")
#         return self.cleaned_docs

#     def create_chunks(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
#         """Chunk cleaned documents"""
#         logger.info("Creating chunks...")
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len,
#             separators=["\n\n", "\n", ". ", " ", ""]
#         )
#         self.chunks = text_splitter.split_documents(self.cleaned_docs)

#         for i, chunk in enumerate(self.chunks):
#             chunk.metadata['chunk_id'] = i
#             chunk.metadata['total_chunks'] = len(self.chunks)

#         logger.info(f"Created {len(self.chunks)} chunks")
#         self._save_documents(self.chunks, "chunks/document_chunks.json")
#         self._save_embedding_ready()
#         return self.chunks

#     def _save_documents(self, docs: List[Document], filename: str):
#         filepath = self.website_folder / filename
#         data = [{'content': doc.page_content, 'metadata': doc.metadata} for doc in docs]
#         with open(filepath, 'w', encoding='utf-8') as f:
#             json.dump(data, f, indent=2, ensure_ascii=False)
#         logger.info(f"Saved to {filepath}")

#     def _save_embedding_ready(self):
#         texts = [chunk.page_content for chunk in self.chunks]
#         metadatas = [chunk.metadata for chunk in self.chunks]
#         embedding_data = {
#             'texts': texts,
#             'metadatas': metadatas,
#             'total_chunks': len(texts),
#             'created_at': datetime.now().isoformat()
#         }
#         filepath = self.website_folder / "chunks" / "embedding_ready.json"
#         with open(filepath, 'w', encoding='utf-8') as f:
#             json.dump(embedding_data, f, indent=2, ensure_ascii=False)
#         logger.info(f"Saved embedding-ready data to {filepath}")
