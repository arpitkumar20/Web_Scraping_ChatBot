# Enterprise Folder Structure for Nisaa Chatbot

```
nisaa-chatbot/
│
├── backend/
│   ├── app/
│   │   ├── **init**.py
│   │   ├── wsgi.py
│   │   ├── auth/
│   │   │   └── api\_key.py
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   └── logging\_conf.py
│   │   ├── api/
│   │   │   ├── routes\_chat.py
│   │   │   ├── routes\_whatsapp.py
│   │   │   ├── routes\_admin.py
│   │   │   └── routes\_booking.py
│   │   ├── services/
│   │   │   ├── rag.py
│   │   │   ├── vectorstore.py
│   │   │   ├── ingestion.py
│   │   │   ├── crawler.py
│   │   │   └── whatsapp.py
│   │   ├── models/
│   │   │   └── booking.py
│   │   ├── utils/
│   │   │   └── validators.py
│   │   └── scripts/
│   │       └── seed\_data.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── docker-compose.yml
├── .env.example
└── README.md

```

---

## How This Maps to Features

* **Conversational Intelligence (Q\&A, RAG)** → `services/rag.py`, `services/vectorstore.py`, `api/routes_chat.py`
* **Website Crawling & Indexing** → `services/crawler.py`, `services/ingestion.py`
* **HMS & Booking** → `services/hms_integration.py`, `services/hotel_integration.py`, `models/booking.py`
* **WhatsApp** → `services/whatsapp.py`, `api/routes_whatsapp.py`
* **Admin Panel** → `admin/`, `api/routes_admin.py`, frontend dashboard
* **360° Room Tours / Hotel Enhancements** → `services/hotel_integration.py` + frontend UI
* **Hospital Features** → `models/doctor.py`, `services/hms_integration.py`
* **DevOps/Infra** → `infra/` (Docker, K8s, CI/CD, monitoring)
* **Testing** → `tests/unit`, `tests/integration`, `tests/performance`, `tests/security`

---

```markdown

````
### 🔹 `backend/app/__init__.py`

```python
from flask import Flask
from .core.config import Config
from .core.logging_conf import configure_logging

from .api.routes_chat import chat_bp
from .api.routes_whatsapp import whatsapp_bp
from .api.routes_admin import admin_bp
from .api.routes_booking import booking_bp

def create_app():
    app = Flask(__name__, static_folder="../static", static_url_path="/static")
    app.config.from_object(Config)

    configure_logging(app)

    app.register_blueprint(chat_bp, url_prefix="/api/v1/chat")
    app.register_blueprint(whatsapp_bp, url_prefix="/api/v1/whatsapp")
    app.register_blueprint(admin_bp, url_prefix="/api/v1/admin")
    app.register_blueprint(booking_bp, url_prefix="/api/v1/booking")

    @app.route("/health", methods=["GET"])
    def health():
        return {"status":"ok"}, 200

    @app.route("/ready", methods=["GET"])
    def ready():
        return {"ready": True}, 200

    return app
```

---

### 🔹 `backend/app/wsgi.py`

```python
from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
```

---

### 🔹 `backend/app/core/config.py`

```python
import os

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "nisaa")
    API_KEY = os.getenv("API_KEY", "changeme")
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "/data/vectorstore")
    REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "/app/uploads")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
```

---

### 🔹 `backend/app/core/logging_conf.py`

```python
import logging
import sys

def configure_logging(app=None):
    level = getattr(logging, (app.config.get("LOG_LEVEL") if app else "INFO"))
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s %(message)s"
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
```

---

### 🔹 `backend/app/auth/api_key.py`

```python
from functools import wraps
from flask import request, jsonify, current_app

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("x-api-key") or request.args.get("api_key")
        if not key or key != current_app.config.get("API_KEY"):
            return jsonify({"error":"Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated
```

---

### 🔹 `backend/app/services/vectorstore.py`

```python
import os, threading, logging
log = logging.getLogger("nisaa.vectorstore")
_vector_lock = threading.Lock()
_vector_store = None

def _create_faiss_vectorstore(path):
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    if os.path.exists(path):
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    vs = FAISS.from_texts([], embeddings)
    vs.save_local(path)
    return vs

def load_vectorstore(path=None):
    global _vector_store
    if path is None:
        path = os.getenv("VECTOR_DB_PATH", "/data/vectorstore")
    if _vector_store is None:
        with _vector_lock:
            if _vector_store is None:
                _vector_store = _create_faiss_vectorstore(path)
    return _vector_store

def add_documents(text_chunks, metas=None, path=None):
    vs = load_vectorstore(path)
    if metas is None:
        metas = [{}] * len(text_chunks)
    vs.add_texts(text_chunks, metadatas=metas)
    vs.save_local(path or os.getenv("VECTOR_DB_PATH", "/data/vectorstore"))
```

---

### 🔹 `backend/app/services/rag.py`

```python
import logging
log = logging.getLogger("nisaa.rag")
from .vectorstore import load_vectorstore

class RAGPipeline:
    def __init__(self, vector_db_path=None):
        self.vector_db_path = vector_db_path
        self._qa = None

    def _ensure(self):
        if self._qa is None:
            from langchain.chains import RetrievalQA
            from langchain_openai import ChatOpenAI
            vs = load_vectorstore(self.vector_db_path)
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
            self._qa = RetrievalQA.from_chain_type(llm=llm, retriever=vs.as_retriever(), return_source_documents=True)

    def answer(self, query):
        self._ensure()
        result = self._qa({"query": query})
        return {
            "answer": result.get("result"),
            "sources": [
                {"page_content": doc.page_content[:300], "metadata": getattr(doc, "metadata", {})}
                for doc in result.get("source_documents", [])
            ]
        }
```

---

### 🔹 `backend/app/services/ingestion.py`

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .vectorstore import add_documents
import logging
log = logging.getLogger("nisaa.ingest")

def ingest_text(text, chunk_size=500, chunk_overlap=50, path=None, metadata=None):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    metas = [metadata or {} for _ in chunks]
    add_documents(chunks, metas, path)
    log.info("Ingested %d chunks", len(chunks))
    return len(chunks)
```

---

### 🔹 `backend/app/services/crawler.py`

```python
import requests, logging
from bs4 import BeautifulSoup
log = logging.getLogger("nisaa.crawler")

def crawl_url(url, selectors=None):
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception:
        log.exception("Failed to fetch %s", url)
        return None
    soup = BeautifulSoup(resp.text, "html.parser")
    if selectors:
        return "\n\n".join(
            el.get_text(separator=" ", strip=True)
            for sel in selectors for el in soup.select(sel)
        )
    return soup.get_text(separator="\n", strip=True)
```

---

### 🔹 `backend/app/services/whatsapp.py`

```python
import logging
log = logging.getLogger("nisaa.whatsapp")

def handle_incoming_message(payload):
    log.info("Received whatsapp payload: %s", payload)
    return {"session_id": payload.get("From"), "text": payload.get("Body")}
```

---

### 🔹 `backend/app/api/routes_chat.py`

```python
from flask import Blueprint, request, jsonify, current_app
from ..services.rag import RAGPipeline
from ..auth.api_key import require_api_key

chat_bp = Blueprint("chat", __name__)
_rag = RAGPipeline()

@chat_bp.route("/ask", methods=["POST"])
@require_api_key
def ask():
    query = (request.get_json() or {}).get("query")
    if not query:
        return jsonify({"error":"query required"}), 400
    try:
        res = _rag.answer(query)
        return jsonify({"query": query, "answer": res["answer"], "sources": res["sources"]})
    except Exception:
        current_app.logger.exception("chat ask failed")
        return jsonify({"error":"internal error"}), 500
```

---

### 🔹 `backend/app/api/routes_whatsapp.py`

```python
from flask import Blueprint, request, jsonify
from ..services.whatsapp import handle_incoming_message
from ..services.rag import RAGPipeline

whatsapp_bp = Blueprint("whatsapp", __name__)
_rag = RAGPipeline()

@whatsapp_bp.route("/webhook", methods=["POST"])
def webhook():
    normalized = handle_incoming_message(request.get_json() or {})
    query = normalized.get("text")
    if not query:
        return jsonify({"error":"no text"}), 400
    res = _rag.answer(query)
    return jsonify({"answer": res["answer"]})
```

---

### 🔹 `backend/app/api/routes_admin.py`

```python
from flask import Blueprint, request, jsonify
from ..auth.api_key import require_api_key
from ..services.ingestion import ingest_text
from ..services.crawler import crawl_url

admin_bp = Blueprint("admin", __name__)

@admin_bp.route("/ingest/url", methods=["POST"])
@require_api_key
def ingest_url():
    data = request.get_json() or {}
    text = crawl_url(data.get("url"), data.get("selectors"))
    if not text:
        return jsonify({"error":"could not crawl"}), 500
    count = ingest_text(text)
    return jsonify({"ingested_chunks": count})
```

---

### 🔹 `backend/app/api/routes_booking.py`

```python
from flask import Blueprint, request, jsonify
from ..auth.api_key import require_api_key
import redis, os
from datetime import datetime, timezone
from ..core.config import Config

booking_bp = Blueprint("booking", __name__)
redis_client = redis.from_url(Config.REDIS_URL)

def now_iso(): return datetime.now(timezone.utc).isoformat()

@booking_bp.route("/start", methods=["POST"])
@require_api_key
def start_booking():
    session_id = (request.get_json() or {}).get("session_id")
    if not session_id:
        return jsonify({"error":"session_id required"}), 400
    redis_client.hset(f"booking:{session_id}", mapping={"state":"started","created_at": now_iso()})
    return jsonify({"status":"started", "session_id": session_id})

@booking_bp.route("/state", methods=["GET"])
@require_api_key
def get_state():
    session_id = request.args.get("session_id")
    if not session_id:
        return jsonify({"error":"session_id required"}), 400
    data = redis_client.hgetall(f"booking:{session_id}")
    return jsonify({k.decode(): v.decode() for k,v in data.items()}) if data else jsonify({"state":"none"})
```

---

### 🔹 `backend/app/models/booking.py`

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class Booking:
    session_id: str
    user_name: Optional[str] = None
    contact: Optional[str] = None
    doctor_name: Optional[str] = None
    appointment_date: Optional[str] = None
    appointment_time: Optional[str] = None
    status: str = "pending"
```

---

### 🔹 `backend/app/utils/validators.py`

```python
import re
from datetime import datetime

def is_valid_date_iso(s):
    try: datetime.fromisoformat(s); return True
    except Exception: return False

_phone_re = re.compile(r"^[0-9\-\+\s\(\)]{7,20}$")
def is_valid_phone(s): return bool(_phone_re.match(s or ""))
```

---

### 🔹 `backend/app/scripts/seed_data.py`

```python
from pymongo import MongoClient
import os

def main():
    client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
    db = client[os.getenv("MONGO_DB_NAME", "nisaa")]
    db.doctors.insert_many([
        {"name":"Dr. A", "specialty":"Cardiology"},
        {"name":"Dr. B", "specialty":"Pediatrics"}
    ])
    print("seeded doctors")

if __name__ == "__main__":
    main()
```

# NGROK
Domain : aeronautic-showier-marquitta.ngrok-free.app
ngrok http 5004 --url aeronautic-showier-marquitta.ngrok-free.app
ngrok http --url=aeronautic-showier-marquitta.ngrok-free.app 5000

---

### 🔹 `backend/Dockerfile`

```dockerfile
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
WORKDIR /app

RUN apt-get update && apt-get install -y build-essential git curl libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt

COPY backend /app
RUN mkdir -p /app/uploads
ENV UPLOAD_FOLDER=/app/uploads

EXPOSE 8000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "app.wsgi:app"]
```

---

### 🔹 `backend/requirements.txt`

```text
Flask==2.2.5
gunicorn==20.1.0
pymongo==4.3.3
redis==4.5.5
langchain==0.0.300
langchain-openai==0.0.4
langchain-community==0.0.9
openai==0.27.0
faiss-cpu==1.7.4
sentence-transformers==2.2.2
beautifulsoup4==4.12.2
requests==2.31.0
python-dotenv==1.0.0
werkzeug==2.2.3
```

---

### 🔹 `docker-compose.yml`

```yaml
version: "3.8"
services:
  backend:
    build:
      context: .
      dockerfile: backend/Dockerfile
    ports:
      - "8000:8000"
    env_file: .env
    volumes:
      - ./backend/uploads:/app/uploads
      - ./data:/data
    depends_on:
      - redis
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
volumes:
  redis-data:
```

---

### 🔹 `.env.example`

```text
OPENAI_API_KEY=sk-REPLACE_ME
MONGODB_URI=mongodb://localhost:27017
MONGO_DB_NAME=nisaa
API_KEY=changeme
VECTOR_DB_PATH=/data/vectorstore
REDIS_URL=redis://redis:6379/0
LOG_LEVEL=INFO
```

---

## ⚡ Quickstart

```bash
git clone <repo-url>
cd nisaa-chatbot
cp .env.example .env
docker-compose up --build
```

---

## 📡 API Reference (Examples)

### Chat with RAG

```bash
curl -X POST http://localhost:8000/api/v1/chat/ask \
  -H "x-api-key: changeme" \
  -H "Content-Type: application/json" \
  -d '{"query": "What services do you provide?"}'
```

### Ingest Website Content

```bash
curl -X POST http://localhost:8000/api/v1/admin/ingest/url \
  -H "x-api-key: changeme" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/services"}'
```

### Start Booking Session

```bash
curl -X POST http://localhost:8000/api/v1/booking/start \
  -H "x-api-key: changeme" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc123"}'
```

### WhatsApp Webhook Simulation

```bash
curl -X POST http://localhost:8000/api/v1/whatsapp/webhook \
  -H "Content-Type: application/json" \
  -d '{"From":"+15551234567","Body":"Hello"}'
```

```
