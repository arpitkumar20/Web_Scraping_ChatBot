import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings


from app.services.pg_embed_logic import embed_postgres_like_json

# ---------- Logging Setup ----------
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"pg_embed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("pg_embed_runner")

# ---------- Load env ----------
BACKEND_DIR = os.path.dirname(__file__)
DOTENV_PATH = os.path.join(BACKEND_DIR, ".env")
load_dotenv(dotenv_path=DOTENV_PATH)

# ---------- Env variables ----------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "nisaa-knowledge")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")  # <-- your .env setting

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set. Please add it to your environment or .env file.")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is not set. Please add it to your environment or .env file.")

# ---------- OpenAI Embeddings Wrapper ----------
class OpenAIEmbeddings:
    """
    Minimal wrapper to match the expected .embed_documents(texts) -> List[List[float]]
    """
    def __init__(self, model_name: str):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = model_name

    def embed_documents(self, texts):
        # The OpenAI embeddings API accepts batching in a single call.
        # If you need micro-batching, you can slice texts here.
        resp = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        # Ensure order alignment with inputs
        return [item.embedding for item in resp.data]

    # ---------- Main ----------
    def postgres_main(payload: str, namespace: str):
        logger.info("🚀 Starting PostgreSQL embedding pipeline.")

        embeddings = OpenAIEmbeddings(model_name=EMBEDDING_MODEL)
        pc_client = Pinecone(api_key=PINECONE_API_KEY)

        embed_postgres_like_json(
            payload=payload,
            embeddings=embeddings,
            pc_client=pc_client,
            namespace=namespace,
            chunk_size=1000,
            chunk_overlap=120,
            embed_batch=100,
            embed_dim=1536,
            index_name=PINECONE_INDEX,
            env_region=PINECONE_ENV,
        )

        logger.info("✅ Embedding process completed successfully.")