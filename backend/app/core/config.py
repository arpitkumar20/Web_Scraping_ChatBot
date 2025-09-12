# from pydantic import BaseModel
# import os

# class Settings(BaseModel):
#     SECRET_KEY: str = os.getenv("SECRET_KEY", "dev")
#     HOST: str = os.getenv("HOST", "0.0.0.0")
#     PORT: int = int(os.getenv("PORT", "8080"))

#     TWILIO_AUTH_TOKEN: str | None = os.getenv("TWILIO_AUTH_TOKEN")
#     TWILIO_ACCOUNT_SID: str | None = os.getenv("TWILIO_ACCOUNT_SID")
#     WHATSAPP_FROM: str = os.getenv("WHATSAPP_FROM", "whatsapp:+14155238886")

#     VECTOR_BACKEND: str = os.getenv("VECTOR_BACKEND", "faiss")
#     VECTOR_INDEX_PATH: str = os.getenv("VECTOR_INDEX_PATH", "/data/faiss/nisaa.index")

#     EMBEDDINGS_PROVIDER: str = os.getenv("EMBEDDINGS_PROVIDER", "sentence")
#     SENTENCE_MODEL: str = os.getenv("SENTENCE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
#     OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

#     CRAWL_OUTPUT_DIR: str = os.getenv("CRAWL_OUTPUT_DIR", "/data/crawls")

#     POSTGRES_DSN: str = os.getenv("POSTGRES_DSN", "postgresql://user:pass@db:5432/nisaa")

#     TOP_K: int = int(os.getenv("TOP_K", "6"))
#     MAX_CHUNK_TOKENS: int = int(os.getenv("MAX_CHUNK_TOKENS", "500"))

#     DEFAULT_TOUR_URL: str = os.getenv("DEFAULT_TOUR_URL", "https://example.com/tour")

# settings = Settings()
