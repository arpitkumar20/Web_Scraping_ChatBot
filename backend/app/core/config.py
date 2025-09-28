import os
from venv import logger
from pinecone import Pinecone
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL')
LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', 0.7))

# ---- Load Environment Variables ----
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
PINECONE_INDEX = os.getenv('PINECONE_INDEX')
# NAMESPACE = os.getenv('NAMESPACE')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')

if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY not set in environment variables.")

# Configure Google GenAI
genai.configure(api_key=GOOGLE_API_KEY)

# # Initialize LLM
# llm = ChatGoogleGenerativeAI(
#     model=GEMINI_MODEL,
#     temperature=LLM_TEMPERATURE,
#     google_api_key=GOOGLE_API_KEY
# )

try:
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=LLM_TEMPERATURE,
        google_api_key=GOOGLE_API_KEY
    )
except Exception as e:
    logger.error(f"Failed to initialize LLM with model {GEMINI_MODEL}: {e}")

# ---- Configure Pinecone Client ----
if not all([PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX]):
    raise EnvironmentError("Pinecone configuration variables missing.")

pinecone_client = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
pinecone_index = pinecone_client.Index(PINECONE_INDEX)

try:
    select_namespace_llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL"),  # ✅ valid public model
        temperature=0,
    )
except Exception as e:
    logger.error(f"Failed to select namespace using LLM with model {GEMINI_MODEL}: {e}")
