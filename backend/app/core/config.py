# import os
# from pinecone import Pinecone
# from dotenv import load_dotenv
# import google.generativeai as genai
# from app.core.logging import get_logger
# from langchain_google_genai import ChatGoogleGenerativeAI

# # Load environment variables
# load_dotenv()
# logger = get_logger(__name__)

# GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
# GEMINI_MODEL = os.getenv('GEMINI_MODEL')
# LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', 0.7))

# # ---- Load Environment Variables ----
# PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
# PINECONE_ENV = os.getenv('PINECONE_ENV')
# PINECONE_INDEX = os.getenv('PINECONE_INDEX')
# # NAMESPACE = os.getenv('NAMESPACE')
# EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')

# if not GOOGLE_API_KEY:
#     raise EnvironmentError("GOOGLE_API_KEY not set in environment variables.")

# # Configure Google GenAI
# genai.configure(api_key=GOOGLE_API_KEY)

# # # Initialize LLM
# # llm = ChatGoogleGenerativeAI(
# #     model=GEMINI_MODEL,
# #     temperature=LLM_TEMPERATURE,
# #     google_api_key=GOOGLE_API_KEY
# # )

# try:
#     llm = ChatGoogleGenerativeAI(
#         model=GEMINI_MODEL,
#         temperature=LLM_TEMPERATURE,
#         google_api_key=GOOGLE_API_KEY
#     )
# except Exception as e:
#     logger.error(f"Failed to initialize LLM with model {GEMINI_MODEL}: {e}")

# # ---- Configure Pinecone Client ----
# if not all([PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX]):
#     raise EnvironmentError("Pinecone configuration variables missing.")

# pinecone_client = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
# pinecone_index = pinecone_client.Index(PINECONE_INDEX)

# try:
#     select_namespace_llm = ChatGoogleGenerativeAI(
#         model=os.getenv("GEMINI_MODEL"),  # ✅ valid public model
#         temperature=0,
#     )
# except Exception as e:
#     logger.error(f"Failed to select namespace using LLM with model {GEMINI_MODEL}: {e}")



import os
from pinecone import Pinecone
from dotenv import load_dotenv
from app.core.logging import get_logger
from langchain_openai import ChatOpenAI

# ---------------- Load Environment Variables ----------------
load_dotenv()
logger = get_logger(__name__)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')  # ✅ default model
LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', 0.7))

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
PINECONE_INDEX = os.getenv('PINECONE_INDEX')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')

# ---------------- Validate Keys ----------------
if not OPENAI_API_KEY:
    raise EnvironmentError("❌ OPENAI_API_KEY not set in environment variables.")
if not all([PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX]):
    raise EnvironmentError("❌ Missing Pinecone configuration variables.")

# ---------------- Initialize OpenAI LLM ----------------
try:
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=LLM_TEMPERATURE,
        openai_api_key=OPENAI_API_KEY
    )
    logger.info(f"✅ Initialized OpenAI LLM model '{OPENAI_MODEL}' successfully.")
except Exception as e:
    logger.error(f"❌ Failed to initialize OpenAI LLM with model '{OPENAI_MODEL}': {e}")
    llm = None

# ---------------- Initialize Pinecone ----------------
try:
    pinecone_client = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    pinecone_index = pinecone_client.Index(PINECONE_INDEX)
    logger.info(f"✅ Connected to Pinecone index '{PINECONE_INDEX}'.")
except Exception as e:
    logger.error(f"❌ Failed to connect to Pinecone: {e}")
    pinecone_client = None
    pinecone_index = None

# # ---------------- Optional: LLM for Namespace Selection ----------------
# try:
#     select_namespace_llm = ChatOpenAI(
#         model=OPENAI_MODEL,
#         temperature=0,
#         openai_api_key=OPENAI_API_KEY
#     )
#     logger.info(f"✅ Namespace selection LLM initialized with model '{OPENAI_MODEL}'.")
# except Exception as e:
#     logger.error(f"❌ Failed to initialize namespace selection LLM: {e}")
#     select_namespace_llm = None
