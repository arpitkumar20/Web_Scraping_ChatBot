# import json
# import os
# import logging
# from dotenv import load_dotenv

# import google.generativeai as genai
# from langchain_google_genai import ChatGoogleGenerativeAI

# # ---- Load Environment Variables ----
# load_dotenv()

# # ---- Configure Logging ----
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s"
# )
# logger = logging.getLogger(__name__)


# def run_namespace_selector(user_query: str) -> str:
#     """
#     Use Google GenAI (Gemini) + LangChain to intelligently detect
#     the most relevant namespace based on user query and website metadata.
#     """
#     logger.info("Starting namespace selection process...")

#     # ---- Load Namespaces (static path) ----
#     namespaces_file = "web_info/web_info.json"
#     with open(namespaces_file, "r") as f:
#         namespaces = json.load(f)  # direct list of dicts
#     logger.info(f"Loaded {len(namespaces)} namespaces.")

#     if not namespaces:
#         raise ValueError("No namespaces found in the JSON file.")

#     # ---- Configure Google GenAI ----
#     GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
#     if not GOOGLE_API_KEY:
#         raise EnvironmentError("GOOGLE_API_KEY not set in environment variables.")

#     genai.configure(api_key=GOOGLE_API_KEY)

#     llm = ChatGoogleGenerativeAI(
#         model=os.getenv("GEMINI_MODEL", "gemini-1.5-pro"),
#         temperature=0,
#     )
#     logger.info("Initialized Google GenAI LLM.")

#     # ---- Build Prompt ----
#     context_blocks = []
#     for ns in namespaces:
#         block = f"""
#         Namespace: {ns.get('namespace')}
#         URL: {ns.get('url')}
#         Title: {ns.get('title')}
#         Description: {ns.get('description')}
#         Keywords: {ns.get('keywords')}
#         Main Heading: {ns.get('main_heading')}
#         """
#         context_blocks.append(block)

#     context = "\n".join(context_blocks)

#     system_prompt = f"""
#     You are an expert AI system that selects the correct namespace based on a user's query. 
#     You are given multiple website metadata entries (url, title, description, keywords, main heading, namespace).
#     Your task is to analyze the user query and match it with the MOST relevant namespace.

#     Rules:
#     - Return ONLY the namespace string.
#     - Do NOT include explanations or extra text.

#     Namespaces dataset:
#     {context}

#     User query: "{user_query}"
#         """

#     logger.info("Sending prompt to GenAI for namespace detection...")

#     response = llm.invoke(system_prompt)
#     selected_namespace = response.content.strip()

#     # ---- Validate fallback ----
#     all_ns = [ns.get("namespace") for ns in namespaces]
#     if selected_namespace not in all_ns:
#         logger.warning(
#             f"GenAI response '{selected_namespace}' not found in namespaces list. "
#             f"Falling back to first namespace."
#         )
#         selected_namespace = all_ns[0]

#     logger.info(f"✅ GenAI selected namespace: {selected_namespace}")
#     return selected_namespace


# if __name__ == "__main__":
#     query = "I want to know about IT company"
#     namespace = run_namespace_selector(query)
#     print("\n👉 Selected Namespace:", namespace)


# import json
# import logging
# from app.core.config import select_namespace_llm
# from app.prompts.namespace_prompt import build_namespace_prompt

# logger = logging.getLogger(__name__)

# def load_namespaces(file_path: str) -> list:
#     """Load namespace metadata from JSON file."""
#     with open(file_path, "r", encoding="utf-8") as f:
#         namespaces = json.load(f)
#     if not namespaces:
#         raise ValueError("No namespaces found in the JSON file.")
#     return namespaces

# def run_namespace_selector(user_query: str, namespaces_file: str = "web_info/web_info.json") -> str:
#     """
#     Detect the most relevant namespace for a given user query.

#     Args:
#         user_query (str): User's input query.
#         namespaces_file (str): Path to JSON file containing namespace metadata.

#     Returns:
#         str: Selected namespace.
#     """
#     logger.info("Starting namespace selection...")

#     # Load namespaces
#     namespaces = load_namespaces(namespaces_file)
#     logger.info(f"Loaded {len(namespaces)} namespaces.")

#     # Build prompt
#     prompt = build_namespace_prompt(user_query, namespaces)

#     # Send prompt to GenAI
#     logger.info("Sending prompt to GenAI for namespace detection...")
#     response = select_namespace_llm.invoke(prompt)
#     selected_namespace = response.content.strip()

#     # Validate response
#     all_ns = [ns.get("namespace") for ns in namespaces]
#     if selected_namespace not in all_ns:
#         logger.warning(
#             f"GenAI response '{selected_namespace}' not found in namespaces list. "
#             f"Falling back to first namespace."
#         )
#         selected_namespace = all_ns[0]

#     logger.info(f"✅ Selected namespace: {selected_namespace}")
#     return selected_namespace

import os
import sys
import json
import logging

# ✅ Ensure project root is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.config import select_namespace_llm
from app.prompts.namespace_prompt import build_namespace_prompt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_namespaces(file_path: str) -> list:
    """Load namespace metadata from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        namespaces = json.load(f)
    if not namespaces:
        raise ValueError("No namespaces found in the JSON file.")
    return namespaces


def run_namespace_selector(user_query: str, namespaces_file: str = "web_info/web_info.json") -> str:
    """
    Detect the most relevant namespace for a given user query.

    Args:
        user_query (str): User's input query.
        namespaces_file (str): Path to JSON file containing namespace metadata.

    Returns:
        str: Selected namespace.
    """
    logger.info("Starting namespace selection...")

    # Load namespaces
    namespaces = load_namespaces(namespaces_file)
    logger.info(f"Loaded {len(namespaces)} namespaces.")

    # Build prompt
    prompt = build_namespace_prompt(user_query, namespaces)

    # Send prompt to GenAI
    logger.info("Sending prompt to GenAI for namespace detection...")
    response = select_namespace_llm.invoke(prompt)
    selected_namespace = response.content.strip()

    # Validate response
    all_ns = [ns.get("namespace") for ns in namespaces]
    if selected_namespace not in all_ns:
        logger.warning(
            f"GenAI response '{selected_namespace}' not found in namespaces list. "
            f"Falling back to first namespace."
        )
        selected_namespace = all_ns[0]

    logger.info(f"✅ Selected namespace: {selected_namespace}")
    return selected_namespace


# if __name__ == "__main__":
#     print(run_namespace_selector("Hello tell me about you"))


# import json
# import os
# import logging
# from dotenv import load_dotenv

# import google.generativeai as genai
# from langchain_google_genai import ChatGoogleGenerativeAI

# # ---- Load Environment Variables ----
# load_dotenv()

# # ---- Configure Logging ----
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s"
# )
# logger = logging.getLogger(__name__)


# def embed_text(text: str):
#     """Generate embeddings using Google GenAI."""
#     embedding_model = "models/embedding-001"
#     response = genai.embed_content(model=embedding_model, content=text)
#     return response["embedding"]


# def cosine_similarity(vec1, vec2):
#     """Compute cosine similarity between two vectors."""
#     import numpy as np
#     v1, v2 = np.array(vec1), np.array(vec2)
#     return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


# def run_namespace_selector(user_query: str, threshold: float = 0.70) -> str:
#     """
#     Smart namespace selector:
#     - Uses embeddings for similarity search
#     - Uses Gemini for reasoning
#     - Falls back to 'general' for irrelevant queries
#     """
#     logger.info("Starting namespace selection process...")

#     # ---- Load Namespaces ----
#     namespaces_file = "web_info/web_info.json"
#     with open(namespaces_file, "r") as f:
#         namespaces = json.load(f)
#     logger.info(f"Loaded {len(namespaces)} namespaces.")

#     if not namespaces:
#         raise ValueError("No namespaces found in the JSON file.")

#     # ---- Configure Google GenAI ----
#     GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
#     if not GOOGLE_API_KEY:
#         raise EnvironmentError("GOOGLE_API_KEY not set in environment variables.")

#     genai.configure(api_key=GOOGLE_API_KEY)

#     llm = ChatGoogleGenerativeAI(
#         model=os.getenv("GEMINI_MODEL", "gemini-1.5-pro"),
#         temperature=0,
#     )
#     logger.info("Initialized Google GenAI LLM.")

#     # ---- Embedding-based Filtering ----
#     user_embedding = embed_text(user_query)

#     scored_namespaces = []
#     for ns in namespaces:
#         meta_text = f"{ns.get('title')} {ns.get('description')} {ns.get('keywords')} {ns.get('main_heading')}"
#         ns_embedding = embed_text(meta_text)
#         score = cosine_similarity(user_embedding, ns_embedding)
#         scored_namespaces.append((ns["namespace"], score, ns))

#     # Sort by score
#     scored_namespaces.sort(key=lambda x: x[1], reverse=True)

#     # Best match
#     best_namespace, best_score, best_meta = scored_namespaces[0]

#     if best_score < threshold:
#         logger.info(f"No strong match found (best score={best_score:.2f}). Returning 'general'.")
#         return "general"

#     # ---- Final Check with Gemini ----
#     context = f"""
# User query: "{user_query}"

# Top candidate namespace:
# Namespace: {best_namespace}
# URL: {best_meta.get('url')}
# Title: {best_meta.get('title')}
# Description: {best_meta.get('description')}
# Keywords: {best_meta.get('keywords')}
# Main Heading: {best_meta.get('main_heading')}

# Task: Decide if this namespace is truly relevant. 
# Answer strictly in JSON:
# {{"namespace": "namespace_string", "confidence": 0.0-1.0}}
# If not relevant, return:
# {{"namespace": "general", "confidence": 0.0}}
#     """

#     response = llm.invoke(context).content.strip()

#     try:
#         result = json.loads(response)
#         final_namespace = result.get("namespace", "general")
#     except Exception:
#         logger.warning("Failed to parse JSON from Gemini. Falling back to best match.")
#         final_namespace = best_namespace

#     logger.info(f"✅ Final selected namespace: {final_namespace}")
#     return final_namespace
