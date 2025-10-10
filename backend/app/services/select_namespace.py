
# import os
# import sys
# import json
# import logging

# # ✅ Ensure project root is on sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from app.core.config import select_namespace_llm
# from app.prompts.namespace_prompt import build_namespace_prompt

# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)


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
import json

def run_namespace_selector(namespaces_file: str = "web_info/web_info.json") -> str:
    """
    Read the namespace from a JSON file containing a single dictionary
    and return its 'namespace' value. Checks if file exists.
    """
    if not os.path.exists(namespaces_file):
        raise FileNotFoundError(f"JSON file not found: {namespaces_file}")
    
    with open(namespaces_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data.get("namespace")
