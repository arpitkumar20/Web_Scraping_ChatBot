import os
import json

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.logger import logger

def run_namespace_selector(user_query: str) -> str:
    """
    Use Google GenAI (Gemini) + LangChain to intelligently detect
    the most relevant namespace based on user query and website metadata.
    """
    logger.info("Starting namespace selection process...")

    # ---- Load Namespaces (static path) ----
    namespaces_file = "web_info/web_info.json"
    with open(namespaces_file, "r") as f:
        namespaces = json.load(f)  # direct list of dicts
    logger.info(f"Loaded {len(namespaces)} namespaces.")

    if not namespaces:
        raise ValueError("No namespaces found in the JSON file.")

    # ---- Configure Google GenAI ----
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise EnvironmentError("GOOGLE_API_KEY not set in environment variables.")

    genai.configure(api_key=GOOGLE_API_KEY)

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL"),
        temperature=0,
    )
    logger.info("Initialized Google GenAI LLM.")

    # ---- Build Prompt ----
    context_blocks = []
    for ns in namespaces:
        block = f"""
        Namespace: {ns.get('namespace')}
        URL: {ns.get('url')}
        Title: {ns.get('title')}
        Description: {ns.get('description')}
        Keywords: {ns.get('keywords')}
        Main Heading: {ns.get('main_heading')}
        """
        context_blocks.append(block)

    context = "\n".join(context_blocks)

    system_prompt = f"""
    You are an expert AI system that selects the correct namespace based on a user's query. 
    You are given multiple website metadata entries (url, title, description, keywords, main heading, namespace).
    Your task is to analyze the user query and match it with the MOST relevant namespace.

    Rules:
    - Return ONLY the namespace string.
    - Do NOT include explanations or extra text.

    Namespaces dataset:
    {context}

    User query: "{user_query}"
        """

    logger.info("Sending prompt to GenAI for namespace detection...")

    response = llm.invoke(system_prompt)
    selected_namespace = response.content.strip()

    # ---- Validate fallback ----
    all_ns = [ns.get("namespace") for ns in namespaces]
    if selected_namespace not in all_ns:
        logger.warning(
            f"GenAI response '{selected_namespace}' not found in namespaces list. "
            f"Falling back to first namespace."
        )
        selected_namespace = all_ns[0]

    logger.info(f"✅ GenAI selected namespace: {selected_namespace}")
    return selected_namespace


