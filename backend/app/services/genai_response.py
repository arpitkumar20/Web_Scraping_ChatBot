import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# === Configuration ===
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL')
LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', 0.7))

# Configure GenAI
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize GenAI model
llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=LLM_TEMPERATURE,
    google_api_key=GOOGLE_API_KEY
)
'''
CORRECT
'''

# SYSTEM_PROMPT = """
# You are a helpful retrieval-augmented chatbot.

# 1. Use ONLY the provided "Retriever results" context. Do NOT use outside knowledge or invent facts.
# 2. Provide a single, concise answer that directly addresses the user’s question.
# 3. Do NOT include "Evidence:", "Need more info:", "Confidence:", or similar sections.
# 4. If the context lacks sufficient information, reply politely asking the user to rephrase or provide more details.
# 5. Keep your answers short, clear, and friendly (≤50 words).
# """

# qa_prompt = ChatPromptTemplate.from_messages([
#     ("system", SYSTEM_PROMPT),
#     ("human", """Retriever results (top K):
# {context}

# User question:
# {query}

# INSTRUCTIONS FOR YOU:
# - Answer concisely using ONLY the Retriever results above.
# - If the context is insufficient, reply with a polite request for clarification or more details.
# - Output only the final answer.
# """)
# ])


SYSTEM_PROMPT = """
You are a retrieval-augmented chatbot.

1. Use ONLY the provided "Retriever results" and the conversation history. Do NOT use outside knowledge or invent facts.
2. Always give a clear, concise, and direct answer to the user’s query.
3. Prefer exact answers over generic explanations.
4. If the context and history lack sufficient information, politely ask the user for clarification.
5. Keep answers short, precise, and user-friendly (≤40 words).
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", """Retriever results (top K):
{context}

User question:
{query}

INSTRUCTIONS FOR YOU:
- Answer directly using ONLY the retriever results.
- If information is missing, ask the user to clarify instead of guessing.
- Output only the final, concise answer.
""")
])

# Core function to handle the query
def handle_user_query(retrieved_context: str, query: str) -> str:
    # Ensure inputs are valid strings
    retrieved_context = retrieved_context or "No relevant context available."
    query = query or "No query provided."

    # Format the prompt
    formatted_prompt = qa_prompt.format(
        context=retrieved_context,
        query=query
    )

    # Generate response
    response = llm.predict(formatted_prompt).strip()

    return response