# # # # import os
# # # # import json
# # # # from langchain_google_genai import ChatGoogleGenerativeAI
# # # # from langchain.prompts import ChatPromptTemplate
# # # # import google.generativeai as genai
# # # # from dotenv import load_dotenv

# # # # # Load environment variables
# # # # load_dotenv()

# # # # # === Configuration ===
# # # # GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
# # # # GEMINI_MODEL = os.getenv('GEMINI_MODEL')
# # # # LLM_TEMPERATURE = float(os.getenv('LLM_TEMPERATURE', 0.7))

# # # # # Configure GenAI
# # # # genai.configure(api_key=GOOGLE_API_KEY)

# # # # # Initialize GenAI model
# # # # llm = ChatGoogleGenerativeAI(
# # # #     model=GEMINI_MODEL,
# # # #     temperature=LLM_TEMPERATURE,
# # # #     google_api_key=GOOGLE_API_KEY
# # # # )

# # # # '''
# # # # CORRECT

# # # # '''


# # # # # SYSTEM_PROMPT = """
# # # # # You are a helpful retrieval-augmented chatbot.

# # # # # 1. Use ONLY the provided "Retriever results" context. Do NOT use outside knowledge or invent facts.
# # # # # 2. Provide a single, concise answer that directly addresses the user’s question.
# # # # # 3. Do NOT include "Evidence:", "Need more info:", "Confidence:", or similar sections.
# # # # # 4. If the context lacks sufficient information, reply politely asking the user to rephrase or provide more details.
# # # # # 5. Keep your answers short, clear, and friendly (≤50 words).
# # # # # """

# # # # # qa_prompt = ChatPromptTemplate.from_messages([
# # # # #     ("system", SYSTEM_PROMPT),
# # # # #     ("human", """Retriever results (top K):
# # # # # {context}

# # # # # User question:
# # # # # {query}

# # # # # INSTRUCTIONS FOR YOU:
# # # # # - Answer concisely using ONLY the Retriever results above.
# # # # # - If the context is insufficient, reply with a polite request for clarification or more details.
# # # # # - Output only the final answer.
# # # # # """)
# # # # # ])


# # # # SYSTEM_PROMPT = """
# # # # You are a retrieval-augmented chatbot.

# # # # 1. Use ONLY the provided "Retriever results" and the conversation history. Do NOT use outside knowledge or invent facts.
# # # # 2. Always give a clear, concise, and direct answer to the user’s query.
# # # # 3. Prefer exact answers over generic explanations.
# # # # 4. If the context and history lack sufficient information, politely ask the user for clarification.
# # # # 5. Keep answers short, precise, and user-friendly (≤40 words).
# # # # """

# # # # qa_prompt = ChatPromptTemplate.from_messages([
# # # #     ("system", SYSTEM_PROMPT),
# # # #     ("human", """Retriever results (top K):
# # # # {context}

# # # # User question:
# # # # {query}

# # # # INSTRUCTIONS FOR YOU:
# # # # - Answer directly using ONLY the retriever results.
# # # # - If information is missing, ask the user to clarify instead of guessing.
# # # # - Output only the final, concise answer.
# # # # """)
# # # # ])



# # # # # Core function to handle the query
# # # # def handle_user_query(retrieved_context: str, query: str) -> str:
# # # #     # Ensure inputs are valid strings
# # # #     retrieved_context = retrieved_context or "No relevant context available."
# # # #     query = query or "No query provided."

# # # #     # Format the prompt
# # # #     formatted_prompt = qa_prompt.format(
# # # #         context=retrieved_context,
# # # #         query=query
# # # #     )

# # # #     # Generate response
# # # #     response = llm.predict(formatted_prompt).strip()

# # # #     return response


# # # # # === Example Usage ===
# # # # # pinecone_context = "Previous queries show available appointment slots on Friday from 10 AM to 3 PM."
# # # # # user_query = "Can I book an appointment on Friday afternoon?"

# # # # # response = handle_user_query(
# # # # #     retrieved_context=pinecone_context,
# # # # #     query=user_query
# # # # # )

# # # # # print("\n🌟 GenAI Response:\n", response)



# # # # from app.prompts.qa_prompt import qa_prompt
# # # # from app.core.config import llm

# # # # def handle_user_query(retrieved_context: str, query: str) -> str:
# # # #     """
# # # #     Generate a concise response using only the provided retriever context.

# # # #     Args:
# # # #         retrieved_context (str): The context retrieved from a vector DB or knowledge source.
# # # #         query (str): The user’s question.

# # # #     Returns:
# # # #         str: LLM-generated response.
# # # #     """
# # # #     # Ensure inputs are valid strings
# # # #     retrieved_context = retrieved_context or "No relevant context available."
# # # #     query = query or "No query provided."

# # # #     # Format the prompt
# # # #     formatted_prompt = qa_prompt.format(
# # # #         context=retrieved_context,
# # # #         query=query
# # # #     )

# # # #     # Generate response
# # # #     response = llm.predict(formatted_prompt).strip()
# # # #     return response



# # # # from app.prompts.qa_prompt import qa_prompt
# # # # from app.core.config import llm

# # # # # In-memory chat history: key=user_id, value=list of {"query":..., "response":...}
# # # # chat_history = {}

# # # # def handle_user_query(user_id: str, retrieved_context: str, query: str) -> str:
# # # #     """
# # # #     Generate a concise response using provided context and user's chat history.

# # # #     Args:
# # # #         user_id (str): Unique identifier for the user.
# # # #         retrieved_context (str): Context from vector DB or knowledge source.
# # # #         query (str): The user’s current question.

# # # #     Returns:
# # # #         str: LLM-generated response.
# # # #     """
# # # #     # Initialize user history if not present
# # # #     if user_id not in chat_history:
# # # #         chat_history[user_id] = []

# # # #     history = chat_history[user_id]  # guaranteed to be a list

# # # #     # Build history string for prompt (last 5 exchanges)
# # # #     history_text = ""
# # # #     if history:
# # # #         history_text = "\nPrevious conversation:\n"
# # # #         for i, turn in enumerate(history[-5:]):
# # # #             history_text += f"Q{i+1}: {turn['query']}\nA{i+1}: {turn['response']}\n"

# # # #     # Ensure retrieved context and query are valid strings
# # # #     retrieved_context = str(retrieved_context or "No relevant context available.")
# # # #     query = str(query or "No query provided.")

# # # #     # Format the prompt including context and history
# # # #     formatted_prompt = qa_prompt.format(
# # # #         context=retrieved_context + history_text,
# # # #         query=query
# # # #     )
# # # #     print("...............formatted_prompt.............",formatted_prompt)
# # # #     # Generate response safely
# # # #     response_raw = llm.predict(formatted_prompt)

# # # #     # Handle cases where LLM returns list or string
# # # #     if isinstance(response_raw, list):
# # # #         response = " ".join([str(r) for r in response_raw]).strip()
# # # #     else:
# # # #         response = str(response_raw).strip()

# # # #     # Append the current turn to user's history
# # # #     history.append({"query": query, "response": response})

# # # #     return response


# # # # import re
# # # # import hashlib
# # # # import json
# # # # from typing import List, Dict, Any, Optional
# # # # import redis
# # # # import logging

# # # # from app.prompts.qa_prompt import qa_prompt
# # # # from app.core.config import llm

# # # # # -------------------------------
# # # # # Logging configuration
# # # # # -------------------------------
# # # # logging.basicConfig(
# # # #     level=logging.INFO,
# # # #     format="%(asctime)s [%(levelname)s] %(message)s"
# # # # )
# # # # logger = logging.getLogger(__name__)

# # # # # -------------------------------
# # # # # Redis connection
# # # # # -------------------------------
# # # # redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# # # # # -------------------------------
# # # # # Patterns
# # # # # -------------------------------
# # # # DOCTOR_PATTERN = r"Dr\.?\s+[A-Z][a-zA-Z]+"
# # # # TIME_PATTERN = r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b"

# # # # # -------------------------------
# # # # # Embedding adapter
# # # # # -------------------------------
# # # # class EmbeddingModel:
# # # #     """Pseudo embedding; replace with real LLM embedding in production"""
# # # #     def embed_text(self, text: str) -> List[float]:
# # # #         h = hashlib.sha256(text.encode("utf-8")).hexdigest()
# # # #         return [(int(h[i:i+2], 16)/255.0) for i in range(0, 16, 2)]

# # # # embedding_model = EmbeddingModel()

# # # # def _safe_embed(text: str) -> List[float]:
# # # #     try:
# # # #         return embedding_model.embed_text(text or "")
# # # #     except Exception:
# # # #         logger.error(f"Embedding failed for text: {text}")
# # # #         return embedding_model.embed_text(text or "")

# # # # # -------------------------------
# # # # # Utilities
# # # # # -------------------------------
# # # # def _extract_doctor_and_time(text: str) -> (Optional[str]):
# # # #     doctor_match = re.search(DOCTOR_PATTERN, text or "")
# # # #     time_match = re.search(TIME_PATTERN, text or "")
# # # #     doctor = doctor_match.group().strip() if doctor_match else None
# # # #     time = time_match.group().strip() if time_match else None
# # # #     return doctor, time

# # # # def _parse_doctor_availability(retrieved_context: str) -> Dict[str, List[str]]:
# # # #     availability: Dict[str, List[str]] = {}
# # # #     if not retrieved_context:
# # # #         return availability
# # # #     sentences = re.split(r'[.\n]+', retrieved_context)
# # # #     for s in sentences:
# # # #         doc_match = re.search(DOCTOR_PATTERN, s)
# # # #         if not doc_match:
# # # #             continue
# # # #         doc = doc_match.group().strip()
# # # #         slots = re.findall(TIME_PATTERN, s)
# # # #         if slots:
# # # #             availability.setdefault(doc, []).extend(sorted(set(slots)))
# # # #     return availability

# # # # # -------------------------------
# # # # # Chat History with Redis
# # # # # -------------------------------
# # # # def save_chat_history(user_id: str, query: str, response: str, q_emb: List[float], r_emb: List[float]):
# # # #     entry = {
# # # #         "query": query,
# # # #         "response": response,
# # # #         "query_embedding": q_emb,
# # # #         "response_embedding": r_emb
# # # #     }
# # # #     redis_client.rpush(f"chat:{user_id}", json.dumps(entry))
# # # #     logger.info(f"Saved chat history for user {user_id}: query='{query}' response='{response}'")

# # # # def get_chat_history(user_id: str, last_n: int = 5) -> List[Dict[str, Any]]:
# # # #     entries = redis_client.lrange(f"chat:{user_id}", -last_n, -1)
# # # #     return [json.loads(e) for e in entries]

# # # # # -------------------------------
# # # # # Booking System with Redis
# # # # # -------------------------------
# # # # def book_doctor(user_id: str, doctor: str, slot: str) -> str:
# # # #     logger.info(f"Attempting booking: user={user_id}, doctor={doctor}, slot={slot}")

# # # #     if redis_client.sismember(f"user_bookings:{user_id}", doctor):
# # # #         logger.warning(f"User {user_id} already has a booking with {doctor}")
# # # #         return f"❌ You already have a booking with {doctor}."

# # # #     if redis_client.sismember(f"doctor_slots:{doctor}", slot):
# # # #         logger.warning(f"Slot {slot} for {doctor} already booked")
# # # #         return f"❌ Sorry, {doctor} at {slot} is already booked."

# # # #     redis_client.sadd(f"doctor_slots:{doctor}", slot)
# # # #     redis_client.sadd(f"user_bookings:{user_id}", doctor)
# # # #     logger.info(f"Booking confirmed: user={user_id}, doctor={doctor}, slot={slot}")
# # # #     return f"✅ Your booking with {doctor} at {slot} is confirmed."

# # # # def get_user_bookings(user_id: str) -> List[str]:
# # # #     return list(redis_client.smembers(f"user_bookings:{user_id}"))

# # # # def get_doctor_booked_slots(doctor: str) -> List[str]:
# # # #     return list(redis_client.smembers(f"doctor_slots:{doctor}"))

# # # # # -------------------------------
# # # # # Booking Logic
# # # # # -------------------------------
# # # # def _process_dynamic_booking(user_id: str, query: str, response: str, retrieved_context: str) -> str:
# # # #     doctor, slot = _extract_doctor_and_time(query)
# # # #     if not doctor or not slot:
# # # #         doc2, slot2 = _extract_doctor_and_time(response)
# # # #         doctor = doctor or doc2
# # # #         slot = slot or slot2
# # # #     if not doctor or not slot:
# # # #         return response

# # # #     availability = _parse_doctor_availability(retrieved_context)
# # # #     matched_doctor = next((d for d in availability if d.lower() == doctor.lower()), None)
# # # #     if not matched_doctor:
# # # #         logger.info(f"Doctor {doctor} not available in retrieved context")
# # # #         return f"Sorry, {doctor} is not available."

# # # #     if slot not in availability.get(matched_doctor, []):
# # # #         possible_slots = availability.get(matched_doctor, [])
# # # #         if possible_slots:
# # # #             logger.info(f"Requested slot {slot} not available for {matched_doctor}")
# # # #             return f"Sorry, {matched_doctor} is not available at {slot}. Available slots: {', '.join(possible_slots)}."
# # # #         return f"Sorry, {matched_doctor} has no available slots."

# # # #     return book_doctor(user_id, matched_doctor, slot)

# # # # # -------------------------------
# # # # # Main Query Handler
# # # # # -------------------------------
# # # # def handle_user_query(user_id: str, retrieved_context: str, query: str) -> str:
# # # #     if not user_id or not query:
# # # #         logger.warning("User ID or query missing")
# # # #         return "Please provide a valid user ID and query."
    
# # # #     if not isinstance(retrieved_context, str):
# # # #         import json
# # # #         retrieved_context = json.dumps(retrieved_context)

# # # #     history = get_chat_history(user_id)
# # # #     history_text = ""
# # # #     if history:
# # # #         recent = history[-5:]
# # # #         history_text = "\nRecent conversation:\n" + "\n".join(
# # # #             f"User: {turn['query']}\nBot: {turn['response']}" for turn in recent
# # # #         )
# # # #         logger.info(f"Recent history for {user_id}: {history_text}")

# # # #     final_prompt = qa_prompt.format(
# # # #         context=f"Relevant medical context:\n{retrieved_context}\n{history_text}",
# # # #         query=query
# # # #     )
# # # #     logger.info(f"Final prompt for LLM: ============================ {final_prompt}")

# # # #     raw_response = llm.predict(final_prompt)
# # # #     response_text = " ".join(raw_response) if isinstance(raw_response, (list, tuple)) else str(raw_response)
# # # #     response_text = response_text.strip()
# # # #     logger.info(f"LLM response: {response_text}")

# # # #     updated_response = _process_dynamic_booking(user_id, query, response_text, retrieved_context)

# # # #     q_emb = _safe_embed(query)
# # # #     r_emb = _safe_embed(updated_response)
# # # #     save_chat_history(user_id, query, updated_response, q_emb, r_emb)

# # # #     return updated_response

# # # # # -------------------------------
# # # # # Helper functions
# # # # # -------------------------------
# # # # def list_available_slots_for_doctor(doctor_name: str, retrieved_context: str) -> List[str]:
# # # #     availability = _parse_doctor_availability(retrieved_context)
# # # #     return availability.get(doctor_name, [])

# # # # def get_user_history(user_id: str, include_embeddings: bool = False) -> List[Dict[str, Any]]:
# # # #     hist = get_chat_history(user_id, last_n=100)
# # # #     if not include_embeddings:
# # # #         return [{"query": e["query"], "response": e["response"]} for e in hist]
# # # #     return hist

# # # # def get_booking_summary() -> Dict[str, List[str]]:
# # # #     keys = redis_client.keys("doctor_slots:*")
# # # #     summary = {}
# # # #     for key in keys:
# # # #         doctor = key.split(":")[1]
# # # #         summary[doctor] = list(redis_client.smembers(key))
# # # #     return summary

# # # # -------------------------------
# # # # # Example Usage (for testing)
# # # # # -------------------------------
# # # # if __name__ == "__main__":
# # # #     sample_context = "Dr. Smith available at 09:00, 10:00, 11:00.\nDr. Priya available at 10:00, 11:00."
# # # #     print(handle_user_query("user1", sample_context, "I want to book with Dr. Smith at 10:00"))
# # # #     print(handle_user_query("user1", sample_context, "Book Dr. Smith at 11:00"))
# # # #     print(handle_user_query("user2", sample_context, "Book Dr. Smith at 10:00"))
# # # #     print(list_available_slots_for_doctor("Dr. Priya", sample_context))
# # # #     print(get_user_history("user1"))
# # # #     print(get_booking_summary())












# # # import os
# # # import re
# # # import hashlib
# # # import json
# # # import math
# # # import logging
# # # from typing import List, Dict, Any, Optional, Tuple

# # # import redis
# # # from langchain.prompts import ChatPromptTemplate

# # # from app.core.config import llm  # your LLM interface (must implement .predict(prompt) or .predict/ .generate as used)

# # # # ==============================
# # # # Logging Configuration
# # # # ==============================
# # # logging.basicConfig(
# # #     level=logging.INFO,
# # #     format="%(asctime)s [%(levelname)s] %(message)s"
# # # )
# # # logger = logging.getLogger(__name__)

# # # # ==============================
# # # # Redis Connection
# # # # ==============================

# # # from dotenv import load_dotenv

# # # load_dotenv()

# # # REDIS_HOST = os.getenv("REDIS_HOST")
# # # REDIS_PORT = os.getenv("REDIS_PORT")
# # # REDIS_USER = os.getenv("REDIS_USER")
# # # REDIS_PASS = os.getenv("REDIS_PASS")

# # # redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# # # # Create Redis client
# # # # redis_client = redis.Redis(
# # # #     host=REDIS_HOST,
# # # #     port=REDIS_PORT,
# # # #     username=REDIS_USER,
# # # #     password=REDIS_PASS,
# # # #     db=0,
# # # #     decode_responses=True
# # # # )

# # # try:
# # #     response = redis_client.ping()
# # #     if response:
# # #         print("✅ Redis connection successful!")
# # #     else:
# # #         print("❌ Redis connection failed.")
# # # except redis.exceptions.AuthenticationError:
# # #     print("❌ Authentication failed. Check username/password.")
# # # except redis.exceptions.ConnectionError:
# # #     print("❌ Cannot connect to Redis server. Check host/port.")


# # # # redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# # # # ==============================
# # # # Regex Patterns
# # # # ==============================
# # # DOCTOR_PATTERN = r"Dr\.?\s+[A-Z][a-zA-Z]+"
# # # TIME_PATTERN = r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b"

# # # # ==============================
# # # # Prompt Template (updated)
# # # # ==============================
# # # SYSTEM_PROMPT = """
# # # You are a retrieval-augmented medical assistant chatbot.

# # # Your responsibilities:
# # # 1. Use ONLY the retriever context, chat history, and the user’s latest query.
# # # 2. Avoid repeating responses already given in prior conversation history; if you must repeat some necessary info, rephrase it.
# # # 3. If a booking was already confirmed for the same doctor, acknowledge that instead of reconfirming.
# # # 4. Be concise, clear, and human-like (≤40 words).
# # # 5. If doctor, time, or information is missing, ask for clarification.
# # # 6. Never fabricate data. If unsure, politely say so.
# # # """

# # # qa_prompt = ChatPromptTemplate.from_messages([
# # #     ("system", SYSTEM_PROMPT),
# # #     ("human", """Retriever context:
# # # {context}

# # # Conversation history (most recent first — includes the user's current message):
# # # {history}

# # # User query:
# # # {query}

# # # INSTRUCTIONS:
# # # - The 'Conversation history' includes previous user queries and bot responses (most recent first).
# # # - Do NOT repeat earlier bot responses verbatim; rephrase or provide new clarifying info when possible.
# # # - If booking was already confirmed with the doctor, say so directly.
# # # - If slot unavailable, provide available slots from the context.
# # # - If clarification is needed, ask a short direct question.
# # # """)
# # # ])


# # # # ==============================
# # # # Simple deterministic pseudo-embedding (replace with real embeddings in production)
# # # # ==============================
# # # class EmbeddingModel:
# # #     """Simple pseudo-embedding (replace with real LLM embedding in production)."""
# # #     def embed_text(self, text: str) -> List[float]:
# # #         # create deterministic vector from SHA256 -> list of floats
# # #         h = hashlib.sha256((text or "").encode("utf-8")).hexdigest()
# # #         return [(int(h[i:i+2], 16) / 255.0) for i in range(0, 32, 2)]  # 16-d floats

# # # embedding_model = EmbeddingModel()

# # # def _safe_embed(text: str) -> List[float]:
# # #     try:
# # #         return embedding_model.embed_text(text or "")
# # #     except Exception:
# # #         logger.exception("Embedding failed, returning zeros")
# # #         return [0.0] * 16

# # # def _cosine_sim(a: List[float], b: List[float]) -> float:
# # #     # compute cosine similarity
# # #     if not a or not b:
# # #         return 0.0
# # #     dot = sum(x * y for x, y in zip(a, b))
# # #     na = math.sqrt(sum(x * x for x in a))
# # #     nb = math.sqrt(sum(y * y for y in b))
# # #     if na == 0 or nb == 0:
# # #         return 0.0
# # #     return dot / (na * nb)

# # # # ==============================
# # # # Utility Functions
# # # # ==============================
# # # def _extract_doctor_and_time(text: str) -> Tuple[Optional[str], Optional[str]]:
# # #     doctor_match = re.search(DOCTOR_PATTERN, text or "")
# # #     time_match = re.search(TIME_PATTERN, text or "")
# # #     doctor = doctor_match.group().strip() if doctor_match else None
# # #     time = time_match.group().strip() if time_match else None
# # #     return doctor, time

# # # def _parse_doctor_availability(retrieved_context: str) -> Dict[str, List[str]]:
# # #     availability: Dict[str, List[str]] = {}
# # #     if not retrieved_context:
# # #         return availability
# # #     sentences = re.split(r'[.\n]+', retrieved_context)
# # #     for s in sentences:
# # #         doc_match = re.search(DOCTOR_PATTERN, s)
# # #         if not doc_match:
# # #             continue
# # #         doc = doc_match.group().strip()
# # #         slots = re.findall(TIME_PATTERN, s)
# # #         if slots:
# # #             availability.setdefault(doc, []).extend(sorted(set(slots)))
# # #     return availability

# # # # ==============================
# # # # Redis Chat History Management (improved)
# # # # ==============================
# # # # Each user's history list key: chat:{user_id} stores JSON entries:
# # # # {"query": str, "response": Optional[str], "query_embedding": [...], "response_embedding": [...], "ts": <optional>}
# # # def _chat_key(user_id: str) -> str:
# # #     return f"chat:{user_id}"

# # # def push_pending_query(user_id: str, query: str, q_emb: List[float]) -> int:
# # #     """
# # #     Push a pending query into user's chat list with response=null.
# # #     Returns the absolute index (0-based) of the pushed element so it can be updated later.
# # #     """
# # #     key = _chat_key(user_id)
# # #     entry = {
# # #         "query": query,
# # #         "response": None,
# # #     }
# # #     # rpush then compute index
# # #     pipe = redis_client.pipeline()
# # #     pipe.rpush(key, json.dumps(entry))
# # #     pipe.llen(key)
# # #     _, length = pipe.execute()
# # #     index = length - 1
# # #     logger.debug(f"Pushed pending query for user {user_id} at index {index}")
# # #     return index

# # # def update_history_response(user_id: str, index: int, response: str, r_emb: List[float]) -> None:
# # #     """
# # #     Update the list element at absolute index with the response and response_embedding.
# # #     """
# # #     key = _chat_key(user_id)
# # #     entries = redis_client.lrange(key, index, index)
# # #     if not entries:
# # #         logger.error(f"No entry found at index {index} for user {user_id}")
# # #         return
# # #     entry = json.loads(entries[0])
# # #     entry["response"] = response
# # #     entry["response_embedding"] = r_emb
# # #     redis_client.lset(key, index, json.dumps(entry))
# # #     logger.info(f"Updated chat history for user {user_id} at index {index}")

# # # def get_chat_history(user_id: str, last_n: int = 10) -> List[Dict[str, Any]]:
# # #     entries = redis_client.lrange(_chat_key(user_id), -last_n, -1)
# # #     return [json.loads(e) for e in entries if e]

# # # def clear_user_history(user_id: str) -> None:
# # #     redis_client.delete(_chat_key(user_id))

# # # # ==============================
# # # # Booking Management (unchanged semantics, but optimized)
# # # # ==============================
# # # def book_doctor(user_id: str, doctor: str, slot: str) -> str:
# # #     logger.info(f"Attempting booking: user={user_id}, doctor={doctor}, slot={slot}")
# # #     pipe = redis_client.pipeline()

# # #     # If user already has booking with doctor -> acknowledge
# # #     if redis_client.sismember(f"user_bookings:{user_id}", doctor):
# # #         return f"✅ You already have a confirmed booking with {doctor}."

# # #     # if doctor_slots already contains the slot -> slot busy
# # #     if redis_client.sismember(f"doctor_slots:{doctor}", slot):
# # #         return f"❌ Sorry, {doctor} at {slot} is already booked."

# # #     # If any user_bookings:* contains this doctor (doctor is fully booked in your model) -> deny
# # #     all_users = redis_client.keys("user_bookings:*")
# # #     for key in all_users:
# # #         if redis_client.sismember(key, doctor):
# # #             return f"❌ Sorry, {doctor} is already booked by another user."

# # #     # Confirm booking: add slot to doctor_slots and add doctor to user_bookings
# # #     pipe.sadd(f"doctor_slots:{doctor}", slot)
# # #     pipe.sadd(f"user_bookings:{user_id}", doctor)
# # #     pipe.execute()
# # #     return f"✅ Your booking with {doctor} at {slot} is confirmed."

# # # def get_user_bookings(user_id: str) -> List[str]:
# # #     return list(redis_client.smembers(f"user_bookings:{user_id}"))

# # # def get_doctor_booked_slots(doctor: str) -> List[str]:
# # #     return list(redis_client.smembers(f"doctor_slots:{doctor}"))

# # # # ==============================
# # # # Repetition detection & safe re-query
# # # # ==============================
# # # REPEAT_SIM_THRESHOLD = 0.92  # high threshold for "too similar"

# # # def _is_repetition(user_id: str, candidate_response: str) -> Tuple[bool, Optional[str]]:
# # #     """
# # #     Returns (is_repetition, matched_previous_response_text_or_None)
# # #     Use cosine similarity between candidate_response embedding and previous response embeddings.
# # #     """
# # #     cand_emb = _safe_embed(candidate_response)
# # #     history = get_chat_history(user_id, last_n=50)
# # #     best_sim = -1.0
# # #     best_resp = None
# # #     for turn in history:
# # #         prev_resp = turn.get("response") or ""
# # #         prev_emb = turn.get("response_embedding")
# # #         if not prev_resp or not prev_emb:
# # #             continue
# # #         sim = _cosine_sim(cand_emb, prev_emb)
# # #         if sim > best_sim:
# # #             best_sim = sim
# # #             best_resp = prev_resp
# # #     logger.debug(f"Repetition check best_sim={best_sim} for user {user_id}")
# # #     return (best_sim >= REPEAT_SIM_THRESHOLD, best_resp if best_sim >= REPEAT_SIM_THRESHOLD else None)

# # # # ==============================
# # # # Dynamic Booking Logic (uses availability parsed from retrieved_context)
# # # # ==============================
# # # def _process_dynamic_booking(user_id: str, query: str, response: str, retrieved_context: str) -> str:
# # #     # Try to extract doctor and slot from query or the model response
# # #     doctor, slot = _extract_doctor_and_time(query)
# # #     if not doctor or not slot:
# # #         doc2, slot2 = _extract_doctor_and_time(response)
# # #         doctor = doctor or doc2
# # #         slot = slot or slot2
# # #     if not doctor or not slot:
# # #         return response

# # #     availability = _parse_doctor_availability(retrieved_context)
# # #     matched_doctor = next((d for d in availability if d.lower() == doctor.lower()), None)
# # #     if not matched_doctor:
# # #         return f"Sorry, {doctor} is not available."

# # #     if slot not in availability.get(matched_doctor, []):
# # #         possible_slots = availability.get(matched_doctor, [])
# # #         if possible_slots:
# # #             return f"Sorry, {matched_doctor} is not available at {slot}. Available slots: {', '.join(possible_slots)}."
# # #         return f"Sorry, {matched_doctor} has no available slots."

# # #     # Cross-user booking check and attempt to book
# # #     # (book_doctor will do necessary checks and return appropriate message)
# # #     return book_doctor(user_id, matched_doctor, slot)

# # # # ==============================
# # # # Main LLM Query Handler (rewritten to push pending query, call LLM with updated history, update response)
# # # # ==============================
# # # def handle_user_query(user_id: str, retrieved_context: str, query: str) -> str:
# # #     if not user_id or not query:
# # #         logger.warning("User ID or query missing")
# # #         return "Please provide a valid user ID and query."

# # #     if not isinstance(retrieved_context, str):
# # #         retrieved_context = json.dumps(retrieved_context)

# # #     # === Prepare embeddings & push pending query to Redis (so history includes current user query) ===
# # #     q_emb = _safe_embed(query)
# # #     pending_index = push_pending_query(user_id, query, q_emb)  # will store entry with response=None

# # #     # === Build history text from Redis (most recent first) and include the newly pushed pending query ===
# # #     history = get_chat_history(user_id, last_n=10)
# # #     # Format history most recent first
# # #     formatted_history = []
# # #     for turn in reversed(history):  # get most recent first
# # #         u = turn.get("query")
# # #         b = turn.get("response") or ""
# # #         if b:
# # #             formatted_history.append(f"User: {u}\nBot: {b}")
# # #         else:
# # #             formatted_history.append(f"User: {u}\nBot: <pending response>")
# # #     history_text = "\n\n".join(reversed(formatted_history)) if formatted_history else "No prior conversation."

# # #     # === Add user's current bookings ===
# # #     user_bookings = get_user_bookings(user_id)
# # #     if user_bookings:
# # #         user_booking_info = f"Your current bookings: {', '.join(user_bookings)}"
# # #     else:
# # #         user_booking_info = "You have no current bookings."

# # #     # === Build final prompt (LLM sees the current query as part of history) ===
# # #     final_prompt = qa_prompt.format(
# # #         context=f"{retrieved_context}\n\n{user_booking_info}",
# # #         history=history_text,
# # #         query=query
# # #     )
# # #     logger.info(f"Final prompt for LLM (User {user_id}):\n{final_prompt}")

# # #     # === Get initial LLM Response ===
# # #     raw_response = llm.predict(final_prompt)
# # #     response_text = " ".join(raw_response) if isinstance(raw_response, (list, tuple)) else str(raw_response)
# # #     response_text = response_text.strip()
# # #     logger.info(f"LLM raw response: {response_text}")

# # #     # === Save LLM response back to the pending Redis entry ===
# # #     r_emb = _safe_embed(response_text)
# # #     update_history_response(user_id, pending_index, response_text, r_emb)

# # #     # === Repetition check: if the new response is too similar to any previous response, ask LLM to rephrase once ===
# # #     is_rep, matched_prev = _is_repetition(user_id, response_text)
# # #     if is_rep:
# # #         logger.info(f"Detected repetition for user {user_id}. Asking LLM to rephrase.")
# # #         rephrase_prompt = qa_prompt.format(
# # #             context=f"{retrieved_context}\n\n{user_booking_info}",
# # #             history=history_text + f"\n\nBot (previous similar response): {matched_prev}",
# # #             query=f"{query}\n\nPlease rephrase the answer avoiding repeating past responses. Keep <=40 words."
# # #         )
# # #         logger.debug(f"Rephrase prompt:\n{rephrase_prompt}")
# # #         raw_re_resp = llm.predict(rephrase_prompt)
# # #         rephrase_text = " ".join(raw_re_resp) if isinstance(raw_re_resp, (list, tuple)) else str(raw_re_resp)
# # #         rephrase_text = rephrase_text.strip()
# # #         # update Redis with rephrased final answer
# # #         r_emb2 = _safe_embed(rephrase_text)
# # #         update_history_response(user_id, pending_index, rephrase_text, r_emb2)
# # #         final_response = rephrase_text
# # #     else:
# # #         final_response = response_text

# # #     # === Booking logic: attempt or report booking (booking function uses Redis sets and will reflect new booking) ===
# # #     booking_result = _process_dynamic_booking(user_id, query, final_response, retrieved_context)

# # #     # If booking_result is a booking message (starts with ✅ or ❌ or 'Sorry') - we should also persist that as bot response
# # #     # i.e., if booking changed the final user-visible response, update both Redis and returned text.
# # #     # Prefer to return the booking_result if it indicates booking activity or conflict, else return final_response.
# # #     booking_indicators = ("✅", "❌", "Sorry", "sorry")
# # #     if isinstance(booking_result, str) and booking_result.startswith(booking_indicators):
# # #         # update Redis entry and return booking_result
# # #         r_emb_booking = _safe_embed(booking_result)
# # #         update_history_response(user_id, pending_index, booking_result, r_emb_booking)
# # #         return booking_result

# # #     # Otherwise verify final_response (already saved) and return
# # #     return final_response

# # # # ==============================
# # # # Helper functions for external use
# # # # ==============================
# # # def list_available_slots_for_doctor(doctor_name: str, retrieved_context: str) -> List[str]:
# # #     availability = _parse_doctor_availability(retrieved_context)
# # #     return availability.get(doctor_name, [])

# # # def get_user_history(user_id: str, include_embeddings: bool = False) -> List[Dict[str, Any]]:
# # #     hist = get_chat_history(user_id, last_n=100)
# # #     if not include_embeddings:
# # #         return [{"query": e["query"], "response": e["response"]} for e in hist]
# # #     return hist

# # # def get_booking_summary() -> Dict[str, List[str]]:
# # #     keys = redis_client.keys("doctor_slots:*")
# # #     summary = {}
# # #     for key in keys:
# # #         doctor = key.split(":")[1]
# # #         summary[doctor] = list(redis_client.smembers(key))
# # #     return summary

# # # # ---------------- NEW: booking-intent + field extraction ---------------- #

# # # def _strip_code_fences(text: str) -> str:
# # #     if not isinstance(text, str):
# # #         return ""
# # #     t = text.strip()
# # #     if t.startswith("```"):
# # #         # remove leading/trailing ``` and an optional "json" tag
# # #         t = t.strip("`")
# # #         if t.lower().startswith("json"):
# # #             t = t[4:].lstrip()
# # #     return t

# # # def detect_booking_intent_and_fields(retrieved_context: str, query: str) -> dict:
# # #     """
# # #     Ask the LLM to:
# # #       1) classify intent ("book" | "other"),
# # #       2) when "book", extract the fields required by validate_and_book(...).

# # #     Returns a dict with this exact shape (use directly in webhook):
# # #       {
# # #         "intent": "book" | "other",
# # #         "doctor": str | null,
# # #         "date": "YYYY-MM-DD" | null,
# # #         "start": "HH:MM" | null,          # 24h
# # #         "duration_min": int | null,       # default to 30 when missing
# # #         "window_start": "HH:MM" | null,   # 24h, derived from context
# # #         "window_end": "HH:MM" | null      # 24h
# # #       }
# # #     """
# # #     retrieved_context = retrieved_context or "No relevant context available."
# # #     query = query or ""

# # #     system_msg = (
# # #         "You are a strict JSON extractor for doctor appointment bookings.\n"
# # #         "Decide if the user is trying to BOOK an appointment. If yes, extract normalized fields.\n"
# # #         "Return STRICT JSON only. No prose, no markdown."
# # #     )

# # #     schema_block = (
# # #         '{\n'
# # #         '  "intent": "book" | "other",\n'
# # #         '  "doctor": string | null,\n'
# # #         '  "date": "YYYY-MM-DD" | null,\n'
# # #         '  "start": "HH:MM" | null,\n'
# # #         '  "duration_min": number | null,\n'
# # #         '  "window_start": "HH:MM" | null,\n'
# # #         '  "window_end": "HH:MM" | null\n'
# # #         '}'
# # #     )

# # #     rules = (
# # #         "- If it is clearly a booking request, set intent to 'book'; otherwise 'other'.\n"
# # #         "- Use CONTEXT to determine the doctor's availability window; convert AM/PM to 24h HH:MM.\n"
# # #         "- Convert any user-provided time to 24h HH:MM.\n"
# # #         "- If a field is unknown, set it to null. Do NOT invent data.\n"
# # #         "- Output ONLY JSON matching the schema keys."
# # #     )

# # #     user_prompt = (
# # #         f"CONTEXT:\n{retrieved_context}\n\n"
# # #         f"USER_MESSAGE:\n{query}\n\n"
# # #         f"Output JSON matching this schema:\n{schema_block}\n\nRules:\n{rules}"
# # #     )

# # #     # We use llm.predict with a single prompt to keep it simple (consistent with your file)
# # #     raw = llm.predict(f"[SYSTEM]\n{system_msg}\n\n[USER]\n{user_prompt}").strip()
# # #     raw = _strip_code_fences(raw)

# # #     try:
# # #         payload = json.loads(raw)
# # #     except Exception:
# # #         # If the model fails JSON, fall back to non-booking
# # #         return {"intent": "other"}

# # #     # Normalize duration
# # #     if payload.get("duration_min") is None:
# # #         payload["duration_min"] = 30

# # #     # Ensure all keys exist
# # #     for k in ("doctor", "date", "start", "window_start", "window_end"):
# # #         payload.setdefault(k, None)

# # #     # Ensure intent
# # #     if payload.get("intent") not in ("book", "other"):
# # #         payload["intent"] = "other"

# # #     return payload


# # # def detect_booking_intent_and_fields(user_id: str, retrieved_context: str, query: str) -> dict:
# # #     """
# # #     Detect booking intent, extract fields, and store query/response in Redis chat history.
# # #     Returns dict matching the JSON schema for booking intent extraction.
# # #     """
# # #     retrieved_context = retrieved_context or "No relevant context available."
# # #     query = query or ""

# # #     # -----------------------------
# # #     # Push pending query to Redis
# # #     # -----------------------------
# # #     q_emb = _safe_embed(query)
# # #     pending_index = push_pending_query(user_id, query, q_emb)

# # #     # -----------------------------
# # #     # Gather user history and bookings
# # #     # -----------------------------
# # #     user_history = get_user_history(user_id, include_embeddings=False)
# # #     user_history_text = "\n".join(
# # #         [f"User: {h['query']}\nBot: {h.get('response','')}" for h in user_history[-10:]]
# # #     ) or "No prior conversation."

# # #     user_bookings = get_user_bookings(user_id)
# # #     user_booking_info = f"Your confirmed bookings: {', '.join(user_bookings)}" if user_bookings else "You have no confirmed bookings."

# # #     # -----------------------------
# # #     # Cross-user booked doctors
# # #     # -----------------------------
# # #     all_doctor_keys = redis_client.keys("doctor_slots:*")
# # #     booked_doctors = [key.split(":")[1] for key in all_doctor_keys if redis_client.scard(key) > 0]
# # #     booked_info = f"Doctors already booked by other users: {', '.join(booked_doctors)}" if booked_doctors else "No doctors are currently fully booked."

# # #     # -----------------------------
# # #     # LLM system + user prompt
# # #     # -----------------------------
# # #     system_msg = (
# # #         "You are a strict JSON extractor for doctor appointment bookings.\n"
# # #         "Decide if the user is trying to BOOK an appointment. If yes, extract normalized fields.\n"
# # #         "Return STRICT JSON only. No prose, no markdown."
# # #     )

# # #     schema_block = (
# # #         '{\n'
# # #         '  "intent": "book" | "other",\n'
# # #         '  "doctor": string | null,\n'
# # #         '  "date": "YYYY-MM-DD" | null,\n'
# # #         '  "start": "HH:MM" | null,\n'
# # #         '  "duration_min": number | null,\n'
# # #         '  "window_start": "HH:MM" | null,\n'
# # #         '  "window_end": "HH:MM" | null\n'
# # #         '}'
# # #     )

# # #     rules = (
# # #         "- If it is clearly a booking request, set intent to 'book'; otherwise 'other'.\n"
# # #         "- Use CONTEXT to determine doctor's availability window; convert AM/PM to 24h HH:MM.\n"
# # #         "- Convert any user-provided time to 24h HH:MM.\n"
# # #         "- Do NOT attempt booking if the doctor is already booked by another user.\n"
# # #         "- Include user's previous bookings in context.\n"
# # #         "- If a field is unknown, set it to null. Do NOT invent data.\n"
# # #         "- Output ONLY JSON matching the schema keys."
# # #     )

# # #     user_prompt = (
# # #         f"CONTEXT:\n{retrieved_context}\n{user_booking_info}\n{booked_info}\n\n"
# # #         f"Conversation History:\n{user_history_text}\n\n"
# # #         f"USER_MESSAGE:\n{query}\n\n"
# # #         f"Output JSON matching this schema:\n{schema_block}\n\nRules:\n{rules}"
# # #     )

# # #     # -----------------------------
# # #     # Call LLM
# # #     # -----------------------------
# # #     raw = llm.predict(f"[SYSTEM]\n{system_msg}\n\n[USER]\n{user_prompt}").strip()
# # #     raw = _strip_code_fences(raw)

# # #     try:
# # #         payload = json.loads(raw)
# # #     except Exception:
# # #         payload = {"intent": "other"}

# # #     # -----------------------------
# # #     # Normalize fields
# # #     # -----------------------------
# # #     if payload.get("duration_min") is None:
# # #         payload["duration_min"] = 30
# # #     for k in ("doctor", "date", "start", "window_start", "window_end"):
# # #         payload.setdefault(k, None)
# # #     if payload.get("intent") not in ("book", "other"):
# # #         payload["intent"] = "other"

# # #     # -----------------------------
# # #     # Cross-user booking check
# # #     # -----------------------------
# # #     doctor_name = payload.get("doctor")
# # #     if doctor_name and doctor_name in booked_doctors:
# # #         payload["intent"] = "other"
# # #         payload["doctor"] = None
# # #         payload["start"] = None
# # #         payload["window_start"] = None
# # #         payload["window_end"] = None
# # #         payload["duration_min"] = None
# # #         response_text = f"❌ Sorry, {doctor_name} is already booked by another user."
# # #     else:
# # #         response_text = f"Detected intent: {payload['intent']}"  # or you can craft any short response

# # #     # -----------------------------
# # #     # Save LLM response in Redis
# # #     # -----------------------------
# # #     r_emb = _safe_embed(response_text)
# # #     update_history_response(user_id, pending_index, response_text, r_emb)

# # #     return payload



# # # from app.prompts.qa_prompt import qa_prompt
# # # from app.core.config import llm

# # # def handle_user_query(retrieved_context: str, query: str) -> str:
# # #     """
# # #     Generate a concise response using only the provided retriever context.

# # #     Args:
# # #         retrieved_context (str): The context retrieved from a vector DB or knowledge source.
# # #         query (str): The user’s question.

# # #     Returns:
# # #         str: LLM-generated response.
# # #     """
# # #     # Ensure inputs are valid strings
# # #     retrieved_context = retrieved_context
# # #     query = query
# # #     # Format the prompt
# # #     formatted_prompt = qa_prompt.format(
# # #         context=retrieved_context,
# # #         query=query
# # #     )

# # #     # Generate response
# # #     response = llm.predict(formatted_prompt).strip()
# # #     return response














# # import os
# # import json
# # import logging
# # from typing import Dict, Any, Optional, List, Union
# # from datetime import datetime
# # import redis
# # from dotenv import load_dotenv

# # from langchain.prompts import ChatPromptTemplate
# # from langchain.schema import HumanMessage, AIMessage
# # from langchain.memory import ConversationSummaryBufferMemory
# # from langchain_core.chat_history import BaseChatMessageHistory
# # from langchain_core.messages import BaseMessage

# # from langgraph.graph import StateGraph, END
# # from langgraph.checkpoint.memory import MemorySaver
# # from typing_extensions import TypedDict

# # from app.core.config import llm

# # load_dotenv()

# # # ==============================
# # # Configuration
# # # ==============================
# # MEMORY_WINDOW_SIZE = 10
# # MAX_TOKEN_LIMIT = 4000
# # REDIS_TTL = 86400  # 24 hours

# # logging.basicConfig(
# #     level=logging.INFO,
# #     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# # )
# # logger = logging.getLogger(__name__)

# # # ==============================
# # # Debug Helper Function
# # # ==============================
# # def debug_print(section: str, data: Any, user_id: str = ""):
# #     """Enhanced debug printing with clear formatting"""
# #     separator = "=" * 80
# #     logger.info(f"\n{separator}")
# #     logger.info(f"🔍 DEBUG [{section}] - User: {user_id[:8] if user_id else 'N/A'}...")
# #     logger.info(f"{separator}")
    
# #     if isinstance(data, dict):
# #         for key, value in data.items():
# #             if isinstance(value, str) and len(value) > 200:
# #                 logger.info(f"  {key}: {value[:200]}... (truncated)")
# #             else:
# #                 logger.info(f"  {key}: {value}")
# #     elif isinstance(data, list):
# #         logger.info(f"  Items count: {len(data)}")
# #         for idx, item in enumerate(data[:5]):  # Show first 5 items
# #             logger.info(f"  [{idx}]: {item}")
# #         if len(data) > 5:
# #             logger.info(f"  ... and {len(data) - 5} more items")
# #     else:
# #         logger.info(f"  {data}")
    
# #     logger.info(f"{separator}\n")

# # # ==============================
# # # Redis Setup with Multiple Connection Methods
# # # ==============================
# # def get_redis_client():
# #     """Create Redis client with proper error handling - supports URL and username/password"""
# #     try:
# #         # Method 1: Try REDIS_URL first (preferred for cloud services)
# #         redis_url = os.getenv("REDIS_URL")
# #         if redis_url:
# #             logger.info(f"Connecting to Redis via URL...")
# #             client = redis.StrictRedis.from_url(
# #                 redis_url,
# #                 decode_responses=True,
# #                 socket_connect_timeout=5,
# #                 socket_timeout=5
# #             )
# #             client.ping()
# #             logger.info(f"✅ Redis connected via URL")
# #             logger.info(f"   Host: {client.connection_pool.connection_kwargs.get('host')}")
# #             logger.info(f"   Port: {client.connection_pool.connection_kwargs.get('port')}")
# #             return client
        
# #         # Method 2: Try username/password authentication
# #         redis_host = os.getenv("REDIS_HOST", "localhost")
# #         redis_port = int(os.getenv("REDIS_PORT", 6379))
# #         redis_db = int(os.getenv("REDIS_DB", 0))
# #         redis_user = os.getenv("REDIS_USER",  None)
# #         redis_password = os.getenv("REDIS_PASS", None)
        
# #         logger.info(f"Connecting to Redis at {redis_host}:{redis_port}")
        
# #         # Build connection parameters
# #         conn_params = {
# #             "host": redis_host,
# #             "port": redis_port,
# #             "db": 0,
# #             "decode_responses": True,
# #             "socket_connect_timeout": 5,
# #             "socket_timeout": 5
# #         }
# #         # Add authentication if provided
# #         if redis_user:
# #             conn_params["username"] = redis_user
# #             logger.info(f"Using username: {redis_user}")
        
# #         if redis_password and redis_password.strip():
# #             conn_params["password"] = redis_password
# #             logger.info(f"Using password authentication")
        
# #         client = redis.Redis(**conn_params)
        
# #         # Test connection
# #         client.ping()
# #         logger.info(f"✅ Redis connected successfully")
# #         logger.info(f"   Host: {redis_host}")
# #         logger.info(f"   Port: {redis_port}")
# #         logger.info(f"   DB: {redis_db}")
# #         return client
        
# #     except redis.ConnectionError as e:
# #         logger.error(f"❌ Redis connection failed: {e}")
# #         logger.warning("⚠️ Using in-memory fallback")
# #         return None
# #     except redis.AuthenticationError as e:
# #         logger.error(f"❌ Redis authentication failed: {e}")
# #         logger.warning("⚠️ Check REDIS_USER and REDIS_PASSWORD in .env file")
# #         logger.warning("⚠️ Using in-memory fallback")
# #         return None
# #     except Exception as e:
# #         logger.error(f"❌ Redis error: {e}")
# #         logger.warning("⚠️ Using in-memory fallback")
# #         return None

# # redis_client = get_redis_client()

# # # In-memory fallback storage
# # _memory_store = {}

# # # ==============================
# # # Custom Redis Chat History with Fallback
# # # ==============================
# # class OptimizedRedisChatHistory(BaseChatMessageHistory):
# #     """Optimized Redis-backed chat history with in-memory fallback"""
   
# #     def __init__(self, session_id: str, window_size: int = MEMORY_WINDOW_SIZE):
# #         self.session_id = session_id
# #         self.key = f"chat:{session_id}"
# #         self.window_size = window_size
# #         self.use_redis = redis_client is not None
   
# #     @property
# #     def messages(self):
# #         """Get messages within window"""
# #         if self.use_redis:
# #             try:
# #                 raw = redis_client.lrange(self.key, -self.window_size * 2, -1)
# #                 messages = []
# #                 for msg_str in raw:
# #                     msg = json.loads(msg_str)
# #                     if msg["type"] == "human":
# #                         messages.append(HumanMessage(content=msg["content"]))
# #                     elif msg["type"] == "ai":
# #                         messages.append(AIMessage(content=msg["content"]))
                
# #                 debug_print("REDIS MESSAGES LOADED", {
# #                     "session_id": self.session_id,
# #                     "message_count": len(messages),
# #                     "storage": "Redis"
# #                 }, self.session_id)
                
# #                 return messages
# #             except Exception as e:
# #                 logger.error(f"Redis read error: {e}, falling back to memory")
# #                 self.use_redis = False
        
# #         # In-memory fallback
# #         if self.key not in _memory_store:
# #             _memory_store[self.key] = []
        
# #         raw = _memory_store[self.key][-self.window_size * 2:]
# #         messages = []
# #         for msg in raw:
# #             if msg["type"] == "human":
# #                 messages.append(HumanMessage(content=msg["content"]))
# #             elif msg["type"] == "ai":
# #                 messages.append(AIMessage(content=msg["content"]))
        
# #         debug_print("MEMORY MESSAGES LOADED", {
# #             "session_id": self.session_id,
# #             "message_count": len(messages),
# #             "storage": "In-Memory"
# #         }, self.session_id)
        
# #         return messages
   
# #     def add_message(self, message: BaseMessage):
# #         """Add message with automatic trimming"""
# #         msg_dict = {
# #             "type": "human" if isinstance(message, HumanMessage) else "ai",
# #             "content": message.content,
# #             "ts": datetime.utcnow().isoformat()
# #         }
# #         logger.info("============================msg_dict====================================")
# #         logger.info(msg_dict)
# #         logger.info("============================msg_dict====================================")
# #         if self.use_redis:
# #             try:
# #                 pipe = redis_client.pipeline()
# #                 pipe.rpush(self.key, json.dumps(msg_dict))
# #                 pipe.ltrim(self.key, -self.window_size * 2, -1)
# #                 pipe.expire(self.key, REDIS_TTL)
# #                 pipe.execute()
                
# #                 debug_print("MESSAGE SAVED TO REDIS", {
# #                     "type": msg_dict["type"],
# #                     "content_preview": msg_dict["content"][:100]
# #                 }, self.session_id)
# #                 return
# #             except Exception as e:
# #                 logger.error(f"Redis write error: {e}, falling back to memory")
# #                 self.use_redis = False
        
# #         # In-memory fallback
# #         if self.key not in _memory_store:
# #             _memory_store[self.key] = []
        
# #         _memory_store[self.key].append(msg_dict)
        
# #         # Trim to window size
# #         if len(_memory_store[self.key]) > self.window_size * 2:
# #             _memory_store[self.key] = _memory_store[self.key][-self.window_size * 2:]
        
# #         debug_print("MESSAGE SAVED TO MEMORY", {
# #             "type": msg_dict["type"],
# #             "content_preview": msg_dict["content"][:100]
# #         }, self.session_id)
   
# #     def clear(self):
# #         """Clear history"""
# #         if self.use_redis:
# #             try:
# #                 redis_client.delete(self.key)
# #                 logger.info(f"Cleared Redis key: {self.key}")
# #                 return
# #             except Exception as e:
# #                 logger.error(f"Redis delete error: {e}")
        
# #         # In-memory fallback
# #         if self.key in _memory_store:
# #             del _memory_store[self.key]
# #             logger.info(f"Cleared memory key: {self.key}")

# # # ==============================
# # # Memory Management
# # # ==============================
# # def get_memory(user_id: str) -> ConversationSummaryBufferMemory:
# #     """Get or create conversation memory with auto-summarization"""
# #     chat_history = OptimizedRedisChatHistory(user_id)
   
# #     return ConversationSummaryBufferMemory(
# #         llm=llm,
# #         chat_memory=chat_history,
# #         max_token_limit=MAX_TOKEN_LIMIT,
# #         return_messages=True,
# #         memory_key="history"
# #     )

# # def load_memory_context(user_id: str) -> Dict[str, str]:
# #     """Load recent history and summary"""
# #     try:
# #         memory = get_memory(user_id)
# #         vars = memory.load_memory_variables({})
       
# #         messages = vars.get("history", [])
# #         recent = []
# #         for msg in messages[-MEMORY_WINDOW_SIZE * 2:]:
# #             prefix = "User" if isinstance(msg, HumanMessage) else "Bot"
# #             recent.append(f"{prefix}: {msg.content}")
        
# #         result = {
# #             "recent_history": "\n".join(recent) or "No recent conversation.",
# #             "summary": getattr(memory, 'moving_summary_buffer', "")
# #         }
        
# #         debug_print("MEMORY CONTEXT LOADED", {
# #             "user_id": user_id,
# #             "message_count": len(messages),
# #             "recent_history_length": len(result["recent_history"]),
# #             "has_summary": bool(result["summary"])
# #         }, user_id)
        
# #         return result
# #     except Exception as e:
# #         logger.error(f"Error loading memory context: {e}")
# #         return {
# #             "recent_history": "No recent conversation.",
# #             "summary": ""
# #         }

# # def save_to_memory(user_id: str, query: str, response: str):
# #     """Save interaction to memory"""
# #     try:
# #         memory = get_memory(user_id)
# #         memory.save_context({"input": query}, {"output": response})
        
# #         debug_print("SAVED TO MEMORY", {
# #             "user_id": user_id,
# #             "query_preview": query[:100],
# #             "response_preview": response[:100]
# #         }, user_id)
# #     except Exception as e:
# #         logger.error(f"Error saving to memory: {e}")

# # # ==============================
# # # Context Formatting
# # # ==============================
# # def format_context_for_llm(retrieved_context: Union[List, str]) -> str:
# #     """Format retrieved context into readable text for LLM"""
# #     if not retrieved_context:
# #         return "No relevant information found."
    
# #     if isinstance(retrieved_context, str):
# #         return retrieved_context
    
# #     formatted_parts = []
    
# #     for idx, item in enumerate(retrieved_context, 1):
# #         if not isinstance(item, dict):
# #             continue
            
# #         # Extract text content
# #         source_text = item.get('source_text', '')
# #         source = item.get('source', 'Unknown')
# #         score = item.get('score', 0)
        
# #         # Handle page_content format (from raw_text_input)
# #         if not source_text and 'source' in item:
# #             raw_source = item.get('source', '')
# #             if isinstance(raw_source, str) and 'page_content=' in raw_source:
# #                 # Extract the actual content after page_content=
# #                 try:
# #                     source_text = raw_source.split("page_content='")[1].split("' metadata=")[0]
# #                 except:
# #                     source_text = raw_source
        
# #         if source_text:
# #             formatted_parts.append(f"[Source {idx}] (Relevance: {score:.2f})\n{source_text}\n")
    
# #     result = "\n".join(formatted_parts) if formatted_parts else "No detailed information available."
    
# #     debug_print("FORMATTED CONTEXT", {
# #         "original_items": len(retrieved_context) if isinstance(retrieved_context, list) else "string",
# #         "formatted_length": len(result),
# #         "preview": result[:300]
# #     })
    
# #     return result

# # # ==============================
# # # LangGraph State
# # # ==============================
# # class ConversationState(TypedDict):
# #     user_id: str
# #     query: str
# #     context: str
# #     recent_history: str
# #     summary: str
# #     response: str

# # # ==============================
# # # Prompt Templates
# # # ==============================
# # SYSTEM_PROMPT = """
# # You are a retrieval-augmented medical assistant chatbot for Indo-US Multi-Specialty Hospital.

# # Your responsibilities:
# # 1. **Extract and use information** from the retriever context, including structured data like doctor lists, schedules, and specialties.
# # 2. When asked for lists (e.g., "top 5 female doctors"), extract and present the information directly from the context.
# # 3. Use chat history to avoid repetition - if you must repeat necessary info, rephrase it.
# # 4. If a booking was already confirmed, acknowledge that instead of reconfirming.
# # 5. Be concise, clear, and human-like (≤50 words for simple queries, longer for lists/details).
# # 6. If doctor, time, or information is missing from context, ask for clarification.
# # 7. Never fabricate data. If unsure, politely say so.

# # **Important**: When the context contains doctor information with names, specialties, contact details, or schedules, extract and present it directly in a clear, formatted way.
# # """

# # qa_prompt = ChatPromptTemplate.from_messages([
# #     ("system", SYSTEM_PROMPT),
# #     ("human", """Retrieved Information:
# # {context}

# # Recent Conversation (last {window_size} exchanges):
# # {recent_history}

# # Memory Summary (older context):
# # {summary}

# # User query: {query}

# # INSTRUCTIONS:
# # - If the Retrieved Information contains structured data (like doctor lists, schedules), extract and format it clearly
# # - For "list" or "top X" requests, provide the information directly if available in the context
# # - Use Memory Summary for background context from earlier conversations
# # - Use Recent Conversation for immediate context
# # - Do NOT repeat earlier bot responses verbatim; rephrase or provide new clarifying info
# # - If information is incomplete, ask specific clarifying questions
# # - Format doctor information clearly with name, specialty, and relevant details
# # """)
# # ])

# # # ==============================
# # # Workflow Nodes
# # # ==============================
# # def load_context_node(state: ConversationState) -> ConversationState:
# #     """Load memory context"""
# #     mem_ctx = load_memory_context(state["user_id"])
# #     state["recent_history"] = mem_ctx["recent_history"]
# #     state["summary"] = mem_ctx["summary"]
    
# #     debug_print("LOAD CONTEXT NODE", {
# #         "user_id": state["user_id"],
# #         "recent_history": state["recent_history"],
# #         "summary": state["summary"] or "No summary"
# #     }, state["user_id"])
    
# #     return state

# # def generate_response_node(state: ConversationState) -> ConversationState:
# #     """Generate LLM response using qa_prompt"""
# #     formatted = qa_prompt.format(
# #         context=state["context"],
# #         window_size=MEMORY_WINDOW_SIZE,
# #         recent_history=state["recent_history"],
# #         summary=state["summary"] or "No previous context.",
# #         query=state["query"]
# #     )
    
# #     debug_print("PROMPT SENT TO LLM", {
# #         "user_id": state["user_id"],
# #         "formatted_prompt": formatted[:500] + "..." if len(formatted) > 500 else formatted
# #     }, state["user_id"])
   
# #     raw = llm.predict(formatted)
# #     state["response"] = str(raw).strip()
    
# #     debug_print("LLM RESPONSE GENERATED", {
# #         "user_id": state["user_id"],
# #         "response": state["response"]
# #     }, state["user_id"])
    
# #     return state

# # def save_memory_node(state: ConversationState) -> ConversationState:
# #     """Save to memory"""
# #     save_to_memory(state["user_id"], state["query"], state["response"])
# #     return state

# # # ==============================
# # # Workflow Creation
# # # ==============================
# # def create_workflow() -> StateGraph:
# #     """Create optimized conversation workflow"""
# #     workflow = StateGraph(ConversationState)
   
# #     workflow.add_node("load_context", load_context_node)
# #     workflow.add_node("generate", generate_response_node)
# #     workflow.add_node("save", save_memory_node)
   
# #     workflow.set_entry_point("load_context")
# #     workflow.add_edge("load_context", "generate")
# #     workflow.add_edge("generate", "save")
# #     workflow.add_edge("save", END)
   
# #     return workflow.compile()

# # conversation_flow = create_workflow()

# # # ==============================
# # # Main Handler
# # # ==============================
# # def handle_user_query(user_id: str, retrieved_context: Union[str, List, Dict], query: str) -> str:
# #     """
# #     Main query handler with optimized memory.
   
# #     Args:
# #         user_id: Unique user identifier
# #         retrieved_context: Context from vector DB (can be list, dict, or string)
# #         query: User's question
   
# #     Returns:
# #         AI response string
# #     """
# #     debug_print("INCOMING REQUEST", {
# #         "user_id": user_id,
# #         "query": query,
# #         "context_type": type(retrieved_context).__name__,
# #         "context_preview": str(retrieved_context)[:200]
# #     }, user_id)
    
# #     if not user_id or not query:
# #         return "Invalid request."
   
# #     # Format context properly for LLM
# #     if isinstance(retrieved_context, list):
# #         formatted_context = format_context_for_llm(retrieved_context)
# #     elif isinstance(retrieved_context, dict):
# #         formatted_context = json.dumps(retrieved_context, indent=2)
# #     elif isinstance(retrieved_context, str):
# #         formatted_context = retrieved_context
# #     else:
# #         formatted_context = str(retrieved_context)
   
# #     state: ConversationState = {
# #         "user_id": user_id,
# #         "query": query,
# #         "context": formatted_context,
# #         "recent_history": "",
# #         "summary": "",
# #         "response": ""
# #     }
   
# #     final = conversation_flow.invoke(state)
    
# #     debug_print("FINAL RESPONSE", {
# #         "user_id": user_id,
# #         "query": query,
# #         "response": final["response"],
# #         "context_used": formatted_context[:200]
# #     }, user_id)
    
# #     return final["response"]

# # # ==============================
# # # Utility Functions
# # # ==============================
# # def clear_user_memory(user_id: str):
# #     """Clear user's conversation history"""
# #     OptimizedRedisChatHistory(user_id).clear()
# #     logger.info(f"Cleared memory: {user_id}")

# # def get_user_history(user_id: str) -> list:
# #     """Get user's conversation history"""
# #     try:
# #         memory = get_memory(user_id)
# #         messages = memory.chat_memory.messages
       
# #         history = []
# #         for i in range(0, len(messages), 2):
# #             if i + 1 < len(messages):
# #                 history.append({
# #                     "query": messages[i].content,
# #                     "response": messages[i + 1].content
# #                 })
        
# #         debug_print("USER HISTORY RETRIEVED", {
# #             "user_id": user_id,
# #             "conversation_count": len(history)
# #         }, user_id)
        
# #         return history
# #     except Exception as e:
# #         logger.error(f"Error getting user history: {e}")
# #         return []

# # # ==============================
# # # Debug Function
# # # ==============================
# # def debug_redis_connection():
# #     """Debug function to test Redis connection"""
# #     if redis_client is None:
# #         logger.error("Redis client is None - connection failed")
# #         return False
    
# #     try:
# #         # Test basic operations
# #         test_key = "test:connection"
# #         redis_client.set(test_key, "test_value", ex=10)
# #         value = redis_client.get(test_key)
# #         redis_client.delete(test_key)
        
# #         logger.info(f"✅ Redis test successful: {value}")
        
# #         # List all keys (for debugging)
# #         all_keys = redis_client.keys("chat:*")
# #         logger.info(f"📋 Current chat keys in Redis: {len(all_keys)}")
        
# #         return True
# #     except Exception as e:
# #         logger.error(f"❌ Redis test failed: {e}")
# #         return False































# import os
# import json
# import logging
# from typing import Dict, Any, Optional, List, Union
# from datetime import datetime
# import redis
# from dotenv import load_dotenv

# from langchain.prompts import ChatPromptTemplate
# from langchain.schema import HumanMessage, AIMessage
# from langchain.memory import ConversationSummaryBufferMemory
# from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_core.messages import BaseMessage

# from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.memory import MemorySaver
# from typing_extensions import TypedDict

# from app.core.config import llm

# load_dotenv()

# # ==============================
# # Configuration
# # ==============================
# MEMORY_WINDOW_SIZE = 10
# MAX_TOKEN_LIMIT = 4000
# REDIS_TTL = 86400  # 24 hours

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# # ==============================
# # Debug Helper Function
# # ==============================
# def debug_print(section: str, data: Any, user_id: str = ""):
#     """Enhanced debug printing with clear formatting"""
#     separator = "=" * 80
#     logger.info(f"\n{separator}")
#     logger.info(f"🔍 DEBUG [{section}] - User: {user_id[:8] if user_id else 'N/A'}...")
#     logger.info(f"{separator}")
    
#     if isinstance(data, dict):
#         for key, value in data.items():
#             if isinstance(value, str) and len(value) > 200:
#                 logger.info(f"  {key}: {value[:200]}... (truncated)")
#             else:
#                 logger.info(f"  {key}: {value}")
#     elif isinstance(data, list):
#         logger.info(f"  Items count: {len(data)}")
#         for idx, item in enumerate(data[:5]):  # Show first 5 items
#             logger.info(f"  [{idx}]: {item}")
#         if len(data) > 5:
#             logger.info(f"  ... and {len(data) - 5} more items")
#     else:
#         logger.info(f"  {data}")
    
#     logger.info(f"{separator}\n")

# # ==============================
# # Redis Setup with Multiple Connection Methods
# # ==============================
# def get_redis_client():
#     """Create Redis client with proper error handling - supports URL and username/password"""
#     try:
#         # Method 1: Try REDIS_URL first (preferred for cloud services)
#         redis_url = os.getenv("REDIS_URL")
#         if redis_url:
#             logger.info(f"Connecting to Redis via URL...")
#             client = redis.StrictRedis.from_url(
#                 redis_url,
#                 decode_responses=True,
#                 socket_connect_timeout=5,
#                 socket_timeout=5
#             )
#             client.ping()
#             logger.info(f"✅ Redis connected via URL")
#             logger.info(f"   Host: {client.connection_pool.connection_kwargs.get('host')}")
#             logger.info(f"   Port: {client.connection_pool.connection_kwargs.get('port')}")
#             return client
        
#         # Method 2: Try username/password authentication
#         redis_host = os.getenv("REDIS_HOST", "localhost")
#         redis_port = int(os.getenv("REDIS_PORT", 6379))
#         redis_db = int(os.getenv("REDIS_DB", 0))
        
#         logger.info(f"Connecting to Redis at {redis_host}:{redis_port}")
        
#         # Build connection parameters
#         conn_params = {
#             "host": redis_host,
#             "port": redis_port,
#             "db": redis_db,
#             "decode_responses": True,
#         }
        
#         client = redis.Redis(**conn_params)
        
#         # Test connection
#         client.ping()
#         logger.info(f"✅ Redis connected successfully")
#         logger.info(f"   Host: {redis_host}")
#         logger.info(f"   Port: {redis_port}")
#         logger.info(f"   DB: {redis_db}")
#         return client
        
#     except redis.ConnectionError as e:
#         logger.error(f"❌ Redis connection failed: {e}")
#         logger.warning("⚠️ Using in-memory fallback")
#         return None
#     except redis.AuthenticationError as e:
#         logger.error(f"❌ Redis authentication failed: {e}")
#         logger.warning("⚠️ Check REDIS_USER and REDIS_PASSWORD in .env file")
#         logger.warning("⚠️ Using in-memory fallback")
#         return None
#     except Exception as e:
#         logger.error(f"❌ Redis error: {e}")
#         logger.warning("⚠️ Using in-memory fallback")
#         return None

# redis_client = get_redis_client()

# # In-memory fallback storage
# _memory_store = {}

# # ==============================
# # Custom Redis Chat History with Fallback
# # ==============================
# class OptimizedRedisChatHistory(BaseChatMessageHistory):
#     """Optimized Redis-backed chat history with in-memory fallback"""
   
#     def __init__(self, session_id: str, window_size: int = MEMORY_WINDOW_SIZE):
#         self.session_id = session_id
#         self.key = f"chat:{session_id}"
#         self.window_size = window_size
#         self.use_redis = redis_client is not None
   
#     @property
#     def messages(self):
#         """Get messages within window"""
#         if self.use_redis:
#             try:
#                 raw = redis_client.lrange(self.key, -self.window_size * 2, -1)
#                 messages = []
#                 for msg_str in raw:
#                     msg = json.loads(msg_str)
#                     if msg["type"] == "human":
#                         messages.append(HumanMessage(content=msg["content"]))
#                     elif msg["type"] == "ai":
#                         messages.append(AIMessage(content=msg["content"]))
                
#                 debug_print("REDIS MESSAGES LOADED", {
#                     "session_id": self.session_id,
#                     "message_count": len(messages),
#                     "storage": "Redis"
#                 }, self.session_id)
                
#                 return messages
#             except Exception as e:
#                 logger.error(f"Redis read error: {e}, falling back to memory")
#                 self.use_redis = False
        
#         # In-memory fallback
#         if self.key not in _memory_store:
#             _memory_store[self.key] = []
        
#         raw = _memory_store[self.key][-self.window_size * 2:]
#         messages = []
#         for msg in raw:
#             if msg["type"] == "human":
#                 messages.append(HumanMessage(content=msg["content"]))
#             elif msg["type"] == "ai":
#                 messages.append(AIMessage(content=msg["content"]))
        
#         debug_print("MEMORY MESSAGES LOADED", {
#             "session_id": self.session_id,
#             "message_count": len(messages),
#             "storage": "In-Memory"
#         }, self.session_id)
        
#         return messages
   
#     def add_message(self, message: BaseMessage):
#         """Add message with automatic trimming"""
#         msg_dict = {
#             "type": "human" if isinstance(message, HumanMessage) else "ai",
#             "content": message.content,
#             "ts": datetime.utcnow().isoformat()
#         }
       
#         if self.use_redis:
#             try:
#                 pipe = redis_client.pipeline()
#                 pipe.rpush(self.key, json.dumps(msg_dict))
#                 pipe.ltrim(self.key, -self.window_size * 2, -1)
#                 pipe.expire(self.key, REDIS_TTL)
#                 pipe.execute()
                
#                 debug_print("MESSAGE SAVED TO REDIS", {
#                     "type": msg_dict["type"],
#                     "content_preview": msg_dict["content"][:100]
#                 }, self.session_id)
#                 return
#             except Exception as e:
#                 logger.error(f"Redis write error: {e}, falling back to memory")
#                 self.use_redis = False
        
#         # In-memory fallback
#         if self.key not in _memory_store:
#             _memory_store[self.key] = []
        
#         _memory_store[self.key].append(msg_dict)
        
#         # Trim to window size
#         if len(_memory_store[self.key]) > self.window_size * 2:
#             _memory_store[self.key] = _memory_store[self.key][-self.window_size * 2:]
        
#         debug_print("MESSAGE SAVED TO MEMORY", {
#             "type": msg_dict["type"],
#             "content_preview": msg_dict["content"][:100]
#         }, self.session_id)
   
#     def clear(self):
#         """Clear history"""
#         if self.use_redis:
#             try:
#                 redis_client.delete(self.key)
#                 logger.info(f"Cleared Redis key: {self.key}")
#                 return
#             except Exception as e:
#                 logger.error(f"Redis delete error: {e}")
        
#         # In-memory fallback
#         if self.key in _memory_store:
#             del _memory_store[self.key]
#             logger.info(f"Cleared memory key: {self.key}")

# # ==============================
# # Memory Management
# # ==============================
# def get_memory(user_id: str) -> ConversationSummaryBufferMemory:
#     """Get or create conversation memory with auto-summarization"""
#     chat_history = OptimizedRedisChatHistory(user_id)
   
#     return ConversationSummaryBufferMemory(
#         llm=llm,
#         chat_memory=chat_history,
#         max_token_limit=MAX_TOKEN_LIMIT,
#         return_messages=True,
#         memory_key="history"
#     )

# def load_memory_context(user_id: str) -> Dict[str, str]:
#     """Load recent history and summary"""
#     try:
#         memory = get_memory(user_id)
#         vars = memory.load_memory_variables({})
       
#         messages = vars.get("history", [])
#         recent = []
#         for msg in messages[-MEMORY_WINDOW_SIZE * 2:]:
#             prefix = "User" if isinstance(msg, HumanMessage) else "Bot"
#             recent.append(f"{prefix}: {msg.content}")
        
#         result = {
#             "recent_history": "\n".join(recent) or "No recent conversation.",
#             "summary": getattr(memory, 'moving_summary_buffer', "")
#         }
        
#         debug_print("MEMORY CONTEXT LOADED", {
#             "user_id": user_id,
#             "message_count": len(messages),
#             "recent_history_length": len(result["recent_history"]),
#             "has_summary": bool(result["summary"])
#         }, user_id)
        
#         return result
#     except Exception as e:
#         logger.error(f"Error loading memory context: {e}")
#         return {
#             "recent_history": "No recent conversation.",
#             "summary": ""
#         }

# def save_to_memory(user_id: str, query: str, response: str):
#     """Save interaction to memory"""
#     try:
#         memory = get_memory(user_id)
#         memory.save_context({"input": query}, {"output": response})
        
#         debug_print("SAVED TO MEMORY", {
#             "user_id": user_id,
#             "query_preview": query[:100],
#             "response_preview": response[:100]
#         }, user_id)
#     except Exception as e:
#         logger.error(f"Error saving to memory: {e}")

# # ==============================
# # Context Formatting
# # ==============================
# def format_context_for_llm(retrieved_context: Union[List, str]) -> str:
#     """Format retrieved context into readable text for LLM"""
#     if not retrieved_context:
#         return "No relevant information found."
    
#     if isinstance(retrieved_context, str):
#         return retrieved_context
    
#     formatted_parts = []
    
#     for idx, item in enumerate(retrieved_context, 1):
#         if not isinstance(item, dict):
#             continue
            
#         # Extract text content
#         source_text = item.get('source_text', '')
#         source = item.get('source', 'Unknown')
#         score = item.get('score', 0)
        
#         # Handle page_content format (from raw_text_input)
#         if not source_text and 'source' in item:
#             raw_source = item.get('source', '')
#             if isinstance(raw_source, str) and 'page_content=' in raw_source:
#                 # Extract the actual content after page_content=
#                 try:
#                     source_text = raw_source.split("page_content='")[1].split("' metadata=")[0]
#                 except:
#                     source_text = raw_source
        
#         if source_text:
#             formatted_parts.append(f"[Source {idx}] (Relevance: {score:.2f})\n{source_text}\n")
    
#     result = "\n".join(formatted_parts) if formatted_parts else "No detailed information available."
    
#     debug_print("FORMATTED CONTEXT", {
#         "original_items": len(retrieved_context) if isinstance(retrieved_context, list) else "string",
#         "formatted_length": len(result),
#         "preview": result[:300]
#     })
    
#     return result

# # ==============================
# # LangGraph State
# # ==============================
# class ConversationState(TypedDict):
#     user_id: str
#     query: str
#     context: str
#     recent_history: str
#     summary: str
#     response: str

# # ==============================
# # Prompt Templates
# # ==============================
# SYSTEM_PROMPT = """
# You are a retrieval-augmented medical assistant chatbot for Indo-US Multi-Specialty Hospital.

# Your responsibilities:
# 1. **Extract and use information** from the retriever context, including structured data like doctor lists, schedules, and specialties.
# 2. When asked for lists (e.g., "top 5 female doctors"), extract and present the information directly from the context.
# 3. Use chat history to avoid repetition - if you must repeat necessary info, rephrase it.
# 4. If a booking was already confirmed, acknowledge that instead of reconfirming.
# 5. Be concise, clear, and human-like (≤50 words for simple queries, longer for lists/details).
# 6. If doctor, time, or information is missing from context, ask for clarification.
# 7. Never fabricate data. If unsure, politely say so.

# **Important**: When the context contains doctor information with names, specialties, contact details, or schedules, extract and present it directly in a clear, formatted way.
# """

# qa_prompt = ChatPromptTemplate.from_messages([
#     ("system", SYSTEM_PROMPT),
#     ("human", """Retrieved Information:
# {context}

# Recent Conversation (last {window_size} exchanges):
# {recent_history}

# Memory Summary (older context):
# {summary}

# User query: {query}

# INSTRUCTIONS:
# - If the Retrieved Information contains structured data (like doctor lists, schedules), extract and format it clearly
# - For "list" or "top X" requests, provide the information directly if available in the context
# - Use Memory Summary for background context from earlier conversations
# - Use Recent Conversation for immediate context
# - Do NOT repeat earlier bot responses verbatim; rephrase or provide new clarifying info
# - If booking was confirmed earlier, acknowledge it directly
# - If slot unavailable, provide available slots from the context
# - If information is incomplete, ask specific clarifying questions
# - Format doctor information clearly with name, specialty, and relevant details
# """)
# ])

# # ==============================
# # Workflow Nodes
# # ==============================
# def load_context_node(state: ConversationState) -> ConversationState:
#     """Load memory context"""
#     mem_ctx = load_memory_context(state["user_id"])
#     state["recent_history"] = mem_ctx["recent_history"]
#     state["summary"] = mem_ctx["summary"]
    
#     debug_print("LOAD CONTEXT NODE", {
#         "user_id": state["user_id"],
#         "recent_history": state["recent_history"],
#         "summary": state["summary"] or "No summary"
#     }, state["user_id"])
    
#     return state

# def generate_response_node(state: ConversationState) -> ConversationState:
#     """Generate LLM response using qa_prompt"""
#     formatted = qa_prompt.format(
#         context=state["context"],
#         window_size=MEMORY_WINDOW_SIZE,
#         recent_history=state["recent_history"],
#         summary=state["summary"] or "No previous context.",
#         query=state["query"]
#     )
    
#     debug_print("PROMPT SENT TO LLM", {
#         "user_id": state["user_id"],
#         "formatted_prompt": formatted[:500] + "..." if len(formatted) > 500 else formatted
#     }, state["user_id"])
   
#     raw = llm.predict(formatted)
#     state["response"] = str(raw).strip()
    
#     debug_print("LLM RESPONSE GENERATED", {
#         "user_id": state["user_id"],
#         "response": state["response"]
#     }, state["user_id"])
    
#     return state

# def save_memory_node(state: ConversationState) -> ConversationState:
#     """Save to memory"""
#     save_to_memory(state["user_id"], state["query"], state["response"])
#     return state

# # ==============================
# # Workflow Creation
# # ==============================
# def create_workflow() -> StateGraph:
#     """Create optimized conversation workflow"""
#     workflow = StateGraph(ConversationState)
   
#     workflow.add_node("load_context", load_context_node)
#     workflow.add_node("generate", generate_response_node)
#     workflow.add_node("save", save_memory_node)
   
#     workflow.set_entry_point("load_context")
#     workflow.add_edge("load_context", "generate")
#     workflow.add_edge("generate", "save")
#     workflow.add_edge("save", END)
   
#     return workflow.compile()

# conversation_flow = create_workflow()

# # ==============================
# # Main Handler
# # ==============================
# def handle_user_query(user_id: str, retrieved_context: Union[str, List, Dict], query: str) -> str:
#     """
#     Main query handler with optimized memory.
   
#     Args:
#         user_id: Unique user identifier
#         retrieved_context: Context from vector DB (can be list, dict, or string)
#         query: User's question
   
#     Returns:
#         AI response string
#     """
#     debug_print("INCOMING REQUEST", {
#         "user_id": user_id,
#         "query": query,
#         "context_type": type(retrieved_context).__name__,
#         "context_preview": str(retrieved_context)[:200]
#     }, user_id)
    
#     if not user_id or not query:
#         return "Invalid request."
   
#     # Format context properly for LLM
#     if isinstance(retrieved_context, list):
#         formatted_context = format_context_for_llm(retrieved_context)
#     elif isinstance(retrieved_context, dict):
#         formatted_context = json.dumps(retrieved_context, indent=2)
#     elif isinstance(retrieved_context, str):
#         formatted_context = retrieved_context
#     else:
#         formatted_context = str(retrieved_context)
   
#     state: ConversationState = {
#         "user_id": user_id,
#         "query": query,
#         "context": formatted_context,
#         "recent_history": "",
#         "summary": "",
#         "response": ""
#     }
   
#     final = conversation_flow.invoke(state)
    
#     debug_print("FINAL RESPONSE", {
#         "user_id": user_id,
#         "query": query,
#         "response": final["response"],
#         "context_used": formatted_context[:200]
#     }, user_id)
    
#     return final["response"]

# # ==============================
# # Booking Intent Detection (Optimized)
# # ==============================
# def detect_booking_intent_and_fields(user_id: str, retrieved_context: Union[str, List, Dict], query: str) -> dict:
#     """Detect booking intent with memory context"""
#     debug_print("BOOKING INTENT DETECTION", {
#         "user_id": user_id,
#         "query": query,
#         "context_type": type(retrieved_context).__name__
#     }, user_id)
    
#     # Format context if it's a list
#     if isinstance(retrieved_context, list):
#         formatted_context = format_context_for_llm(retrieved_context)
#     elif isinstance(retrieved_context, dict):
#         formatted_context = json.dumps(retrieved_context, indent=2)
#     else:
#         formatted_context = str(retrieved_context)
    
#     mem_ctx = load_memory_context(user_id)
   
#     system = (
#         "You are a JSON extractor for appointment bookings.\n"
#         "Return ONLY valid JSON. No markdown, no prose."
#     )
   
#     schema = '''{
#         "intent": "book" | "other",
#         "doctor": string | null,
#         "date": "YYYY-MM-DD" | null,
#         "start": "HH:MM" | null,
#         "duration_min": number | null,
#         "window_start": "HH:MM" | null,
#         "window_end": "HH:MM" | null
#     }'''
   
#     user_prompt = f"""Context:
#         {formatted_context[:1000]}

#         Recent History:
#         {mem_ctx['recent_history']}

#         Summary:
#         {mem_ctx['summary']}

#         User: {query}

#         Extract booking intent and fields as JSON:
#         {schema}

#         Rules:
#         - intent='book' only if clearly requesting appointment
#         - Convert AM/PM to 24h format
#         - Use null for unknown fields
#         - Default duration_min to 30 if not specified"""
   
#     raw = llm.predict(f"[SYSTEM]\n{system}\n\n[USER]\n{user_prompt}").strip()
#     raw = raw.strip("`").replace("```json", "").replace("```", "").strip()
   
#     try:
#         payload = json.loads(raw)
#     except:
#         logger.error(f"JSON parse failed: {raw}")
#         payload = {"intent": "other"}
   
#     # Set defaults
#     payload.setdefault("duration_min", 30)
#     for k in ["doctor", "date", "start", "window_start", "window_end"]:
#         payload.setdefault(k, None)
   
#     if payload.get("intent") not in ["book", "other"]:
#         payload["intent"] = "other"
    
#     debug_print("BOOKING INTENT RESULT", payload, user_id)
   
#     return payload

# # ==============================
# # Utility Functions
# # ==============================
# def clear_user_memory(user_id: str):
#     """Clear user's conversation history"""
#     OptimizedRedisChatHistory(user_id).clear()
#     logger.info(f"Cleared memory: {user_id}")

# def get_user_history(user_id: str) -> list:
#     """Get user's conversation history"""
#     try:
#         memory = get_memory(user_id)
#         messages = memory.chat_memory.messages
       
#         history = []
#         for i in range(0, len(messages), 2):
#             if i + 1 < len(messages):
#                 history.append({
#                     "query": messages[i].content,
#                     "response": messages[i + 1].content
#                 })
        
#         debug_print("USER HISTORY RETRIEVED", {
#             "user_id": user_id,
#             "conversation_count": len(history)
#         }, user_id)
        
#         return history
#     except Exception as e:
#         logger.error(f"Error getting user history: {e}")
#         return []

# # ==============================
# # Debug Function
# # ==============================
# def debug_redis_connection():
#     """Debug function to test Redis connection"""
#     if redis_client is None:
#         logger.error("Redis client is None - connection failed")
#         return False
    
#     try:
#         # Test basic operations
#         test_key = "test:connection"
#         redis_client.set(test_key, "test_value", ex=10)
#         value = redis_client.get(test_key)
#         redis_client.delete(test_key)
        
#         logger.info(f"✅ Redis test successful: {value}")
        
#         # List all keys (for debugging)
#         all_keys = redis_client.keys("chat:*")
#         logger.info(f"📋 Current chat keys in Redis: {len(all_keys)}")
        
#         return True
#     except Exception as e:
#         logger.error(f"❌ Redis test failed: {e}")
#         return False

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import redis
from dotenv import load_dotenv

from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from app.core.config import llm

load_dotenv()

# ==============================
# Configuration
# ==============================
MEMORY_WINDOW_SIZE = 10  # Short-term memory window
MAX_TOKEN_LIMIT = 4000
REDIS_TTL = 2592000  # 30 days for long-term storage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==============================
# Debug Helper Function
# ==============================
def debug_print(section: str, data: Any, user_id: str = ""):
    """Enhanced debug printing with clear formatting"""
    separator = "=" * 80
    logger.info(f"\n{separator}")
    logger.info(f"🔍 DEBUG [{section}] - User: {user_id[:8] if user_id else 'N/A'}...")
    logger.info(f"{separator}")
    
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str) and len(value) > 200:
                logger.info(f"  {key}: {value[:200]}... (truncated)")
            else:
                logger.info(f"  {key}: {value}")
    elif isinstance(data, list):
        logger.info(f"  Items count: {len(data)}")
        for idx, item in enumerate(data[:5]):
            logger.info(f"  [{idx}]: {item}")
        if len(data) > 5:
            logger.info(f"  ... and {len(data) - 5} more items")
    else:
        logger.info(f"  {data}")
    
    logger.info(f"{separator}\n")

# ==============================
# Redis Setup
# ==============================
def get_redis_client():
    """Create Redis client with proper error handling"""
    try:
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            logger.info(f"Connecting to Redis via URL...")
            client = redis.StrictRedis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            client.ping()
            logger.info(f"✅ Redis connected via URL")
            return client
        
        redis_host = os.getenv("REDIS_HOST", "127.0.0.1")
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        redis_db = int(os.getenv("REDIS_DB", 0))
        
        logger.info(f"Connecting to Redis at {redis_host}:{redis_port}")
        
        conn_params = {
            "host": redis_host,
            "port": redis_port,
            "db": redis_db,
            "decode_responses": True,
        }
        
        client = redis.Redis(**conn_params)
        client.ping()
        logger.info(f"✅ Redis connected successfully")
        return client
        
    except Exception as e:
        logger.error(f"❌ Redis error: {e}")
        logger.warning("⚠️ Using in-memory fallback")
        return None

redis_client = get_redis_client()

# In-memory fallback storage
_memory_store = {}

# ==============================
# Long-Term Memory Storage (Redis)
# ==============================
class LongTermMemoryStore:
    """Redis-backed long-term memory for complete conversation history"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.history_key = f"longterm:chat:{user_id}"
        self.context_key = f"longterm:context:{user_id}"
        self.metadata_key = f"longterm:meta:{user_id}"
        self.use_redis = redis_client is not None
    
    def save_interaction(self, query: str, response: str, context: str, metadata: dict = None):
        """Save complete interaction to long-term storage"""
        interaction = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "response": response,
            "context": context[:5000],  # Store first 5000 chars of context
            "metadata": metadata or {}
        }
        
        if self.use_redis:
            try:
                # Store in Redis list
                redis_client.rpush(self.history_key, json.dumps(interaction))
                redis_client.expire(self.history_key, REDIS_TTL)
                
                # Update metadata
                meta = {
                    "user_id": self.user_id,
                    "last_interaction": datetime.utcnow().isoformat(),
                    "total_interactions": redis_client.llen(self.history_key)
                }
                redis_client.set(self.metadata_key, json.dumps(meta), ex=REDIS_TTL)
                
                debug_print("LONG-TERM MEMORY SAVED", {
                    "user_id": self.user_id,
                    "total_stored": meta["total_interactions"],
                    "storage": "Redis"
                }, self.user_id)
                return True
            except Exception as e:
                logger.error(f"Failed to save to Redis: {e}")
                self.use_redis = False
        
        # In-memory fallback
        if self.history_key not in _memory_store:
            _memory_store[self.history_key] = []
        _memory_store[self.history_key].append(interaction)
        
        debug_print("LONG-TERM MEMORY SAVED", {
            "user_id": self.user_id,
            "total_stored": len(_memory_store[self.history_key]),
            "storage": "In-Memory"
        }, self.user_id)
        return True
    
    def get_full_history(self, limit: int = None) -> List[Dict]:
        """Retrieve full conversation history from long-term storage"""
        if self.use_redis:
            try:
                if limit:
                    raw = redis_client.lrange(self.history_key, -limit, -1)
                else:
                    raw = redis_client.lrange(self.history_key, 0, -1)
                
                history = [json.loads(item) for item in raw]
                
                debug_print("LONG-TERM MEMORY LOADED", {
                    "user_id": self.user_id,
                    "items_loaded": len(history),
                    "storage": "Redis"
                }, self.user_id)
                
                return history
            except Exception as e:
                logger.error(f"Failed to load from Redis: {e}")
                self.use_redis = False
        
        # In-memory fallback
        if self.history_key not in _memory_store:
            return []
        
        history = _memory_store[self.history_key]
        if limit:
            history = history[-limit:]
        
        debug_print("LONG-TERM MEMORY LOADED", {
            "user_id": self.user_id,
            "items_loaded": len(history),
            "storage": "In-Memory"
        }, self.user_id)
        
        return history
    
    def get_recent_context(self, n: int = 5) -> str:
        """Get recent conversation context for continuity"""
        history = self.get_full_history(limit=n)
        
        context_parts = []
        for item in history:
            context_parts.append(f"User: {item['query']}")
            context_parts.append(f"Assistant: {item['response']}")
        
        return "\n".join(context_parts) if context_parts else "No previous conversation."
    
    def clear(self):
        """Clear long-term memory"""
        if self.use_redis:
            try:
                redis_client.delete(self.history_key, self.context_key, self.metadata_key)
                logger.info(f"Cleared long-term memory for {self.user_id}")
                return
            except Exception as e:
                logger.error(f"Failed to clear Redis: {e}")
        
        # In-memory fallback
        if self.history_key in _memory_store:
            del _memory_store[self.history_key]
        logger.info(f"Cleared in-memory long-term storage for {self.user_id}")

# ==============================
# Short-Term Memory (LangChain Buffer)
# ==============================
class HybridChatHistory(BaseChatMessageHistory):
    """Hybrid chat history: Short-term buffer + Long-term Redis"""
    
    def __init__(self, user_id: str, window_size: int = MEMORY_WINDOW_SIZE):
        self.user_id = user_id
        self.window_size = window_size
        self.buffer_key = f"shortterm:buffer:{user_id}"
        self.use_redis = redis_client is not None
        self.long_term = LongTermMemoryStore(user_id)
        
    @property
    def messages(self):
        """Get messages from short-term buffer or load from long-term"""
        messages = []
        
        # Try to load from short-term buffer (Redis)
        if self.use_redis:
            try:
                raw = redis_client.lrange(self.buffer_key, -self.window_size * 2, -1)
                for msg_str in raw:
                    msg = json.loads(msg_str)
                    if msg["type"] == "human":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg["type"] == "ai":
                        messages.append(AIMessage(content=msg["content"]))
                
                if messages:
                    debug_print("SHORT-TERM BUFFER LOADED", {
                        "user_id": self.user_id,
                        "message_count": len(messages),
                        "source": "Redis Buffer"
                    }, self.user_id)
                    return messages
            except Exception as e:
                logger.error(f"Short-term buffer read error: {e}")
                self.use_redis = False
        
        # If buffer is empty, try to restore from long-term memory
        if not messages:
            logger.info(f"🔄 Short-term buffer empty, restoring from long-term memory...")
            history = self.long_term.get_full_history(limit=self.window_size)
            
            for item in history:
                messages.append(HumanMessage(content=item["query"]))
                messages.append(AIMessage(content=item["response"]))
            
            # Repopulate short-term buffer
            if messages and self.use_redis:
                try:
                    for msg in messages:
                        msg_dict = {
                            "type": "human" if isinstance(msg, HumanMessage) else "ai",
                            "content": msg.content,
                            "ts": datetime.utcnow().isoformat()
                        }
                        redis_client.rpush(self.buffer_key, json.dumps(msg_dict))
                    redis_client.ltrim(self.buffer_key, -self.window_size * 2, -1)
                    redis_client.expire(self.buffer_key, 3600)  # 1 hour TTL for buffer
                    logger.info(f"✅ Short-term buffer restored with {len(messages)} messages")
                except Exception as e:
                    logger.error(f"Failed to restore buffer: {e}")
            
            debug_print("SHORT-TERM BUFFER RESTORED", {
                "user_id": self.user_id,
                "message_count": len(messages),
                "source": "Long-Term Memory"
            }, self.user_id)
        
        return messages
    
    def add_message(self, message: BaseMessage):
        """Add message to short-term buffer"""
        msg_dict = {
            "type": "human" if isinstance(message, HumanMessage) else "ai",
            "content": message.content,
            "ts": datetime.utcnow().isoformat()
        }
        
        if self.use_redis:
            try:
                pipe = redis_client.pipeline()
                pipe.rpush(self.buffer_key, json.dumps(msg_dict))
                pipe.ltrim(self.buffer_key, -self.window_size * 2, -1)
                pipe.expire(self.buffer_key, 3600)  # 1 hour TTL
                pipe.execute()
                
                debug_print("SHORT-TERM BUFFER UPDATED", {
                    "type": msg_dict["type"],
                    "content_preview": msg_dict["content"][:100]
                }, self.user_id)
                return
            except Exception as e:
                logger.error(f"Buffer write error: {e}")
                self.use_redis = False
        
        # In-memory fallback
        if self.buffer_key not in _memory_store:
            _memory_store[self.buffer_key] = []
        
        _memory_store[self.buffer_key].append(msg_dict)
        if len(_memory_store[self.buffer_key]) > self.window_size * 2:
            _memory_store[self.buffer_key] = _memory_store[self.buffer_key][-self.window_size * 2:]
    
    def clear(self):
        """Clear short-term buffer only"""
        if self.use_redis:
            try:
                redis_client.delete(self.buffer_key)
                logger.info(f"Cleared short-term buffer: {self.buffer_key}")
                return
            except Exception as e:
                logger.error(f"Buffer delete error: {e}")
        
        if self.buffer_key in _memory_store:
            del _memory_store[self.buffer_key]
            logger.info(f"Cleared in-memory buffer: {self.buffer_key}")

# ==============================
# Memory Management Functions
# ==============================
def get_memory(user_id: str) -> ConversationBufferWindowMemory:
    """Get short-term conversation buffer memory"""
    chat_history = HybridChatHistory(user_id, window_size=MEMORY_WINDOW_SIZE)
    
    return ConversationBufferWindowMemory(
        chat_memory=chat_history,
        k=MEMORY_WINDOW_SIZE,
        return_messages=True,
        memory_key="history"
    )

def load_memory_context(user_id: str) -> Dict[str, str]:
    """Load recent history from buffer and older context from long-term"""
    try:
        # Get short-term memory
        memory = get_memory(user_id)
        vars = memory.load_memory_variables({})
        messages = vars.get("history", [])
        
        # Format recent messages
        recent = []
        for msg in messages[-MEMORY_WINDOW_SIZE * 2:]:
            prefix = "User" if isinstance(msg, HumanMessage) else "Assistant"
            recent.append(f"{prefix}: {msg.content}")
        
        # Get older context from long-term memory if short-term is limited
        long_term = LongTermMemoryStore(user_id)
        older_context = ""
        
        total_interactions = long_term.get_full_history(limit=1)
        if len(messages) < 4 and len(total_interactions) > 0:
            # Load older context for continuity
            older_context = long_term.get_recent_context(n=5)
        
        result = {
            "recent_history": "\n".join(recent) or "No recent conversation.",
            "older_context": older_context,
            "total_interactions": len(long_term.get_full_history())
        }
        
        debug_print("MEMORY CONTEXT LOADED", {
            "user_id": user_id,
            "recent_messages": len(messages),
            "has_older_context": bool(older_context),
            "total_interactions": result["total_interactions"]
        }, user_id)
        
        return result
    except Exception as e:
        logger.error(f"Error loading memory context: {e}")
        return {
            "recent_history": "No recent conversation.",
            "older_context": "",
            "total_interactions": 0
        }

def save_to_memory(user_id: str, query: str, response: str, context: str = ""):
    """Save interaction to both short-term buffer and long-term storage"""
    try:
        # Save to short-term buffer
        memory = get_memory(user_id)
        memory.save_context({"input": query}, {"output": response})
        
        # Save to long-term storage with context
        long_term = LongTermMemoryStore(user_id)
        long_term.save_interaction(
            query=query,
            response=response,
            context=context,
            metadata={"timestamp": datetime.utcnow().isoformat()}
        )
        
        debug_print("SAVED TO HYBRID MEMORY", {
            "user_id": user_id,
            "query_preview": query[:100],
            "response_preview": response[:100],
            "saved_to": "Short-term Buffer + Long-term Storage"
        }, user_id)
    except Exception as e:
        logger.error(f"Error saving to memory: {e}")

# ==============================
# Context Formatting
# ==============================
def format_context_for_llm(retrieved_context: Union[List, str]) -> str:
    """Format retrieved context into readable text for LLM"""
    if not retrieved_context:
        return "No relevant information found."
    
    if isinstance(retrieved_context, str):
        return retrieved_context
    
    formatted_parts = []
    
    for idx, item in enumerate(retrieved_context, 1):
        if not isinstance(item, dict):
            continue
            
        source_text = item.get('source_text', '')
        source = item.get('source', 'Unknown')
        score = item.get('score', 0)
        
        # Handle page_content format
        if not source_text and 'source' in item:
            raw_source = item.get('source', '')
            if isinstance(raw_source, str) and 'page_content=' in raw_source:
                try:
                    source_text = raw_source.split("page_content='")[1].split("' metadata=")[0]
                except:
                    source_text = raw_source
        
        if source_text:
            formatted_parts.append(f"[Source {idx}] (Relevance: {score:.2f})\n{source_text}\n")
    
    result = "\n".join(formatted_parts) if formatted_parts else "No detailed information available."
    
    debug_print("FORMATTED CONTEXT", {
        "original_items": len(retrieved_context) if isinstance(retrieved_context, list) else "string",
        "formatted_length": len(result),
        "preview": result[:300]
    })
    
    return result

# ==============================
# LangGraph State
# ==============================
class ConversationState(TypedDict):
    user_id: str
    query: str
    context: str
    recent_history: str
    older_context: str
    response: str

# ==============================
# Prompt Templates
# ==============================
SYSTEM_PROMPT = """
You are a retrieval-augmented medical assistant chatbot for Indo-US Multi-Specialty Hospital.

Your responsibilities:
1. **Extract and use information** from the retriever context, including structured data like doctor lists, schedules, and specialties.
2. When asked for lists (e.g., "top 5 female doctors"), extract and present the information directly from the context.
3. Use conversation history to maintain continuity and avoid repetition.
4. If a booking was already confirmed, acknowledge that instead of reconfirming.
5. Be concise, clear, and human-like (≤50 words for simple queries, longer for lists/details).
6. If information is missing from context, ask for clarification.
7. Never fabricate data. If unsure, politely say so.

**Important**: When the context contains doctor information with names, specialties, contact details, or schedules, extract and present it directly in a clear, formatted way.
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", """Retrieved Information:
{context}

Recent Conversation (last few exchanges):
{recent_history}

Older Conversation Context:
{older_context}

User query: {query}

INSTRUCTIONS:
- If the Retrieved Information contains structured data (like doctor lists), extract and format it clearly
- For "list" requests, provide the information directly if available in context
- Use Older Conversation Context for background understanding
- Use Recent Conversation for immediate context
- Do NOT repeat earlier responses verbatim
- If booking was confirmed earlier, acknowledge it
- If information is incomplete, ask specific questions
- Format doctor information clearly with name, specialty, and relevant details
""")
])

# ==============================
# Workflow Nodes
# ==============================
def load_context_node(state: ConversationState) -> ConversationState:
    """Load memory context from hybrid storage"""
    mem_ctx = load_memory_context(state["user_id"])
    state["recent_history"] = mem_ctx["recent_history"]
    state["older_context"] = mem_ctx.get("older_context", "")
    
    debug_print("LOAD CONTEXT NODE", {
        "user_id": state["user_id"],
        "recent_history_length": len(state["recent_history"]),
        "has_older_context": bool(state["older_context"]),
        "total_interactions": mem_ctx.get("total_interactions", 0)
    }, state["user_id"])
    
    return state

def generate_response_node(state: ConversationState) -> ConversationState:
    """Generate LLM response using qa_prompt"""
    formatted = qa_prompt.format(
        context=state["context"],
        recent_history=state["recent_history"],
        older_context=state["older_context"] or "No older context.",
        query=state["query"]
    )
    
    debug_print("PROMPT SENT TO LLM", {
        "user_id": state["user_id"],
        "formatted_prompt": formatted[:500] + "..." if len(formatted) > 500 else formatted
    }, state["user_id"])
    
    raw = llm.predict(formatted)
    state["response"] = str(raw).strip()
    
    debug_print("LLM RESPONSE GENERATED", {
        "user_id": state["user_id"],
        "response": state["response"]
    }, state["user_id"])
    
    return state

def save_memory_node(state: ConversationState) -> ConversationState:
    """Save to hybrid memory system"""
    save_to_memory(
        state["user_id"], 
        state["query"], 
        state["response"],
        state["context"]
    )
    return state

# ==============================
# Workflow Creation
# ==============================
def create_workflow() -> StateGraph:
    """Create optimized conversation workflow"""
    workflow = StateGraph(ConversationState)
    
    workflow.add_node("load_context", load_context_node)
    workflow.add_node("generate", generate_response_node)
    workflow.add_node("save", save_memory_node)
    
    workflow.set_entry_point("load_context")
    workflow.add_edge("load_context", "generate")
    workflow.add_edge("generate", "save")
    workflow.add_edge("save", END)
    
    return workflow.compile()

conversation_flow = create_workflow()

# ==============================
# Main Handler
# ==============================
def handle_user_query(user_id: str, retrieved_context: Union[str, List, Dict], query: str) -> str:
    """
    Main query handler with hybrid memory system.
    
    Args:
        user_id: Unique user identifier
        retrieved_context: Context from vector DB
        query: User's question
    
    Returns:
        AI response string
    """
    debug_print("INCOMING REQUEST", {
        "user_id": user_id,
        "query": query,
        "context_type": type(retrieved_context).__name__
    }, user_id)
    
    if not user_id or not query:
        return "Invalid request."
    
    # Format context properly for LLM
    if isinstance(retrieved_context, list):
        formatted_context = format_context_for_llm(retrieved_context)
    elif isinstance(retrieved_context, dict):
        formatted_context = json.dumps(retrieved_context, indent=2)
    elif isinstance(retrieved_context, str):
        formatted_context = retrieved_context
    else:
        formatted_context = str(retrieved_context)
    
    state: ConversationState = {
        "user_id": user_id,
        "query": query,
        "context": formatted_context,
        "recent_history": "",
        "older_context": "",
        "response": ""
    }
    
    final = conversation_flow.invoke(state)
    
    debug_print("FINAL RESPONSE", {
        "user_id": user_id,
        "query": query,
        "response": final["response"]
    }, user_id)
    
    return final["response"]

# ==============================
# Booking Intent Detection
# ==============================
def detect_booking_intent_and_fields(user_id: str, retrieved_context: Union[str, List, Dict], query: str) -> dict:
    """Detect booking intent with memory context"""
    debug_print("BOOKING INTENT DETECTION", {
        "user_id": user_id,
        "query": query
    }, user_id)
    
    # Format context
    if isinstance(retrieved_context, list):
        formatted_context = format_context_for_llm(retrieved_context)
    elif isinstance(retrieved_context, dict):
        formatted_context = json.dumps(retrieved_context, indent=2)
    else:
        formatted_context = str(retrieved_context)
    
    mem_ctx = load_memory_context(user_id)
    
    system = (
        "You are a JSON extractor for appointment bookings.\n"
        "Return ONLY valid JSON. No markdown, no prose."
    )
    
    schema = '''{
  "intent": "book" | "other",
  "doctor": string | null,
  "date": "YYYY-MM-DD" | null,
  "start": "HH:MM" | null,
  "duration_min": number | null,
  "window_start": "HH:MM" | null,
  "window_end": "HH:MM" | null
}'''
    
    user_prompt = f"""Context:
{formatted_context[:1000]}

Recent History:
{mem_ctx['recent_history']}

Older Context:
{mem_ctx.get('older_context', '')}

User: {query}

Extract booking intent and fields as JSON:
{schema}

Rules:
- intent='book' only if clearly requesting appointment
- Convert AM/PM to 24h format
- Use null for unknown fields
- Default duration_min to 30 if not specified"""
    
    raw = llm.predict(f"[SYSTEM]\n{system}\n\n[USER]\n{user_prompt}").strip()
    raw = raw.strip("`").replace("```json", "").replace("```", "").strip()
    
    try:
        payload = json.loads(raw)
    except:
        logger.error(f"JSON parse failed: {raw}")
        payload = {"intent": "other"}
    
    # Set defaults
    payload.setdefault("duration_min", 30)
    for k in ["doctor", "date", "start", "window_start", "window_end"]:
        payload.setdefault(k, None)
    
    if payload.get("intent") not in ["book", "other"]:
        payload["intent"] = "other"
    
    debug_print("BOOKING INTENT RESULT", payload, user_id)
    
    return payload

# ==============================
# Utility Functions
# ==============================
def clear_user_memory(user_id: str, clear_long_term: bool = False):
    """
    Clear user's conversation history
    
    Args:
        user_id: User identifier
        clear_long_term: If True, also clear long-term storage (default: False)
    """
    # Clear short-term buffer
    HybridChatHistory(user_id).clear()
    logger.info(f"Cleared short-term memory for: {user_id}")
    
    # Optionally clear long-term storage
    if clear_long_term:
        LongTermMemoryStore(user_id).clear()
        logger.info(f"Cleared long-term memory for: {user_id}")

def get_user_history(user_id: str, full_history: bool = False) -> list:
    """
    Get user's conversation history
    
    Args:
        user_id: User identifier
        full_history: If True, return complete long-term history
    
    Returns:
        List of conversation exchanges
    """
    try:
        if full_history:
            # Get from long-term storage
            long_term = LongTermMemoryStore(user_id)
            raw_history = long_term.get_full_history()
            
            history = []
            for item in raw_history:
                history.append({
                    "query": item["query"],
                    "response": item["response"],
                    "timestamp": item["timestamp"],
                    "context_preview": item.get("context", "")[:200]
                })
        else:
            # Get from short-term buffer
            memory = get_memory(user_id)
            messages = memory.chat_memory.messages
            
            history = []
            for i in range(0, len(messages), 2):
                if i + 1 < len(messages):
                    history.append({
                        "query": messages[i].content,
                        "response": messages[i + 1].content
                    })
        
        debug_print("USER HISTORY RETRIEVED", {
            "user_id": user_id,
            "conversation_count": len(history),
            "source": "Long-term" if full_history else "Short-term"
        }, user_id)
        
        return history
    except Exception as e:
        logger.error(f"Error getting user history: {e}")
        return []

def get_memory_stats(user_id: str) -> dict:
    """Get memory statistics for a user"""
    try:
        long_term = LongTermMemoryStore(user_id)
        full_history = long_term.get_full_history()
        
        memory = get_memory(user_id)
        short_term_messages = memory.chat_memory.messages
        
        stats = {
            "user_id": user_id,
            "short_term_messages": len(short_term_messages),
            "long_term_interactions": len(full_history),
            "oldest_interaction": full_history[0]["timestamp"] if full_history else None,
            "latest_interaction": full_history[-1]["timestamp"] if full_history else None
        }
        
        return stats
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        return {"error": str(e)}

# ==============================
# Debug Function
# ==============================
def debug_redis_connection():
    """Debug function to test Redis connection"""
    if redis_client is None:
        logger.error("Redis client is None - connection failed")
        return False
    
    try:
        # Test basic operations
        test_key = "test:connection"
        redis_client.set(test_key, "test_value", ex=10)
        value = redis_client.get(test_key)
        redis_client.delete(test_key)
        
        logger.info(f"✅ Redis test successful: {value}")
        
        # List all keys
        short_term_keys = redis_client.keys("shortterm:*")
        long_term_keys = redis_client.keys("longterm:*")
        
        logger.info(f"📋 Short-term buffer keys: {len(short_term_keys)}")
        logger.info(f"📋 Long-term storage keys: {len(long_term_keys)}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Redis test failed: {e}")
        return False