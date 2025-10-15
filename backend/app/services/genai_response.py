import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import redis
from dotenv import load_dotenv

from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from app.core.config import llm

load_dotenv()

# ==============================
# Configuration
# ==============================
MEMORY_WINDOW_SIZE = 10
MAX_TOKEN_LIMIT = 4000
REDIS_TTL = 86400  # 24 hours

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================
# Redis Setup with Multiple Connection Methods
# ==============================
def get_redis_client():
    """Create Redis client with proper error handling - supports URL and username/password"""
    try:
        # Method 1: Try REDIS_URL first (preferred for cloud services)
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
            logger.info(f"   Host: {client.connection_pool.connection_kwargs.get('host')}")
            logger.info(f"   Port: {client.connection_pool.connection_kwargs.get('port')}")
            return client
        
        # Method 2: Try username/password authentication
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        redis_db = int(os.getenv("REDIS_DB", 0))
        redis_user = os.getenv("REDIS_USER")
        redis_password = os.getenv("REDIS_PASS")
        
        logger.info(f"Connecting to Redis at {redis_host}:{redis_port}")
        
        # Build connection parameters
        conn_params = {
            "host": redis_host,
            "port": redis_port,
            "db": redis_db,
            "decode_responses": True,
            "socket_connect_timeout": 5,
            "socket_timeout": 5
        }
        
        # Add authentication if provided
        if redis_user:
            conn_params["username"] = redis_user
            logger.info(f"   Using username: {redis_user}")
        
        if redis_password and redis_password.strip():
            conn_params["password"] = redis_password
            logger.info(f"   Using password authentication")
        
        client = redis.Redis(**conn_params)
        
        # Test connection
        client.ping()
        logger.info(f"✅ Redis connected successfully")
        logger.info(f"   Host: {redis_host}")
        logger.info(f"   Port: {redis_port}")
        logger.info(f"   DB: {redis_db}")
        return client
        
    except redis.ConnectionError as e:
        logger.error(f"❌ Redis connection failed: {e}")
        logger.warning("⚠️ Using in-memory fallback")
        return None
    except redis.AuthenticationError as e:
        logger.error(f"❌ Redis authentication failed: {e}")
        logger.warning("⚠️ Check REDIS_USER and REDIS_PASSWORD in .env file")
        logger.warning("⚠️ Using in-memory fallback")
        return None
    except Exception as e:
        logger.error(f"❌ Redis error: {e}")
        logger.warning("⚠️ Using in-memory fallback")
        return None

redis_client = get_redis_client()

# In-memory fallback storage
_memory_store = {}

# ==============================
# Custom Redis Chat History with Fallback
# ==============================
class OptimizedRedisChatHistory(BaseChatMessageHistory):
    """Optimized Redis-backed chat history with in-memory fallback"""
   
    def __init__(self, session_id: str, window_size: int = MEMORY_WINDOW_SIZE):
        self.session_id = session_id
        self.key = f"chat:{session_id}"
        self.window_size = window_size
        self.use_redis = redis_client is not None
   
    @property
    def messages(self):
        """Get messages within window"""
        if self.use_redis:
            try:
                raw = redis_client.lrange(self.key, -self.window_size * 2, -1)
                messages = []
                for msg_str in raw:
                    msg = json.loads(msg_str)
                    if msg["type"] == "human":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg["type"] == "ai":
                        messages.append(AIMessage(content=msg["content"]))
                logger.debug(f"Loaded {len(messages)} messages from Redis for {self.session_id[:8]}")
                return messages
            except Exception as e:
                logger.error(f"Redis read error: {e}, falling back to memory")
                self.use_redis = False
        
        # In-memory fallback
        if self.key not in _memory_store:
            _memory_store[self.key] = []
        
        raw = _memory_store[self.key][-self.window_size * 2:]
        messages = []
        for msg in raw:
            if msg["type"] == "human":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["type"] == "ai":
                messages.append(AIMessage(content=msg["content"]))
        logger.debug(f"Loaded {len(messages)} messages from memory for {self.session_id[:8]}")
        return messages
   
    def add_message(self, message: BaseMessage):
        """Add message with automatic trimming"""
        msg_dict = {
            "type": "human" if isinstance(message, HumanMessage) else "ai",
            "content": message.content,
            "ts": datetime.utcnow().isoformat()
        }
       
        if self.use_redis:
            try:
                pipe = redis_client.pipeline()
                pipe.rpush(self.key, json.dumps(msg_dict))
                pipe.ltrim(self.key, -self.window_size * 2, -1)
                pipe.expire(self.key, REDIS_TTL)
                result = pipe.execute()
                logger.debug(f"Saved message to Redis: {self.key}")
                return
            except Exception as e:
                logger.error(f"Redis write error: {e}, falling back to memory")
                self.use_redis = False
        
        # In-memory fallback
        if self.key not in _memory_store:
            _memory_store[self.key] = []
        
        _memory_store[self.key].append(msg_dict)
        
        # Trim to window size
        if len(_memory_store[self.key]) > self.window_size * 2:
            _memory_store[self.key] = _memory_store[self.key][-self.window_size * 2:]
        
        logger.debug(f"Saved message to memory: {self.key}")
   
    def clear(self):
        """Clear history"""
        if self.use_redis:
            try:
                redis_client.delete(self.key)
                logger.info(f"Cleared Redis key: {self.key}")
                return
            except Exception as e:
                logger.error(f"Redis delete error: {e}")
        
        # In-memory fallback
        if self.key in _memory_store:
            del _memory_store[self.key]
            logger.info(f"Cleared memory key: {self.key}")

# ==============================
# Memory Management
# ==============================
def get_memory(user_id: str) -> ConversationSummaryBufferMemory:
    """Get or create conversation memory with auto-summarization"""
    chat_history = OptimizedRedisChatHistory(user_id)
   
    return ConversationSummaryBufferMemory(
        llm=llm,
        chat_memory=chat_history,
        max_token_limit=MAX_TOKEN_LIMIT,
        return_messages=True,
        memory_key="history"
    )

def load_memory_context(user_id: str) -> Dict[str, str]:
    """Load recent history and summary"""
    try:
        memory = get_memory(user_id)
        vars = memory.load_memory_variables({})
       
        messages = vars.get("history", [])
        recent = []
        for msg in messages[-MEMORY_WINDOW_SIZE * 2:]:
            prefix = "User" if isinstance(msg, HumanMessage) else "Bot"
            recent.append(f"{prefix}: {msg.content}")
       
        return {
            "recent_history": "\n".join(recent) or "No recent conversation.",
            "summary": getattr(memory, 'moving_summary_buffer', "")
        }
    except Exception as e:
        logger.error(f"Error loading memory context: {e}")
        return {
            "recent_history": "No recent conversation.",
            "summary": ""
        }

def save_to_memory(user_id: str, query: str, response: str):
    """Save interaction to memory"""
    try:
        memory = get_memory(user_id)
        memory.save_context({"input": query}, {"output": response})
        logger.info(f"Saved: user={user_id[:8]}...")
    except Exception as e:
        logger.error(f"Error saving to memory: {e}")

# ==============================
# LangGraph State
# ==============================
class ConversationState(TypedDict):
    user_id: str
    query: str
    context: str
    recent_history: str
    summary: str
    response: str

# ==============================
# Prompt Templates
# ==============================
SYSTEM_PROMPT = """
You are a retrieval-augmented medical assistant chatbot.

Your responsibilities:
1. Use ONLY the retriever context, chat history, and the user's latest query.
2. Avoid repeating responses already given in prior conversation history; if you must repeat necessary info, rephrase it.
3. If a booking was already confirmed for the same doctor, acknowledge that instead of reconfirming.
4. Be concise, clear, and human-like (≤40 words).
5. If doctor, time, or information is missing, ask for clarification.
6. Never fabricate data. If unsure, politely say so.
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", """Retriever context:
{context}

Recent Conversation (last {window_size} exchanges):
{recent_history}

Memory Summary (older context):
{summary}

User query:
{query}

INSTRUCTIONS:
- Use Memory Summary for background context from earlier conversations.
- Use Recent Conversation for immediate context.
- Do NOT repeat earlier bot responses verbatim; rephrase or provide new clarifying info when possible.
- If booking was already confirmed with the doctor, say so directly.
- If slot unavailable, provide available slots from the context.
- If clarification is needed, ask a short direct question.
""")
])

# ==============================
# Workflow Nodes
# ==============================
def load_context_node(state: ConversationState) -> ConversationState:
    """Load memory context"""
    mem_ctx = load_memory_context(state["user_id"])
    state["recent_history"] = mem_ctx["recent_history"]
    state["summary"] = mem_ctx["summary"]
    logger.info(f"Context loaded: {state['user_id'][:8]}...")
    return state

def generate_response_node(state: ConversationState) -> ConversationState:
    """Generate LLM response using qa_prompt"""
    formatted = qa_prompt.format(
        context=state["context"],
        window_size=MEMORY_WINDOW_SIZE,
        recent_history=state["recent_history"],
        summary=state["summary"] or "No previous context.",
        query=state["query"]
    )
   
    raw = llm.predict(formatted)
    state["response"] = str(raw).strip()
    logger.info(f"Response generated: {state['user_id'][:8]}...")
    return state

def save_memory_node(state: ConversationState) -> ConversationState:
    """Save to memory"""
    save_to_memory(state["user_id"], state["query"], state["response"])
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
def handle_user_query(user_id: str, retrieved_context: str, query: str) -> str:
    """
    Main query handler with optimized memory.
   
    Args:
        user_id: Unique user identifier
        retrieved_context: Context from vector DB
        query: User's question
   
    Returns:
        AI response string
    """
    if not user_id or not query:
        return "Invalid request."
   
    if not isinstance(retrieved_context, str):
        retrieved_context = json.dumps(retrieved_context)
   
    state: ConversationState = {
        "user_id": user_id,
        "query": query,
        "context": retrieved_context,
        "recent_history": "",
        "summary": "",
        "response": ""
    }
   
    final = conversation_flow.invoke(state)
    return final["response"]

# ==============================
# Booking Intent Detection (Optimized)
# ==============================
def detect_booking_intent_and_fields(user_id: str, retrieved_context: str, query: str) -> dict:
    """Detect booking intent with memory context"""
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
{retrieved_context}

Recent History:
{mem_ctx['recent_history']}

Summary:
{mem_ctx['summary']}

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
   
    return payload

# ==============================
# Utility Functions
# ==============================
def clear_user_memory(user_id: str):
    """Clear user's conversation history"""
    OptimizedRedisChatHistory(user_id).clear()
    logger.info(f"Cleared memory: {user_id}")

def get_user_history(user_id: str) -> list:
    """Get user's conversation history"""
    try:
        memory = get_memory(user_id)
        messages = memory.chat_memory.messages
       
        history = []
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                history.append({
                    "query": messages[i].content,
                    "response": messages[i + 1].content
                })
        return history
    except Exception as e:
        logger.error(f"Error getting user history: {e}")
        return []

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
        
        # List all keys (for debugging)
        all_keys = redis_client.keys("chat:*")
        logger.info(f"📋 Current chat keys in Redis: {len(all_keys)}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Redis test failed: {e}")
        return False