from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
import os
import psutil

# 1. Immediate Memory Check
print("--- SYSTEM CHECK ---")
process = psutil.Process(os.getpid())
print(f"Initial Memory Usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

# Define global placeholders
retriever = None
agent = None

# 2. The Lifespan Manager (The Secret Sauce for 512MB RAM)
@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, agent
    print("--- STARTUP: Initializing AI Components ---")
    
    # We import and initialize INSIDE here so the port opens first
    from retriever import HybridRetriever
    from agent import AssessmentAgent
    
    retriever = HybridRetriever()
    agent = AssessmentAgent()
    
    mem = process.memory_info().rss / 1024 / 1024
    print(f"--- STARTUP COMPLETE: Models Loaded. Memory: {mem:.2f} MB ---")
    yield
    print("--- SHUTTING DOWN ---")

# 3. Initialize FastAPI with the lifespan
app = FastAPI(title="SHL Conversational Assessment Recommender", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Schemas (Keep as you had them) ---
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool

# --- API Endpoints ---

@app.get("/health")
async def health_check():
    # Adding a check to see if agent is actually ready
    status = "ok" if agent is not None else "initializing"
    return {"status": status}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    # Safety check: If agent isn't loaded yet, don't crash
    if agent is None or retriever is None:
        raise HTTPException(status_code=503, detail="AI Models are still loading. Please try again in 30 seconds.")

    try:
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        turn_count = sum(1 for msg in messages if msg["role"] == "user")
        
        analysis = await agent.analyze_conversation(messages)
        intent = analysis.get("intent", "CLARIFY")
        
        if turn_count == 1:
            intent = "CLARIFY"
            
        curated_docs = []
        
        if intent in ["RECOMMEND", "COMPARE"]:
            candidates = retriever.retrieve(analysis, top_k=50)
            if candidates:
                curated_docs = await agent.llm_rerank(messages, candidates)
        
        raw_response = await agent.generate_response(messages, intent, curated_docs[:10], turn_count)
        is_end_of_conversation = (turn_count >= 8) or (intent == "RECOMMEND")
        
        return ChatResponse(
            reply=raw_response.get("reply", "I'm sorry, I encountered an error formatting my response."),
            recommendations=raw_response.get("recommendations", []),
            end_of_conversation=is_end_of_conversation
        )

    except Exception as e:
        print(f"Server Error: {e}")
        return ChatResponse(
            reply="I'm sorry, I'm having trouble processing that right now.",
            recommendations=[],
            end_of_conversation=False
        )