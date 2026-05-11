from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import time
import asyncio

# Import your advanced RAG components
from retriever import HybridRetriever
from agent import AssessmentAgent

# Initialize FastAPI app
app = FastAPI(title="SHL Conversational Assessment Recommender")

# Add CORS Middleware to ensure the grading script/frontend can reach your API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

# Load models globally so they don't reload on every request (Crucial for 30s timeout limit)
print("Initializing Agent and Retriever...")
retriever = HybridRetriever()
agent = AssessmentAgent()
print("System Ready.")

# --- Pydantic Schemas for Strict Validation ---
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
    """Returns status ok to allow Render/deployment cold-start wakeups."""
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Takes full conversation history, returns reply + recommendations."""
    try:
        # Convert Pydantic models to standard dictionaries for the Groq API
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Count user turns to enforce the 8-turn hard cap and Turn 1 behavior
        turn_count = sum(1 for msg in messages if msg["role"] == "user")
        
        # 1. Pipeline Stage 1: Analyze & Extract (MIL & HyDE)
        analysis = await agent.analyze_conversation(messages)
        intent = analysis.get("intent", "CLARIFY")
        
        # Hard Constraint: Force Turn 1 to CLARIFY to prevent eager hallucinations
        if turn_count == 1:
            intent = "CLARIFY"
            
        curated_docs = []
        
        # Only trigger retrieval if the agent has decided to recommend or compare
        if intent in ["RECOMMEND", "COMPARE"]:
            # 2. Pipeline Stage 2: Wide-Net Hybrid Retrieval
            # CRITICAL FIX: Increased top_k to 50 based on MIL pooling research
            candidates = retriever.retrieve(analysis, top_k=50)
            
            if candidates:
                # 3. Pipeline Stage 3: LLM Precision Reranking (Top 10)
                curated_docs = await agent.llm_rerank(messages, candidates)
        
        # 4. Pipeline Stage 4: Generate Strict JSON Response
        raw_response = await agent.generate_response(messages, intent, curated_docs[:10], turn_count)
        
        # 5. Enforce Hard Constraints & Formatting
        # End if we hit the turn limit, OR if we just successfully delivered the final recommendations
        is_end_of_conversation = (turn_count >= 8) or (intent == "RECOMMEND")
        
        # Map to the strict output schema
        final_response = ChatResponse(
            reply=raw_response.get("reply", "I'm sorry, I encountered an error formatting my response."),
            recommendations=raw_response.get("recommendations", []),
            end_of_conversation=is_end_of_conversation
        )
        
        return final_response

    except Exception as e:
        print(f"Server Error: {e}")
        # Return a safe fallback rather than crashing the evaluator
        return ChatResponse(
            reply="I'm sorry, I'm having trouble processing that right now. Could you rephrase your request?",
            recommendations=[],
            end_of_conversation=False
        )