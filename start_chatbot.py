#!/usr/bin/env python
"""
AI Chatbot - All-in-One Starter
Starts Ollama, ngrok tunnel, and the chatbot server in one script.
"""

import json
import numpy as np
import requests
import subprocess
import time
import threading
import sys
import os
import uuid
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn
from openai import OpenAI

# ============================================================
# Set working directory to script directory
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
print(f"[INFO] Working directory: {SCRIPT_DIR}")

# ============================================================
# Question Logging Configuration
# ============================================================
QUESTIONS_LOG = os.path.join(SCRIPT_DIR, 'questions_log.txt')
ACTIVE_SESSIONS = {}  # Track which sessions have been logged

print(f"[OK] Questions log file: {QUESTIONS_LOG}\n")

# ============================================================
# Configuration
# ============================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-3.5-turbo"
PORT = 8000

# Initialize OpenAI client lazily to avoid import issues
openai_client = None

def get_openai_client():
    global openai_client
    if openai_client is None:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return openai_client

# ============================================================
# FastAPI App Setup
# ============================================================
app = FastAPI(title="AI FAQ Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow ALL origins - works on any website
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "ngrok-skip-browser-warning"],
)

# Global variables (loaded at startup)
embedding_model = None
question_embeddings = None
questions = []
answers = []

class Query(BaseModel):
    question: str
    session_id: str = None  # Optional: client provides session ID for conversation tracking

class Response(BaseModel):
    answer: str
    confidence: float
    is_ai_generated: bool = False
    session_id: str = None  # Return session_id to client

def get_relevant_context(user_question: str, top_k: int = 3) -> str:
    """Retrieve relevant FAQ entries (token-efficient - top 3)"""
    user_embedding = embedding_model.encode([user_question])
    similarities = np.dot(question_embeddings, user_embedding.T).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    relevant_docs = []
    for idx in top_indices:
        if similarities[idx] > 0.2:  # Lower threshold to catch more questions
            relevant_docs.append(f"{questions[idx]}\n{answers[idx]}")
    
    if relevant_docs:
        return "\n\n".join(relevant_docs)
    return ""

def detect_question_type(user_question: str) -> str:
    """Detect the type of question to format response appropriately"""
    q_lower = user_question.lower()
    
    if any(word in q_lower for word in ['eat', 'food', 'restaurant', 'dine', 'dining', 'hungry']):
        return "food"
    elif any(word in q_lower for word in ['where', 'location', 'find', 'address', 'visit']):
        return "location"
    elif any(word in q_lower for word in ['what', 'things', 'activities', 'do', 'attractions']):
        return "activity"
    elif any(word in q_lower for word in ['how', 'many', 'cost', 'price', 'fee']):
        return "info"
    elif any(word in q_lower for word in ['best', 'good', 'recommend', 'popular', 'top']):
        return "recommendation"
    else:
        return "general"

def query_openai(prompt: str) -> str:
    """Query OpenAI API (token-optimized)"""
    try:
        print(f"[DEBUG] Calling OpenAI with model: {OPENAI_MODEL}")
        client = get_openai_client()
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are Murray. Give 1 short sentence only. Be brief. Speak with a thick Jersey accent - use Jersey dialect, expressions, and mannerisms (e.g., 'ay', 'fuhgeddabout', 'down the shore', dropping g's from words like 'gettin'', etc.)."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.7
        )
        answer = response.choices[0].message.content.strip()
        print(f"[DEBUG] OpenAI response: {answer[:100]}...")
        return answer
    except Exception as e:
        print(f"[ERROR] OpenAI API Error: {e}")
    return None

def log_question(session_id: str, user_question: str, answer: str):
    """
    Log user question and answer grouped by unique session ID
    """
    try:
        with open(QUESTIONS_LOG, 'a', encoding='utf-8') as f:
            # Write session header if this is the first message in this session
            if session_id not in ACTIVE_SESSIONS:
                ACTIVE_SESSIONS[session_id] = True
                f.write(f"\n{'='*60}\n")
                f.write(f"CONVERSATION: {session_id}\n")
                f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*60}\n\n")
            
            # Write Q and A
            f.write(f"Q: {user_question}\n")
            f.write(f"A: {answer}\n\n")
            f.flush()  # Flush to ensure data is written immediately
        
        print(f"[OK] Logged to session {session_id[:8]}")
    except Exception as e:
        print(f"[ERROR] Could not write to log: {e}")





@app.post("/ask", response_model=Response)
async def ask_question(query: Query, request: Request):
    user_question = query.question
    
    # Debug: Log all relevant headers
    print(f"\n[DEBUG] === Request Headers ===")
    print(f"  X-Forwarded-For: {request.headers.get('X-Forwarded-For', 'NOT SET')}")
    print(f"  X-Real-IP: {request.headers.get('X-Real-IP', 'NOT SET')}")
    print(f"  CF-Connecting-IP: {request.headers.get('CF-Connecting-IP', 'NOT SET')}")
    print(f"  Direct client.host: {request.client.host}")
    
    # Get real client IP (handles proxies like ngrok, Render, etc.)
    client_ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or \
                request.headers.get("X-Real-IP", "") or \
                request.headers.get("CF-Connecting-IP", "") or \
                request.client.host
    
    session_id = client_ip
    
    # Debug: Log the IP address being received
    print(f"[DEBUG] Final IP detected: {client_ip}")
    print(f"[DEBUG] Question: {user_question[:50]}\n")
    
    # Get relevant context
    relevant_context = get_relevant_context(user_question)
    question_type = detect_question_type(user_question)
    
    print(f"[DEBUG] Detected question type: {question_type}")
    print(f"[DEBUG] Found context: {len(relevant_context)} chars")
    
    # Create concise prompts that allow multiple options
    if relevant_context:
        if question_type == "food":
            prompt = f"""Based on this info, suggest 1-2 restaurants in 1-2 sentences:
{relevant_context}

Q: {user_question}
A:"""
        elif question_type == "location":
            prompt = f"""SUMMARIZE in 1-2 sentences (describe location types):
{relevant_context}

Q: {user_question}
A:"""
        elif question_type == "recommendation" or question_type == "activity":
            prompt = f"""SUMMARIZE in 1-2 sentences (describe types of places):
{relevant_context}

Q: {user_question}
A:"""
        else:
            prompt = f"""SUMMARIZE in 1-2 sentences (rephrase the information):
{relevant_context}

Q: {user_question}
A:"""
    else:
        # No FAQ found - use conversational helper agent with Meadowlands context
        prompt = f"""You are Murray, a friendly helper for the Meadowlands region in NJ. Answer helpfully in 1-2 sentences. Speak with a thick Jersey accent - use Jersey dialect, expressions, and mannerisms (e.g., 'ay', 'fuhgeddabout', 'down the shore', dropping g's from words like 'gettin'', etc.). If asked about food/restaurants, mention there are diners and restaurants near the stadium.

Q: {user_question}
A:"""
    
    # Query OpenAI
    ai_answer = query_openai(prompt)
    
    if ai_answer:
        log_question(session_id, user_question, ai_answer)
        # Use high confidence for FAQ-based answers, lower for pure conversational
        confidence = 0.95 if relevant_context else 0.85
        return Response(answer=ai_answer, confidence=confidence, is_ai_generated=True, session_id=session_id)
    
    # Only fallback to semantic search if we had FAQ context and LLM failed
    if relevant_context:
        user_embedding = embedding_model.encode([user_question])
        similarities = np.dot(question_embeddings, user_embedding.T).flatten()
        best_idx = np.argmax(similarities)
        best_sim = float(similarities[best_idx])
        
        if best_sim >= 0.5:
            log_question(session_id, user_question, answers[best_idx])
            return Response(answer=answers[best_idx], confidence=best_sim, is_ai_generated=False, session_id=session_id)
    
    # No FAQ match and LLM failed - return Murray's fallback
    no_answer = "Hi there! I'm Murray. I'm here to help, but I didn't quite understand that. Can you ask me about things to do or places to visit?"
    log_question(session_id, user_question, no_answer)
    return Response(
        answer=no_answer,
        confidence=0.5,
        is_ai_generated=True,
        session_id=session_id
    )


@app.get("/")
def root():
    return {"message": "AI FAQ Chatbot is running", "model": OPENAI_MODEL}

@app.get("/test_client.html")
def serve_test_client():
    """Serve the test client HTML"""
    client_path = os.path.join(SCRIPT_DIR, "test_client.html")
    if os.path.exists(client_path):
        return FileResponse(client_path, media_type="text/html")
    return {"error": "Test client not found"}

@app.get("/health")
def health():
    return {"status": "healthy", "openai": "configured" if OPENAI_API_KEY else "not_configured"}

# ============================================================
# Startup Functions
# ============================================================

def check_openai_api():
    """Check if OpenAI API key is set"""
    try:
        if OPENAI_API_KEY:
            print(f"[OK] OpenAI API key is configured")
            return True
        print(f"[ERROR] OpenAI API key not found")
        return False
    except Exception as e:
        print(f"[ERROR] Could not validate OpenAI API: {e}")
        return False

def load_faq_data():
    """Load FAQ data and embeddings"""
    global embedding_model, question_embeddings, questions, answers
    
    print("[...] Loading FAQ data...")
    faq_path = os.path.join(os.path.dirname(__file__), "faq_data.json")
    with open(faq_path, "r") as f:
        faq_data = json.load(f)
    
    questions = [item["question"] for item in faq_data]
    answers = [item["answer"] for item in faq_data]
    print(f"[OK] Loaded {len(questions)} FAQ entries")
    
    print("[...] Loading embedding model (this may take a moment)...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("[OK] Embedding model loaded")
    
    print("[...] Computing FAQ embeddings...")
    question_embeddings = embedding_model.encode(questions)
    print("[OK] Embeddings computed")

def find_cloudflared():
    """Find cloudflared executable in common locations"""
    import shutil
    
    # First try to find it in PATH
    cf = shutil.which("cloudflared")
    if cf:
        return cf
    
    # Common Windows installation paths
    common_paths = [
        r"C:\Program Files\cloudflared\cloudflared.exe",
        r"C:\Program Files (x86)\cloudflared\cloudflared.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\cloudflared\cloudflared.exe"),
        os.path.expanduser(r"~\AppData\Local\cloudflared\cloudflared.exe"),
        os.path.expanduser(r"~\Downloads\cloudflared-windows-amd64.exe"),
        os.path.expanduser(r"~\Downloads\cloudflared-windows-386.exe"),
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    return None

def start_cloudflare_tunnel():
    """Start Cloudflare tunnel using cloudflared"""
    cloudflared_path = find_cloudflared()
    
    if not cloudflared_path:
        print(f"[WARN] cloudflared not found")
        print("[INFO] Download: https://developers.cloudflare.com/cloudflare-one/connections/connect-applications/install-and-setup/")
        print("[INFO] Extract to Downloads folder or add to PATH")
        print("[INFO] Running on localhost only\n")
        return None
    
    try:
        print(f"[...] Starting Cloudflare tunnel using: {cloudflared_path}")
        # Start cloudflared in a visible window (non-blocking)
        subprocess.Popen(
            [cloudflared_path, "tunnel", "--url", f"http://localhost:{PORT}"],
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        
        print(f"[OK] Cloudflare tunnel started in a new window")
        print(f"\n{'='*60}")
        print(f"[INFO] Look at the new terminal window for the public URL:")
        print(f"[INFO] It will show: 'https://xxxx.trycloudflare.com'")
        print(f"[INFO] Use that URL to access your chatbot publicly!")
        print(f"{'='*60}\n")
        return True  # Return True since we don't need to manage the process
    except Exception as e:
        print(f"[WARN] Could not start cloudflared: {e}")
        print("[INFO] Running on localhost only\n")
        return None

# ============================================================
# Main
# ============================================================

@app.on_event("startup")
def startup_event():
    """Load data when FastAPI starts"""
    load_faq_data()

def main():
    print("\n" + "="*60)
    print("AI CHATBOT - OPENAI + CLOUDFLARE TUNNEL")
    print("="*60 + "\n")
    
    # Step 1: Check OpenAI API
    if not check_openai_api():
        print("\n[FATAL] Cannot access OpenAI API. Check your API key. Exiting.")
        sys.exit(1)
    
    # Step 2: Cloudflare tunnel is started by START_SERVER.bat
    # We don't start it here to avoid conflicts
    print("[INFO] Cloudflare tunnel should be started in a separate window by START_SERVER.bat")
    print("[INFO] If you see a cloudflared window, the tunnel is running")
    
    # Step 3: Start server
    print(f"\n[OK] Starting server on port {PORT}...")
    print(f"[OK] Local URL: http://localhost:{PORT}")
    print(f"[OK] Test endpoint: http://localhost:{PORT}/test_client.html")
    print("\nPress Ctrl+C to stop the server.\n")
    
    try:
        # Run uvicorn directly
        uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
    finally:
        print("\n[INFO] Server shut down")
        print("[INFO] The cloudflared tunnel window will continue running")
        print("[INFO] Close it manually if needed")

if __name__ == "__main__":
    main()
