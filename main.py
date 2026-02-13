import json
import os
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import Optional
import logging
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


def _parse_frontend_origins() -> list[str]:
    raw_origins = os.getenv("FRONTEND_ORIGINS", "")
    return [origin.strip() for origin in raw_origins.split(",") if origin.strip()]


def _init_openai_client() -> Optional[OpenAI]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY is not set; OpenAI client is disabled")
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception as exc:
        logger.error(f"Failed to initialize OpenAI client: {exc}")
        return None


openai_client = _init_openai_client()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

# Enable CORS for configured frontend domains
frontend_origins = _parse_frontend_origins()
if frontend_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=frontend_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Load FAQ data
with open("faq_data.json", "r") as f:
    faq_data = json.load(f)

questions = [item["question"] for item in faq_data]
answers = [item["answer"] for item in faq_data]

# Initialize the model
# 'all-MiniLM-L6-v2' is a small, fast model suitable for this task
print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

# Pre-compute embeddings for FAQ questions
question_embeddings = model.encode(questions)

class Query(BaseModel):
    question: str

class Response(BaseModel):
    answer: str
    confidence: float

def _generate_answer(user_question: str) -> Response:
    user_embedding = model.encode([user_question])
    similarities = np.dot(question_embeddings, user_embedding.T).flatten()
    best_match_index = np.argmax(similarities)
    best_similarity = float(similarities[best_match_index])

    threshold = 0.5
    if best_similarity < threshold:
        return Response(
            answer="I'm sorry, I don't have an answer for that. Please contact support.",
            confidence=best_similarity,
        )

    return Response(answer=answers[best_match_index], confidence=best_similarity)


@app.post("/ask", response_model=Response)
async def ask_question(query: Query):
    return _generate_answer(query.question)


@app.post("/chat", response_model=Response)
async def chat(query: Query, x_api_key: Optional[str] = Header(default=None)):
    expected_api_key = os.getenv("WIDGET_API_KEY")
    if not expected_api_key:
        raise HTTPException(status_code=503, detail="WIDGET_API_KEY is not configured")
    if not x_api_key or x_api_key != expected_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return _generate_answer(query.question)

@app.get("/")
def read_root():
    return {"message": "FAQ Chatbot API is running"}


@app.get("/health")
def health():
    return {"ok": True}

if __name__ == "__main__":
    import uvicorn
    print("Starting server...")
    try:
        port = int(os.getenv("PORT", "8000"))
        uvicorn.run(app, host="0.0.0.0", port=port)
    except Exception as e:
        print(f"Server failed to start: {e}")
