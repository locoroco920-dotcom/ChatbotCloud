import json
import os
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from typing import Any
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
faq_data = None
questions: list[str] = []
answers: list[str] = []
model: Any = None
question_embeddings = None

class Query(BaseModel):
    question: Optional[str] = None
    message: Optional[str] = None

class Response(BaseModel):
    answer: str
    confidence: float


def _load_faq_data_once() -> None:
    global faq_data, questions, answers
    if faq_data is not None:
        return

    faq_path = os.path.join(SCRIPT_DIR, "faq_data.json")
    with open(faq_path, "r", encoding="utf-8") as f:
        faq_data = json.load(f)

    questions = [item["question"] for item in faq_data]
    answers = [item["answer"] for item in faq_data]


def _load_model_once() -> None:
    global model
    if model is not None:
        return

    logger.info("Loading sentence-transformer model")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")


def _load_embeddings_once() -> None:
    global question_embeddings
    if question_embeddings is not None:
        return

    _load_faq_data_once()
    _load_model_once()
    if not questions:
        raise HTTPException(status_code=500, detail="FAQ data is empty")
    question_embeddings = model.encode(questions)


def _extract_user_question(query: Query) -> str:
    user_question = (query.question or query.message or "").strip()
    if not user_question:
        raise HTTPException(status_code=400, detail="question is required")
    return user_question

def _generate_answer(user_question: str) -> Response:
    _load_embeddings_once()
    if not answers:
        raise HTTPException(status_code=500, detail="FAQ answers are unavailable")
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
    user_question = _extract_user_question(query)
    return _generate_answer(user_question)


@app.post("/chat", response_model=Response)
async def chat(query: Query, x_api_key: Optional[str] = Header(default=None)):
    expected_api_key = os.getenv("WIDGET_API_KEY")
    if not expected_api_key:
        raise HTTPException(status_code=503, detail="WIDGET_API_KEY is not configured")
    if not x_api_key or x_api_key != expected_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    user_question = _extract_user_question(query)
    return _generate_answer(user_question)

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
