import json
import os
import threading
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import Response as FastAPIResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from typing import Any
import logging
from difflib import SequenceMatcher
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
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

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
_model_loading_started = False
_model_load_lock = threading.Lock()

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


def _load_embeddings_background() -> None:
    try:
        _load_embeddings_once()
        logger.info("Sentence-transformer model and embeddings are ready")
    except Exception as exc:
        logger.error(f"Background model load failed: {exc}")


def _start_background_model_load() -> None:
    global _model_loading_started
    if question_embeddings is not None:
        return

    with _model_load_lock:
        if _model_loading_started or question_embeddings is not None:
            return
        _model_loading_started = True

    thread = threading.Thread(target=_load_embeddings_background, daemon=True)
    thread.start()


def _fallback_answer(user_question: str) -> Response:
    _load_faq_data_once()
    if not questions or not answers:
        return Response(
            answer="I'm sorry, I don't have an answer for that. Please contact support.",
            confidence=0.0,
        )

    normalized_input = user_question.lower().strip()
    best_match_index = 0
    best_score = 0.0

    for idx, candidate in enumerate(questions):
        score = SequenceMatcher(None, normalized_input, candidate.lower()).ratio()
        if score > best_score:
            best_score = score
            best_match_index = idx

    threshold = 0.3
    if best_score < threshold:
        openai_response = _openai_fallback(user_question)
        if openai_response is not None:
            return openai_response
        return Response(
            answer="I'm sorry, I don't have an answer for that. Please contact support.",
            confidence=best_score,
        )
    return Response(answer=answers[best_match_index], confidence=best_score)


def _openai_fallback(user_question: str) -> Optional[Response]:
    if openai_client is None:
        return None
    try:
        completion = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are Murray the Meadowlands ambassador. Reply in 1-2 short, helpful sentences.",
                },
                {"role": "user", "content": user_question},
            ],
            max_tokens=80,
            temperature=0.7,
            timeout=12,
        )
        content = completion.choices[0].message.content
        if content:
            return Response(answer=content.strip(), confidence=0.7)
    except Exception as exc:
        logger.warning(f"OpenAI fallback failed: {exc}")
    return None


def _extract_user_question(query: Query) -> str:
    user_question = (query.question or query.message or "").strip()
    if not user_question:
        raise HTTPException(status_code=400, detail="question is required")
    return user_question

def _generate_answer(user_question: str) -> Response:
    _load_faq_data_once()

    if question_embeddings is None:
        _start_background_model_load()
        return _fallback_answer(user_question)

    if not answers:
        raise HTTPException(status_code=500, detail="FAQ answers are unavailable")
    if question_embeddings is None:
        raise HTTPException(status_code=500, detail="Question embeddings are unavailable")
    user_embedding = model.encode([user_question])
    similarities = np.dot(question_embeddings, user_embedding.T).flatten()
    best_match_index = np.argmax(similarities)
    best_similarity = float(similarities[best_match_index])

    threshold = 0.5
    if best_similarity < threshold:
        openai_response = _openai_fallback(user_question)
        if openai_response is not None:
            return openai_response
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

@app.api_route("/", methods=["GET", "HEAD"])
def read_root():
    return {"message": "FAQ Chatbot API is running"}


@app.head("/")
def head_root():
    return FastAPIResponse(status_code=200)


@app.get("/favicon.ico")
def favicon():
    return FastAPIResponse(status_code=204)


@app.get("/health")
def health():
    return {"ok": True}


@app.on_event("startup")
def startup_warmup():
    _load_faq_data_once()
    _start_background_model_load()

if __name__ == "__main__":
    import uvicorn
    print("Starting server...")
    try:
        port = int(os.getenv("PORT", "8000"))
        uvicorn.run(app, host="0.0.0.0", port=port)
    except Exception as e:
        print(f"Server failed to start: {e}")
