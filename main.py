import json
import os
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import Response as FastAPIResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
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
OPENAI_MODEL = "gpt-3.5-turbo"

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


def _fallback_answer(user_question: str) -> Response:
    _load_faq_data_once()
    if not questions or not answers:
        return Response(
            answer="I'm sorry, I don't have an answer for that. Please contact support.",
            confidence=0.0,
        )

    ranked = _rank_faq_candidates(user_question, top_k=1)
    if not ranked:
        return Response(
            answer="I'm sorry, I don't have an answer for that. Please contact support.",
            confidence=0.0,
        )

    best_match_index, best_score = ranked[0]

    threshold = 0.3
    if best_score < threshold:
        openai_response = _openai_fallback(user_question, None)
        if openai_response is not None:
            return openai_response
        return Response(
            answer="I'm sorry, I don't have an answer for that. Please contact support.",
            confidence=best_score,
        )
    return Response(answer=answers[best_match_index], confidence=best_score)


def _rank_faq_candidates(user_question: str, top_k: int = 3) -> list[tuple[int, float]]:
    normalized_input = user_question.lower().strip()
    scored: list[tuple[int, float]] = []
    for idx, candidate in enumerate(questions):
        score = SequenceMatcher(None, normalized_input, candidate.lower()).ratio()
        scored.append((idx, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]


def _openai_fallback(user_question: str, faq_context: Optional[str]) -> Optional[Response]:
    if openai_client is None:
        return None
    model_candidates = [OPENAI_MODEL]
    tried = set()
    last_error = None

    for model_name in model_candidates:
        if model_name in tried:
            continue
        tried.add(model_name)
        try:
            if faq_context:
                user_content = (
                    "Use ONLY this FAQ context for facts, and answer in 1-2 short sentences. "
                    "If context is missing details, say so briefly.\n\n"
                    f"Context:\n{faq_context}\n\nUser question: {user_question}"
                )
            else:
                user_content = user_question

            completion = openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are Murray, a Meadowlands ambassador with a friendly Jersey voice. Keep replies to 1-2 short sentences, use light Jersey phrasing (like 'ya know', 'down the shore', 'fuhgeddaboutit' sparingly), and stay helpful.",
                    },
                    {"role": "user", "content": user_content},
                ],
                max_tokens=80,
                temperature=0.7,
                timeout=12,
            )
            content = completion.choices[0].message.content
            if content:
                return Response(answer=content.strip(), confidence=0.7)
        except Exception as exc:
            last_error = exc
            if "model_not_found" in str(exc) or "does not have access to model" in str(exc):
                continue
            logger.warning(f"OpenAI fallback failed: {exc}")
            return None

    if last_error is not None:
        logger.warning(f"OpenAI fallback failed: {last_error}")
    return None


def _extract_user_question(query: Query) -> str:
    user_question = (query.question or query.message or "").strip()
    if not user_question:
        raise HTTPException(status_code=400, detail="question is required")
    return user_question

def _generate_answer(user_question: str) -> Response:
    _load_faq_data_once()

    greeting_inputs = {"hi", "hello", "hey", "yo", "sup"}
    if user_question.lower().strip() in greeting_inputs:
        openai_response = _openai_fallback(user_question, None)
        if openai_response is not None:
            return openai_response
        return Response(
            answer="Hey! I'm Murray the Meadowlands ambassador. Ask me about places to eat, things to do, or local spots.",
            confidence=0.8,
        )

    ranked = _rank_faq_candidates(user_question, top_k=3)
    best_similarity = ranked[0][1] if ranked else 0.0

    faq_context = None
    if ranked and best_similarity >= 0.12:
        snippets = []
        for idx, _score in ranked:
            snippets.append(f"Q: {questions[idx]}\nA: {answers[idx]}")
        faq_context = "\n\n".join(snippets)

    openai_response = _openai_fallback(user_question, faq_context)
    if openai_response is not None:
        if faq_context:
            openai_response.confidence = max(openai_response.confidence, min(0.92, best_similarity + 0.2))
        return openai_response

    if ranked and best_similarity >= 0.15:
        best_idx = ranked[0][0]
        return Response(answer=answers[best_idx], confidence=max(0.55, best_similarity))

    return _fallback_answer(user_question)


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

if __name__ == "__main__":
    import uvicorn
    print("Starting server...")
    try:
        port = int(os.getenv("PORT", "8000"))
        uvicorn.run(app, host="0.0.0.0", port=port)
    except Exception as e:
        print(f"Server failed to start: {e}")
