import json
import os
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import Response as FastAPIResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging
import re
from difflib import SequenceMatcher
import requests
from bs4 import BeautifulSoup
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

KNOWN_CITIES = [
    "Secaucus",
    "East Rutherford",
    "Rutherford",
    "Lyndhurst",
    "Carlstadt",
    "Moonachie",
    "Jersey City",
    "Weehawken",
    "Hackensack",
    "Paramus",
    "North Bergen",
    "Teterboro",
    "Hasbrouck Heights",
    "Clifton",
    "Wayne",
    "Teaneck",
    "Harrison",
    "Saddle Brook",
]

EVENTS_CALENDAR_URL = "https://dev.mlcvb.com/meadowlands-calendar-of-events/"
FOOD_DIRECTORY_URL = "https://dev.mlcvb.com/dining-culinary-experiences-in-the-meadowlands/"

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
    input_tokens = set(re.findall(r"[a-z0-9]+", normalized_input))

    def score_question(candidate_text: str) -> float:
        candidate_lower = candidate_text.lower()
        seq_score = SequenceMatcher(None, normalized_input, candidate_lower).ratio()

        candidate_tokens = set(re.findall(r"[a-z0-9]+", candidate_lower))
        if input_tokens:
            overlap = len(input_tokens.intersection(candidate_tokens)) / max(1, len(input_tokens))
        else:
            overlap = 0.0

        return (0.6 * overlap) + (0.4 * seq_score)

    scored: list[tuple[int, float]] = []
    for idx, candidate in enumerate(questions):
        score = score_question(candidate)
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
                    "If context is missing details, say so briefly. If the user asks for the best/favorite, do not choose one winner—give multiple options fairly. "
                    "Only include links when referring to specific places/events.\n\n"
                    f"Context:\n{faq_context}\n\nUser question: {user_question}"
                )
            else:
                user_content = user_question

            completion = openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are Murray, a friendly Meadowlands local with a conversational Jersey accent. Sound natural, warm, and casual. If asked for the best/favorite, say it's a tough call and give multiple good options instead of one winner.",
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


def _extract_urls(text: str) -> list[str]:
    return re.findall(r"https?://[^\s)]+", text)


def _extract_source_url(text: str) -> Optional[str]:
    match = re.search(r"Source:\s*(https?://\S+)", text)
    if not match:
        return None
    return match.group(1).rstrip(".,)")


def _clean_place_title(question_text: str) -> str:
    title = question_text.replace("What should visitors know about ", "")
    title = title.replace("Where can I find information about ", "")
    title = re.sub(r"\s*-\s*MLCVB.*$", "", title, flags=re.IGNORECASE).strip(" ?")
    return title


def _extract_short_description(answer_text: str) -> str:
    text = re.sub(r"Source:\s*https?://\S+", "", answer_text).strip()
    text = re.sub(r"Check out what other travelers say about", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" .")
    if not text:
        return "Local Meadowlands dining option."
    first_sentence = re.split(r"(?<=[.!?])\s+", text)[0].strip()
    if len(first_sentence) > 95:
        first_sentence = first_sentence[:92].rstrip() + "..."
    return first_sentence or "Local Meadowlands dining option."


def _extract_location(answer_text: str, url: str) -> str:
    for city in KNOWN_CITIES:
        if re.search(rf"\b{re.escape(city)}\b", answer_text, flags=re.IGNORECASE):
            return f"{city}, NJ"
    if "/secaucus" in url:
        return "Secaucus, NJ"
    if "/east-rutherford" in url:
        return "East Rutherford, NJ"
    if "/jersey-city" in url:
        return "Jersey City, NJ"
    if "/weehawken" in url:
        return "Weehawken, NJ"
    return "Meadowlands, NJ"


def _collect_food_places(user_question: str) -> list[dict[str, str]]:
    lower = user_question.lower()
    pizza_mode = "pizza" in lower
    places: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    for question_text, answer_text in zip(questions, answers):
        source_url = _extract_source_url(answer_text)
        if not source_url:
            continue
        if "/restaurant/" not in source_url:
            continue

        searchable = f"{question_text} {answer_text}".lower()
        if pizza_mode and not any(token in searchable for token in ["pizza", "pizzeria", "wood fired", "italian"]):
            continue

        if source_url in seen_urls:
            continue
        seen_urls.add(source_url)

        place_name = _clean_place_title(question_text)
        location = _extract_location(answer_text, source_url)
        short_description = _extract_short_description(answer_text)

        places.append(
            {
                "name": place_name,
                "url": source_url,
                "location": location,
                "description": short_description,
            }
        )

    return places


def _build_food_listing_answer(user_question: str) -> Optional[str]:
    if _intent_label(user_question) != "food":
        return None

    places = _collect_food_places(user_question)
    if not places:
        return None

    if len(places) > 5:
        if "pizza" in user_question.lower():
            return (
                "Ya got a ton of pizza options around here, so I’ll point ya straight to the dining directory: "
                f"{FOOD_DIRECTORY_URL}"
            )
        return (
            "There are a lotta food spots in the Meadowlands, so here’s the full dining directory: "
            f"{FOOD_DIRECTORY_URL}"
        )

    intro = "That's a tough call, ya know—there's no one best spot. Here are food options around the Meadowlands:"
    if "pizza" in user_question.lower():
        intro = "That's a tough call, ya know—no single best pizza spot. Here are pizza options around the Meadowlands:"

    lines = [intro, ""]
    display_places = places[:5]
    for idx, place in enumerate(display_places, start=1):
        lines.append(f"{idx}. [{place['name']}]({place['url']})")
        lines.append(place["location"])
        lines.append(place["description"])
        if idx < len(display_places):
            lines.append("--------------------")
        lines.append("")

    return "\n".join(lines).strip()


def _is_upcoming_events_query(user_question: str) -> bool:
    lower = user_question.lower()
    return _intent_label(user_question) == "event" and any(
        token in lower for token in ["upcoming", "next", "coming up", "this week", "events"]
    )


def _fetch_upcoming_events(limit: int = 5) -> list[tuple[str, str]]:
    candidate_urls = ["https://dev.mlcvb.com/", EVENTS_CALENDAR_URL]
    seen_names: set[str] = set()
    events: list[tuple[str, str]] = []

    for url in candidate_urls:
        try:
            response = requests.get(url, timeout=12)
            if response.status_code >= 400:
                continue
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text("\n", strip=True)

            matches = re.findall(r"\n([^\n]{4,140})\n\s*Date:\s*([^\n]{4,120})", "\n" + text)
            for raw_name, raw_date in matches:
                name = re.sub(r"\s+", " ", raw_name).strip()
                date_text = re.sub(r"\s+", " ", raw_date).strip()

                if name.lower() in seen_names:
                    continue
                if any(block in name.lower() for block in ["upcoming events", "image", "menu", "cookie"]):
                    continue

                seen_names.add(name.lower())
                events.append((name, date_text))
                if len(events) >= limit:
                    return events
        except Exception:
            continue

    return events[:limit]


def _build_upcoming_events_answer(user_question: str) -> Optional[str]:
    if not _is_upcoming_events_query(user_question):
        return None

    events = _fetch_upcoming_events(limit=5)
    if not events:
        return (
            "Lemme check the schedule for ya — I couldn't pull the live list right this second, "
            f"but here's the full events calendar: {EVENTS_CALENDAR_URL}"
        )

    lines = ["You got it — here are the next 5 upcoming events in the Meadowlands:", ""]
    for index, (name, date_text) in enumerate(events, start=1):
        lines.append(f"{index}. {name} ({date_text})")
    lines.append("")
    lines.append(f"Full events calendar: {EVENTS_CALENDAR_URL}")
    return "\n".join(lines)


def _is_place_query(user_question: str) -> bool:
    lower = user_question.lower()
    place_terms = [
        "eat",
        "food",
        "restaurant",
        "dining",
        "pizza",
        "hotel",
        "stay",
        "event",
        "concert",
        "show",
        "attraction",
        "visit",
        "shop",
        "shopping",
        "where",
        "place",
    ]
    return any(term in lower for term in place_terms)


def _has_preference_intent(user_question: str) -> bool:
    lower = user_question.lower()
    return any(term in lower for term in ["best", "favorite", "top", "number one", "recommend"]) 


def _intent_label(user_question: str) -> str:
    lower = user_question.lower()
    if any(word in lower for word in ["eat", "food", "restaurant", "dining", "pizza", "burger"]):
        return "food"
    if any(word in lower for word in ["event", "show", "concert", "game", "calendar"]):
        return "event"
    if any(word in lower for word in ["hotel", "stay", "lodging"]):
        return "hotel"
    if any(word in lower for word in ["shop", "shopping", "store", "mall"]):
        return "shopping"
    return "general"


def _is_generic_info_question(question_text: str) -> bool:
    lower = question_text.lower().strip()
    return lower.startswith("where can i find information about")


def _is_candidate_relevant(user_question: str, idx: int) -> bool:
    intent = _intent_label(user_question)
    question_text = questions[idx].lower()
    answer_text = answers[idx].lower()
    text = f"{question_text} {answer_text}"

    if intent == "food":
        return any(token in text for token in ["restaurant", "dining", "pizza", "culinary", "/restaurant/"])
    if intent == "event":
        return any(token in text for token in ["event", "concert", "calendar", "show", "/event/"])
    if intent == "hotel":
        return any(token in text for token in ["hotel", "hospitality", "/hotel/"])
    if intent == "shopping":
        return any(token in text for token in ["shopping", "shop", "store", "mall"])
    return True


def _should_add_follow_up(user_question: str) -> bool:
    normalized = user_question.lower().strip()
    greeting_inputs = {"hi", "hello", "hey", "yo", "sup", "thanks", "thank you"}
    if normalized in greeting_inputs:
        return False
    if "?" in user_question:
        return True
    return normalized.startswith(("what", "where", "when", "how", "who", "which", "can", "do", "is", "are"))


def _build_related_question(user_question: str) -> str:
    intent = _intent_label(user_question)
    if intent == "food":
        return "Want to know about other places to eat?"
    if intent == "event":
        return "Want me to share a few more events around the same time?"
    if intent == "hotel":
        return "Want me to list a few more hotel options nearby?"
    if intent == "shopping":
        return "Want a couple more shopping spots to compare?"
    return "Want me to help with another question about Meadowlands places or events?"


def _finalize_answer(answer_text: str, user_question: str, ranked: list[tuple[int, float]]) -> str:
    text = answer_text.strip()
    existing_urls = _extract_urls(text)
    place_query = _is_place_query(user_question)

    top_urls: list[str] = []
    for idx, _score in ranked[:3]:
        for url in _extract_urls(answers[idx]):
            if url not in top_urls:
                top_urls.append(url)

    if place_query and not existing_urls and top_urls:
        link_lines = "\n".join(top_urls[:3])
        text = f"{text}\n\nLinks:\n{link_lines}"

    if _should_add_follow_up(user_question):
        follow_up = _build_related_question(user_question)
        if follow_up.lower() not in text.lower():
            text = f"{text}\n\n{follow_up}"

    return text

def _generate_answer(user_question: str) -> Response:
    _load_faq_data_once()

    greeting_inputs = {"hi", "hello", "hey", "yo", "sup"}
    if user_question.lower().strip() in greeting_inputs:
        openai_response = _openai_fallback(user_question, None)
        if openai_response is not None:
            openai_response.answer = _finalize_answer(openai_response.answer, user_question, [])
            return openai_response
        return Response(
            answer=_finalize_answer(
                "Hey! I'm Murray the Meadowlands ambassador. Ask me about places to eat, things to do, or local spots.",
                user_question,
                [],
            ),
            confidence=0.8,
        )

    ranked = _rank_faq_candidates(user_question, top_k=3)
    best_similarity = ranked[0][1] if ranked else 0.0

    upcoming_events_answer = _build_upcoming_events_answer(user_question)
    if upcoming_events_answer is not None:
        return Response(
            answer=_finalize_answer(upcoming_events_answer, user_question, ranked),
            confidence=max(0.75, best_similarity),
        )

    food_listing_answer = _build_food_listing_answer(user_question)
    if food_listing_answer is not None:
        return Response(
            answer=_finalize_answer(food_listing_answer, user_question, ranked),
            confidence=max(0.72, best_similarity),
        )

    faq_context = None
    if ranked and best_similarity >= 0.12:
        snippets = []
        for idx, _score in ranked:
            snippets.append(f"Q: {questions[idx]}\nA: {answers[idx]}")
        faq_context = "\n\n".join(snippets)

    if _has_preference_intent(user_question) and ranked:
        filtered_ranked = [
            (idx, score)
            for idx, score in ranked
            if not _is_generic_info_question(questions[idx]) and _is_candidate_relevant(user_question, idx)
        ]
        candidate_ranked = filtered_ranked if filtered_ranked else ranked

        top_entries = []
        seen_urls = set()
        seen_titles = set()
        for idx, _score in candidate_ranked:
            title = questions[idx].replace("What should visitors know about ", "").strip(" ?")
            if title.lower() in seen_titles:
                continue
            urls = _extract_urls(answers[idx])
            selected_url = None
            for url in urls:
                if url not in seen_urls:
                    selected_url = url
                    seen_urls.add(url)
                    break
            if selected_url:
                top_entries.append(f"- {title}: {selected_url}")
            else:
                top_entries.append(f"- {title}")
            seen_titles.add(title.lower())
            if len(top_entries) >= 3:
                break

        if not top_entries:
            top_entries = ["- I can share a few options if you tell me your preferred area or vibe."]

        answer_text = "That's a tough call, ya know—there isn't just one best spot. Here are a few solid options:\n" + "\n".join(top_entries)
        return Response(
            answer=_finalize_answer(answer_text, user_question, candidate_ranked),
            confidence=max(0.65, best_similarity),
        )

    openai_response = _openai_fallback(user_question, faq_context)
    if openai_response is not None:
        if faq_context:
            openai_response.confidence = max(openai_response.confidence, min(0.92, best_similarity + 0.2))
        openai_response.answer = _finalize_answer(openai_response.answer, user_question, ranked)
        return openai_response

    if ranked and best_similarity >= 0.15:
        best_idx = ranked[0][0]
        return Response(
            answer=_finalize_answer(answers[best_idx], user_question, ranked),
            confidence=max(0.55, best_similarity),
        )

    fallback = _fallback_answer(user_question)
    fallback.answer = _finalize_answer(fallback.answer, user_question, ranked)
    return fallback


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
