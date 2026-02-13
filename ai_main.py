import json
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import logging
import uvicorn
import requests
from datetime import datetime
import csv
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot_questions.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# CSV file for question tracking
QUESTIONS_CSV = 'questions_log.csv'

# Initialize CSV file with headers if it doesn't exist
if not os.path.exists(QUESTIONS_CSV):
    with open(QUESTIONS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Question', 'Answer_Confidence', 'Is_AI_Generated', 'Matched_FAQ_Index'])

app = FastAPI()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

# Enable CORS for website embedding
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex='https?://.*',
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load FAQ data
with open("faq_data.json", "r") as f:
    faq_data = json.load(f)

questions = [item["question"] for item in faq_data]
answers = [item["answer"] for item in faq_data]
context_text = "\n\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in faq_data])

# Initialize the embedding model for semantic search
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded.")

# Pre-compute embeddings for FAQ questions
question_embeddings = embedding_model.encode(questions)

# Ollama API endpoint
OLLAMA_API = "http://localhost:11434/api/generate"

class Query(BaseModel):
    question: str

class Response(BaseModel):
    answer: str
    confidence: float
    is_ai_generated: bool = False

def get_relevant_context(user_question: str, top_k: int = 3) -> str:
    """
    Retrieve relevant FAQ entries based on semantic similarity
    """
    user_embedding = embedding_model.encode([user_question])
    similarities = np.dot(question_embeddings, user_embedding.T).flatten()
    
    # Get top k most similar questions
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    relevant_docs = []
    for idx in top_indices:
        if similarities[idx] > 0.3:  # Only include if similarity is above threshold
            relevant_docs.append(f"Q: {questions[idx]}\nA: {answers[idx]}")
    
    return "\n\n".join(relevant_docs) if relevant_docs else context_text

def query_ollama(prompt: str) -> str:
    """
    Query the local Ollama instance
    """
    try:
        print(f"[DEBUG] Sending prompt to Ollama...")
        response = requests.post(
            OLLAMA_API,
            json={
                "model": "gemma3:4b",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.7,
            },
            timeout=60
        )
        print(f"[DEBUG] Ollama responded with status {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip()
        else:
            logger.error(f"Ollama error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.Timeout:
        logger.error("Ollama request timed out")
        return None
    except requests.exceptions.ConnectionError:
        logger.error("Ollama is not running. Start it with: ollama serve")
        return None
    except Exception as e:
        logger.error(f"Error querying Ollama: {e}")
        return None

def log_question(user_question: str, confidence: float, is_ai_generated: bool, matched_faq_index: int = None):
    """
    Log user question to both logger and CSV file
    """
    timestamp = datetime.now().isoformat()
    
    # Log to console/log file
    logger.info(f"Question: {user_question} | Confidence: {confidence:.2f} | AI Generated: {is_ai_generated}")
    
    # Log to CSV file
    try:
        with open(QUESTIONS_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, user_question, f"{confidence:.4f}", is_ai_generated, matched_faq_index or ""])
    except Exception as e:
        logger.error(f"Error writing to CSV: {e}")

@app.post("/ask", response_model=Response)
async def ask_question(query: Query):
    user_question = query.question
    
    # Get relevant context from FAQ
    relevant_context = get_relevant_context(user_question)
    
    # Create prompt for Ollama
    prompt = f"""Based on the following FAQ information, answer the user's question. 
If the answer is in the FAQ, answer based on that. If not, provide a helpful response based on general knowledge.
Be concise and helpful.

FAQ Information:
{relevant_context}

User Question: {user_question}

Answer:"""
    
    # Query Ollama
    ai_answer = query_ollama(prompt)
    
    if ai_answer:
        log_question(user_question, confidence=0.9, is_ai_generated=True)
        return Response(
            answer=ai_answer,
            confidence=0.9,
            is_ai_generated=True
        )
    else:
        # Fallback to semantic search if Ollama fails
        user_embedding = embedding_model.encode([user_question])
        similarities = np.dot(question_embeddings, user_embedding.T).flatten()
        best_match_index = np.argmax(similarities)
        best_similarity = float(similarities[best_match_index])
        
        if best_similarity < 0.5:
            log_question(user_question, confidence=best_similarity, is_ai_generated=False, matched_faq_index=None)
            return Response(
                answer="I'm sorry, I don't have an answer for that. Please contact support.",
                confidence=best_similarity,
                is_ai_generated=False
            )
        
        log_question(user_question, confidence=best_similarity, is_ai_generated=False, matched_faq_index=best_match_index)
        return Response(
            answer=answers[best_match_index],
            confidence=best_similarity,
            is_ai_generated=False
        )

@app.get("/")
def read_root():
    return {"message": "AI FAQ Chatbot API is running"}

@app.get("/health")
def health():
    """Check if Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            return {"status": "healthy", "ollama": "running"}
    except:
        pass
    return {"status": "healthy", "ollama": "not_running - start with 'ollama serve'"}

if __name__ == "__main__":
    print("\nStarting AI Chatbot with Ollama (gemma3:4b)")
    print("Server will run on http://localhost:8000")
    print("Make sure Ollama is running: ollama serve\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
