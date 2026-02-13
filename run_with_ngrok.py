import json
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import logging
import uvicorn
from pyngrok import ngrok

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Initialize the model
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

@app.post("/ask", response_model=Response)
async def ask_question(query: Query):
    user_question = query.question
    
    # Encode the user's question
    user_embedding = model.encode([user_question])
    
    # Calculate cosine similarities
    similarities = np.dot(question_embeddings, user_embedding.T).flatten()
    
    # Find the index of the highest similarity
    best_match_index = np.argmax(similarities)
    best_similarity = float(similarities[best_match_index])
    
    # Threshold for a valid match
    threshold = 0.5
    
    if best_similarity < threshold:
        return Response(answer="I'm sorry, I don't have an answer for that. Please contact support.", confidence=best_similarity)
    
    return Response(answer=answers[best_match_index], confidence=best_similarity)

@app.get("/")
def read_root():
    return {"message": "FAQ Chatbot API is running"}

if __name__ == "__main__":
    # Start ngrok tunnel
    print("Starting ngrok tunnel...")
    public_url = ngrok.connect(8000)
    print(f"\n{'='*60}")
    print(f"🚀 Your PUBLIC HTTPS URL is: {public_url}")
    print(f"{'='*60}")
    print(f"\nUse this URL in your frontend: {public_url}/ask\n")
    
    # Start the server
    print("Starting server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
