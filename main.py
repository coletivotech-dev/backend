from fastapi import FastAPI
from qdrant_client import QdrantClient
from google import genai
import os

app = FastAPI()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

client = genai.Client(api_key=GEMINI_API_KEY)

@app.get("/")
def home():
    return {"message": "API rodando"}

@app.get("/test-qdrant")
def test_qdrant():
    collections = qdrant.get_collections().collections
    return {"collections": [c.name for c in collections]}

@app.get("/test-llm")
def test_llm():
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents="Explique em uma frase o que é IA"
    )
    return {"response": response.text}
