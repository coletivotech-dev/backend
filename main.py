from fastapi import FastAPI
from qdrant_client import QdrantClient

app = FastAPI()

QDRANT_URL = "https://d30aaf2b-635c-4835-aa2d-6da958f4b9bb.sa-east-1-0.aws.cloud.qdrant.io" 
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.DVJIVyYIiJIhkNwgDEHHhuIPzRo7FvHtZmUz7fIQ-lg"

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

@app.get("/")
def home():
    return {"message": "API rodando"}

@app.get("/test-qdrant")
def test_qdrant():
    collections = client.get_collections().collections
    return {"collections": [c.name for c in collections]}
