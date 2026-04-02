from fastapi import FastAPI
from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint
from google import genai
import os
import openai

app = FastAPI()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

client = genai.Client(api_key=GEMINI_API_KEY)

openai.api_key = OPENAI_API_KEY

def get_openai_embedding(text: str):
    response = openai.Embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    embedding = response.data[0].embedding
    return embedding

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
        contents="Para quando está prevista a nova onda de unicórnios?"
    )
    return {"response": response.text}

@app.get("/ask")
def ask(question: str):

    embedding = get_openai_embedding(question)

    search_result = qdrant.search_points(
        collection_name="dados",
        query_vector=embedding,
        limit=3,
        with_payload=True
    )

    context_list = []
    for hit in search_result:
        if hit.payload and "content" in hit.payload:
            context_list.append(hit.payload["content"])

    context = "\n\n".join(context_list) if context_list else "Nenhuma informação encontrada na base."

    return {"question": question, "context": context}

    prompt = f"""
Responda com base no contexto abaixo. 
Se não souber, diga que não encontrou a informação.

Contexto:
{context}

Pergunta:
{question}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )

    return {
        "question": question,
        "context": context,
        "response": response.text
    }
