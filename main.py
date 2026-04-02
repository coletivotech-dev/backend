from fastapi import FastAPI
from qdrant_client import QdrantClient
from google import genai
import os
import requests

app = FastAPI()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

client = genai.Client(api_key=GEMINI_API_KEY)

HF_MODEL_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-m3/pipeline/feature-extraction"

def get_bge_embedding(text: str):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": text,
        "options": {"wait_for_model": True}
    }
    response = requests.post(HF_MODEL_URL, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()

    # bge-m3 retorna lista de listas (uma por token), precisamos do vetor [CLS] = primeiro
    if isinstance(result[0], list):
        return result[0]  # vetor do token [CLS]
    return result  # já é o vetor direto

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
    # 1. Gera embedding da pergunta com bge-m3 via HF
    embedding = get_bge_embedding(question)

    # 2. Busca no Qdrant
    search_result = qdrant.search(
        collection_name="dados",
        query_vector=embedding,
        limit=3,
        with_payload=True
    )

    # 3. Monta contexto
    context_list = []
    for hit in search_result:
        if hit.payload and "content" in hit.payload:
            context_list.append(hit.payload["content"])

    context = "\n\n".join(context_list) if context_list else "Nenhuma informação encontrada na base."

    # 4. Chama o Gemini com o contexto
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
