from fastapi import FastAPI
from qdrant_client import QdrantClient
from qdrant_client.http.models import SparseVector, FusionQuery, Fusion
from qdrant_client.models import Prefetch
from google import genai
from pydantic import BaseModel
import os
import requests

class AskRequest(BaseModel):
    question: str

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
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
    if isinstance(result[0], list):
        return result[0]
    return result

@app.get("/")
def home():
    return {"message": "API rodando"}

@app.get("/test-qdrant")
def test_qdrant():
    collections = qdrant.get_collections().collections
    return {"collections": [c.name for c in collections]}

@app.post("/ask")
def ask(body: AskRequest):
    question = body.question
    # 1. Gera vetor denso via HF
    dense_vector = get_bge_embedding(question)

    # 2. Busca híbrida: denso + esparso combinados com RRF
    search_result = qdrant.query_points(
        collection_name="dados",
        prefetch=[
            Prefetch(
                query=dense_vector,
                using="dense",
                limit=10
            ),
            Prefetch(
                query=SparseVector(indices=[], values=[]),
                using="sparse",
                limit=10
            )
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=5,
        with_payload=True
    ).points

    # 3. Monta contexto
    context_list = []
    for hit in search_result:
        if hit.payload and "content" in hit.payload:
            site = hit.payload.get("site", "")
            date = hit.payload.get("date", "")
            source = hit.payload.get("source", "")
            conteudo = hit.payload["content"]
            context_list.append(
                f"Site: {site}\nData: {date}\nURL: {source}\nConteúdo: {conteudo}"
            )

    context = "\n\n---\n\n".join(context_list) if context_list else None

    # 4. Chama o Gemini
    if not context:
        return {
            "response": "Nossa base de dados ainda não contempla informações sobre esse tema. Estamos sempre expandindo nosso acervo — tente reformular a pergunta ou explore outros tópicos relacionados."
        }

prompt = f"""
Você é um jornalista especializado em negócios, tecnologia e sustentabilidade.
Seu estilo é profissional e direto, com linguagem acessível mas sem perder a precisão — como uma boa reportagem da Harvard Business Review ou do MIT Technology Review em português.

Regras que você deve seguir à risca:

1. Responda EXCLUSIVAMENTE com base no contexto fornecido abaixo. Não acrescente informações externas, opiniões próprias ou dados que não estejam no contexto.

2. Cada trecho do contexto contém um campo "Site" com o nome da fonte. Ao citar uma informação, mencione esse nome naturalmente dentro do texto, como parte da narrativa. Exemplos:
   - "De acordo com o Gartner, ..."
   - "O MIT Technology Review aponta que ..."
   - "Segundo o Fórum Econômico Mundial, ..."
   - "Ao observarmos os dados da Mackinsey ..."
   - "... conforme apresentado pelo Gartner..."
   Nunca mencione a fonte de forma mecânica ou repetitiva — integre-a ao fluxo do texto.

3. Cada trecho do contexto contém um campo "Data" com a data de publicação no formato yyyy-mm-dd. Ao usar datas, simplifique: use apenas o ano, ou mês e ano quando relevante. Nunca escreva a data no formato técnico yyyy-mm-dd na resposta.

4. Atenção especial a datas e projeções mencionadas dentro dos próprios textos do contexto — como previsões, estimativas e metas futuras. Preserve essas referências temporais com precisão, pois são parte essencial da informação.

5. Quando a mesma informação aparecer em mais de um trecho ou em mais de uma fonte, traga todas as perspectivas de forma cronológica, indicando a origem e o período de cada uma. Mostre a evolução ou convergência entre as fontes de forma natural. Exemplo: "Em 2025, o Gartner já apontava que... Esse cenário foi reforçado pelo MIT Technology Review em 2026, que observou..."

6. Calibre o tamanho da resposta proporcionalmente à riqueza do contexto disponível:
   - Contexto rico e variado: desenvolva bem os pontos em parágrafos completos.
   - Contexto limitado ou repetitivo: seja conciso, sem forçar um texto longo.
   - Em nenhum caso repita informações para preencher espaço.

7. Organize a resposta de forma fluida, como um texto jornalístico — evite listas com marcadores sempre que possível. Prefira parágrafos bem construídos.

8. Se o contexto não contiver informações suficientes para responder, diga exatamente isso: "Nossa base de dados ainda não contempla informações suficientes sobre esse tema. Estamos sempre expandindo nosso acervo — tente reformular a pergunta ou explore outros tópicos relacionados."

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
        "response": response.text
    }
