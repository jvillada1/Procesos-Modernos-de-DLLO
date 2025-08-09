import os
import json
import numpy as np
from typing import List, Dict, Tuple
import requests
from sentence_transformers import SentenceTransformer 
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn


# =========================
# CONFIG
# =========================
# Modelo: para español/inglés va bien el multilingüe; si todo es en inglés, MiniLM-L6-v2 es muy rápido.
EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 6
SIM_THRESHOLD = 0.2  # si la similitud máxima es menor, consideramos que no hay buen contexto

# === API(rellenar) ===
#API_KEY = os.getenv("LLM_API_KEY", "") 
API_KEY=""
#url = os.getenv("LLM_URL", "")  # p.ej. "https://api.groq.com/openai/v1/chat/completions" 
url="https://api.groq.com/openai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# =========================
# 1) DATOS (ejemplo). En producción, carga desde archivos .md/.txt y trocea.
# ========================= 
with open("Data.json", "r", encoding="utf-8") as f:
    DOCS = json.load(f)

# =========================
# 2) MODELO + INDEXACIÓN EN MEMORIA
# =========================
print("Cargando modelo de embeddings...")
emb_model = SentenceTransformer(EMB_MODEL)

def build_index(docs: List[Dict]) -> np.ndarray:
    texts = [d["text"] for d in docs]
    # normalize_embeddings=True produce vectores L2-normalizados (facilita similitud coseno con producto punto)
    doc_embs = emb_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(doc_embs, dtype=np.float32)

DOC_EMBS = build_index(DOCS)
print(f"Indexados {len(DOCS)} fragmentos.")

# =========================
#BÚSQUEDA SEMÁNTICA (en memoria)
# =========================
def search_similar(query: str, k: int = TOP_K) -> List[Tuple[Dict, float]]:
    q_emb = emb_model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    sims = DOC_EMBS @ q_emb  # coseno porque ya están normalizados
    order = np.argsort(-sims)[:k]
    results = [(DOCS[i], float(sims[i])) for i in order]
    return results

# =========================
# 4) RAG BÁSICO (armar contexto + llamar al LLM con requests)
# =========================
def build_context(hits: List[Tuple[Dict, float]]) -> str:
    bloques = []
    for d, score in hits:
        tag = f"[{d['title']} • sim={score:.2f}]"
        bloques.append(f"{tag}\n{d['text']}")
    return "\n\n---\n\n".join(bloques)

def ask_llm_with_rag(question: str) -> Dict:
    hits = search_similar(question, k=TOP_K)
    # Chequeo de calidad mínimo: sin buen match, pide aclaración
    if not hits or hits[0][1] < SIM_THRESHOLD:
        return {
            "answer": "",
            "sources": [],
            "used_rag": False
        }

    context = build_context(hits)
    prompt_user = f"""Responde SOLO usando el contexto. Si falta información o es ambiguo, dilo.
Pregunta: {question}

Contexto:
{context}
"""

    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [
            {"role": "system", "content": "Eres un asistente útil y conciso con conocimientos sobre minecraft. Cita la sección/título que uses."},
            {"role": "user", "content": prompt_user}
        ],
        "temperature": 0.3,
        "max_tokens": 2000
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)

    if resp.status_code != 200:
        raise RuntimeError(f"LLM error {resp.status_code}: {resp.text}")

    content = resp.json()["choices"][0]["message"]["content"]
    fuentes = [d["title"] for d, _ in hits]
    return {
        "answer": content,
        "sources": fuentes,
        "used_rag": True
    }


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en local no pasa nada
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskReq(BaseModel):
    question: str

@app.get("/")
def root():
    return FileResponse(os.path.join("static", "index.html"))

@app.post("/ask")
def ask(req: AskReq):
    out = ask_llm_with_rag(req.question)
    return out

if __name__ == "__main__":
    # Esto levanta FastAPI en local
    uvicorn.run(app, host="127.0.0.1", port=8000)

