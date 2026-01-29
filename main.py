import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.config import Config
from src.ingestion import IngestionPipeline
from src.rag_engine import RAGEngine

# 1. Inicialización
app = FastAPI(title="AI appi RAG")

# Validamos configuración al arrancar
try:
    Config.validate()
    # Instanciamos el motor una sola vez para que sea rápido
    engine = RAGEngine()
except Exception as e:
    print(f"❌ Error inicial: {e}")
    sys.exit(1)

# 2. Modelos de datos (lo que recibimos por internet)
class ChatRequest(BaseModel):
    message: str

class IngestRequest(BaseModel):
    file_path: str

# --- ENDPOINT PARA EL CHAT (Opción 2 de tu menú) ---
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Usamos tu lógica del RAGEngine
        answer, sources = engine.chat(request.message)
        return {
            "answer": answer,
            "sources": list(set(sources)) if sources else []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ENDPOINT PARA INGESTA (Opción 1 de tu menú) ---
@app.post("/ingest")
async def ingest_endpoint(request: IngestRequest):
    try:
        pipeline = IngestionPipeline()
        pipeline.process_pdf(request.file_path)
        return {"status": "success", "message": f"Archivo {request.file_path} procesado."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ENDPOINT DE SALUD ---
@app.get("/")
async def root():
    return {"status": "online", "message": "API de IA para Vendure lista"}