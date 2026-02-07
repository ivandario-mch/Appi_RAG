import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.config import Config
#from src.ingestion import IngestionPipeline
from src.rag_engine import RAGEngine

# Initialization
app = FastAPI(title="AI appi RAG")

# Validate configuration on startup
try:
    Config.validate()
    # Instantiate engine once
    engine = RAGEngine()
except Exception as e:
    print(f"❌ Initial Error: {e}")
    sys.exit(1)

# Data Models
class ChatRequest(BaseModel):
    message: str

class IngestRequest(BaseModel):
    file_path: str

# Chat Endpoint
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        answer, sources = engine.chat(request.message)
        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e) )

# Ingestion Endpoint
"""@app.post("/ingest")
async def ingest_endpoint(request: IngestRequest):
    try:
        pipeline = IngestionPipeline()
        pipeline.process_pdf(request.file_path)
        return {"status": "success", "message": f"File {request.file_path} processed."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))"""

# Health Check Endpoint
@app.get("/")
async def root():
    return {"status": "online", "message": "RAG API Ready"}