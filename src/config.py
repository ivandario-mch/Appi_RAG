import os
from dotenv import load_dotenv

if os.path.exists('.env'):
    load_dotenv()

class Config:
    # Qdrant Cloud
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    
    # Azure AI Foundry / OpenAI
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
    AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    
    # Groq (backup opcional)
    GROQ_KEY = os.getenv("GROQ_API_KEY")
    
    # LLM Provider: "azure" o "groq"
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # Por defecto Groq
    
    # Hyperparameters
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ecommerce-rag-collection")
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    VECTOR_SIZE = 384
    
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

    @classmethod
    def validate(cls):
        missing = []
        
        # Validar Qdrant (siempre requerido)
        if not cls.QDRANT_URL:
            missing.append("QDRANT_URL")
        if not cls.QDRANT_API_KEY:
            missing.append("QDRANT_API_KEY")
        
        # Validar LLM según provider
        if cls.LLM_PROVIDER == "azure":
            if not cls.AZURE_OPENAI_ENDPOINT:
                missing.append("AZURE_OPENAI_ENDPOINT")
            if not cls.AZURE_OPENAI_KEY:
                missing.append("AZURE_OPENAI_KEY")
            if not cls.AZURE_OPENAI_DEPLOYMENT:
                missing.append("AZURE_OPENAI_DEPLOYMENT")
        elif cls.LLM_PROVIDER == "groq":
            if not cls.GROQ_KEY:
                missing.append("GROQ_API_KEY")
        else:
            missing.append(f"LLM_PROVIDER inválido: {cls.LLM_PROVIDER} (debe ser 'azure' o 'groq')")
        
        if missing:
            raise ValueError(f"Missing or invalid environment variables: {', '.join(missing)}")
        
        print(f"✅ Config validated")
        print(f"   LLM Provider: {cls.LLM_PROVIDER}")
        print(f"   Qdrant URL: {cls.QDRANT_URL}")
        print(f"   Collection: {cls.COLLECTION_NAME}")