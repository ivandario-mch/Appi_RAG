import httpx
from groq import Groq
from openai import AzureOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding
from src.config import Config
import uuid

class RAGEngine:
    def __init__(self):
        print("Inicializando RAG Engine...")
        
        # Load embedding model
        model_name = "sentence-transformers/all-MiniLM-L6-v2" if Config.EMBEDDING_MODEL == "all-MiniLM-L6-v2" else Config.EMBEDDING_MODEL
        self.embed_model = TextEmbedding(model_name=model_name)
        
        # Inicializar LLM según provider configurado
        if Config.LLM_PROVIDER == "azure":
            print("🔷 Usando Azure AI Foundry")
            self.llm_client = AzureOpenAI(
                api_key=Config.AZURE_OPENAI_KEY,
                api_version=Config.AZURE_OPENAI_API_VERSION,
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT
            )
            self.model_name = Config.AZURE_OPENAI_DEPLOYMENT
            self.is_azure = True
        else:
            print("🟢 Usando Groq")
            self.llm_client = Groq(
                api_key=Config.GROQ_KEY,
                http_client=httpx.Client()
            )
            self.model_name = "llama-3.3-70b-versatile"
            self.is_azure = False
        
        # Qdrant Client
        print(f"Conectando a Qdrant: {Config.QDRANT_URL}")
        self.qdrant_client = QdrantClient(
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY,
            timeout=10.0
        )
        
        self._ensure_collection_exists()
        print("✅ RAG Engine inicializado correctamente")
    
    def _ensure_collection_exists(self):
        """Crea la colección en Qdrant si no existe"""
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if Config.COLLECTION_NAME not in collection_names:
                print(f"Creando colección: {Config.COLLECTION_NAME}")
                self.qdrant_client.create_collection(
                    collection_name=Config.COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=Config.VECTOR_SIZE,
                        distance=Distance.COSINE
                    )
                )
                print(f"✅ Colección '{Config.COLLECTION_NAME}' creada")
            else:
                print(f"✅ Colección '{Config.COLLECTION_NAME}' existe")
        except Exception as e:
            print(f"❌ Error con colección: {e}")
            raise

    def chat(self, query):
        """
        Realiza búsqueda vectorial y genera respuesta
        """
        print(f"\n🔍 Query recibida: {query}")
        
        # 1. Embedizar la pregunta
        try:
            vector = list(self.embed_model.embed([query]))[0].tolist()
            print(f"✅ Vector generado (dimensión: {len(vector)})")
        except Exception as e:
            error_msg = f"Error al embedizar: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg, []
        
        # 2. Buscar en Qdrant
        try:
            print(f"🔎 Buscando en colección: {Config.COLLECTION_NAME}")
            
            # Verificar que la colección tiene documentos
            collection_info = self.qdrant_client.get_collection(Config.COLLECTION_NAME)
            print(f"📊 Documentos en colección: {collection_info.points_count}")
            
            if collection_info.points_count == 0:
                return "La base de conocimiento está vacía. Por favor, ejecuta la ingesta de documentos primero con: python run_ingestion.py", []
            
            # Búsqueda vectorial con query_points
            query_response = self.qdrant_client.query_points(
                collection_name=Config.COLLECTION_NAME,
                query=vector,
                limit=5,
                score_threshold=0.3
            )
            search_results = query_response.points
            
            print(f"✅ Encontrados {len(search_results)} resultados")
            
        except Exception as e:
            error_msg = f"Error en búsqueda Qdrant: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg, []
        
        # 3. Procesar resultados
        if not search_results:
            msg = "No encontré información relevante. Intenta reformular tu pregunta."
            print(f"⚠️ {msg}")
            return msg, []
        
        context_parts = []
        sources = []
        
        for i, result in enumerate(search_results):
            content = result.payload.get("content", "")
            source = result.payload.get("source", "Unknown")
            score = result.score
            
            print(f"  [{i+1}] Fuente: {source} | Score: {score:.3f}")
            context_parts.append(f"[{source}] {content}")
            sources.append(source)
        
        context = "\n\n".join(context_parts)
        
        # 4. Generar respuesta (Azure o Groq)
        system_prompt = """Eres un Asistente de Compras Experto para [ecommer]. 

Instrucciones:
- Usa SOLO la información del contexto proporcionado
- Sé profesional, entusiasta y servicial
- Menciona precios y características específicas
- Si la información no está en el contexto, dilo honestamente
- Usa formato markdown para productos y precios
- Finaliza con una pregunta para guiar la venta"""

        user_prompt = f"""CONTEXTO:
{context}

PREGUNTA DEL USUARIO: {query}

Responde basándote exclusivamente en el contexto anterior."""
        
        try:
            provider_name = "Azure AI Foundry" if self.is_azure else "Groq"
            print(f"🤖 Generando respuesta con {provider_name}...")
            
            # Llamada compatible con Azure OpenAI y Groq
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                max_tokens=512
            )
            
            answer = response.choices[0].message.content
            print(f"✅ Respuesta generada ({len(answer)} caracteres)")
            
            return answer, list(set(sources))
            
        except Exception as e:
            error_msg = f"Error en generación {provider_name}: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg, sources