import httpx
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SearchRequest, NamedVector
from fastembed import TextEmbedding
from src.config import Config
import uuid

class RAGEngine:
    def __init__(self):
        print("Inicializando RAG Engine...")
        
        # Load embedding model
        model_name = "sentence-transformers/all-MiniLM-L6-v2" if Config.EMBEDDING_MODEL == "all-MiniLM-L6-v2" else Config.EMBEDDING_MODEL
        self.embed_model = TextEmbedding(model_name=model_name)
        
        # Initialize Groq
        self.groq_client = Groq(
            api_key=Config.GROQ_KEY,
            http_client=httpx.Client()
        )
        
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
        
        # 2. Buscar en Qdrant (método correcto)
        try:
            print(f"🔎 Buscando en colección: {Config.COLLECTION_NAME}")
            
            # Verificar que la colección tiene documentos
            collection_info = self.qdrant_client.get_collection(Config.COLLECTION_NAME)
            print(f"📊 Documentos en colección: {collection_info.points_count}")
            
            if collection_info.points_count == 0:
                return "La base de conocimiento está vacía. Por favor, ejecuta la ingesta de documentos primero con: python run_ingestion.py", []
            
            # Búsqueda vectorial
            # Utilizando query_points ya que search puede no estar disponible
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
        
        # 4. Generar respuesta con Groq
        prompt = f"""
      Eres un Asistente de Compras Experto para [ecommerce]. Tu objetivo es ayudar a los usuarios a encontrar el producto ideal, resolver dudas técnicas y facilitar el proceso de compra utilizando únicamente la información proporcionada en el contexto.

        1. Instrucciones de Comportamiento
        Veracidad: Utiliza exclusivamente la información del "Contexto" proporcionado abajo. Si la respuesta no está en el contexto, di amablemente que no tienes esa información y ofrece contactar con soporte humano.
        Eres un Asistente de Compras Experto. Ayuda al usuario basándote SOLO en la información del contexto.

        Tono: Sé profesional, entusiasta y servicial. Usa un lenguaje que invite a la compra sin ser agresivo.
        CONTEXTO:
        {context}

        Formato: Usa negritas para nombres de productos y precios. Utiliza listas con viñetas para comparar características.
        PREGUNTA: {query}

        2. Reglas de Negocio
        Stock: Si el contexto indica que no hay stock, ofrece una alternativa similar presente en los documentos.

        Venta Cruzada: Siempre que recomiendes un producto, menciona brevemente un accesorio o producto complementario que aparezca en el contexto.

        Alucinaciones: No inventes descuentos, códigos promocionales ni fechas de entrega que no estén explícitamente en el texto.

        3. Estructura de la Respuesta
        Empatía: Valida la necesidad del usuario (ej: "Entiendo que buscas una cámara para viajes...").

        Respuesta Directa: Entrega la información solicitada o la recomendación basada en los datos.

        Llamado a la Acción (CTA): Finaliza con una pregunta para guiar la venta (ej: "¿Te gustaría que lo añada al carrito?" o "¿Deseas saber más sobre la garantía?").

        CONTEXTO RECUPERADO: {context}

        PREGUNTA DEL USUARIO: {query}
        """
        
        try:
            print("🤖 Generando respuesta con Groq...")
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=Config.CHAT_MODEL,
                temperature=0.5,
                max_tokens=1024
            )
            
            answer = response.choices[0].message.content
            print(f"✅ Respuesta generada ({len(answer)} caracteres)")
            
            return answer, list(set(sources))
            
        except Exception as e:
            error_msg = f"Error en generación Groq: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg, sources