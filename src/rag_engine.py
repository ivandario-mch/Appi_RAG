import httpx
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from fastembed import TextEmbedding
from src.config import Config

class RAGEngine:
    def __init__(self):
        print("🚀 Inicializando RAG Engine...")
        
        # Cargar modelo de embedding
        model_name = "sentence-transformers/all-MiniLM-L6-v2" if Config.EMBEDDING_MODEL == "all-MiniLM-L6-v2" else Config.EMBEDDING_MODEL
        self.embed_model = TextEmbedding(model_name=model_name)
        
        # Cliente Groq
        self.groq_client = Groq(
            api_key=Config.GROQ_KEY,
            http_client=httpx.Client()
        )
        
        # Cliente Qdrant
        self.qdrant_client = QdrantClient(
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY,
            timeout=10.0
        )
        
        self._ensure_collection_exists()
        
        # Historial simple en memoria (Opcional: para recordar la conversación actual)
        self.chat_history = [] 

    def _ensure_collection_exists(self):
        try:
            if not self.qdrant_client.collection_exists(Config.COLLECTION_NAME):
                print(f"📦 Creando colección: {Config.COLLECTION_NAME}")
                self.qdrant_client.create_collection(
                    collection_name=Config.COLLECTION_NAME,
                    vectors_config=VectorParams(size=Config.VECTOR_SIZE, distance=Distance.COSINE)
                )
        except Exception as e:
            print(f"❌ Error verificando colección: {e}")

    def chat(self, query):
        print(f"\n👤 Usuario: {query}")

        # 1. Detección rápida de saludos (Para ser más amigable y rápido)
        saludos = ["hola", "buenos dias", "buenas", "que tal", "hello"]
        if query.lower().strip() in saludos:
            return "¡Hola! 👋 Soy tu asistente de compras experto. ¿En qué producto estás interesado hoy o qué necesitas encontrar?", []

        # 2. Embedizar
        try:
            vector = list(self.embed_model.embed([query]))[0].tolist()
        except Exception as e:
            return "Tuve un pequeño problema técnico procesando tu pregunta. ¿Podrías intentarlo de nuevo?", []

        # 3. Buscar en Qdrant
        try:
            search_results = self.qdrant_client.query_points(
                collection_name=Config.COLLECTION_NAME,
                query=vector,
                limit=4, # 4 resultados suelen ser suficientes y confunden menos al modelo
                score_threshold=0.4 # Subimos un poco el umbral para calidad
            ).points
        except Exception as e:
            print(f"❌ Error Qdrant: {e}")
            return "No pude acceder a mi catálogo en este momento. Intenta más tarde.", []

        # 4. Construir Contexto
        if not search_results:
            # Fallback amigable si no hay info
            return "Mmm, revisé mi catálogo y no encontré nada exacto sobre eso. 🧐 ¿Podrías darme más detalles o buscar una categoría diferente?", []

        context_text = ""
        sources = []
        for result in search_results:
            source = result.payload.get("source", "General")
            content = result.payload.get("content", "")
            # Añadimos metadatos útiles al contexto (ej. precio si existe en payload)
            price = result.payload.get("price", "Consultar") 
            
            context_text += f"- [Fuente: {source}] (Precio ref: {price}): {content}\n"
            sources.append(source)

        sources = list(set(sources))

        # 5. Generar Respuesta (EL CAMBIO CLAVE: SYSTEM PROMPT)
        
        # Definimos la personalidad del bot en el System Prompt
        system_prompt = """
        Eres Alex, un asistente de compras virtual súper útil, experto y carismático para [TU ECOMMERCE].
        
        TUS OBJETIVOS:
        1. Ayudar al usuario a decidir su compra basándote EXCLUSIVAMENTE en el CONTEXTO proporcionado.
        2. Si el contexto tiene la respuesta, sé directo y entusiasta.
        3. Si el contexto NO tiene la respuesta, admítelo amablemente y sugiere contactar a soporte, no inventes datos.
        4. Usa emojis moderadamente para dar calidez (👋, ✅, 🚀).
        
        FORMATO:
        - Usa **negritas** para resaltar productos y precios clave.
        - Sé conciso. No escribas párrafos gigantes.
        - Siempre termina con una pregunta o sugerencia que invite a la acción (ej: "¿Te lo agrego al carrito?", "¿Quieres ver modelos similares?").
        """

        user_prompt = f"""
        CONTEXTO DEL CATÁLOGO:
        {context_text}

        PREGUNTA DEL CLIENTE:
        {query}
        """

        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    # Aquí podrías inyectar self.chat_history si quisieras memoria
                    {"role": "user", "content": user_prompt}
                ],
                model=Config.CHAT_MODEL,
                temperature=0.3, # Bajamos temperatura para que sea creativo pero fiel a los datos
                max_tokens=800
            )
            
            answer = response.choices[0].message.content
            
            # Guardar historial simple (Opcional)
            self.chat_history.append({"role": "user", "content": query})
            self.chat_history.append({"role": "assistant", "content": answer})
            
            return answer, sources

        except Exception as e:
            print(f"❌ Error Groq: {e}")
            return "Tuve un problema generando la respuesta. ¿Podemos intentar con otra pregunta?", sources