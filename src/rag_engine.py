import httpx
from groq import Groq
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from fastembed import TextEmbedding
from src.config import Config

class RAGEngine:
    def __init__(self):
        # Load embedding model (FastEmbed is lighter and faster)
        # using the same model name as before if supported, or default
        # "sentence-transformers/all-MiniLM-L6-v2" is supported by fastembed
        model_name = "sentence-transformers/all-MiniLM-L6-v2" if Config.EMBEDDING_MODEL == "all-MiniLM-L6-v2" else Config.EMBEDDING_MODEL
        self.embed_model = TextEmbedding(model_name=model_name)
        
        # Initialize Groq with clean client
        self.groq_client = Groq(
            api_key=Config.GROQ_KEY,
            http_client=httpx.Client()
        )
        
        # Azure Client
        self.search_client = SearchClient(
            Config.AZURE_ENDPOINT, 
            Config.INDEX_NAME, 
            AzureKeyCredential(Config.AZURE_KEY)
        )

    def chat(self, query):
        # 1. Vector Projection
        # fastembed returns a generator, so we convert to list and take the first item
        vector = list(self.embed_model.embed([query]))[0].tolist()
        
        # 2. Azure Search
        vector_query = VectorizedQuery(
            vector=vector, 
            k_nearest_neighbors=5, 
            fields="content_vector"
        )
        
        results = list(self.search_client.search(
            vector_queries=[vector_query], 
            select=["content", "source"]
        ))
        
        context = "\n".join([f"[{r['source']}] {r['content']}" for r in results])
        
        if not context:
            return "No relevant context found in Azure.", []

        # 3. Generation
        prompt = f"""
      Eres un Asistente de Compras Experto para [ecommerce]. Tu objetivo es ayudar a los usuarios a encontrar el producto ideal, resolver dudas técnicas y facilitar el proceso de compra utilizando únicamente la información proporcionada en el contexto.

1. Instrucciones de Comportamiento
Veracidad: Utiliza exclusivamente la información del "Contexto" proporcionado abajo. Si la respuesta no está en el contexto, di amablemente que no tienes esa información y ofrece contactar con soporte humano.

Tono: Sé profesional, entusiasta y servicial. Usa un lenguaje que invite a la compra sin ser agresivo.

Formato: Usa negritas para nombres de productos y precios. Utiliza listas con viñetas para comparar características.

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
        
        response = self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=Config.CHAT_MODEL,
            temperature=0.5
        )
        
        return response.choices[0].message.content, [r['source'] for r in results]