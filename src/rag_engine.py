import httpx
from groq import Groq
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from sentence_transformers import SentenceTransformer
from src.config import Config

class RAGEngine:
    def __init__(self):
        # Load embedding model
        self.embed_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        
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
        vector = self.embed_model.encode(query).tolist()
        
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
        You are an expert assistant. Use the context to answer in detail.
        
        Context:
        {context}
        
        Question: {query}
        """
        
        response = self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=Config.CHAT_MODEL,
            temperature=0.5
        )
        
        return response.choices[0].message.content, [r['source'] for r in results]