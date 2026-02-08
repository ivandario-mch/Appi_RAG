import os
from pypdf import PdfReader
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.config import Config
import uuid

class IngestionPipeline:
    def __init__(self):
        print(f"Loading Embedding Model: {Config.EMBEDDING_MODEL}...")
        # Using the same model name as before if supported, or default
        model_name = "sentence-transformers/all-MiniLM-L6-v2" if Config.EMBEDDING_MODEL == "all-MiniLM-L6-v2" else Config.EMBEDDING_MODEL
        self.embed_model = TextEmbedding(model_name=model_name)
        
        # Qdrant Client (reemplaza Azure)
        print(f"Connecting to Qdrant at {Config.QDRANT_URL}...")
        self.qdrant_client = QdrantClient(
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY,
            timeout=10.0
        )
        
        self._create_collection_if_not_exists()

    def _create_collection_if_not_exists(self):
        """Crea la colección en Qdrant si no existe"""
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if Config.COLLECTION_NAME not in collection_names:
                print(f"Creating Collection: {Config.COLLECTION_NAME}")
                self.qdrant_client.create_collection(
                    collection_name=Config.COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=Config.VECTOR_SIZE,  # 384 para all-MiniLM-L6-v2
                        distance=Distance.COSINE
                    )
                )
                print(f"✅ Collection '{Config.COLLECTION_NAME}' created successfully")
            else:
                print(f"✅ Collection '{Config.COLLECTION_NAME}' already exists")
        except Exception as e:
            print(f"❌ Error creating/checking collection: {e}")
            raise

    def process_pdf(self, file_path):
        """Procesa un archivo PDF y lo indexa en Qdrant"""
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return

        print(f"📄 Processing: {file_path}")
        
        try:
            reader = PdfReader(file_path)
            text = "".join([page.extract_text() for page in reader.pages])
        except Exception as e:
            print(f"❌ Error reading PDF: {e}")
            return
        
        if not text.strip():
            print(f"⚠️  Warning: No text extracted from {file_path}")
            return
        
        # Chunking Strategy
        chunks = []
        for i in range(0, len(text), Config.CHUNK_SIZE - Config.CHUNK_OVERLAP):
            chunk = text[i : i + Config.CHUNK_SIZE].strip()
            if chunk:  # Solo agregar chunks no vacíos
                chunks.append(chunk)
            
        print(f"📝 Generated {len(chunks)} chunks. Embedding and uploading...")
        
        # Procesar en batches
        batch = []
        batch_size = 50
        total_uploaded = 0
        
        for i, chunk in enumerate(chunks):
            try:
                # Embedizar el chunk
                vector = list(self.embed_model.embed([chunk]))[0].tolist()
                
                # Crear punto para Qdrant
                point = PointStruct(
                    id=str(uuid.uuid4()),  # ID único
                    vector=vector,
                    payload={
                        "content": chunk,
                        "source": os.path.basename(file_path),
                        "chunk_index": i,
                        "file_path": file_path
                    }
                )
                batch.append(point)
                
                # Subir batch cuando alcance el tamaño
                if len(batch) >= batch_size:
                    self.qdrant_client.upsert(
                        collection_name=Config.COLLECTION_NAME,
                        points=batch
                    )
                    total_uploaded += len(batch)
                    print(f"  ✓ Uploaded {total_uploaded}/{len(chunks)} chunks...")
                    batch = []
                    
            except Exception as e:
                print(f"⚠️  Error processing chunk {i}: {e}")
                continue
        
        # Subir el último batch si existe
        if batch:
            try:
                self.qdrant_client.upsert(
                    collection_name=Config.COLLECTION_NAME,
                    points=batch
                )
                total_uploaded += len(batch)
                print(f"  ✓ Uploaded {total_uploaded}/{len(chunks)} chunks")
            except Exception as e:
                print(f"❌ Error uploading final batch: {e}")
        
        print(f"✅ Ingestion Complete: {total_uploaded} chunks from {os.path.basename(file_path)}")
    
    def get_collection_stats(self):
        """Obtiene estadísticas de la colección"""
        try:
            info = self.qdrant_client.get_collection(Config.COLLECTION_NAME)
            return {
                "name": Config.COLLECTION_NAME,
                "points_count": info.points_count,
                "indexed_vectors_count": info.indexed_vectors_count if hasattr(info, 'indexed_vectors_count') else info.points_count,
                "status": info.status
            }
        except Exception as e:
            return {"error": str(e)}
    
    def delete_collection(self):
        """Elimina la colección (útil para resetear)"""
        try:
            self.qdrant_client.delete_collection(Config.COLLECTION_NAME)
            print(f"✅ Collection '{Config.COLLECTION_NAME}' deleted")
        except Exception as e:
            print(f"❌ Error deleting collection: {e}")