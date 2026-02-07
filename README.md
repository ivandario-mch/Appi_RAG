# 🧠 Azure Hybrid RAG System (Cost-Optimized)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Azure](https://img.shields.io/badge/Azure-AI%20Search-0078D4)
![Groq](https://img.shields.io/badge/Groq-LPU-orange)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)


###  Clone the Repository
```bash
git clone https://github.com/kevinestebanpotosi/Appi_RAG.git
cd Appi_RAG
```



###  Local Installation

#### Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### Run the Application
Start the development server:
```bash
uvicorn main:app --reload
```
The API will be available at: `http://localhost:8000`

### 📂 Configuración de Carpetas Locales

Para que el proyecto funcione correctamente y no se suban archivos sensibles o pesados al repositorio, debes asegurarte de tener la siguiente estructura local (estas carpetas están ignoradas por git):

1.  **`data/`**: Crea esta carpeta en la raíz. Aquí es donde debes colocar los archivos PDF que deseas procesar.
    ```bash
    mkdir data
    ```
2.  **`.env`**: Crea este archivo con tus credenciales (ver sección de Configuración al final).
3.  **`venv/`**: Se crea automáticamente al instalar el entorno virtual (ver paso anterior).

> **Nota**: `data/*.pdf` y `.env` están configurados en `.gitignore` y `.dockerignore` para seguridad.

---

## 🐳 Docker Deployment

To build and run the application in a production-like environment:

### Build Image
```bash
docker build -t appi-rag .
```

### Run Container
Make sure your `.env` file is present, then run:
```bash
docker run -p 8000:8000 --env-file .env appi-rag
```

---

## 📡 API Usage

### Health Check
```bash
curl http://localhost:8000/docs
# Output: {"status": "online", ...}
```

### 1. Ingest a PDF (Local File)
**Note:** This runs locally on the server filesystem.
```bash
curl -X POST "http://localhost:8000/ingest" \
     -H "Content-Type: application/json" \
     -d '{"file_path": "data/sample_document.pdf"}'
```

### 2. Chat with Documents
```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "What are the key takeaways from the document?"}'
```

---

## 📦 Project Structure
'''
Appi_RAG/
├── data/                      # 📁 Coloca tus PDFs aquí
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── ingestion.py
│   └── rag_engine.py
├── main.py                   # API FastAPI
├── probe.py                  # 🛠 Script de prueba (Nuevo)
├── run_ingestion.py
├── .env
├── requirements.txt
└── README.md
'''
## 🛠 Tech Stack
| Component | Technology | Purpose |
|-----------|------------|---------|
| **Core** | Python 3.9+, FastAPI | Framework de API |
| **Embeddings** | FastEmbed (Qdrant) | Vectorización local ligera |
| **Vector DB** | Qdrant | Base de datos vectorial (Reemplaza a Azure) |
| **LLM** | Groq (Llama 3.3) | Inferencia ultra-rápida |

> [!WARNING]
> **Nota sobre Qdrant Client**:
> En versiones recientes de `qdrant-client` (1.16+), el método `search` puede no estar disponible o comportarse de manera diferente. Este proyecto utiliza `client.query_points(...)` en `src/rag_engine.py` para garantizar la compatibilidad y estabilidad.


## 🔑 Configuration (.env)

Create a `.env` file in the root directory with the following variables:

```ini
QDRANT_URL="your_qdrant_url"
QDRANT_API_KEY="your_qdrant_api_key"
GROQ_API_KEY="your_groq_api_key"
COLLECTION_NAME="ecommerce-rag-collection"
```


