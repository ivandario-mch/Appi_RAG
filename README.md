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
curl http://localhost:8000/
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
```
Appi_RAG/
├── main.py                 # FastAPI Application Application entry point
├── Dockerfile              # Container definition
├── requirements.txt        # Python dependencies (Optimized)
├── .env                    # Environment variables (DO NOT COMMIT)
├── README.md               # Documentation
│
├── src/
│   ├── config.py           # Configuration loader
│   ├── ingestion.py        # PDF processing & Azure upload pipeline
│   └── rag_engine.py       # RAG logic (Retrieval + Generation)
│
└── data/                   # Directory for storing local PDFs
```

---

## 🛠 Tech Stack
| Component | Technology | Purpose |
|-----------|------------|---------|
| **Core** | Python 3.9, FastAPI | API Framework |
| **Embeddings** | FastEmbed (Qdrant) | Lightweight local vectorization |
| **Vector DB** | Azure AI Search | Enterprise-grade vector storage |
| **LLM** | Groq (Llama 3.3) | Ultra-fast inference |

