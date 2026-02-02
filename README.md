# 🧠 Azure Hybrid RAG System (Cost-Optimized)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Azure](https://img.shields.io/badge/Azure-AI%20Search-0078D4)
![Groq](https://img.shields.io/badge/Groq-LPU-orange)
![Status](https://img.shields.io/badge/Status-Production%20Ready-green)

**Key Features:**
- 🎯 Local PDF Processing (Edge Computing)
- 🚀 Locally Generated Embeddings (No API required)
- ☁️ Scalable Vector Storage in Azure AI Search
- 💬 Interactive Chat powered by Groq LPU (Llama 3.3 70B)

---

## 1️⃣ Installation

```bash
# Clone repository
git clone https://github.com/kevinestebanpotosi/RAG.git
cd RAG
```

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

```bash
# Install dependencies
pip install -r requirements.txt
```

---

## 📂 Project Structure

```
RAG/
├── main.py                 # Entry point (FastAPI App)
├── requirements.txt        # Project dependencies
├── .env.example           # Environment variables template
├── .gitignore            # Git ignore file
├── README.md             # This file
│
├── src/
│   ├── __init__.py
│   ├── config.py         # Centralized configuration
│   ├── ingestion.py      # PDF ingestion pipeline
│   └── rag_engine.py     # RAG Engine (retrieval + generation)
│
├── data/
│   └── *.pdf             # PDFs to process (not in git)
│
├── notebooks/            # Jupyter notebooks for experimentation
│   └── (analysis and tests)
```

---

## 📦 Main Dependencies

| Package | Version | Purpose |
|---------|---------|-----------|
| `azure-search-documents` | ≥11.4.0 | Azure AI Search Client |
| `groq` | ≥0.5.0 | Groq API for inference |
| `sentence-transformers` | ≥2.2.0 | Local embedding generation |
| `pypdf` | ≥4.0.0 | PDF text extraction |
| `torch` | ≥2.0.0 | Transformers dependency |
| `python-dotenv` | ≥1.0.0 | Environment variable management |

---

## ⚙️ Advanced Configuration

### Adjustable Parameters (src/config.py)

```python
CHUNK_SIZE = 500           # Chunk size in characters
CHUNK_OVERLAP = 50         # Overlap between chunks
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Embedding model
CHAT_MODEL = "llama-3.3-70b-versatile"  # LLM Model
INDEX_NAME = "portfolio-rag-index"     # Azure Index Name
```
