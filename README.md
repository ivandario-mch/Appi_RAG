# 🧠 Azure Hybrid RAG System (Cost-Optimized)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Azure](https://img.shields.io/badge/Azure-AI%20Search-0078D4)
![Groq](https://img.shields.io/badge/Groq-LPU-orange)
![Status](https://img.shields.io/badge/Status-Production%20Ready-green)



##
### 1️⃣ Instalación

```bash
# Clonar repositorio
git clone https://github.com/kevinestebanpotosi/RAG.git
cd RAG
```

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate
```

```bash

```


```
RAG/
├── main.py                 # Punto de entrada (CLI interactiva)
├── requirements.txt        # Dependencias del proyecto
├── .env.example           # Template de variables de entorno
├── .gitignore            # Archivos ignorados en git
├── README.md             # Este archivo
│
├── src/
│   ├── __init__.py
│   ├── config.py         # Configuración centralizada
│   ├── ingestion.py      # Pipeline de ingesta de PDFs
│   └── rag_engine.py     # Motor de RAG (retrieval + generation)
│
├── data/
│   └── *.pdf             # PDFs para procesar (no se suben a git)
│
├── notebooks/            # Jupyter notebooks para experimentación
│   └── (análisis y pruebas)
```

---

## 📦 Dependencias Principales

| Paquete | Versión | Propósito |
|---------|---------|-----------|
| `azure-search-documents` | ≥11.4.0 | Cliente de Azure AI Search |
| `groq` | ≥0.5.0 | API de Groq para inferencia |
| `sentence-transformers` | ≥2.2.0 | Generación de embeddings locales |
| `pypdf` | ≥4.0.0 | Extracción de texto de PDFs |
| `torch` | ≥2.0.0 | Dependencia de transformers |
| `python-dotenv` | ≥1.0.0 | Gestión de variables de entorno |

---

## ⚙️ Configuración Avanzada

### Parámetros Ajustables (src/config.py)

```python
CHUNK_SIZE = 500           # Tamaño de cada chunk en caracteres
CHUNK_OVERLAP = 50         # Superposición entre chunks
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Modelo de embeddings
CHAT_MODEL = "llama-3.3-70b-versatile"  # Modelo LLM
INDEX_NAME = "portfolio-rag-index"     # Nombre del índice en Azure
```



