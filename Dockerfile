# 1. IMAGEN BASE:

FROM python:3.9-slim

# 2. DIRECTORIO DE TRABAJO:
WORKDIR /app

# 3. GESTIÓN DE DEPENDENCIAS:
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 4. CÓDIGO FUENTE:
COPY . .

# 5. PUERTO:
EXPOSE 8000

# 6. COMANDO DE INICIO:
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]