# 1. IMAGEN BASE:
# Imagina que compras una computadora vacía con Linux y Python 3.9 instalado.
# "slim" significa que es una versión ligera (ocupa menos espacio/memoria).
FROM python:3.9-slim

# 2. DIRECTORIO DE TRABAJO:
# Creas una carpeta llamada "app" dentro de esa computadora y te metes ahí.
WORKDIR /app

# 3. GESTIÓN DE DEPENDENCIAS:
# Copias tu lista de ingredientes (librerías) a la computadora.
COPY requirements.txt .

# Instalas las librerías. --no-cache-dir es para no guardar basura y que pese menos.
RUN pip install --no-cache-dir -r requirements.txt

# 4. CÓDIGO FUENTE:
# Copias todos tus archivos (.py) de tu PC real a la carpeta "app" del contenedor.
COPY . .

# 5. PUERTO:
# Le dices a Docker que este contenedor "escuchará" por el puerto 8000.
EXPOSE 8000

# 6. COMANDO DE INICIO:
# ¿Qué pasa cuando le das al botón de "ON"? Ejecuta uvicorn (el servidor web).
# 0.0.0.0 significa "acepto conexiones de cualquier lugar" (necesario para la nube).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]