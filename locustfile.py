from locust import HttpUser, task, between
import random

# Preguntas típicas de un e-commerce
PREGUNTAS = [
    "¿Que es ecommer?",
    "¿Que beneficions tengo como vendedor?",
    "¿que puede hacer un comprador en ecommer?",
    "tienen facturacion electronica"
]

class UsuarioEcommerce(HttpUser):
    # Tiempo de espera entre preguntas (simula a una persona pensando)
    # Entre 5 y 15 segundos
    wait_time = between(5, 15)

    @task
    def preguntar_al_rag(self):
        pregunta = random.choice(PREGUNTAS)
        
        # Asumiendo que tu backend RAG está en /api/chat
        with self.client.post("/api/chat", json={"message": pregunta}, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 429:
                # Esto es lo que buscas detectar: Límites de Groq
                response.failure("Rate Limit Exceeded (Groq)")
            else:
                response.failure(f"Error: {response.status_code}")