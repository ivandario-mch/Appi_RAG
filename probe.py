import sys
import os

# Ensure the current directory is in sys.path to allow imports from main
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from main import app

def test_chat():
    print("Initializing TestClient...")
    client = TestClient(app)
    
    query = "Hola, ¿qué productos tienes?"
    print(f"\nSending query: '{query}'")
    
    try:
        response = client.post("/chat", json={"message": query})
        
        if response.status_code == 200:
            data = response.json()
            print("\n✅ Success!")
            print(f"Answer: {data.get('answer')}")
            print(f"Sources: {data.get('sources')}")
        else:
            print(f"\n❌ Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"\n❌ Exception during request: {e}")

if __name__ == "__main__":
    test_chat()
