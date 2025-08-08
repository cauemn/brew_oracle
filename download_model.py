# download_model.py
from sentence_transformers import SentenceTransformer
import os

os.makedirs("./models", exist_ok=True)

print("⬇️ Baixando modelo all-MiniLM-L6-v2...")
model = SentenceTransformer("all-MiniLM-L6-v2")
model.save("./models/all-MiniLM-L6-v2")
print("✅ Modelo salvo em ./models/all-MiniLM-L6-v2")
