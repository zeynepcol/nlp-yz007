import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -----------------------------
# AYARLAR
# -----------------------------
DATA_PATH = "data/digital_transformation.txt"
SIMILARITY_THRESHOLD = 0.35  # konu dışı ayıklamak için
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# -----------------------------
# TXT OKUMA & CHUNK
# -----------------------------
def load_and_chunk_text(path, chunk_size=400):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks

# -----------------------------
# RAG SINIFI
# -----------------------------
class DigitalTransformationRAG:
    def __init__(self, chunks):
        self.model = SentenceTransformer(MODEL_NAME)
        self.chunks = chunks
        self.chunk_embeddings = self.model.encode(chunks)

    def query(self, question):
        question_embedding = self.model.encode([question])
        similarities = cosine_similarity(
            question_embedding, self.chunk_embeddings
        )[0]

        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        if best_score < SIMILARITY_THRESHOLD:
            return None, best_score

        return self.chunks[best_idx], best_score

# -----------------------------
# BENCHMARK SORULARI
# -----------------------------
benchmark_questions = [
    # KONU İÇİ (8)
    "Dijital dönüşüm stratejisi olmayan bir işletme hangi risklerle karşılaşır?",
    "Dijital dönüşüm neden sadece teknoloji yatırımı değildir?",
    "Dijital dönüşüm çalışan direnciyle karşılaşırsa ne olur?",
    "Dijital dönüşüm işletmelerin rekabet gücünü nasıl etkiler?",
    "Dijital dönüşüm süreçlerinde insan faktörü neden kritiktir?",
    "Dijitalleşme işletme kültürünü nasıl değiştirir?",
    "Dijital dönüşüm karar alma süreçlerini nasıl etkiler?",
    "Dijital dönüşüm başarısız olursa olası nedenler nelerdir?",

    # KONU DIŞI (2)
    "Fransız Devrimi’nin Avrupa’ya etkileri nelerdir?",
    "Kuantum bilgisayarlar nasıl çalışır?"
]

# -----------------------------
# ANA ÇALIŞMA
# -----------------------------
if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("TXT dosyası bulunamadı.")

    chunks = load_and_chunk_text(DATA_PATH)
    rag = DigitalTransformationRAG(chunks)

    print("\n===== DİJİTAL DÖNÜŞÜM RAG BENCHMARK TESTİ =====\n")

    for i, question in enumerate(benchmark_questions, 1):
        print(f"SORU {i}: {question}")
        print("-" * 60)

        result, score = rag.query(question)

        if result is None:
            print("CEVAP: Bu soru doküman kapsamı dışındadır.")
        else:
            print("CEVAP (RAG):")
            print(result[:700] + "...")
            print(f"\nBenzerlik Skoru: {score:.3f}")

        print("\n" + "=" * 60 + "\n")