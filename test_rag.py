# test_rag_gemini.py
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from typing import List

# 🔑 Gemini API anahtarını buraya yapıştır
genai.configure(api_key="AIzaSyC4cfKSsS_4ebWRdvAP3WJE0PBDytYNXRo")

DB_DIR = "db_gemini"

# ===============================
# 🧠 Gemini Embeddings (API üzerinden)
# ===============================
class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [
            genai.embed_content(
                model="models/text-embedding-004",
                content=t,
                task_type="retrieval_document"
            )["embedding"]
            for t in texts
        ]

    def embed_query(self, text: str) -> List[float]:
        return genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_query"
        )["embedding"]

# ===============================
# 🔍 Veritabanını yükle
# ===============================
print("🔍 Vektör veritabanı yükleniyor (Gemini API ile)...")
embeddings = GeminiEmbeddings()
vectordb = Chroma(embedding_function=embeddings, persist_directory=DB_DIR)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

print("\n✅ Gemini RAG sistemi hazır!")
print("Örnek: kulağın yapısında neler var, fotosentez nedir, DNA replikasyonu nasıl gerçekleşir?")
print("(Çıkmak için 'q' yaz.)\n")

# ===============================
# 💬 Sorgu döngüsü
# ===============================
while True:
    soru = input("🔹 Sorunu yaz: ")
    if soru.lower() == "q":
        print("🧩 Çıkış yapıldı.")
        break

    # 🔹 En alakalı bölümleri getir
    results = retriever.invoke(soru)

    print("\n📘 İlgili bölümler:\n")
    for i, doc in enumerate(results, start=1):
        print(f"--- Parça {i} (sayfa: {doc.metadata.get('page', 'bilinmiyor')}) ---")
        print(doc.page_content[:700], "\n")

    print("───────────────────────────────\n")
