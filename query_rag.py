import os
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Gemini API key'i ortam değişkeninden al
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Vektör veritabanı yolu
DB_DIR = "db_gemini"

# Türkçe uyumlu embedding modeli (dilersen Gemini'yi de kullanabiliriz)
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

# Chroma veritabanını yükle
vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)


def ask_gemini(question):
    try:
        # Sorguya en yakın 3 belgeyi bul
        docs = vectordb.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Prompt oluştur
        prompt = f"""
        MEB AYT Biyoloji kitabındaki bilgiler temel alınarak bu soruya sade ve kısa bir açıklama yap.
        Kaynakta tam bilgi yoksa, konuya uygun genel bir açıklama da ekleyebilirsin.

        Soru: {question}
        Kaynak metinler:
        {context}
        """

        # Gemini modelini çağır
        model = genai.GenerativeModel("models/gemini-2.0-flash")
        response = model.generate_content(prompt)

        return response.text.strip()

    except Exception as e:
        return f"⚠️ Hata: {e}"
