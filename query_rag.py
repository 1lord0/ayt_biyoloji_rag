import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from build_index_gemini import GeminiEmbeddings
import os

# 🔑 API key'i Streamlit Cloud'dan secrets'tan al (lokalde istersen direkt yazabilirsin)
genai.configure(api_key=os.getenv("GEMINI_API_KEY", "AIzaSyC4cfKSsS_4ebWRdvAP3WJE0PBDytYNXRo"))

# 📦 Veritabanı dizini
DB_DIR = "db_gemini"

# 🔹 Chroma veritabanını yükle
vectordb = Chroma(persist_directory=DB_DIR, embedding_function=GeminiEmbeddings())

def ask_gemini(question, k=5):
    """PDF veritabanından bilgi çekip Gemini ile cevap oluşturur"""
    docs = vectordb.similarity_search(question, k=k)

    # İlgili parçaları birleştir
    context = "\n\n".join([f"Parça {i+1}: {d.page_content}" for i, d in enumerate(docs)])

    # Prompt
    prompt = f"""
    Aşağıda MEB AYT Biyoloji kitabından alınan bilgiler yer alıyor.
    Sadece bu bilgilere dayanarak soruya net, sade ve doğru bir yanıt ver.
    Uydurma yapma, emin değilsen "Bilmiyorum" de.

    Soru: {question}

    Kaynak Bilgiler:
    {context}
    """

    # Gemini API çağrısı
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    return response.text, docs
