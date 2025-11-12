import os
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from build_index_gemini import GeminiEmbeddings

# 🔑 API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

DB_DIR = "db_gemini"

# 📦 Vektör veritabanını yükle
vectordb = Chroma(persist_directory=DB_DIR, embedding_function=GeminiEmbeddings())

def ask_gemini(question, k=5):
    """PDF veritabanından bilgi çekip Gemini ile cevap oluşturur"""
    try:
        docs = vectordb.similarity_search(question, k=k)
    except Exception as e:
        return f"Veritabanı hatası: {e}", []

    # 🔹 Bağlam birleştirme
    if not docs:
        context = "PDF içeriğinde bu soruyla doğrudan ilgili bilgi bulunamadı."
    else:
        context = "\n\n".join([f"{i+1}. {d.page_content}" for i, d in enumerate(docs)])

    # 🔹 Optimize edilmiş prompt
    prompt = f"""
    Aşağıda MEB AYT Biyoloji kitabından alınmış bilgiler yer alıyor.
    Bu bilgiler ışığında aşağıdaki soruyu açıklayıcı ve sade bir Türkçe ile cevapla.
    Gereksiz tekrarlardan kaçın. Sadece PDF içeriğine dayan, uydurma bilgi ekleme.
    Eğer kaynaklarda doğrudan bilgi yoksa "Kitapta bu konuda net bilgi bulunmamaktadır." de.

    📘 Soru:
    {question}

    📚 Kaynak Metinler:
    {context}
    """

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text.strip(), docs
    except Exception as e:
        return f"Model hatası: {e}", []
