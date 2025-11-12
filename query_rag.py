import os
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from build_index_gemini import GeminiEmbeddings

# 🔑 API key'i al (Streamlit secrets üzerinden)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# 📂 Veritabanı klasörü
DB_DIR = "db_gemini"

# 📦 Chroma veritabanını yükle
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

# 📦 RAM tabanlı (in-memory) veritabanı
vectordb = Chroma(
    persist_directory=None,   # 💡 disk yok
    embedding_function=GeminiEmbeddings()
)

def ask_gemini(question, k=2):
    """
    AYT Biyoloji PDF veritabanından bilgi çekip
    Gemini API ile hızlı şekilde cevap oluşturur.
    """

    try:
        # En alakalı k adet parçayı bul
        docs = vectordb.similarity_search(question, k=k)
    except Exception as e:
        return f"⚠️ Veritabanı hatası: {e}", []

    # Bağlam birleştirme
    if not docs:
        context = "PDF içeriğinde bu soruyla ilgili doğrudan bilgi bulunamadı."
    else:
        context = "\n\n".join([f"{i+1}. {d.page_content}" for i, d in enumerate(docs)])

    # 🔹 Optimize edilmiş prompt
    prompt = f"""
    Aşağıda MEB AYT Biyoloji kitabından alınmış bilgiler bulunuyor.
    Bu bilgilere dayanarak aşağıdaki soruyu sade, net ve bilimsel bir dille yanıtla.
    Eğer kaynaklarda bilgi yoksa "Kitapta bu konuda net bilgi bulunmamaktadır." de.

    🔹 Soru:
    {question}

    📘 Kaynak Bilgiler:
    {context}
    """

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        # ⏱️ Timeout koruması (20 sn)
        response = model.generate_content(
            prompt,
            request_options={"timeout": 20}
        )
        return response.text.strip(), docs

    except Exception as e:
        return f"⚠️ Model hatası: {e}", []

