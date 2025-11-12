import os
import google.generativeai as genai
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from typing import List

# ===============================
# 🔑 Gemini API key (Streamlit secrets üzerinden)
# ===============================
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ===============================
# 📘 Dosya yolları
# ===============================
PDF_PATH = "data/miyelin_kılıf.pdf"
DB_DIR = "db_gemini"

# ===============================
# 🧠 Gemini embedding sınıfı
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
            model="models/embedding-gecko-001",
            content=text,
            task_type="retrieval_query"
        )["embedding"]

# ===============================
# 🚀 (Opsiyonel) İndeks oluşturma fonksiyonu
# ===============================
def build_index():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF bulunamadı: {PDF_PATH}")

    print("📖 PDF yükleniyor (PyMuPDF ile)...")
    loader = PyMuPDFLoader(PDF_PATH)
    documents = loader.load()
    print(f"Toplam sayfa: {len(documents)}")

    print("✂️ Sayfalar parçalara bölünüyor...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1600,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?"]
    )
    docs = splitter.split_documents(documents)
    print(f"Toplam parça: {len(docs)}")

    print("🧠 Gemini API ile embedding oluşturuluyor...")
    embeddings = GeminiEmbeddings()

    print("📦 Chroma veritabanı oluşturuluyor...")
    vectordb = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    vectordb.persist()
    print("✅ İndeks oluşturma tamamlandı!")
