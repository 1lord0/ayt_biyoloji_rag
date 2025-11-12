import os
import pickle
import numpy as np
import google.generativeai as genai
from PyPDF2 import PdfReader

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class BiologyRAG:
    def __init__(self):
        self.chunks = []
        self.embeddings = []
        
    def process_pdf(self, pdf_path):
        # PDF'i oku
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        # Chunk'la
        self.chunks = self.chunk_text(text, chunk_size=400)
        
        # Embedding'leri oluştur
        print("Embedding'ler oluşturuluyor...")
        for i, chunk in enumerate(self.chunks):
            emb = genai.embed_content(
                model="models/text-embedding-004",
                content=chunk,
                task_type="retrieval_document"
            )
            self.embeddings.append(emb['embedding'])
            if i % 10 == 0:
                print(f"{i}/{len(self.chunks)} tamamlandı")
        
        # Kaydet
        self.save_db()
    
    def chunk_text(self, text, chunk_size=400, overlap=50):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.strip()) > 100:  # Çok kısa chunk'ları atla
                chunks.append(chunk)
        return chunks
    
    def save_db(self):
        with open("bio_rag.pkl", "wb") as f:
            pickle.dump({"chunks": self.chunks, "embeddings": self.embeddings}, f)
        print("✅ Veritabanı kaydedildi")
    
    def load_db(self):
        with open("bio_rag.pkl", "rb") as f:
            data = pickle.load(f)
            self.chunks = data["chunks"]
            self.embeddings = data["embeddings"]
        print("✅ Veritabanı yüklendi")
    
    def search(self, query, top_k=3):
        # Query embedding
        query_emb = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="retrieval_query"
        )['embedding']
        
        # Cosine similarity hesapla
        similarities = []
        for emb in self.embeddings:
            sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
            similarities.append(sim)
        
        # En yakın k tanesini bul
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.chunks[i] for i in top_indices]
    
    def ask(self, question):
        # İlgili chunk'ları bul
        relevant_chunks = self.search(question, top_k=3)
        context = "\n\n---\n\n".join(relevant_chunks)
        
        # Gemini'ye sor
        prompt = f"""Sen bir AYT Biyoloji öğretmenisin. Aşağıdaki kaynaklara dayanarak soruyu yanıtla.

SORU: {question}

KAYNAK METİNLER:
{context}

Kurallar:
- Sade ve anlaşılır açıkla
- Öğrenci seviyesine uygun konuş
- Kaynaklarda bilgi yoksa "Bu konu kaynaklarda detaylı yok ama..." diyerek genel bilgi ver
- Kısa ve öz tut (max 200 kelime)
"""
        
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        response = model.generate_content(prompt)
        return response.text
