import streamlit as st
from query_rag import ask_gemini

# ----------------------------
# 🧠 Sayfa ayarları
# ----------------------------
st.set_page_config(page_title="AYT Biyoloji RAG Asistanı", page_icon="🧬", layout="centered")

st.title("🧬 AYT Biyoloji RAG Asistanı")
st.markdown("""
Bu uygulama, **MEB AYT Biyoloji kitabına** dayalı olarak geliştirilmiş bir **RAG (Retrieval-Augmented Generation)** sistemidir.  
Sorularına sadece kitapta yer alan bilgiler doğrultusunda yanıt verir.  
---
""")

# ----------------------------
# 📝 Kullanıcı girişi
# ----------------------------
question = st.text_input("🔹 Sorunu yaz:", placeholder="örnek: Miyelin kılıfın görevi nedir?")

# ----------------------------
# 🚀 Cevaplama işlemi
# ----------------------------
if st.button("Cevapla") and question.strip():
    with st.spinner("Yanıt aranıyor..."):
        try:
            answer, docs = ask_gemini(question)

            # --- Cevap bölümü ---
            st.markdown("### ✳️ Cevap")
            if answer:
                st.write(answer)
            else:
                st.warning("⚠️ Model bir cevap üretemedi veya kaynak bulamadı.")

            # --- Kaynak bölümü ---
            if docs and len(docs) > 0:
                with st.expander("📘 Kullanılan kaynak parçaları"):
                    for i, d in enumerate(docs, 1):
                        snippet = d.page_content[:600].strip().replace("\n", " ")
                        st.markdown(f"**Parça {i}:** {snippet}...")
            else:
                st.info("🔎 Bu soruya uygun kaynak bulunamadı veya doğrudan cevap üretildi.")

        except Exception as e:
            st.error(f"Hata oluştu: {e}")

# ----------------------------
# 📎 Alt bilgi
# ----------------------------
st.markdown("---")
st.caption("💡 Bu uygulama Gemini API + Chroma RAG sistemi ile çalışmaktadır.")
