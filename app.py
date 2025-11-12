import streamlit as st
from query_rag import ask_gemini

# Sayfa ayarları
st.set_page_config(
    page_title="AYT Biyoloji RAG Asistanı",
    page_icon="🧠",
    layout="centered"
)

# Başlık
st.title("🧠 AYT Biyoloji RAG Asistanı")
st.markdown(
    "MEB **AYT Biyoloji** kitabına dayalı yapay zekâ destekli bilgi asistanı.\n\n"
    "Sorularını yaz, sistem yalnızca **kitaptaki bilgilerle** cevap versin."
)

# Kullanıcıdan soru al
question = st.text_input("🔹 Soru:", placeholder="örnek: Miyelin kılıfın görevi nedir?")

# Cevaplama butonu
if st.button("Cevapla") and question.strip():
    with st.spinner("Yanıt aranıyor..."):
        try:
            answer, docs = ask_gemini(question)
            st.markdown("### ✳️ Cevap")
            st.write(answer)

            with st.expander("📘 Kullanılan kaynak parçaları"):
                for i, d in enumerate(docs, 1):
                    st.markdown(f"**Parça {i}:** {d.page_content[:500]}...")
        except Exception as e:
            st.error(f"Hata oluştu: {e}")

# Alt bilgi
st.markdown("---")
st.markdown("💡 *Bu uygulama Gemini API ve Chroma tabanlı RAG sistemiyle çalışır.*")
