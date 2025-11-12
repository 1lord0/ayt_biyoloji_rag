# --- Kullanıcı girişi ---
question = st.text_input("🔹 Soru:", placeholder="örnek: Miyelin kılıfın görevi nedir?")

if st.button("Cevapla") and question.strip():
    with st.spinner("Yanıt aranıyor..."):
        try:
            answer, docs = ask_gemini(question)

            # ✅ Cevap bölümü
            st.markdown("### ✳️ Cevap")
            if answer:
                st.write(answer)
            else:
                st.warning("⚠️ Model bir cevap üretemedi veya kaynak bulamadı.")

            # ✅ Kaynak parçaları bölümü
            if docs and len(docs) > 0:
                with st.expander("📘 Kullanılan kaynak parçaları"):
                    for i, d in enumerate(docs, 1):
                        st.markdown(f"**Parça {i}:** {d.page_content[:600]}...")
            else:
                st.info("🔎 Bu soruya uygun kaynak bulunamadı veya doğrudan cevap üretildi.")

        except Exception as e:
            st.error(f"Hata oluştu: {e}")

st.markdown("---")
st.caption("💡 Bu uygulama Gemini API + Chroma RAG sistemiyle çalışmaktadır.")
