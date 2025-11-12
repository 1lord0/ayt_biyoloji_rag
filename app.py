import streamlit as st

st.set_page_config(page_title="AYT Biyoloji Asistanı", page_icon="🧬")

st.title("🧬 AYT Biyoloji Asistanı")
st.caption("MEB 11. Sınıf Kitabı - Gemini RAG")

# RAG sistemini yükle
@st.cache_resource
def load_rag():
    rag = BiologyRAG()
    rag.load_db()
    return rag

rag = load_rag()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mesajları göster
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcı inputu
if prompt := st.chat_input("Biyoloji hakkında bir şey sor..."):
    # Kullanıcı mesajını ekle
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Bot yanıtı
    with st.chat_message("assistant"):
        with st.spinner("Düşünüyorum..."):
            response = rag.ask(prompt)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
