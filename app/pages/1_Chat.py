import streamlit as st
from src.rag_chain import RAGPipeline

st.title("Chat con RAG Híbrido")
if "rag" not in st.session_state:
    st.session_state["rag"] = RAGPipeline()

q = st.chat_input("Pregunta algo…")
if q:
    with st.spinner("Pensando…"):
        ans, hits = st.session_state["rag"].answer(q)
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(ans)
        with st.expander("Fuentes"):
            for _, r in hits.iterrows():
                st.markdown(f"- **{r.titulo}** — {r.autor}, *{r.periodico}*, {str(r.fecha)[:10]}  \n`{r.doc_id}`")
