import streamlit as st
from src.rag_chain import RAGPipeline
from src.nlp_tools import ner, sentiment

st.set_page_config(page_title="Agente Opinión 🇨🇴", layout="wide")
st.title("Agente de Opinión (2018–2020) – RAG Híbrido")


if "rag" not in st.session_state:
    st.session_state["rag"] = RAGPipeline()

q = st.text_input("Haz una pregunta sobre las columnas…")
if st.button("Responder") and q:
    with st.spinner("Buscando y generando…"):
        ans, hits = st.session_state["rag"].answer(q)
    st.markdown("### Respuesta")
    st.write(ans)
    with st.expander("Fuentes"):
        for _, r in hits.iterrows():
            st.markdown(f"- **{r.titulo}** — {r.autor}, *{r.periodico}*, {str(r.fecha)[:10]}  \n`{r.doc_id}`")


st.divider()
st.markdown("### Análisis rápido")
txt = st.text_area("Pega un texto o selecciona un documento en la pestaña Buscador", height=120)
col1, col2 = st.columns(2)
with col1:
    if st.button("NER"):
        st.json(ner(txt))
with col2:
    if st.button("Sentimiento"):
        st.json(sentiment(txt))
