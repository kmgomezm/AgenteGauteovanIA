import streamlit as st
from src.rag_chain import RAGHybridPipeline
from src.nlp_tools import ner, sentiment

st.set_page_config(page_title="Agente Gauteovan IA", layout="wide")
st.title("Agente de Opinión Gauteovan (2018–2020) – RAG Híbrido")


if "rag" not in st.session_state:
    st.session_state["rag"] = RAGHybridPipeline()


# Checkbox para permitir búsqueda web, aparece justo encima del botón de responder
allow_web = st.checkbox("Permitir búsqueda web", value=False)
q = st.text_input("Haz una pregunta sobre las columnas de interés…")
if st.button("Responder") and q:
    with st.spinner("Buscando y generando…"):
        ans_dict = st.session_state["rag"].answer(q, allow_web=allow_web)
        ans = ans_dict["answer"]
        hits = ans_dict.get("hits", [])
    st.markdown("### Respuesta")
    st.write(ans)
    with st.expander("Fuentes"):
        # Si hits es un DataFrame
        if hasattr(hits, 'iterrows'):
            for _, r in hits.iterrows():
                st.markdown(f"- **{r.título}** — {r.autor}, *{r.diario}*, {str(r.fecha)[:10]}  \n`{r.doc_id}`")
        # Si hits es una lista de dicts
        elif isinstance(hits, list):
            for r in hits:
                st.markdown(f"- **{r.get('título','')}** — {r.get('autor','')}, *{r.get('diario','')}*, {str(r.get('fecha',''))[:10]}  \n`{r.get('doc_id','')}`")

# Checklist para opciones avanzadas
options = st.multiselect(
    "Opciones de búsqueda:",
    ["Permitir búsqueda web", "Razonamiento profundo (resúmenes estructurados)"],
    default=[]
)
allow_web = "Permitir búsqueda web" in options
use_deep_reason = "Razonamiento profundo (resúmenes estructurados)" in options

q = st.text_input("Haz una pregunta sobre las columnas de interés…")
if st.button("Responder") and q:
    with st.spinner("Buscando y generando…"):
        ans_dict = st.session_state["rag"].answer(q, allow_web=allow_web, use_deep_reason=use_deep_reason)
        ans = ans_dict["answer"]
        hits = ans_dict.get("hits", [])
        briefs = ans_dict.get("briefs", None)
    st.markdown("### Respuesta")
    st.write(ans)
    if use_deep_reason and briefs:
        st.markdown("#### Resúmenes estructurados")
        st.json(briefs)
    with st.expander("Fuentes"):
        # Si hits es un DataFrame
        if hasattr(hits, 'iterrows'):
            for _, r in hits.iterrows():
                st.markdown(f"- **{r.título}** — {r.autor}, *{r.diario}*, {str(r.fecha)[:10]}  \n`{r.doc_id}`")
        # Si hits es una lista de dicts
        elif isinstance(hits, list):
            for r in hits:
                st.markdown(f"- **{r.get('título','')}** — {r.get('autor','')}, *{r.get('diario','')}*, {str(r.get('fecha',''))[:10]}  \n`{r.get('doc_id','')}`")

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
