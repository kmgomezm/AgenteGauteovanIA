import streamlit as st
from src.nlp_tools import ner, sentiment, classify

st.title("Análisis NLP (NER / Sentimiento / Clasificación)")
txt = st.text_area("Texto para analizar", height=160)
labels = st.text_input("Etiquetas separadas por coma (para zero-shot)", "política, economía, salud, cultura")

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("NER") and txt:
        st.json(ner(txt))
with c2:
    if st.button("Sentimiento") and txt:
        st.json(sentiment(txt))
with c3:
    if st.button("Clasificar (zero-shot)") and txt:
        labs = [s.strip() for s in labels.split(',') if s.strip()]
        st.json(classify(txt, labs))
