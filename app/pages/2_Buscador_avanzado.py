import streamlit as st
from src.search_hybrid import HybridSearcher
from src.plots import plot_counts
import pandas as pd

st.title("Buscador avanzado (FAISS + BM25)")
searcher = HybridSearcher()

query = st.text_input("Consulta (soporta keywords y semántica)", "protestas 2019 Bogotá")
k = st.slider("Documentos a mostrar", 1, 20, 8)
if st.button("Buscar"):
    hits = searcher.search(query, final_k=k)
    st.dataframe(hits)
    col = st.selectbox("Graficar conteos por…", ["periodico","autor"]) 
    fig = plot_counts(hits, by=col)
    st.pyplot(fig)
