# 4_Reportes_y_Graficas.py
import streamlit as st
import pandas as pd
from src.plots import plot_counts

st.title("Reportes y Gráficas (POR TERMINAR)")
st.write("Carga el parquet procesado para explorar conteos.")
uploaded = st.file_uploader("Sube chunks.parquet", type=["parquet"])
if uploaded:
    df = pd.read_parquet(uploaded)
    by = st.selectbox("Agrupar por", ["diario","autor"])
    fig = plot_counts(df, by=by)
    st.pyplot(fig)
