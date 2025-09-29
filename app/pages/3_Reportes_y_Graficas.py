# pages/3_Reportes_y_Graficas.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

st.set_page_config(
    page_title="Reportes y Gráficas",
    page_icon="📈",
    layout="wide"
)

# ===== ESTILO GLOBAL =====
plt.style.use("seaborn-v0_8")
sns.set_palette("tab10")

# ===== TÍTULO =====
st.title("📈 Reportes y Gráficas del Corpus")
st.markdown("Análisis estadístico descriptivo del corpus de columnas de opinión")

# ===== CARGAR DATOS =====
@st.cache_data
def load_corpus_data():
    try:
        df = pd.read_parquet("data/processed/chunks.parquet")
        return df, None
    except Exception as e:
        return None, f"Error al cargar datos: {e}"

df, error = load_corpus_data()
if error:
    st.error(error)
    st.stop()

# ===== ESTADÍSTICAS =====
df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
df['chunk_length'] = df['chunk'].astype(str).apply(len)
df['word_count'] = df['chunk'].astype(str).apply(lambda x: len(x.split()))

# Palabras distintas
all_words = " ".join(df['chunk'].astype(str)).split()
unique_words = set(all_words)

stats = {
    "total_chunks": len(df),
    "total_documents": df['doc_id'].nunique(),
    "total_authors": df['autor'].nunique(),
    "total_newspapers": df['diario'].nunique(),
    "avg_chunk_length": df['chunk_length'].mean(),
    "avg_word_count": df['word_count'].mean(),
    "total_words": df['word_count'].sum(),
    "unique_words": len(unique_words),
    "lexical_density": len(unique_words) / len(all_words) if all_words else 0,
    "avg_docs_per_author": df.drop_duplicates("doc_id").groupby("autor")["doc_id"].count().mean(),
    "avg_docs_per_newspaper": df.drop_duplicates("doc_id").groupby("diario")["doc_id"].count().mean(),
    "date_range": {"start": df['fecha'].min(), "end": df['fecha'].max()},
    "top_authors": df.drop_duplicates("doc_id")['autor'].value_counts(),
    "top_newspapers": df.drop_duplicates("doc_id")['diario'].value_counts(),
    "authors_per_media": df.drop_duplicates("doc_id").groupby("diario")["autor"].nunique()
}

# ===== RESUMEN EJECUTIVO =====
st.markdown("## 📊 Resumen Ejecutivo")

col1, col2, col3, col4 = st.columns(4)
col1.metric("📄 Fragmentos", f"{stats['total_chunks']:,}")
col2.metric("📰 Documentos", f"{stats['total_documents']:,}")
col3.metric("✍️ Autores", f"{stats['total_authors']:,}")
col4.metric("🏢 Medios", f"{stats['total_newspapers']:,}")

col5, col6, col7, col8 = st.columns(4)
col5.metric("💬 Palabras", f"{stats['total_words']:,}")
col6.metric("🔤 Palabras distintas", f"{stats['unique_words']:,}")
col7.metric("📏 Promedio Palabras/Fragmento", f"{stats['avg_word_count']:.1f}")
col8.metric("📚 Densidad Léxica", f"{stats['lexical_density']:.2%}")

col9, col10 = st.columns(2)
col9.metric("📖 Promedio docs/autor", f"{stats['avg_docs_per_author']:.1f}")
col10.metric("📰 Promedio docs/medio", f"{stats['avg_docs_per_newspaper']:.1f}")

# ===== VISUALIZACIONES =====
st.markdown("---")
st.markdown("## 📈 Visualizaciones")

tab1, tab2, tab3, tab4 = st.tabs([
    "📰 Por Medio",
    "✍️ Por Autor",
    "📅 Temporal",
    "📊 Distribuciones"
])

# ---- Por Medio ----
with tab1:
    st.markdown("### 📰 Documentos por Medio de Comunicación")
    counts = stats['top_newspapers']

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax.set_ylabel("Número de documentos")
    ax.set_xlabel("Medio")
    st.pyplot(fig)

    st.dataframe(counts.rename("Documentos"))

# ---- Por Autor ----
with tab2:
    st.markdown("### ✍️ Documentos por Autor (Top general)")
    n_authors = st.slider("Número de autores a mostrar:", 5, 20, 10)
    top_authors = stats['top_authors'].head(n_authors)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=top_authors.values, y=top_authors.index, ax=ax)
    ax.set_xlabel("Número de documentos")
    ax.set_ylabel("Autor")
    st.pyplot(fig)

    st.dataframe(top_authors.rename("Documentos"))

    st.markdown("### 📊 Número de autores distintos por medio")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=stats['authors_per_media'].index, y=stats['authors_per_media'].values, ax=ax)
    ax.set_ylabel("Autores distintos")
    ax.set_xlabel("Medio")
    st.pyplot(fig)

    st.dataframe(stats['authors_per_media'].rename("Autores distintos"))

    st.markdown("### 🏆 Top autores por periódico")
    for medio in stats['top_newspapers'].index:
        st.subheader(medio)
        top_autores_medio = (
            df[df['diario'] == medio]
            .drop_duplicates("doc_id")['autor']
            .value_counts()
            .head(5)
        )
        st.dataframe(top_autores_medio.rename("Documentos"))

# ---- Temporal ----
with tab3:
    st.markdown("### 📅 Análisis Temporal")
    docs = df.drop_duplicates("doc_id")

    yearly = docs.groupby(docs['fecha'].dt.year)["doc_id"].nunique()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(x=yearly.index, y=yearly.values, marker="o", ax=ax)
    ax.set_ylabel("Número de documentos")
    ax.set_title("Documentos por año")
    st.pyplot(fig)

    monthly = docs.groupby(docs['fecha'].dt.month)["doc_id"].nunique()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=monthly.index, y=monthly.values, ax=ax)
    ax.set_xlabel("Mes")
    ax.set_ylabel("Número de documentos")
    ax.set_title("Documentos por mes (agregados)")
    st.pyplot(fig)

# ---- Distribuciones ----
with tab4:
    st.markdown("### 📊 Distribuciones Estadísticas de Fragmentos")

    colA, colB = st.columns(2)

    with colA:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df['chunk_length'], bins=40, kde=True, ax=ax, color="skyblue")
        ax.axvline(df['chunk_length'].mean(), color="red", linestyle="--", label="Media")
        ax.axvline(df['chunk_length'].median(), color="green", linestyle="--", label="Mediana")
        ax.set_title("Longitud de fragmentos (caracteres)")
        ax.legend()
        st.pyplot(fig)

    with colB:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df['word_count'], bins=40, kde=True, ax=ax, color="lightcoral")
        ax.axvline(df['word_count'].mean(), color="blue", linestyle="--", label="Media")
        ax.axvline(df['word_count'].median(), color="orange", linestyle="--", label="Mediana")
        ax.set_title("Número de palabras por fragmento")
        ax.legend()
        st.pyplot(fig)

    st.markdown("#### 📈 Estadísticas Descriptivas")
    descriptive_stats = pd.DataFrame({
        "Métrica": ["Longitud (caracteres)", "Número de palabras"],
        "Media": [df['chunk_length'].mean(), df['word_count'].mean()],
        "Mediana": [df['chunk_length'].median(), df['word_count'].median()],
        "Desv. Estándar": [df['chunk_length'].std(), df['word_count'].std()],
        "Mínimo": [df['chunk_length'].min(), df['word_count'].min()],
        "Máximo": [df['chunk_length'].max(), df['word_count'].max()],
    }).round(1)
    st.dataframe(descriptive_stats)

# ===== EXPORTAR =====
st.markdown("---")
st.markdown("## 💾 Exportar Reportes")

csv_authors = stats['top_authors'].to_csv()
st.download_button("📥 Descargar Top Autores (CSV)", csv_authors, "top_autores.csv", "text/csv")

csv_media = stats['top_newspapers'].to_csv()
st.download_button("📥 Descargar Top Medios (CSV)", csv_media, "top_medios.csv", "text/csv")

csv_authors_media = stats['authors_per_media'].to_csv()
st.download_button("📥 Descargar Autores por Medio (CSV)", csv_authors_media, "autores_por_medio.csv", "text/csv")
