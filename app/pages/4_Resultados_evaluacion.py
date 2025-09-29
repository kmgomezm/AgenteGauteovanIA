# pages/4_Resultados_evaluacion.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Resultados Evaluación",
    page_icon="✅",
    layout="wide"
)

st.title("✅ Resultados de Evaluación")
st.markdown("Análisis del performance del agente según los criterios definidos (correctness, relevance, coherence, toxicity, harmfulness).")

# ===== CARGAR DATOS =====
@st.cache_data
def load_eval_data():
    path = "data/evals/eval_results.parquet"
    if not os.path.exists(path):
        return None, f"⚠️ No se encontró el archivo {path}"
    try:
        df = pd.read_parquet(path)
        return df, None
    except Exception as e:
        return None, f"Error al cargar {path}: {e}"

df, error = load_eval_data()
if error:
    st.error(error)
    st.stop()

# ===== MÉTRICAS GENERALES =====
criterios = [c for c in df.columns if c.endswith("_score")]

st.markdown("## 📊 Promedios por criterio")
cols = st.columns(len(criterios))
for i, crit in enumerate(criterios):
    mean_score = df[crit].mean()
    cols[i].metric(crit.replace("_score", "").capitalize(), f"{mean_score:.2f}")

# ===== DISTRIBUCIONES =====
st.markdown("## 📈 Distribuciones de scores")
tabs = st.tabs([c.replace("_score", "").capitalize() for c in criterios])
for tab, crit in zip(tabs, criterios):
    with tab:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.histplot(df[crit], bins=10, kde=True, ax=ax, color="skyblue")
        ax.set_title(f"Distribución de {crit}")
        ax.set_xlabel("Score")
        st.pyplot(fig)

# ===== MEJORES Y PEORES =====
st.markdown("## 🏆 Mejores y peores casos")
for crit in criterios:
    st.subheader(crit.replace("_score", "").capitalize())
    top = df.nlargest(3, crit)[["question", "reference", "prediction", crit]]
    low = df.nsmallest(3, crit)[["question", "reference", "prediction", crit]]

    st.markdown("### ✅ Top 3")
    st.dataframe(top)
    st.markdown("### ❌ Peores 3")
    st.dataframe(low)

# ===== DETALLE COMPLETO =====
st.markdown("## 📋 Resultados detallados por pregunta")
with st.expander("Ver tabla completa", expanded=False):
    st.dataframe(df)

