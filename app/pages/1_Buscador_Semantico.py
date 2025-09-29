# pages/1_Buscador_Semantico.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from src.search_hybrid import HybridSearcher
from src.plots import plot_counts

st.set_page_config(
    page_title="Búsqueda Semántica", 
    page_icon="🔍",
    layout="wide"
)

# ================= INICIALIZAR BUSCADOR =================
@st.cache_resource
def init_searcher():
    """Inicializa HybridSearcher una sola vez"""
    try:
        return HybridSearcher(), None
    except Exception as e:
        return None, str(e)

searcher, init_error = init_searcher()
if init_error:
    st.error(f"❌ Error al inicializar el buscador: {init_error}")
    st.stop()

# ================= ENCABEZADO =================
st.title("🔍 Búsqueda Semántica Avanzada")
st.markdown("Busca y exporta fragmentos relevantes de las columnas de opinión usando **FAISS + BM25**")

# ================= CONFIGURACIÓN =================
with st.sidebar:
    st.title("⚙️ Configuración")
    
    k = st.slider(
        "📄 Documentos a mostrar:", 
        min_value=1, 
        max_value=50, 
        value=15,
        help="Número máximo de documentos a recuperar"
    )
    
    show_scores = st.checkbox("📊 Mostrar scores de relevancia", value=True)
    show_metadata = st.checkbox("ℹ️ Mostrar metadatos completos", value=True)
    
    st.markdown("---")
    st.page_link("streamlit_app.py", label="💬 Volver al Chat")

# ================= INPUT DE BÚSQUEDA =================
query = st.text_input("Escribe tu consulta:", placeholder="Ej. acuerdo de paz 2019, protestas estudiantiles…")

# ================= RESULTADOS =================
if query:
    with st.spinner("Buscando en los documentos..."):
        results = searcher.search(query, final_k=k)

    st.subheader(f"Resultados para: **{query}**")
    
    if results is not None and not results.empty:
        for i, row in results.iterrows():
            st.markdown(f"### {i+1}. {row.get('título','Sin título')}")
            st.write(row.get("chunk","[Sin texto]")[:600] + "…")
            
            meta = f"👤 {row.get('autor','?')} • 📰 {row.get('diario','?')} • 📅 {str(row.get('fecha',''))[:10]}"
            st.caption(meta)
            
            if show_metadata:
                with st.expander("📑 Metadatos completos"):
                    st.json(row.to_dict())
            
            if show_scores:
                st.progress(min(1.0, row.get("rrf_score", 0.5)))  # normalizado
        
        # Descargar resultados
        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Descargar resultados (CSV)",
            data=csv,
            file_name="resultados_busqueda.csv",
            mime="text/csv"
        )
        
        # Visualización extra: conteos por autor/medio
        st.markdown("### 📊 Distribución de resultados (Nivel Fragmento)")
        tabs = st.tabs(["Por Autor", "Por Medio", "Por Año"])
        with tabs[0]:
            st.pyplot(plot_counts(results, by="autor"))
        with tabs[1]:
            st.pyplot(plot_counts(results, by="diario"))
        with tabs[2]:
            st.pyplot(plot_counts(results, by="fecha"))
    else:
        st.warning("⚠️ No se encontraron resultados para esta consulta.")
