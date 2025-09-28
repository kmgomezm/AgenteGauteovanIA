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

# Inicializar buscador
@st.cache_resource
def init_searcher():
    """Inicializa HybridSearcher una sola vez"""
    try:
        return HybridSearcher(), None
    except Exception as e:
        return None, str(e)

searcher, init_error = init_searcher()

if init_error:
    st.error(f"Error al inicializar el buscador: {init_error}")
    st.stop()

# ===== BÚSQUEDA SEMÁNTICA =====
st.title("🔍 Búsqueda Semántica Avanzada")
st.markdown("Busca y exporta fragmentos relevantes de las columnas de opinión usando FAISS + BM25")

# Configuración de búsqueda
with st.sidebar:
    st.title("⚙️ Configuración")
    
    k = st.slider(
        "📄 Documentos a mostrar:", 
        min_value=1, 
        max_value=50, 
        value=15,
        help="Número máximo de documentos a recuperar"
    )
    
    st.markdown("**Opciones de visualización:**")
    show_scores = st.checkbox("📊 Mostrar scores de relevancia", value=True)
    show_metadata = st.checkbox("ℹ️ Mostrar metadatos completos", value=True)
    
    st.markdown("---")
    st.page_link("streamlit_app.py", label="💬 Volver al Chat")
    st.page