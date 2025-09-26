import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from src.rag_chain import RAGHybridPipeline
from src.nlp_tools import ner, sentiment

st.set_page_config(page_title="Agente Gauteovan IA", layout="wide")
st.title("Agente de Opinión Gauteovan (2018–2020) – RAG Híbrido")

# Inicializar el pipeline con manejo de errores
if "rag" not in st.session_state:
    try:
        with st.spinner("Inicializando sistema RAG..."):
            st.session_state["rag"] = RAGHybridPipeline()
        st.success("Sistema inicializado correctamente")
    except Exception as e:
        st.error(f"Error al inicializar el sistema: {e}")
        st.stop()

# Opciones de búsqueda
st.markdown("### Configuración de búsqueda")
col1, col2 = st.columns([2, 1])

with col1:
    options = st.multiselect(
        "Opciones avanzadas:",
        ["Permitir búsqueda web", "Razonamiento profundo (resúmenes estructurados)"],
        default=[],
        help="Selecciona las opciones que deseas activar para tu consulta"
    )

with col2:
    k_docs = st.slider("Documentos a recuperar:", 3, 20, 8, help="Número de documentos más relevantes")

allow_web = "Permitir búsqueda web" in options
use_deep_reason = "Razonamiento profundo (resúmenes estructurados)" in options

# Input principal
st.markdown("### Consulta")
q = st.text_input(
    "Haz una pregunta sobre las columnas de opinión:",
    key="main_question",
    placeholder="Ej: ¿Qué se opina sobre la educación en Colombia?"
)

# Botón de respuesta con validación
if st.button("🔍 Responder", type="primary", key="answer_button"):
    if not q.strip():
        st.warning("Por favor, ingresa una pregunta antes de continuar")
    else:
        with st.spinner("Buscando en la base de conocimientos y generando respuesta..."):
            try:
                # Llamar al pipeline con los parámetros correctos
                ans_dict = st.session_state["rag"].answer(
                    question=q, 
                    k_local=k_docs,
                    allow_web=allow_web,
                    use_deep_reason=use_deep_reason 
                )
                
                ans = ans_dict["answer"]
                hits = ans_dict.get("hits", [])
                mode = ans_dict.get("mode", "unknown")
                
                # Mostrar información del modo usado
                if mode == "local":
                    st.success("✅ Respuesta basada en evidencia local")
                elif mode == "local_insufficient":
                    st.warning("⚠️ Evidencia local insuficiente")
                elif mode == "web_not_implemented":
                    st.info("ℹ️ Búsqueda web no disponible")
                    
            except Exception as e:
                st.error(f"Error al procesar la consulta: {e}")
                st.stop()
        
        # Mostrar respuesta
        st.markdown("### 💬 Respuesta")
        st.write(ans)
        
        # Mostrar resúmenes estructurados (si están disponibles)
        if use_deep_reason and ans_dict.get("briefs"):
            st.markdown("### 🧠 Análisis estructurado")
            briefs = ans_dict["briefs"]
            
            # Crear tabs para cada faceta
            tab1, tab2, tab3 = st.tabs(["📅 Por Año", "📰 Por Medio", "👤 Por Autor"])
            
            with tab1:
                if "por_año" in briefs and briefs["por_año"]:
                    for item in briefs["por_año"]:
                        st.write(f"• {item}")
                else:
                    st.info("No hay análisis por año disponible")
            
            with tab2:
                if "por_medio" in briefs and briefs["por_medio"]:
                    for item in briefs["por_medio"]:
                        st.write(f"• {item}")
                else:
                    st.info("No hay análisis por medio disponible")
            
            with tab3:
                if "por_autor" in briefs and briefs["por_autor"]:
                    for item in briefs["por_autor"]:
                        st.write(f"• {item}")
                else:
                    st.info("No hay análisis por autor disponible")
            
            # Mostrar análisis crudo si existe (para debugging)
            if "analisis_crudo" in briefs:
                with st.expander("🔧 Análisis crudo (debug)", expanded=False):
                    st.text(briefs["analisis_crudo"])
        
        # Mostrar fuentes
        if hits:
            with st.expander(f"📚 Fuentes consultadas ({len(hits)} documentos)", expanded=False):
                for i, r in enumerate(hits, 1):
                    if isinstance(r, dict):
                        titulo = r.get('título', 'Sin título')
                        autor = r.get('autor', 'Sin autor')
                        diario = r.get('diario', 'Sin medio')
                        fecha = str(r.get('fecha', ''))[:10]
                        doc_id = r.get('doc_id', 'Sin ID')
                        rrf_score = r.get('rrf_score', 'N/A')
                        
                        st.markdown(f"""
                        **{i}.** **{titulo}**  
                        👤 {autor} • 📰 *{diario}* • 📅 {fecha}  
                        🔗 `{doc_id}` • 📊 Score: {rrf_score}
                        """)
                
        else:
            st.info("No se encontraron fuentes relevantes")

st.divider()
st.markdown("### 🔬 Herramientas de análisis de texto")
st.markdown("Analiza cualquier texto usando herramientas de NLP")

txt = st.text_area(
    "Pega un texto para analizar:", 
    height=120, 
    key="analysis_text",
    placeholder="Ingresa o pega el texto que quieres analizar..."
)

if txt.strip():
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🏷️ Reconocimiento de Entidades (NER)", key="ner_button"):
            with st.spinner("Analizando entidades..."):
                try:
                    ner_result = ner(txt)
                    st.markdown("#### Entidades encontradas:")
                    st.json(ner_result)
                except Exception as e:
                    st.error(f"Error en análisis NER: {e}")
    
    with col2:
        if st.button("😊 Análisis de Sentimiento", key="sentiment_button"):
            with st.spinner("Analizando sentimiento..."):
                try:
                    sentiment_result = sentiment(txt)
                    st.markdown("#### Análisis de sentimiento:")
                    st.json(sentiment_result)
                except Exception as e:
                    st.error(f"Error en análisis de sentimiento: {e}")
else:
    st.info("👆 Ingresa un texto arriba para usar las herramientas de análisis")