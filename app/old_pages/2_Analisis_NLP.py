# pages/2_Analisis_NLP.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from src.nlp_tools import ner, sentiment
import json

st.set_page_config(
    page_title="Análisis NLP", 
    page_icon="📊",
    layout="wide"
)

# ===== ANÁLISIS NLP =====
st.title("📊 Análisis NLP (Sentimiento y Entidades)")
st.markdown("Herramientas de procesamiento de lenguaje natural para análisis de texto")

# Sidebar con información
with st.sidebar:
    st.title("🔧 Herramientas NLP")
    
    st.markdown("""
    **Disponibles:**
    - 🏷️ **NER (Named Entity Recognition)**: Identificación de entidades nombradas
    - 😊 **Análisis de Sentimiento**: Clasificación emocional del texto
    
    **Modelos utilizados:**
    - SpaCy es_core_news_lg
    - pysentimiento/robertuito-sentiment-analysis
    """)
    
    st.markdown("---")
    st.page_link("streamlit_app.py", label="💬 Volver al Chat")
    st.page_link("pages/1_Buscador_Semantico.py", label="🔍 Búsqueda Semántica")

# Verificar si hay fragmento seleccionado desde búsqueda
if 'selected_fragment' in st.session_state and st.session_state.selected_fragment:
    fragment = st.session_state.selected_fragment
    
    st.success(f"✅ **Fragmento seleccionado:** {fragment['titulo']}")
    
    # Mostrar metadatos del fragmento
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👤 Autor", fragment['autor'])
    with col2:
        st.metric("📰 Medio", fragment['diario'])
    with col3:
        st.metric("📅 Fecha", fragment['fecha'])
    with col4:
        st.metric("🔗 ID", fragment.get('doc_id', 'N/A'))
    
    # Mostrar y permitir edición del texto
    st.markdown("### 📄 Texto para análisis")
    texto_analizar = st.text_area(
        "Texto seleccionado (puedes editarlo):",
        value=fragment['text'],
        height=200,
        help="Puedes modificar el texto antes de analizarlo"
    )
    
    # Almacenar historial de análisis
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    # Botones de análisis
    st.markdown("### 🔬 Herramientas de análisis")
    col_ner, col_sentiment = st.columns(2)
    
    with col_ner:
        st.markdown("#### 🏷️ Reconocimiento de Entidades")
        if st.button("🏷️ Ejecutar NER", type="primary", use_container_width=True):
            if texto_analizar.strip():
                with st.spinner("Analizando entidades nombradas..."):
                    try:
                        ner_result = ner(texto_analizar)
                        
                        # Guardar en historial
                        st.session_state.analysis_history.append({
                            'type': 'NER',
                            'text': texto_analizar[:100] + '...',
                            'result': ner_result,
                            'timestamp': st.session_state.get('timestamp', 'N/A')
                        })
                        
                        if ner_result:
                            st.success("✅ Análisis NER completado")
                            
                            # Organizar entidades por tipo
                            entidades_por_tipo = {}
                            for entidad, tipo in ner_result:
                                if tipo not in entidades_por_tipo:
                                    entidades_por_tipo[tipo] = []
                                entidades_por_tipo[tipo].append(entidad)
                            
                            # Mostrar resultados organizados
                            st.markdown("##### 🎯 Entidades encontradas:")
                            for tipo, entidades in entidades_por_tipo.items():
                                entidades_unicas = list(set(entidades))
                                
                                # Mapeo de tipos de entidad a emojis
                                emoji_map = {
                                    'PER': '👤', 'PERSON': '👤',
                                    'LOC': '📍', 'LOCATION': '📍', 'GPE': '📍',
                                    'ORG': '🏢', 'ORGANIZATION': '🏢',
                                    'MISC': '🏷️', 'MISCELLANEOUS': '🏷️',
                                    'DATE': '📅', 'TIME': '⏰',
                                    'MONEY': '💰', 'PERCENT': '📊'
                                }
                                
                                emoji = emoji_map.get(tipo.upper(), '🔸')
                                st.markdown(f"**{emoji} {tipo}:** {', '.join(entidades_unicas)}")
                            
                            # Métricas
                            col_met1, col_met2 = st.columns(2)
                            with col_met1:
                                st.metric("Total entidades", len(ner_result))
                            with col_met2:
                                st.metric("Tipos únicos", len(entidades_por_tipo))
                            
                            # Resultado detallado expandible
                            with st.expander("🔧 Resultado detallado (JSON)", expanded=False):
                                st.json(ner_result)
                                
                        else:
                            st.info("ℹ️ No se encontraron entidades nombradas en el texto")
                            
                    except Exception as e:
                        st.error(f"❌ Error en análisis NER: {e}")
                        st.exception(e)
            else:
                st.warning("⚠️ El texto está vacío")
    
    with col_sentiment:
        st.markdown("#### 😊 Análisis de Sentimiento")
        if st.button("😊 Ejecutar Sentimiento", type="primary", use_container_width=True):
            if texto_analizar.strip():
                with st.spinner("Analizando sentimiento..."):
                    try:
                        sentiment_result = sentiment(texto_analizar)
                        
                        # Guardar en historial
                        st.session_state.analysis_history.append({
                            'type': 'SENTIMENT',
                            'text': texto_analizar[:100] + '...',
                            'result': sentiment_result,
                            'timestamp': st.session_state.get('timestamp', 'N/A')
                        })
                        
                        if isinstance(sentiment_result, list) and sentiment_result:
                            st.success("✅ Análisis de sentimiento completado")
                            
                            result = sentiment_result[0]
                            label = result.get('label', 'N/A')
                            score = result.get('score', 0)
                            
                            # Mapeo de sentimientos
                            sentiment_map = {
                                'POSITIVE': {'emoji': '😊', 'color': 'green', 'desc': 'Positivo'},
                                'NEGATIVE': {'emoji': '😞', 'color': 'red', 'desc': 'Negativo'}, 
                                'NEUTRAL': {'emoji': '😐', 'color': 'gray', 'desc': 'Neutral'}
                            }
                            
                            sentiment_info = sentiment_map.get(label.upper(), {'emoji': '🤔', 'color': 'blue', 'desc': label})
                            
                            # Mostrar resultado visual
                            st.markdown(f"### {sentiment_info['emoji']} **{sentiment_info['desc']}**")
                            
                            # Barra de confianza
                            st.progress(score)
                            st.markdown(f"**Confianza:** {score:.1%}")
                            
                            # Interpretación
                            if score >= 0.8:
                                confianza_desc = "Muy alta"
                            elif score >= 0.6:
                                confianza_desc = "Alta"
                            elif score >= 0.4:
                                confianza_desc = "Media"
                            else:
                                confianza_desc = "Baja"
                            
                            st.info(f"🎯 **Interpretación:** Sentimiento {sentiment_info['desc'].lower()} con confianza {confianza_desc}")
                            
                            # Resultado detallado
                            with st.expander("🔧 Resultado detallado (JSON)", expanded=False):
                                st.json(sentiment_result)
                                
                        else:
                            st.info("ℹ️ No se pudo analizar el sentimiento del texto")
                            
                    except Exception as e:
                        st.error(f"❌ Error en análisis de sentimiento: {e}")
                        st.exception(e)
            else:
                st.warning("⚠️ El texto está vacío")
    
    # Análisis combinado
    st.markdown("### 🔄 Análisis combinado")
    if st.button("🚀 Ejecutar ambos análisis", type="secondary", use_container_width=True):
        if texto_analizar.strip():
            with st.spinner("Ejecutando análisis completo..."):
                col_res1, col_res2 = st.columns(2)
                
                # NER
                with col_res1:
                    try:
                        ner_result = ner(texto_analizar)
                        st.markdown("#### 🏷️ Entidades:")
                        if ner_result:
                            for entidad, tipo in ner_result[:5]:  # Mostrar solo las primeras 5
                                st.write(f"• **{entidad}** ({tipo})")
                            if len(ner_result) > 5:
                                st.write(f"... y {len(ner_result) - 5} más")
                        else:
                            st.info("Sin entidades")
                    except Exception as e:
                        st.error(f"Error NER: {e}")
                
                # Sentimiento
                with col_res2:
                    try:
                        sentiment_result = sentiment(texto_analizar)
                        st.markdown("#### 😊 Sentimiento:")
                        if sentiment_result and isinstance(sentiment_result, list):
                            result = sentiment_result[0]
                            label = result.get('label', 'N/A')
                            score = result.get('score', 0)
                            
                            sentiment_map = {
                                'POSITIVE': {'emoji': '😊', 'desc': 'Positivo'},
                                'NEGATIVE': {'emoji': '😞', 'desc': 'Negativo'}, 
                                'NEUTRAL': {'emoji': '😐', 'desc': 'Neutral'}
                            }
                            
                            info = sentiment_map.get(label.upper(), {'emoji': '🤔', 'desc': label})
                            st.write(f"{info['emoji']} **{info['desc']}** ({score:.1%})")
                        else:
                            st.info("Sin sentimiento")
                    except Exception as e:
                        st.error(f"Error Sentimiento: {e}")
    
    # Opción para limpiar selección
    st.markdown("---")
    col_clear, col_history = st.columns([1, 1])
    
    with col_clear:
        if st.button("🗑️ Limpiar fragmento seleccionado"):
            del st.session_state.selected_fragment
            st.rerun()
    
    with col_history:
        if st.button("📜 Ver historial de análisis"):
            st.session_state.show_history = not st.session_state.get('show_history', False)

else:
    # No hay fragmento seleccionado - mostrar opciones alternativas
    st.info("📋 **No hay fragmento seleccionado desde búsqueda**")
    
    tab1, tab2 = st.tabs(["✍️ Texto libre", "📁 Cargar archivo"])
    
    with tab1:
        st.markdown("### ✍️ Analizar texto libre")
        texto_libre = st.text_area(
            "Pega o escribe el texto que quieres analizar:",
            height=200,
            placeholder="Ingresa aquí el texto para análisis NLP..."
        )
        
        if texto_libre.strip():
            # Análisis de texto libre
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🏷️ Solo NER", key="ner_libre", use_container_width=True):
                    with st.spinner("Analizando entidades..."):
                        try:
                            ner_result = ner(texto_libre)
                            st.markdown("#### 🏷️ Entidades encontradas:")
                            if ner_result:
                                # Mostrar organizadamente
                                entidades_por_tipo = {}
                                for entidad, tipo in ner_result:
                                    if tipo not in entidades_por_tipo:
                                        entidades_por_tipo[tipo] = []
                                    entidades_por_tipo[tipo].append(entidad)
                                
                                for tipo, entidades in entidades_por_tipo.items():
                                    st.write(f"**{tipo}:** {', '.join(set(entidades))}")
                            else:
                                st.info("No se encontraron entidades")
                        except Exception as e:
                            st.error(f"Error: {e}")
            
            with col2:
                if st.button("😊 Solo Sentimiento", key="sentiment_libre", use_container_width=True):
                    with st.spinner("Analizando sentimiento..."):
                        try:
                            sentiment_result = sentiment(texto_libre)
                            st.markdown("#### 😊 Análisis de sentimiento:")
                            if sentiment_result and isinstance(sentiment_result, list):
                                result = sentiment_result[0]
                                label = result.get('label', 'N/A')
                                score = result.get('score', 0)
                                
                                sentiment_map = {
                                    'POSITIVE': {'emoji': '😊', 'desc': 'Positivo'},
                                    'NEGATIVE': {'emoji': '😞', 'desc': 'Negativo'}, 
                                    'NEUTRAL': {'emoji': '😐', 'desc': 'Neutral'}
                                }
                                
                                info = sentiment_map.get(label.upper(), {'emoji': '🤔', 'desc': label})
                                st.write(f"{info['emoji']} **{info['desc']}**")
                                st.progress(score)
                                st.write(f"Confianza: {score:.1%}")
                        except Exception as e:
                            st.error(f"Error: {e}")
            
            with col3:
                if st.button("🚀 Análisis completo", key="complete_libre", use_container_width=True):
                    with st.spinner("Análisis completo..."):
                        col_ner, col_sent = st.columns(2)
                        
                        with col_ner:
                            st.markdown("**🏷️ Entidades:**")
                            try:
                                ner_result = ner(texto_libre)
                                if ner_result:
                                    for entidad, tipo in ner_result[:3]:
                                        st.write(f"• {entidad} ({tipo})")
                                else:
                                    st.info("Sin entidades")
                            except:
                                st.error("Error en NER")
                        
                        with col_sent:
                            st.markdown("**😊 Sentimiento:**")
                            try:
                                sentiment_result = sentiment(texto_libre)
                                if sentiment_result:
                                    result = sentiment_result[0]
                                    label = result.get('label', 'N/A')
                                    score = result.get('score', 0)
                                    st.write(f"{label} ({score:.1%})")
                                else:
                                    st.info("Sin resultado")
                            except:
                                st.error("Error en sentimiento")
    
    with tab2:
        st.markdown("### 📁 Cargar archivo de texto")
        uploaded_file = st.file_uploader(
            "Sube un archivo de texto:",
            type=['txt', 'md'],
            help="Formatos soportados: .txt, .md"
        )
        
        if uploaded_file is not None:
            try:
                texto_archivo = str(uploaded_file.read(), "utf-8")
                st.text_area("Contenido del archivo:", texto_archivo, height=200)
                
                if st.button("🔍 Analizar archivo"):
                    # Aquí puedes reutilizar la lógica de análisis
                    st.info("Funcionalidad de análisis de archivos - por implementar")
            except Exception as e:
                st.error(f"Error al leer archivo: {e}")
    
    # Instrucciones
    st.markdown("---")
    st.markdown("### 📋 Instrucciones")
    st.markdown("""
    **Para analizar fragmentos de búsqueda:**
    1. Ve a **🔍 Búsqueda Semántica**
    2. Realiza una búsqueda
    3. Selecciona un fragmento con el botón **📊 Analizar NLP**
    4. Regresa aquí para ver el análisis
    
    **Para análisis de texto libre:**
    - Usa la pestaña **✍️ Texto libre** arriba
    - Pega cualquier texto y analízalo directamente
    """)

# Mostrar historial si está activado
if st.session_state.get('show_history', False) and st.session_state.get('analysis_history'):
    st.markdown("---")
    st.markdown("### 📜 Historial de análisis")
    
    for i, analysis in enumerate(reversed(st.session_state.analysis_history)):
        with st.expander(f"{analysis['type']} - {analysis['text']}", expanded=False):
            st.json(analysis['result'])

# Footer con información técnica
st.markdown("---")
st.markdown("### 🔧 Información técnica")
st.markdown("""
**Herramientas NLP utilizadas:**

- **🏷️ NER (Named Entity Recognition):**
  - Modelo: SpaCy `es_core_news_lg`
  - Identifica: Personas, Lugares, Organizaciones, Fechas, etc.
  
- **😊 Análisis de Sentimiento:**
  - Modelo: `pysentimiento/robertuito-sentiment-analysis`
  - Clasificación: Positivo, Negativo, Neutral
  - Optimizado para español latinoamericano

**Tipos de entidades detectadas:**
- **👤 PER/PERSON**: Nombres de personas
- **📍 LOC/LOCATION**: Lugares y ubicaciones  
- **🏢 ORG/ORGANIZATION**: Organizaciones e instituciones
- **📅 DATE**: Fechas y períodos temporales
- **💰 MONEY**: Cantidades monetarias
- **📊 PERCENT**: Porcentajes
- **🏷️ MISC**: Otras entidades relevantes
"""))