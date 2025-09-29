# app/streamlit_app.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from src.rag_chain import RAGHybridPipeline

# ================= CONFIGURACIÓN DE PÁGINA =================
st.set_page_config(
    page_title="Agente Gauteovan IA - Chat",
    page_icon="💬",
    layout="wide"
)

# ================= INICIALIZAR RAG =================
@st.cache_resource
def init_rag_system():
    """Inicializa RAG Pipeline una sola vez"""
    try:
        return RAGHybridPipeline(use_memory=False), None
    except TypeError as e:
        if "unexpected keyword argument 'use_memory'" in str(e):
            try:
                return RAGHybridPipeline(), None
            except Exception as e2:
                return None, f"Fallo al crear RAGHybridPipeline sin use_memory: {e2}"
        return None, str(e)
    except Exception as e:
        return None, str(e)

rag, init_error = init_rag_system()
if init_error:
    st.error(f"❌ Error al inicializar el sistema: {init_error}")
    st.stop()

# ================= INTERFAZ PRINCIPAL =================
st.title("💬 Agente de Opinión Gauteovan (2018–2020)")
st.markdown("### Chat con razonamiento profundo y búsqueda web opcional")

# Opciones principales
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    allow_web = st.checkbox("🌐 Permitir búsqueda web", value=False)
with col2:
    use_deep_reason = st.checkbox("🧠 Razonamiento profundo", value=False)
with col3:
    k_docs = st.slider("📄 Documentos locales", 3, 20, 8)

# Inicializar historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# ================= MOSTRAR HISTORIAL =================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            # indicador del modo
            mode = message.get("mode_info")
            if mode == "local":
                st.success("✅ Respuesta basada en evidencia local")
            elif mode == "local_deep_reason":
                st.success("🧠 Respuesta con razonamiento profundo sobre evidencia local")
            elif mode == "local_insufficient":
                st.warning("⚠️ Evidencia local insuficiente")
            elif mode == "web":
                st.info("🌐 Respuesta usando evidencia local + web")
            elif mode in ("web_failure", "web_none"):
                st.error("❌ No se encontraron resultados web suficientes")

            # Mostrar fuentes guardadas
            sources = message.get("sources") or []
            if sources:
                with st.expander(f"📚 Fuentes ({len(sources)})", expanded=False):
                    for i, src in enumerate(sources, 1):
                        if src.get("url"):  # fuente web
                            st.markdown(f"""
**{i}.** 🌐 **{src.get('titulo', 'Sin título')}**  
🔗 {src.get('url', src.get('doc_id',''))} • 📊 Score: {src.get('rrf_score','N/A')}  
📄 *{(src.get('snippet','') or src.get('texto',''))[:200]}...*
""")
                        else:  # fuente local
                            fecha_str = str(src.get('fecha',''))[:10]
                            st.markdown(f"""
**{i}.** **{src.get('título', src.get('titulo','Sin título'))}**  
👤 {src.get('autor', 'Sin autor')} • 📰 *{src.get('diario', 'Sin medio')}* • 📅 {fecha_str}  
🔗 {src.get('vínculo', src.get('doc_id',''))}
""")

            # Mostrar briefs guardados
            if "briefs" in message and message["briefs"]:
                with st.expander("🧠 Análisis estructurado", expanded=False):
                    briefs = message["briefs"]
                    for key, title in [
                        ("por_año", "📅 Por Año"),
                        ("por_medio", "📰 Por Medio"),
                        ("por_autor", "👤 Por Autor"),
                    ]:
                        st.markdown(f"#### {title}")
                        if briefs.get(key):
                            for item in briefs[key]:
                                st.write(f"• {item}")
                        else:
                            st.info("No disponible")

# ================= INPUT DEL CHAT =================
if prompt := st.chat_input("Haz una pregunta sobre las columnas de opinión..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analizando y generando respuesta..."):
            try:
                result = rag.answer(
                    question=prompt,
                    k_local=k_docs,
                    allow_web=allow_web,
                    use_deep_reason=use_deep_reason
                )
                answer = result.get("answer", "")
                mode = result.get("mode", "unknown")

                # Fuentes y briefs
                sources = result.get("hits") or result.get("web_results") or []
                briefs = result.get("briefs")

                # Mostrar respuesta inmediatamente
                st.markdown(answer or "*(Sin respuesta)*")

                # Mostrar fuentes inmediatamente
                if sources:
                    with st.expander(f"📚 Fuentes ({len(sources)})", expanded=False):
                        for i, src in enumerate(sources, 1):
                            if src.get("url"):
                                st.markdown(f"""
**{i}.** 🌐 **{src.get('titulo', 'Sin título')}**  
🔗 {src.get('url', src.get('doc_id',''))} • 📊 Score: {src.get('rrf_score','N/A')}  
📄 *{(src.get('snippet','') or src.get('texto',''))[:200]}...*
""")
                            else:
                                fecha_str = str(src.get('fecha',''))[:10]
                                st.markdown(f"""
**{i}.** **{src.get('título', src.get('titulo','Sin título'))}**  
👤 {src.get('autor', 'Sin autor')} • 📰 *{src.get('diario', 'Sin medio')}* • 📅 {fecha_str}  
🔗 {src.get('vínculo', src.get('doc_id',''))}
""")

                # Mostrar briefs inmediatamente
                if briefs:
                    with st.expander("🧠 Análisis estructurado", expanded=False):
                        for key, title in [
                            ("por_año", "📅 Por Año"),
                            ("por_medio", "📰 Por Medio"),
                            ("por_autor", "👤 Por Autor"),
                        ]:
                            st.markdown(f"#### {title}")
                            if briefs.get(key):
                                for item in briefs[key]:
                                    st.write(f"• {item}")
                            else:
                                st.info("No disponible")

                # Guardar en historial
                msg = {
                    "role": "assistant",
                    "content": answer or "*(Sin respuesta)*",
                    "mode_info": mode,
                }
                if sources:
                    msg["sources"] = sources
                if briefs:
                    msg["briefs"] = briefs
                st.session_state.messages.append(msg)

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# ================= EJEMPLOS =================
st.markdown("---")
st.markdown("### 💡 Ejemplos de preguntas")
ejemplos = [
    "¿Qué opinaban los columnistas sobre el acuerdo de paz en 2019?",
    "¿Cuáles eran las principales críticas a la educación pública?",
    "¿Qué se decía sobre las protestas estudiantiles en 2018?",
    "¿Cómo se abordó el tema de la corrupción en los medios?",
]
for ej in ejemplos:
    st.markdown(f"• {ej}")

# ================= CONTROLES =================
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    if st.button("🗑️ Limpiar conversación", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
with col2:
    if st.session_state.messages:
        conversation = "\n\n".join(
            [
                ("Usuario" if m["role"] == "user" else "Asistente") + ": " + m["content"]
                for m in st.session_state.messages
            ]
        )
        st.download_button("💾 Descargar chat", data=conversation, file_name="chat.txt")

# ================= FOOTER =================
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#666;'>"
    "<small>🎓 Agente Gauteovan IA • RAG híbrido local+web </small>"
    "</div>",
    unsafe_allow_html=True,
)
