# pages/2_Analisis_NLP.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from src.search_hybrid import HybridSearcher
from src import nlp_tools

st.set_page_config(
    page_title="Análisis NLP",
    page_icon="🧠",
    layout="wide"
)

# ================= INICIALIZAR BUSCADOR =================
@st.cache_resource
def init_searcher():
    try:
        return HybridSearcher(), None
    except Exception as e:
        return None, str(e)

searcher, init_error = init_searcher()
if init_error:
    st.error(f"❌ Error al inicializar el buscador: {init_error}")
    st.stop()

# ================= ENCABEZADO =================
st.title("🧠 Análisis NLP de Fragmentos")
st.markdown("Aplica **NER**, **sentimiento** y **zero-shot** a fragmentos recuperados por **FAISS+BM25**.")

# ================= CONTROLES =================
with st.sidebar:
    st.header("⚙️ Configuración")
    k = st.slider("📄 Documentos a recuperar", 3, 50, 12)
    aplicar_ner = st.checkbox("🏷️ Entidades (NER)", True)
    aplicar_sent = st.checkbox("🙂 Sentimiento", True)
    aplicar_zs = st.checkbox("🎯 Zero-shot", True)
    etiquetas_zs = st.text_input(
        "Etiquetas (coma separadas)",
        "política, educación, economía, salud, ciencia, arte, cultura, deportes, tecnología"
    )

# ================= BÚSQUEDA =================
query = st.text_input("Escribe tu consulta:", placeholder="Ej. aborto 2018, educación superior 2019…")

results = None
if query:
    with st.spinner("🔎 Buscando en los documentos..."):
        results = searcher.search(query, final_k=k)

# ================= SELECCIÓN DE FRAGMENTO (ARREGLO) =================
selected_text = None
selected_row = None

if results is not None and not results.empty:
    st.subheader(f"Resultados para: **{query}**")

    # Preparar opciones para UN SOLO selector
    df = results.reset_index(drop=True).copy()
    options = []
    for i, r in df.iterrows():
        titulo = r.get("título", r.get("titulo", "Sin título"))
        diario = r.get("diario", "?")
        fecha = str(r.get("fecha", ""))[:10]
        preview = (r.get("chunk", "") or "").replace("\n", " ")
        preview = (preview[:220] + "…") if len(preview) > 220 else preview
        options.append(f"{i+1}. {titulo} — {diario} — {fecha}\n{preview}")

    # Un solo selectbox con todas las opciones
    selected_idx = st.selectbox(
        "Selecciona un fragmento a analizar:",
        options=list(range(len(options))),
        format_func=lambda i: options[i],
        index=0 if "nlp_sel_idx" not in st.session_state else st.session_state["nlp_sel_idx"]
    )
    st.session_state["nlp_sel_idx"] = selected_idx

    selected_row = df.iloc[selected_idx]
    selected_text = selected_row.get("chunk", "")

    # Mostrar meta y texto elegido
    st.markdown("---")
    st.markdown("### 📄 Fragmento seleccionado")
    meta = f"👤 {selected_row.get('autor','?')} • 📰 {selected_row.get('diario','?')} • 📅 {str(selected_row.get('fecha',''))[:10]}"
    st.caption(meta)
    st.write(selected_text)

    # ================= APLICAR NLP =================
    st.markdown("---")
    st.subheader("🔎 Análisis NLP")

    cols = st.columns(3)
    # --- NER ---
    with cols[0]:
        if aplicar_ner:
            st.markdown("#### 🏷️ Entidades")
            try:
                ents = nlp_tools.ner(selected_text)
                if ents:
                    for ent, label, score in ents:
                        st.write(f"- **{ent}** → {label} (confianza: {score})")
                else:
                    st.info("Sin entidades detectadas.")
            except Exception as e:
                st.error(f"Error en NER: {e}")

    # --- Sentimiento ---
    with cols[1]:
        if aplicar_sent:
            st.markdown("#### 🙂 Sentimiento")
            try:
                sent = nlp_tools.sentiment(selected_text)
                if "error" in sent:
                    st.error(f"Error en sentimiento: {sent['error']}")
                else:
                    st.write(f"**Etiqueta predicha:** {sent['label']}")
                    st.markdown("**Probabilidades:**")
                    for k, v in sent["probs"].items():
                        st.write(f"- {k}: {v}")
            except Exception as e:
                st.error(f"Error en sentimiento: {e}")

    # --- Zero-shot ---
    with cols[2]:
        if aplicar_zs:
            st.markdown("#### 🎯 Zero-shot")
            try:
                labels = [x.strip() for x in etiquetas_zs.split(",") if x.strip()]
                if not labels:
                    st.info("Define al menos una etiqueta.")
                else:
                    zs = nlp_tools.classify(selected_text, labels)
                    if zs:
                        for label, score in zs:
                            st.write(f"- **{label}** → {score}")
            except Exception as e:
                st.error(f"Error en zero-shot: {e}")

else:
    if query:
        st.warning("⚠️ No se encontraron resultados.")
    else:
        st.info("Escribe una consulta para recuperar fragmentos y analizarlos.")
