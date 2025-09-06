# Agente de Opinión 🇨🇴 (2018–2020) — RAG Híbrido (Local, OSS)

Proyecto base para un agente conversacional (Streamlit + LangChain + Ollama) que analiza ~13k columnas de opinión de periódicos colombianos (2018–2020).  
**100% local** (CPU ok), con **FAISS + BM25**, **citas** y herramientas de **NLP** (sentimiento, NER, clasificación).

## 🚀 Puesta en marcha (pasos mínimos)
1) **Instala dependencias**
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -U pip
pip install -r requirements.txt
python -m spacy download es_core_news_lg
```
2) **Instala Ollama** y modelos locales (en otra terminal):
```bash
# https://ollama.com/download
ollama pull mistral:7b-instruct
# opcional:
# ollama pull llama3.1:8b-instruct
```
3) **Coloca el Excel** en `data/raw/opiniones.xlsx` (con columnas: `autor, fecha, titulo, periódico/periodico, texto`).
4) **Construye los índices** (ingesta + FAISS + BM25):
```bash
python scripts/build_index.py
```
5) **Lanza la app**
```bash
streamlit run app/streamlit_app.py
```

## 📁 Estructura
```
colombia-opinion-agent/
├─ app/
│  ├─ streamlit_app.py
│  └─ pages/
│     ├─ 1_Chat.py
│     ├─ 2_Buscador_avanzado.py
│     ├─ 3_Analisis_NLP.py
│     └─ 4_Reportes_y_Graficas.py
├─ data/
│  ├─ raw/opiniones.xlsx        # (pon aquí tu archivo)
│  ├─ processed/chunks.parquet  # (generado)
│  └─ indexes/
│      ├─ faiss.index           # (generado)
│      ├─ faiss_meta.parquet    # (generado)
│      └─ bm25.pkl              # (generado; o sqlite.db si migras a FTS5)
├─ scripts/
│  ├─ build_index.py
│  └─ eval_rag.py               # (plantilla)
├─ src/
│  ├─ __init__.py
│  ├─ ingest.py
│  ├─ index_faiss.py
│  ├─ index_bm25.py
│  ├─ search_hybrid.py
│  ├─ prompts.py
│  ├─ rag_chain.py
│  ├─ nlp_tools.py
│  ├─ plots.py
│  └─ utils.py
├─ tests/
├─ requirements.txt
└─ README.md
```

## 🧠 Conceptos clave
- **RAG híbrido**: FAISS (embeddings `intfloat/multilingual-e5-small`) + BM25 (rank-bm25). Fusión con **RRF**.
- **Citas**: cada afirmación clave cita `[autor, periódico, fecha, título, doc_id]`.
- **NLP**: spaCy (`es_core_news_lg`) para NER; BETO para sentimiento; zero-shot opcional para tópicos.

## 📌 Notas
- Este esqueleto corre **CPU-only**. Si tienes GPU, Transformers y Sentence-Transformers la aprovecharán.
- Si tu Excel difiere, ajusta `src/ingest.py` (renombrado de columnas).
- Para resultados más “seguros”, usa `temperature=0.2` en `rag_chain.py`.

## 🧪 Evaluación (rápida)
- Especifica 30–50 queries de prueba en `scripts/eval_rag.py`, mide `Recall@K` y latencias por etapa.
- Activa el logging en `app/streamlit_app.py` si quieres conservar conversaciones.

¡Listo para iterar! Cualquier mejora (FTS5, reranking, filtros UI) la puedes añadir sin romper el flujo base.
