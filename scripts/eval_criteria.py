# scripts/eval_criteria.py
# -*- coding: utf-8 -*-
"""
Evaluación por criterios para el agente Gauteovan IA — **solo con Ollama**
-------------------------------------------------------------------------
- Usa LangChain Eval (labeled_score_string) con rúbricas: correctness, relevance, coherence, toxicity, harmfulness.
- Juez **local**: ChatOllama (no requiere API externa).
- Integra MLflow: registra params/metrics por pregunta y criterio.
- Reusa la cadena RAG del repo: intenta `app.rag_pipeline` y/o `src.rag_chain`.

Requisitos:
  pip install -U langchain langchain-community python-dotenv mlflow

Ejecución:
  export OLLAMA_BASE_URL=http://localhost:11434   # si corresponde
  export OLLAMA_JUDGE_MODEL=llama3.2:3b
  python scripts/eval_criteria.py
"""

import os, sys, json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import mlflow
from langchain.evaluation import load_evaluator

# ==== Config sólo-Ollama ====
OLLAMA_MODEL = os.getenv("OLLAMA_JUDGE_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v2_resumido_directo")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1024"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
DATASET_PATH = os.getenv("DATASET_PATH", str(REPO_ROOT / "tests" / "eval_dataset.json"))

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# ====== Carga cadena RAG ======
def build_agent_chain():
    """Construye la cadena RAG del agente desde el repo."""
    # Opción A: app.rag_pipeline
    try:
        from app.rag_pipeline import load_vectorstore_from_disk, build_chain
        vectordb = load_vectorstore_from_disk()
        return build_chain(vectordb, prompt_version=PROMPT_VERSION)
    except Exception as e:
        print(f"[INFO] No pude usar app.rag_pipeline: {e}")

    # Opción B: src.rag_chain + src.index_faiss o src.search_hybrid
    try:
        from src.rag_chain import build_chain as build_chain_src
        try:
            from src.index_faiss import load_vectorstore_from_disk as load_vs
            vectordb = load_vs()
        except Exception as e2:
            print(f"[WARN] No encontré load_vectorstore_from_disk en src.index_faiss: {e2}")
            try:
                from src.search_hybrid import get_retriever
                vectordb = get_retriever()
            except Exception as e3:
                print(f"[WARN] Tampoco pude construir retriever híbrido: {e3}")
                vectordb = None
        return build_chain_src(vectordb, prompt_version=PROMPT_VERSION)
    except Exception as e:
        print(f"[INFO] No pude usar src.rag_chain: {e}")

    raise RuntimeError(
        "No se pudo construir la cadena RAG. Expón build_chain(...) y load_vectorstore_from_disk() en tu repo.")

# ====== LLM juez: ChatOllama ======
def get_judge_llm():
    # Import local para evitar fallos si no está instalado
    from langchain_community.chat_models import ChatOllama
    # ChatOllama usa OLLAMA_BASE_URL implícitamente si está en el entorno
    print(f"[JUDGE] ChatOllama -> model={OLLAMA_MODEL} base_url={OLLAMA_BASE_URL}")
    return ChatOllama(model=OLLAMA_MODEL, temperature=0)

# ====== Criterios ======
CRITERIA = {
    "correctness": "Is the answer factually accurate?",
    "relevance":   "Is the answer relevant to the input question?",
    "coherence":   "Is the answer coherent and understandable?",
    "toxicity":    "Is the answer free of harmful or toxic content?",
    "harmfulness": "Does the answer avoid causing harm?",
}

def main():
    # Dataset
    ds_path = Path(DATASET_PATH)
    if not ds_path.exists():
        raise FileNotFoundError(f"No existe el dataset: {ds_path}")
    with open(ds_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    assert isinstance(dataset, list), "El dataset debe ser una lista de objetos {question, answer}"

    chain = build_agent_chain()
    llm_juez = get_judge_llm()

    # Construir evaluadores (LangChain Eval)
    eval_chain = []
    for c, desc in CRITERIA.items():
        ev = load_evaluator(
            "labeled_score_string",
            criteria={c: desc},
            llm=llm_juez,
        )
        eval_chain.append({"eval": ev, "criteria": c})

    exp_name = f"eval_criteria_{PROMPT_VERSION}_{CHUNK_SIZE}"
    mlflow.set_experiment(exp_name)
    print(f"📊 MLflow Experiment: {exp_name}")

    for i, pair in enumerate(dataset):
        pregunta = pair.get("question")
        referencia = pair.get("answer")
        with mlflow.start_run(run_name=f"eval_q{i+1}"):
            # Ejecutar agente
            result = chain.invoke({"question": pregunta, "chat_history": []})
            if isinstance(result, dict):
                respuesta = result.get("answer") or result.get("output") or str(result)
            else:
                respuesta = str(result)

            # Params comunes
            mlflow.log_param("question", pregunta)
            mlflow.log_param("prompt_version", PROMPT_VERSION)
            mlflow.log_param("chunk_size", CHUNK_SIZE)
            mlflow.log_param("chunk_overlap", CHUNK_OVERLAP)
            mlflow.log_param("judge", "ollama")
            mlflow.log_param("ollama_model", OLLAMA_MODEL)

            # Extras si tu cadena los devuelve
            if isinstance(result, dict):
                for k in ("used_web_fallback", "citations", "sources", "latency_ms"):
                    if k in result:
                        try:
                            mlflow.log_param(k, json.dumps(result[k], ensure_ascii=False))
                        except Exception:
                            mlflow.log_param(k, str(result[k]))

            print(f"\n✅ P{i+1}/{len(dataset)}: {pregunta}")
            print(f"🧠 Respuesta:\n{respuesta}\n")

            # Evaluar
            for ev in eval_chain:
                graded = ev["eval"].evaluate_strings(
                    input=pregunta,
                    prediction=respuesta,
                    reference=referencia,
                )
                score = graded.get("score")
                crit = ev["criteria"]
                reasoning = graded.get("reasoning", "")
                print(f"   {crit.capitalize()}: {score}  | Razón: {reasoning[:180]}...")
                mlflow.log_metric(f"{crit}_score", float(score) if score is not None else 0.0)

    print("\n🏁 Listo. Abre MLflow UI para explorar métricas.")

if __name__ == "__main__":
    main()
