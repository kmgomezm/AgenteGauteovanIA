# scripts/eval_rag.py

# Se corre en terminal con: python -m scripts.eval_rag

# Plantilla de evaluación rápida 
from time import perf_counter
from src.rag_chain import RAGHybridPipeline

QUERIES = [
    "¿Qué se opinaba sobre el acuerdo de paz en 2019?",
    "Columnas sobre protestas de 2019 en Bogotá",
]

if __name__ == "__main__":
    rag = RAGHybridPipeline()
    for q in QUERIES:
        t0 = perf_counter()
        result = rag.answer(q)  # devuelve un dict
        dt = perf_counter() - t0

        ans = result.get("answer", "")
        hits = result.get("hits") or result.get("web_results", [])

        print("Q:", q)
        print("t=", round(dt, 2), "s")
        print("RESPUESTA:\n", ans[:400], "…\n")
        print("FUENTES:", len(hits), "\n---\n")