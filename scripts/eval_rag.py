# Plantilla de evaluación rápida
from time import perf_counter
from src.rag_chain import RAGPipeline

QUERIES = [
    "¿Qué se opinaba sobre el acuerdo de paz en 2019?",
    "Columnas sobre protestas de 2019 en Bogotá",
]

if __name__ == "__main__":
    rag = RAGPipeline()
    for q in QUERIES:
        t0 = perf_counter()
        ans, hits = rag.answer(q)
        dt = perf_counter() - t0
        print("Q:", q)
        print("t=", round(dt, 2), "s")
        print(ans[:400], "…\n---\n")
