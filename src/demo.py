# src/main_demo.py
import os
from .rag_pipeline import RAGHybridPipeline

def main():
    pipeline = RAGHybridPipeline(
        model=os.getenv("LLM_MODEL", "llama3.1:8b"),
        temperature=float(os.getenv("LLM_TEMP", "0.2"))
    )

    print("Escribe tu pregunta. Para permitir búsqueda web, añade el token #web (p.ej. '... #web').")
    q = input("Pregunta: ").strip()

    result = pipeline.answer(q, k_local=8, min_local_chars=600, k_web=6)
    print("\n=== MODO ===", result.get("mode"))
    print("\n=== RESPUESTA ===\n", result.get("answer"))
    if result.get("evidence_text"):
        print("\n=== EVIDENCIA (vista previa) ===\n", (result["evidence_text"][:1200] + "…") if len(result["evidence_text"]) > 1200 else result["evidence_text"])

if __name__ == "__main__":
    main()
