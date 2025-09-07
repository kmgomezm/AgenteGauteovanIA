# src/rag_pipeline.py
from typing import Optional, Dict, Any

import pandas as pd
from langchain_community.llms import Ollama

from .prompts import SYSTEM_LOCAL, SYSTEM_WEB, PROMPT
from .formatting import format_evidence_local, format_evidence_web
from .intent import parse_user_intent
from .web_search import web_search_duckduckgo

# Tu buscador híbrido (debes tenerlo implementado en el proyecto)
from .search_hybrid import HybridSearcher


def _evidence_sufficient(df: Optional[pd.DataFrame], min_total_chars: int = 600) -> bool:
    """Heurística para decidir si hay suficiente evidencia local."""
    if df is None or len(df) == 0:
        return False
    total = int(sum(len(str(x)) for x in df.get("chunk", [])))
    return total >= min_total_chars

class RAGHybridPipeline:
    """
    Flujo: RAG local -> (opcional) Fallback Web (solo si el usuario lo pide).
    Devuelve un dict con respuesta + modo + evidencia usada.
    """
    def __init__(
        self,
        model: str = "llama3.1:8b",
        temperature: float = 0.2,
        searcher: Optional[HybridSearcher] = None,
    ):
        self.llm = Ollama(model=model, temperature=temperature)
        self.searcher = searcher or HybridSearcher()

    def answer(
        self,
        question: str,
        k_local: int = 8,
        min_local_chars: int = 600,
        k_web: int = 6,
        allow_web: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        - Primero intenta responder con evidencia LOCAL.
        - Si la evidencia local es insuficiente:
            - Solo hace fallback a WEB si allow_web=True (o si el usuario lo indicó en la pregunta).
            - Si allow_web=False: responde que no encontró suficiente evidencia local y sugiere activar web.
        """
        # 1) Intención del usuario (si no llega allow_web explícito)
        clean_question, intent_allow_web = parse_user_intent(question)
        if allow_web is None:
            allow_web = intent_allow_web

        # 2) RAG local
        hits = self.searcher.search(clean_question, final_k=k_local)
        if _evidence_sufficient(hits, min_total_chars=min_local_chars):
            evidence = format_evidence_local(hits, enumerate_chunks=True)
            prompt   = PROMPT.format(system=SYSTEM_LOCAL, question=clean_question, evidence=evidence)
            answer   = self.llm.invoke(prompt)
            return {
                "mode": "local",
                "answer": answer,
                "evidence_text": evidence,
                "hits": hits.to_dict(orient="records"),
                "allow_web": allow_web,
            }

        # 3) ¿Debe activar Web?
        if not allow_web:
            return {
                "mode": "local_insufficient",
                "answer": (
                    "No encontré evidencia suficiente en la base local para responder con confianza. "
                    "Si deseas que intente buscar en la web, agrega #web a tu pregunta o activa allow_web=True."
                ),
                "evidence_text": "",
                "hits": hits.to_dict(orient="records") if isinstance(hits, pd.DataFrame) else [],
                "allow_web": allow_web,
            }

        # 4) Fallback Web (a petición del usuario)
        web_results = web_search_duckduckgo(clean_question, max_results=k_web)
        if not web_results:
            return {
                "mode": "web_none",
                "answer": (
                    "No encontré información confiable en resultados web para complementar la evidencia local."
                ),
                "evidence_text": "",
                "web_results": [],
                "allow_web": allow_web,
            }

        evidence = format_evidence_web(web_results, enumerate_items=True)
        prompt   = PROMPT.format(system=SYSTEM_WEB, question=clean_question, evidence=evidence)
        answer   = self.llm.invoke(prompt)

        return {
            "mode": "web",
            "answer": answer,
            "evidence_text": evidence,
            "web_results": web_results,
            "allow_web": allow_web,
        }

