# src/rag_chain.py
from typing import Optional, Dict, Any
import pandas as pd
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferWindowMemory

from .utils import _evidence_sufficient
from .prompts import SYSTEM_LOCAL, SYSTEM_WEB, PROMPT
from .briefings import structured_briefs
from .formatting import format_evidence_local, format_evidence_web
from .intent_user import parse_user_intent
from .web_search import web_search_duckduckgo
from .search_hybrid import HybridSearcher


class RAGHybridPipeline:
    """
    Flujo:
      1. Buscar en base local con FAISS+BM25.
      2. Si no hay suficiente evidencia local:
          - Responde con aviso, salvo que allow_web=True.
          - Si allow_web=True → buscar en la web.
    Devuelve un dict con: respuesta, modo, evidencia usada, fuentes, briefs.
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        temperature: float = 0.2,
        searcher: Optional[HybridSearcher] = None,
        use_memory: bool = False,  # 🔧 ahora configurable
    ):
        self.llm = Ollama(model=model, temperature=temperature)
        self.searcher = searcher or HybridSearcher()
        self.memory = ConversationBufferWindowMemory(k=5, return_messages=True) if use_memory else None

    def answer(
        self,
        question: str,
        k_local: int = 8,
        min_local_chars: int = 400,
        k_web: int = 6,
        allow_web: Optional[bool] = None,
        use_deep_reason: bool = False,
    ) -> Dict[str, Any]:
        """
        Devuelve un diccionario con:
          - mode: 'local', 'local_deep_reason', 'local_insufficient', 'web', 'web_none'
          - answer: respuesta en texto
          - evidence_text: evidencia formateada
          - hits / web_results: fuentes consultadas
          - briefs: análisis estructurado (si deep_reason=True)
        """
        # 1) Intención del usuario (para activar web si lo pide explícitamente)
        clean_question, intent_allow_web = parse_user_intent(question)
        if allow_web is None:
            allow_web = intent_allow_web

        # 2) RAG local
        hits = self.searcher.search(clean_question, final_k=k_local)

        if _evidence_sufficient(hits, min_total_chars=min_local_chars):
            evidence = format_evidence_local(hits, enumerate_chunks=True)
            briefs = None

            if use_deep_reason:
                briefs = structured_briefs(self, clean_question, evidence)  # 🔧 fix aquí
                import json
                evidence = json.dumps(briefs, ensure_ascii=False, indent=2)

            prompt = PROMPT.format(system=SYSTEM_LOCAL, question=clean_question, evidence=evidence)
            answer = self.llm.invoke(prompt)

            if self.memory is not None:
                self.memory.save_context({"input": clean_question}, {"output": answer})

            return {
                "mode": "local_deep_reason" if use_deep_reason else "local",
                "answer": answer,
                "evidence_text": evidence,
                "hits": hits.to_dict(orient="records"),
                "allow_web": allow_web,
                "briefs": briefs,
            }

        # 3) Evidencia local insuficiente
        if not allow_web:
            return {
                "mode": "local_insufficient",
                "answer": (
                    "⚠️ No encontré evidencia suficiente en la base local para responder con confianza. "
                    "Puedes activar la búsqueda en web para intentar complementar."
                ),
                "evidence_text": "",
                "hits": hits.to_dict(orient="records") if isinstance(hits, pd.DataFrame) else [],
                "allow_web": allow_web,
            }

        # 4) Fallback Web
        web_results = web_search_duckduckgo(clean_question, max_results=k_web)
        if not web_results:
            return {
                "mode": "web_none",
                "answer": "❌ No encontré información confiable en resultados web.",
                "evidence_text": "",
                "web_results": [],
                "allow_web": allow_web,
            }

        evidence = format_evidence_web(web_results, enumerate_items=True)
        prompt = PROMPT.format(system=SYSTEM_WEB, question=clean_question, evidence=evidence)
        answer = self.llm.invoke(prompt)

        if self.memory is not None:
            self.memory.save_context({"input": clean_question}, {"output": answer})

        return {
            "mode": "web",
            "answer": answer,
            "evidence_text": evidence,
            "web_results": web_results,
            "sources": web_results, 
            "allow_web": allow_web,
        }
