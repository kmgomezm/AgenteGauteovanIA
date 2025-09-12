# src/rag_pipeline.py
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

# Tu buscador híbrido (debes tenerlo implementado en el proyecto)
from .search_hybrid import HybridSearcher


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
        self.memory = ConversationBufferWindowMemory(k=5, return_messages=True)
        self.llm = Ollama(model=model, temperature=temperature)
        self.searcher = searcher or HybridSearcher()

    def answer(
        self,
        question: str,
        k_local: int = 8,
        min_local_chars: int = 400,
        k_web: int = 6,
        allow_web: Optional[bool] = None,
        use_deep_reason: bool = False,  # Nuevo argumento para razonamiento profundo
    ) -> Dict[str, Any]:
        """
        - Primero intenta responder con evidencia LOCAL.
        - Si la evidencia local es insuficiente:
            - Solo hace fallback a WEB si allow_web=True (o si el usuario lo indicó en la pregunta).
            - Si allow_web=False: responde que no encontró suficiente evidencia local y sugiere activar web.
        """
        # 1) Intención del usuario (si no llega allow_web True)
        clean_question, intent_allow_web = parse_user_intent(question)
        if allow_web is None: # Si no se especifica, usar la intención del usuario
            allow_web = intent_allow_web # Esta puede ser True o False

        # 2) RAG local
        hits = self.searcher.search(clean_question, final_k=k_local)
        # Ajustar min_local_chars y k_local para determinar qué tan "suficiente" es suficiente
        if _evidence_sufficient(hits, min_total_chars=min_local_chars):
            evidence = format_evidence_local(hits, enumerate_chunks=True)
            briefs = None
            if use_deep_reason:
                briefs = structured_briefs(clean_question, evidence)
                # Puedes ajustar el formato de briefs según lo que espera el LLM
                import json
                evidence = json.dumps(briefs, ensure_ascii=False, indent=2)
            prompt = PROMPT.format(system=SYSTEM_LOCAL, question=clean_question, evidence=evidence)
            answer = self.llm.invoke(prompt)
            # Guardar en memoria si está habilitada
            if hasattr(self, "memory") and self.memory is not None:
                self.memory.save_context({"input": clean_question}, {"output": answer})
            return {
                "mode": "local_deep_reason" if use_deep_reason else "local",
                "answer": answer,
                "evidence_text": evidence,
                "hits": hits.to_dict(orient="records"),
                "allow_web": allow_web,
                "briefs": briefs if use_deep_reason else None,
            }

        # 3) ¿Debe activar Web?
        if not allow_web:
            return {
                "mode": "local_insufficient",
                "answer": (
                    "No encontré evidencia suficiente en la base local para responder con confianza. "
                    "Indicame si quieres que intente buscar en internet, o activa allow_web=True."
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
                "allow_web": allow_web
            }

        evidence = format_evidence_web(web_results, enumerate_items=True)
        prompt   = PROMPT.format(system=SYSTEM_WEB, question=clean_question, evidence=evidence)
        answer   = self.llm.invoke(prompt)
        # Guardar en memoria si está habilitada
        if hasattr(self, "memory") and self.memory is not None:
            self.memory.save_context({"input": clean_question}, {"output": answer})

        return {
            "mode": "web",
            "answer": answer,
            "evidence_text": evidence,
            "web_results": web_results,
            "allow_web": allow_web,
        }

    

