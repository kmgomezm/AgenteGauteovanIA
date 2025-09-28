# src/rag_chain.py
from typing import Optional, Dict, Any
import pandas as pd
from langchain_ollama import OllamaLLM as Ollama
from .search_hybrid import HybridSearcher
from .prompts import SYSTEM_LOCAL, PROMPT, format_evidence
from .briefings import structured_briefs
from .web_search import web_search_duckduckgo


class RAGPipeline:
    """Pipeline RAG simple para consultas con evidencia local"""
    
    def __init__(self, model="llama3.2:3b", temperature=0.2):
        self.llm = Ollama(model=model, temperature=temperature)
        self.searcher = HybridSearcher()

    def answer(self, question: str, k: int = 8):
        """
        Responde una pregunta usando evidencia local de la base de datos.
        
        Args:
            question: La pregunta a responder
            k: Número de documentos a recuperar
            
        Returns:
            Tuple de (respuesta, hits_dataframe)
        """
        # Buscar evidencia relevante
        hits = self.searcher.search(question, final_k=k)
        
        if hits is None or hits.empty:
            return "No se encontró evidencia relevante para responder la pregunta.", pd.DataFrame()
        
        # Formatear evidencia para el prompt
        evidence = format_evidence(hits)

        # Crear prompt con evidencia
        prompt = PROMPT.format(
            system=SYSTEM_LOCAL, 
            question=question, 
            evidence=evidence
        )
        
        # Generar respuesta
        answer = self.llm.invoke(prompt)
        
        return answer, hits


class RAGHybridPipeline:
    """
    Pipeline RAG híbrido con fallback web y razonamiento profundo opcional.
    """
    
    def __init__(
        self,
        model: str = "llama3.2:3b",
        temperature: float = 0.2,
        searcher: Optional[HybridSearcher] = None,
    ):
        self.llm = Ollama(model=model, temperature=temperature)
        self.searcher = searcher or HybridSearcher()

    def answer(
        self,
        question: str,
        k_local: int = 8,
        min_local_chars: int = 400,
        allow_web: bool = False,
        use_deep_reason: bool = False,
    ) -> Dict[str, Any]:
        """
        Responde usando evidencia local, con fallback web y razonamiento profundo opcionales.
        """
        
        # 1) RAG local
        hits = self.searcher.search(question, final_k=k_local)
        
        # Verificar si la evidencia local es suficiente
        if hits is not None and not hits.empty:
            total_chars = sum(len(str(chunk)) for chunk in hits.get('chunk', []))
            
            if total_chars >= min_local_chars:
                evidence = format_evidence(hits)
                prompt = PROMPT.format(
                    system=SYSTEM_LOCAL, 
                    question=question, 
                    evidence=evidence
                )
                answer = self.llm.invoke(prompt)
                
                # Generar briefs si se solicita razonamiento profundo
                briefs = None
                if use_deep_reason:
                    print("Generando análisis estructurado...")
                    briefs = structured_briefs(self, question, evidence)
                
                return {
                    "mode": "local",
                    "answer": answer,
                    "evidence_text": evidence,
                    "hits": hits.to_dict(orient="records") if isinstance(hits, pd.DataFrame) else [],
                    "allow_web": allow_web,
                    "briefs": briefs,
                }
        
        # 2) Si evidencia local insuficiente
        if not allow_web:
            return {
                "mode": "local_insufficient",
                "answer": (
                    "No encontré evidencia suficiente en la base local para responder con confianza. "
                    "Si deseas, puedes activar la búsqueda web."
                ),
                "evidence_text": "",
                "hits": hits.to_dict(orient="records") if isinstance(hits, pd.DataFrame) else [],
                "allow_web": allow_web,
                "briefs": None,
            }
        
        # 3) Fallback web (si está habilitado)
        try:
            web_hits = web_search_duckduckgo(question, max_results=6)
            return {
                "mode": "web",
                "answer": "Resultados de la web encontrados. Integra con LLM si deseas.",
                "evidence_text": "\n\n".join(h["texto"] for h in web_hits),
                "hits": web_hits,
                "allow_web": allow_web,
                "briefs": None,
            }
        except Exception as e:
            return {
                "mode": "web_failure",
                "answer": f"Error en la búsqueda web: {e}",
                "evidence_text": "",
                "hits": [],
                "allow_web": allow_web,
                "briefs": None,
            }