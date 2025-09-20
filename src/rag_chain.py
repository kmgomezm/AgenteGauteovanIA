# src/rag_chain.py
from typing import Optional, Dict, Any
import pandas as pd
from langchain_ollama import OllamaLLM as Ollama
from langchain.prompts import PromptTemplate
from .search_hybrid import HybridSearcher
from .prompts import SYSTEM_LOCAL, PROMPT, format_evidence


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
    Pipeline RAG híbrido con fallback web (versión más avanzada).
    Flujo: RAG local -> (opcional) Fallback Web si el usuario lo solicita.
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
    ) -> Dict[str, Any]:
        """
        Responde usando evidencia local, con fallback web opcional.
        
        Args:
            question: La pregunta a responder
            k_local: Número de documentos locales a recuperar
            min_local_chars: Mínimo de caracteres para considerar evidencia suficiente
            allow_web: Si permitir búsqueda web como fallback
            
        Returns:
            Dict con respuesta, modo usado, evidencia, etc.
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
                
                return {
                    "mode": "local",
                    "answer": answer,
                    "evidence_text": evidence,
                    "hits": hits.to_dict(orient="records") if isinstance(hits, pd.DataFrame) else [],
                    "allow_web": allow_web,
                }
        
        # 2) Si evidencia local insuficiente
        if not allow_web:
            return {
                "mode": "local_insufficient",
                "answer": (
                    "No encontré evidencia suficiente en la base local para responder con confianza. "
                    "Si deseas, puedes activar la búsqueda web con allow_web=True."
                ),
                "evidence_text": "",
                "hits": hits.to_dict(orient="records") if isinstance(hits, pd.DataFrame) else [],
                "allow_web": allow_web,
            }
        
        # 3) Fallback web (si está habilitado)
        # Nota: Aquí irían las funciones de búsqueda web que no están implementadas aún
        return {
            "mode": "web_not_implemented",
            "answer": "La búsqueda web no está implementada aún. Usa solo la evidencia local disponible.",
            "evidence_text": "",
            "hits": hits.to_dict(orient="records") if isinstance(hits, pd.DataFrame) else [],
            "allow_web": allow_web,
        }

