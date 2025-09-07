# src/rag_chain.py
import os
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from .search_hybrid import HybridSearcher

SYSTEM = """Eres un analista en español. Responde con precisión usando SOLO la evidencia dada.
Cita cada afirmación clave con [autor, periódico, fecha, título, doc_id] al final de la frase.
Si falta evidencia, dilo explícitamente."""
PROMPT = PromptTemplate.from_template(
    "{system}\n\nPregunta: {question}\n\nEvidencia:\n{evidence}\n\nRespuesta (español concisa, con citas):"
)

def format_evidence(df):
    rows = []
    for _, r in df.iterrows():
        meta = f"[{r.autor}, {r.periodico}, {str(r.fecha)[:10]}, {r.titulo}, {r.doc_id}]"
        rows.append(f"- {r.chunk}\n  {meta}")
    return "\n".join(rows)

class RAGPipeline:
    def __init__(self, model="mistral:7b-instruct", temperature=0.2):
        self.llm = Ollama(model=model, temperature=temperature)
        self.searcher = HybridSearcher()

    def answer(self, question: str):
        hits = self.searcher.search(question, final_k=8)
        evidence = format_evidence(hits)
        prompt = PROMPT.format(system=SYSTEM, question=question, evidence=evidence)
        return self.llm.invoke(prompt), hits
