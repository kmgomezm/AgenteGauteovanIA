from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from .search_hybrid import HybridSearcher
from .prompts import SYSTEM, USER_TEMPLATE, format_evidence

PROMPT = PromptTemplate.from_template(USER_TEMPLATE)

class RAGPipeline:
    def __init__(self, model="mistral:7b-instruct", temperature=0.2):
        self.llm = Ollama(model=model, temperature=temperature)
        self.searcher = HybridSearcher()

    def answer(self, question: str):
        hits = self.searcher.search(question, final_k=8)
        evidence = format_evidence(hits)
        prompt = PROMPT.format(system=SYSTEM, question=question, evidence=evidence)
        return self.llm.invoke(prompt), hits
