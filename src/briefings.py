from .prompts import SYSTEM_DEEP_REASON, PROMPT_DEEP_REASON

def structured_briefs(self, question: str, evidence: str) -> dict:
        """
        Usa el LLM para organizar la evidencia en resúmenes estructurados por facetas.
        Devuelve un dict con claves: 'por_año', 'por_medio', 'por_autor'.
        """
        prompt = PROMPT_DEEP_REASON.format(system=SYSTEM_DEEP_REASON, question=question, evidence=evidence)
        result = self.llm.invoke(prompt)
        try:
            import json
            return json.loads(result)
        except Exception:
            return {"raw_output": result}