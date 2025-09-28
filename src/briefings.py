# src/briefings.py
import json
from .prompts import SYSTEM_DEEP_REASON, PROMPT_DEEP_REASON

def structured_briefs(rag_instance, question: str, evidence: str) -> dict:
    """
    Usa el LLM para organizar la evidencia en resúmenes estructurados por facetas.
    Devuelve un dict con claves: 'por_año', 'por_medio', 'por_autor'.
    
    Args:
        rag_instance: Instancia de RAGHybridPipeline con acceso al LLM
        question: Pregunta original del usuario
        evidence: Evidencia formateada para análisis
        
    Returns:
        Dict con resúmenes estructurados o información de error
    """
    prompt = PROMPT_DEEP_REASON.format(
        system=SYSTEM_DEEP_REASON, 
        question=question, 
        evidence=evidence
    )
    
    try:
        result = rag_instance.llm.invoke(prompt)
        
        # Intentar parsear como JSON
        try:
            parsed_result = json.loads(result)
            
            # Validar estructura esperada
            expected_keys = ['por_año', 'por_medio', 'por_autor']
            if all(key in parsed_result for key in expected_keys):
                return parsed_result
            else:
                # Si no tiene la estructura esperada, crear una estructura mínima
                return {
                    "por_año": [f"Análisis temporal disponible: {str(parsed_result)[:200]}..."],
                    "por_medio": [f"Análisis por medio disponible: {str(parsed_result)[:200]}..."],
                    "por_autor": [f"Análisis por autor disponible: {str(parsed_result)[:200]}..."],
                    "analisis_crudo": result
                }
                
        except json.JSONDecodeError:
            # Si no es JSON válido, procesar como texto libre
            return {
                "por_año": [f"Análisis temporal: {result[:300]}..."],
                "por_medio": [f"Análisis por medio: {result[300:600]}..."],
                "por_autor": [f"Análisis por autor: {result[600:900]}..."],
                "analisis_crudo": result
            }
            
    except Exception as e:
        print(f"Error en structured_briefs: {e}")
        return {
            "por_año": [f"Error procesando análisis temporal: {str(e)}"],
            "por_medio": [f"Error procesando análisis por medio: {str(e)}"], 
            "por_autor": [f"Error procesando análisis por autor: {str(e)}"],
            "analisis_crudo": f"Error: {str(e)}"
        }