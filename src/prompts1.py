# src/prompts.py
from langchain.prompts import PromptTemplate

# Prompt para uso con evidencia LOCAL (tu base de columnas)
SYSTEM_LOCAL = """Eres un analista en español. Responde con precisión usando SOLO la evidencia dada.
Cita cada afirmación clave con [autor, diario, fecha, título, doc_id] al final de la frase.
Si falta evidencia, dilo explícitamente y no inventes. No uses conocimiento externo."""

# Prompt para uso con evidencia de la WEB (fallback explícito del usuario)
SYSTEM_WEB = """Eres un analista en español. Responde con precisión usando SOLO la evidencia web dada.
Para cada afirmación, cita la URL entre corchetes al final de la frase, por ejemplo: [https://...].
Si falta evidencia, dilo explícitamente y no inventes."""

# Plantilla común
PROMPT = PromptTemplate.from_template(
    "{system}\n\nPregunta: {question}\n\nEvidencia:\n{evidence}\n\nRespuesta (español concisa, con citas):"
)
