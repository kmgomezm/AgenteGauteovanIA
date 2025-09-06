SYSTEM = """Eres un analista en español. Responde con precisión usando SOLO la evidencia dada.
Cita cada afirmación clave con [autor, periódico, fecha, título, doc_id] al final de la frase.
Si falta evidencia, dilo explícitamente."""

USER_TEMPLATE = """{system}

Pregunta: {question}

Evidencia:
{evidence}

Respuesta (español concisa, con citas):"""

def format_evidence(df):
    rows = []
    for _, r in df.iterrows():
        meta = f"[{r.autor}, {r.periodico}, {str(r.fecha)[:10]}, {r.titulo}, {r.doc_id}]"
        rows.append(f"- {r.chunk}\n  {meta}")
    return "\n".join(rows)
