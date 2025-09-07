# src/intent.py
from typing import Tuple

WEB_TOKENS = {
    "#web", "#buscarweb", "#usarweb", "buscar en la web", "busca en la web",
    "usa web", "usar web", "con web", "web:", "web ->"
}

def parse_user_intent(question: str) -> Tuple[str, bool]:
    """
    Detecta si el usuario pide explícitamente usar la web.
    Devuelve (pregunta_limpia, allow_web)
    """
    if not question:
        return "", False

    q_norm = question.strip()
    low = q_norm.lower()

    allow = any(tok in low for tok in WEB_TOKENS)

    # Limpia tokens comunes si están embebidos al inicio o final
    cleaned = q_norm
    for tok in WEB_TOKENS:
        cleaned = cleaned.replace(tok, "")
    cleaned = " ".join(cleaned.split())

    return cleaned, allow
