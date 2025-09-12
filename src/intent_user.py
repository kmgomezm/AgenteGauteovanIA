# src/intent.py
from typing import Tuple
import re


def parse_user_intent(question: str) -> Tuple[str, bool]:
    """
    Detecta si el usuario pide explícitamente usar la web.
    Devuelve (pregunta_limpia, allow_web)
    """
    if not question:
        return "", False

    q_norm = question.strip()
    low = q_norm.lower()

    # Detecta si la pregunta contiene ('web' o 'internet') y ('busca' o 'usa')

    allow = False
    cleaned = q_norm
    # Patrones para detectar intención de usar web/internet
    pattern = re.compile(r"(?i)(web|internet).*(busca|usa)|(busca|usa).*(web|internet)")
    if pattern.search(low):
        allow = True
        # Limpia las palabras clave detectadas
        cleaned = re.sub(r"(?i)\b(web|internet)\b", "", cleaned)
        cleaned = re.sub(r"(?i)\b(busca|usa)\b", "", cleaned)
        cleaned = " ".join(cleaned.split())
    return cleaned, allow
