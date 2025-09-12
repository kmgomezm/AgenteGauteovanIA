# src/formatting.py
from typing import List, Dict, Any
import pandas as pd

def _truncate(text: str, max_chars: int = 800) -> str:
    """Recorta texto para evitar prompts demasiado largos."""
    if not isinstance(text, str):
        return ""
    if len(text) <= max_chars:
        return text
    # cortar en el último espacio para no romper palabras
    return text[:max_chars].rsplit(" ", 1)[0] + "…"

def format_evidence_local(
    df: pd.DataFrame,
    enumerate_chunks: bool = True,
    max_chars_per_chunk: int = 800
) -> str:
    """
    Formatea resultados locales en bloques con metadatos.
    Se esperan columnas: ['doc_id','autor','fecha','diario','título','vínculo','row_idx','chunk','chunk_id']
    """
    if df is None or len(df) == 0:
        return ""

    rows = []
    for i, (_, r) in enumerate(df.iterrows(), 1):
        # Acceso robusto a columnas (con o sin tildes)
        autor  = r.get("autor", "?")
        diario = r.get("diario", r.get("periodico", "?"))
        fecha  = str(r.get("fecha", "?"))[:10]
        titulo = r.get("título", r.get("titulo", "?"))
        doc_id = r.get("doc_id", "?")
        chunk  = _truncate(r.get("chunk", ""), max_chars_per_chunk)

        meta = f"[{autor}, {diario}, {fecha}, {titulo}, {doc_id}]"
        prefix = f"[{i}] " if enumerate_chunks else ""
        rows.append(f"{prefix}{chunk}\n  {meta}")

    return "\n\n".join(rows)

def format_evidence_web(
    web_results: List[Dict[str, Any]],
    enumerate_items: bool = True,
    max_chars_per_snippet: int = 600
) -> str:
    """
    Formatea resultados de DuckDuckGo (lista de dicts con 'title', 'href', 'body'/'snippet').
    Cita por URL.
    """
    if not web_results:
        return ""

    rows = []
    for i, r in enumerate(web_results, 1):
        title = r.get("title") or r.get("source") or "Resultado"
        url   = r.get("href") or r.get("url") or ""
        body  = r.get("body") or r.get("snippet") or ""
        body  = _truncate(body, max_chars_per_snippet)

        prefix = f"[{i}] " if enumerate_items else ""
        meta = f"[{url}]" if url else "[sin URL]"
        rows.append(f"{prefix}{title}\n{body}\n  {meta}")

    return "\n\n".join(rows)
