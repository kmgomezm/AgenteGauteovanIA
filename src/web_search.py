# src/web_search.py
from typing import List, Dict, Any
from duckduckgo_search import DDGS
import hashlib
import json
import os
import time

CACHE_FILE = "data/web_cache.json"
CACHE_TTL = 60 * 60 * 24  # 1 día en segundos


def _load_cache() -> Dict[str, Any]:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache: Dict[str, Any]):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def web_search_duckduckgo(
    query: str,
    max_results: int = 6,
    region: str = "co-es",
    safesearch: str = "off",
    timeout: int = 60,
) -> List[Dict[str, Any]]:
    """
    Devuelve resultados de búsqueda web usando DuckDuckGo.
    Formato: [{'titulo','autor','diario','fecha','doc_id','texto','url','snippet','rrf_score'}, ...]
    """
    cache = _load_cache()
    key = hashlib.sha256(query.encode("utf-8")).hexdigest()

    # 1. Revisar cache
    if key in cache:
        entry = cache[key]
        if time.time() - entry["timestamp"] < CACHE_TTL:
            return entry["results"]

    try:
        with DDGS() as ddgs:
            results = list(
                ddgs.text(
                    query,
                    region=region,
                    safesearch=safesearch,
                    max_results=max_results,
                )
            )
    except Exception as e:
        print(f"[web_search] Error en búsqueda: {e}")
        return [{"mode": "web_failure", "answer": "Error en la búsqueda web."}]

    # 2. Normalizar resultados
    normalized = []
    for i, r in enumerate(results):
        normalized.append(
            {
                "titulo": r.get("title", ""),
                "autor": None,  # No disponible en DDG
                "diario": None,  # No disponible en DDG
                "fecha": None,  # No disponible en DDG
                "doc_id": f"web_{i}",
                "texto": r.get("body", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
                "rrf_score": 1.0 / (i + 1),  # score simple por ranking
            }
        )

    # 3. Guardar en cache
    cache[key] = {"timestamp": time.time(), "results": normalized}
    _save_cache(cache)

    return normalized
