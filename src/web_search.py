# src/web_search.py
from typing import List, Dict, Any
from duckduckgo_search import DDGS

def web_search_duckduckgo(
    query: str,
    max_results: int = 6,
    region: str = "co-es", #para Colombia o usar wt-wt para resultados globales
    safesearch: str = "off",
) -> List[Dict[str, Any]]:
    """
    Devuelve una lista de resultados DDG: [{'title','href','body',...}, ...]
    No requiere API key.
    """
    with DDGS() as ddgs:
        results = list(ddgs.text(query, region=region, safesearch=safesearch, max_results=max_results))
    return results
