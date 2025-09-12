import re
import pandas as pd
from typing import Any, Dict, List, Optional

def clean_text(t: str) -> str:
    t = re.sub(r'\s+', ' ', str(t)).strip()
    return t

def to_date(x):
    return pd.to_datetime(x, errors='coerce')

def _evidence_sufficient(df: Optional[pd.DataFrame], min_total_chars: int = 600) -> bool:
    """Heurística para decidir si hay suficiente evidencia local."""
    if df is None or len(df) == 0:
        return False
    total = int(sum(len(str(x)) for x in df.get("chunk", [])))
    return total >= min_total_chars
