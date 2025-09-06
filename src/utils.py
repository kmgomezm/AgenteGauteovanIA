import re
import pandas as pd

def clean_text(t: str) -> str:
    t = re.sub(r'\s+', ' ', str(t)).strip()
    return t

def to_date(x):
    return pd.to_datetime(x, errors='coerce')
