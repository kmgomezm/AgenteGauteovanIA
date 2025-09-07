from pathlib import Path
from uuid import uuid4
import pandas as pd
from utils import clean_text, to_date


def load_and_chunk(xlsx_path, out_parquet="data/processed/chunks.parquet", 
                   chunk_size=450, overlap=80):
    df = pd.read_excel(xlsx_path)
    df = df.rename(columns={c: c.lower() for c in df.columns})

    rows = []
    for i, r in df.iterrows():
        doc_id = f"doc_{i:06d}"
        text = clean_text(r.get('texto',''))
        meta = {
            "doc_id": doc_id,
            "autor": r.get('autor',''),
            "fecha": to_date(r.get('fecha', None)),
            "diario": r.get('diario',''),
            "título": str(r.get('título','')),
            "vínculo": r.get('vínculo',''),
            "row_idx": i
        }
        words = str(text).split()
        if not words:
            continue
        start = 0
        while start < len(words):
            end = min(len(words), start + chunk_size)
            chunk = " ".join(words[start:end])
            rows.append({**meta, "chunk": chunk, "chunk_id": str(uuid4())})
            if end == len(words): break
            start = max(0, end - overlap)

    out = pd.DataFrame(rows)
    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_parquet, index=False)
    return out

if __name__ == "__main__":
    load_and_chunk(r"data\raw\Corpus_completo_revisado.xlsx")
