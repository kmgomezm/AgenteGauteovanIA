import pickle, pandas as pd
from rank_bm25 import BM25Okapi
from pathlib import Path

def build_bm25(parquet="data/processed/chunks.parquet", out="data/indexes/bm25.pkl"):
    df = pd.read_parquet(parquet)
    corpus = [str(t).lower().split() for t in df["chunk"].fillna("")]
    bm25 = BM25Okapi(corpus)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": df["chunk_id"].tolist()}, f)

if __name__ == "__main__":
    build_bm25()
