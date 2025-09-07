import faiss, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path

def build_faiss(parquet="data/processed/chunks.parquet", 
                out_index="data/indexes/faiss.index", 
                out_meta="data/indexes/faiss_meta.parquet"):
    df = pd.read_parquet(parquet)
    model = SentenceTransformer("intfloat/multilingual-e5-small")
    texts = ("query: " + df["chunk"].fillna("").astype(str)).tolist()
    X = model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=True).astype("float32")
    index = faiss.index_factory(X.shape[1], "HNSW32,Flat")
    index.train(X)
    index.add(X)
    Path(out_index).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, out_index)
    df[["chunk_id","doc_id","autor","fecha","diario","título", "vínculo",  "row_idx"]].to_parquet(out_meta, index=False)

if __name__ == "__main__":
    build_faiss()
