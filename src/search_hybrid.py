import faiss, numpy as np, pandas as pd, pickle
from sentence_transformers import SentenceTransformer

def rrf(ranks, k=60):
    return sum(1.0/(k+r) for r in ranks)

class HybridSearcher:
    def __init__(self,
                 faiss_index="data/indexes/faiss.index",
                 faiss_meta="data/indexes/faiss_meta.parquet",
                 bm25_path="data/indexes/bm25.pkl",
                 parquet_chunks="data/processed/chunks.parquet"):
        self.df_meta = pd.read_parquet(faiss_meta)
        self.faiss = faiss.read_index(faiss_index)
        self.model = SentenceTransformer("intfloat/multilingual-e5-small")
        with open(bm25_path, "rb") as f:
            obj = pickle.load(f)
        self.bm25, self.bm25_chunk_ids = obj["bm25"], obj["chunk_ids"]
        self.chunks = pd.read_parquet(parquet_chunks).set_index("chunk_id")

    def search(self, query, top_k_vec=20, top_k_kw=20, final_k=8):
        qv = self.model.encode(["query: " + query], normalize_embeddings=True).astype("float32")
        D, I = self.faiss.search(qv, top_k_vec)
        vec_hits = [self.df_meta.iloc[i]["chunk_id"] for i in I[0]]
        toks = query.lower().split()
        scores = self.bm25.get_scores(toks)
        kw_idx = np.argsort(scores)[::-1][:top_k_kw]
        kw_hits = [self.bm25_chunk_ids[i] for i in kw_idx]

        rank_map = {}
        for rank, cid in enumerate(vec_hits): rank_map.setdefault(cid, []).append(rank+1)
        for rank, cid in enumerate(kw_hits):  rank_map.setdefault(cid, []).append(rank+1)

        fused = sorted(((cid, rrf(ranks)) for cid, ranks in rank_map.items()),
                       key=lambda x: x[1], reverse=True)[:final_k]
        cids = [cid for cid,_ in fused]
        meta = self.df_meta.set_index("chunk_id").loc[cids].reset_index()
        meta["chunk"] = [self.chunks.loc[cid]["chunk"] for cid in cids]
        return meta
