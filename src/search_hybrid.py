# src/search_hybrid.py
import faiss
import numpy as np
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

def rrf(ranks, k=60):
    """Reciprocal Rank Fusion"""
    return sum(1.0/(k+r) for r in ranks)

class HybridSearcher:
    def __init__(self,
                 faiss_index="data/indexes/faiss.index",
                 faiss_meta="data/indexes/faiss_meta.parquet",
                 bm25_path="data/indexes/bm25.pkl",
                 parquet_chunks="data/processed/chunks.parquet"):
        self.df_meta = pd.read_parquet(faiss_meta)
        self.faiss = faiss.read_index(faiss_index)
        self.model = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
        
        with open(bm25_path, "rb") as f:
            obj = pickle.load(f)
        self.bm25, self.bm25_chunk_ids = obj["bm25"], obj["chunk_ids"]
        self.chunks = pd.read_parquet(parquet_chunks).set_index("chunk_id")

    def search(self, query, top_k_vec=20, top_k_kw=20, final_k=8):
        """
        Búsqueda híbrida usando vectores + BM25 con fusión por ranking recíproco
        """
        # 1. Búsqueda vectorial
        qv = self.model.encode(["query: " + query], normalize_embeddings=True).astype("float32")
        D, I = self.faiss.search(qv, top_k_vec)
        vec_hits = [self.df_meta.iloc[i]["chunk_id"] for i in I[0]]
        
        # 2. Búsqueda por palabras clave (BM25)
        toks = query.lower().split()
        scores = self.bm25.get_scores(toks)
        kw_idx = np.argsort(scores)[::-1][:top_k_kw]
        kw_hits = [self.bm25_chunk_ids[i] for i in kw_idx]

        # 3. Fusión de rankings
        rank_map = {}
        for rank, cid in enumerate(vec_hits): 
            rank_map.setdefault(cid, []).append(rank+1)
        for rank, cid in enumerate(kw_hits):  
            rank_map.setdefault(cid, []).append(rank+1)

        # 4. Calcular scores RRF y ordenar
        fused = sorted(
            ((cid, rrf(ranks)) for cid, ranks in rank_map.items()),
            key=lambda x: x[1], reverse=True
        )[:final_k]
        
        cids = [cid for cid, _ in fused]
        
        # 5. Filtrar solo los IDs que existen en ambos índices
        valid_cids = []
        for cid in cids:
            if cid in self.df_meta.set_index("chunk_id").index and cid in self.chunks.index:
                valid_cids.append(cid)
            else:
                print(f"Warning: chunk_id {cid} no encontrado en índices")
        
        if not valid_cids:
            print("No se encontraron chunks válidos")
            return pd.DataFrame()
        
        # 6. Construir resultado final
        try:
            meta = self.df_meta.set_index("chunk_id").loc[valid_cids].reset_index()
            meta["chunk"] = [self.chunks.loc[cid]["chunk"] for cid in valid_cids]
            # Mantener el orden de relevancia
            meta["rrf_score"] = [dict(fused)[cid] for cid in valid_cids]
            
            return meta
            
        except Exception as e:
            print(f"Error al construir resultado: {e}")
            return pd.DataFrame()