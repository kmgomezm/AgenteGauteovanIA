from src.ingest import load_and_chunk
from src.index_faiss import build_faiss
from src.index_bm25 import build_bm25

if __name__ == "__main__":
    print("[1/3] Ingesta y particionado…")
    load_and_chunk("data/raw/opiniones.xlsx")
    print("[2/3] Índice FAISS…")
    build_faiss()
    print("[3/3] Índice BM25…")
    build_bm25()
    print("Listo ✅")
