"""
build_index.py
Builds FAISS + BM25 indexes from the normalized SHL catalog.
Run once after fetch_catalog.py.
"""
import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

CATALOG_PATH = Path(__file__).parent / "shl_catalog.json"
INDEX_DIR = Path(__file__).parent / "indexes"
FAISS_PATH = INDEX_DIR / "faiss.index"
BM25_PATH = INDEX_DIR / "bm25.pkl"
IDS_PATH = INDEX_DIR / "ids.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()


def build_indexes():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    catalog: list[dict] = json.loads(CATALOG_PATH.read_text())
    print(f"Building indexes for {len(catalog)} catalog entries...")

    texts = [item["search_text"] for item in catalog]
    ids = [item["entity_id"] for item in catalog]

    # ── BM25 ──────────────────────────────────────────────────────────────────
    print("  Building BM25 index...")
    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print(f"  BM25 saved -> {BM25_PATH}")

    # ── FAISS ─────────────────────────────────────────────────────────────────
    print(f"  Loading sentence-transformer: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print("  Encoding catalog texts (may take ~30s)...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product on normalized vecs = cosine sim
    index.add(embeddings)
    faiss.write_index(index, str(FAISS_PATH))
    print(f"  FAISS index saved -> {FAISS_PATH} ({index.ntotal} vectors, dim={dim})")

    # Save ordered entity IDs so retriever can map index → product
    IDS_PATH.write_text(json.dumps(ids))
    print(f"  IDs saved -> {IDS_PATH}")
    print("Done.")


if __name__ == "__main__":
    build_indexes()
