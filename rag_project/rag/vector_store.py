import os, faiss, json
import numpy as np

STORE_DIR = "vector_store"
DOCS_FILE = os.path.join(STORE_DIR, "docs.json")
INDEX_FILE = os.path.join(STORE_DIR, "index.bin")

def load() -> tuple:
    """Loads FAISS index and document metadata from disk."""
    docs = []
    index = None
    if os.path.exists(DOCS_FILE):
        with open(DOCS_FILE, "r") as f:
            docs = json.load(f)
    if os.path.exists(INDEX_FILE) and docs:
        index = faiss.read_index(INDEX_FILE)
    return docs, index

def save(docs: list, index: faiss.Index):
    """Saves FAISS index and document metadata to disk."""
    os.makedirs(STORE_DIR, exist_ok=True)
    with open(DOCS_FILE, "w") as f:
        json.dump(docs, f)
    if index:
        faiss.write_index(index, INDEX_FILE)

def create_index(dimension: int) -> faiss.Index:
    """Creates a new FAISS FlatIP (inner product) index."""
    print(f"VECTOR_STORE: Creating new IndexFlatIP with dimension {dimension}")
    return faiss.IndexFlatIP(dimension)

def validate_dimensions(index: faiss.Index, vectors: np.ndarray):
    """Ensures vectors match the index dimension."""
    if index is None: return True
    if vectors.shape[1] != index.d:
        raise ValueError(f"Dimension mismatch: Index expects {index.d}, got {vectors.shape[1]}")
    return True

def clear():
    """Deletes stored FAISS index and document metadata from disk."""
    for f in [DOCS_FILE, INDEX_FILE]:
        if os.path.exists(f):
            os.remove(f)
            print(f"RESET: Deleted {f}")
    # Also clean old legacy files
    for legacy in ["index.faiss", "index.pkl"]:
        p = os.path.join(STORE_DIR, legacy)
        if os.path.exists(p):
            os.remove(p)
    print("RESET: Vector store cleared ✅")
