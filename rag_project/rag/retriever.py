import numpy as np
import faiss
from rag.embedder import get_embedding

def retrieve(query: str, index: faiss.Index, docs: list, k=10) -> list:
    """
    Performs similarity search against the FAISS index.
    Improved with higher k and index validation.
    """
    if index is None or not docs:
        print("RETRIEVER: Index or docs empty. Cannot retrieve.")
        return []
        
    # Get query embedding
    try:
        embedding = get_embedding(query)
    except Exception as e:
        print(f"RETRIEVER ERROR: Query embedding failed: {e}")
        return []

    # FAISS search expects a 2D array [batch_size, dimension]
    vec = np.array([embedding], dtype="float32")
    faiss.normalize_L2(vec)
    
    # Search for top k matches
    # distances are squared L2 distances since we normalized (so 0 is perfect, 2 is opposite)
    # With IndexFlatIP + normalize_L2, it behaves like cosine similarity
    scores, ids = index.search(vec, k)
    
    # Map back to documents and add scores
    results = []
    for score, i in zip(scores[0], ids[0]):
        if i != -1 and i < len(docs):
            doc = docs[i].copy()
            doc["score"] = float(score)
            results.append(doc)
            
    print(f"RETRIEVER: Found {len(results)} matches for query.")
    return results
