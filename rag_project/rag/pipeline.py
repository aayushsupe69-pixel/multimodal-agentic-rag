import os, time
import numpy as np
import faiss
from rag import vector_store, embedder, chunker, retriever, generator
from rag.loader import load_pdf, caption_image

# Shared state
docs_store, faiss_index = vector_store.load()

def add_document(content: bytes, filename: str, file_type: str):
    """
    Robust orchestration:
    Extract → Validate → Clean → Chunk → Embed → Validate → Index
    """
    global docs_store, faiss_index
    start_total = time.time()
    
    safe_filename = os.path.basename(filename)
    caption = None

    # 1. Extraction & Cleaning
    if file_type == "pdf":
        t0 = time.time()
        text = load_pdf(content) # Already cleaned inside load_pdf
        print(f"PIPELINE: PDF Extraction took {time.time() - t0:.2f}s")
        
        if not text or len(text) < 20:
            raise ValueError("PDF contains no readable text or is too small.")
        source = safe_filename

    elif file_type == "image":
        t0 = time.time()
        caption = caption_image(content)
        print(f"PIPELINE: Image Captioning took {time.time() - t0:.2f}s")
        text = f"Image description: {caption}"
        source = f"image:{safe_filename}"
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    # 2. Chunking
    t0 = time.time()
    chunks = chunker.split_text(text)
    if not chunks:
        raise ValueError("Chunking failed: No valid text segments found.")
    print(f"PIPELINE: Created {len(chunks)} chunks in {time.time() - t0:.4f}s")
    
    # 3. Embedding
    t0 = time.time()
    try:
        vecs = embedder.get_embedding(chunks)
    except Exception as e:
        raise RuntimeError(f"Embedding failed: {str(e)}")
    print(f"PIPELINE: Batch Embedding took {time.time() - t0:.2f}s")
    
    # 4. Validation & Indexing
    arr = np.array(vecs, dtype="float32")
    faiss.normalize_L2(arr)
    
    if faiss_index is None:
        faiss_index = vector_store.create_index(arr.shape[1])
    
    # Validate dimensions
    vector_store.validate_dimensions(faiss_index, arr)
    
    # Add to index
    faiss_index.add(arr)
    print(f"PIPELINE: FAISS indexing successful. New index size: {faiss_index.ntotal}")
    
    # 5. Metadata & Persistence
    for c in chunks:
        docs_store.append({"text": c, "source": source})
        
    vector_store.save(docs_store, faiss_index)
    print(f"PIPELINE: Total processing time: {time.time() - start_total:.2f}s ✅")
    
    return {"chunks": len(chunks), "caption": caption}

def handle_query(query: str):
    """Executes the full RAG pipeline for a user query."""
    if not faiss_index or not docs_store:
        return {
            "answer": "I don't have any documents indexed yet. Please upload a PDF or image first!",
            "context": None,
            "source": "none"
        }

    print(f"PIPELINE: Retrieving context for query...")
    results = retriever.retrieve(query, faiss_index, docs_store, k=10)
    print(f"PIPELINE: Retrieved {len(results)} context chunks. Calling LLM...")
    answer = generator.generate_answer(query, results)
    print(f"PIPELINE: LLM Answer generated. Length: {len(answer)}")
    
    source = results[0]["source"] if results else "none"
    context_preview = "\n\n".join(d["text"] for d in results)[:300] if results else None
    
    return {
        "answer": answer,
        "context": context_preview,
        "source": source
    }

def reset_all():
    """Clears all in-memory state, persisted vectors, and uploaded files."""
    global docs_store, faiss_index
    docs_store = []
    faiss_index = None
    vector_store.clear()
    
    import shutil
    for folder in ["data/pdfs", "data/images"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            os.makedirs(folder, exist_ok=True)
    
    print("RESET: Full system reset complete ✅")
    return {"status": "reset successful"}
