from rag import llm

def generate_answer(query: str, context_results: list) -> str:
    """Generates an answer using the LLM based on retrieved context."""
    if not context_results:
        return "I couldn't find any relevant information to answer your question."

    context_text = "\n\n".join([f"Source: {res['source']}\nContent: {res['text']}" for res in context_results])
    
    prompt = f"""
    You are a helpful assistant answering questions based ONly on the provided context.
    If the context doesn't contain the answer, say "I don't know based on the provided documents."
    
    Context:
    {context_text}
    
    Question: {query}
    
    Answer:
    """
    
    return llm.ask(prompt)
