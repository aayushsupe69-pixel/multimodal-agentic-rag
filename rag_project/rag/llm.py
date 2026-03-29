from langchain_groq import ChatGroq
from config import GROQ_API_KEY

def get_llm():
    """Returns a LangChain ChatGroq instance."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found. Please check your .env file.")
    
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        api_key=GROQ_API_KEY
    )

def ask(prompt: str) -> str:
    """Legacy helper maintained for backward compatibility (raw prompt)."""
    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content
