from langchain_huggingface import HuggingFaceEndpointEmbeddings
from config import HUGGINGFACE_API_KEY as HF_KEY

def get_embedder():
    """Returns a LangChain HuggingFaceEndpointEmbeddings instance."""
    if not HF_KEY:
        raise ValueError("HUGGINGFACE_API_KEY not found. Please check your .env file.")
    
    return HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=HF_KEY
    )

def get_embedding(input_data) -> list:
    """Legacy helper maintained for compatibility (raw data to list)."""
    embedder = get_embedder()
    if isinstance(input_data, str):
        return embedder.embed_query(input_data)
    return embedder.embed_documents(input_data)
