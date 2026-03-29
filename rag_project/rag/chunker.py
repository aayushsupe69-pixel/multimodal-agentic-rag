from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_splitter(chunk_size=600, chunk_overlap=100):
    """Returns a LangChain RecursiveCharacterTextSplitter instance."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
        add_start_index=True
    )

def split_text(text: str, chunk_size=600, chunk_overlap=100) -> list:
    """Splits text using LangChain's RecursiveCharacterTextSplitter."""
    if not text:
        return []
    
    splitter = get_splitter(chunk_size, chunk_overlap)
    content = splitter.split_text(text)
    print(f"CHUNKER: Created {len(content)} recursive chunks.")
    return content
