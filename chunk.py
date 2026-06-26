from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
   #splits the text into chunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    return text_splitter.split_text(text)