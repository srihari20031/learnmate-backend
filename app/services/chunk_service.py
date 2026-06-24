from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    step = chunk_size - overlap
    
    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i:i + chunk_size]
        if len(chunk_tokens) < 50:
            continue
        chunks.append(tokenizer.decode(chunk_tokens))
    
    return chunks