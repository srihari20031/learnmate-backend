from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# all-MiniLM-L6-v2 has a max sequence length of 256 tokens; anything longer is
# silently truncated at encode time, so keep chunks within that budget.
def chunk_text(text: str, chunk_size: int = 256, overlap: int = 50) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    
    # Tokenize WITH offset mapping: every token carries the (start, end) character
    # positions it came from in the ORIGINAL text. We still size and overlap chunks
    # by token count (to respect the model's 256-token window), but we slice the
    # original string by those offsets instead of calling tokenizer.decode().
    #
    # Why: decode() round-trips through an uncased WordPiece vocabulary, which
    # lowercases text, inserts spaces around punctuation ("uuid4()" -> "uuid4 ( )"),
    # and turns unknown characters into "[UNK]". Slicing the original preserves the
    # real case, punctuation, code, and symbols that later get shown to the LLM.
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]

    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(offsets), step):
        window = offsets[i:i + chunk_size]
        if len(window) < 50:
            continue
        start_char = window[0][0]      # first char of the first token in the window
        end_char = window[-1][1]       # last char of the last token in the window
        chunks.append(text[start_char:end_char])

    return chunks