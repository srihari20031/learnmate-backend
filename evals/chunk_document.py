"""
Stage 1 of building a real eval set: chunk a real document with the PRODUCTION
chunker and freeze the result as the eval corpus.

Usage (from repo root):
    python -m evals.chunk_document docs/rag-async-notes.md

WHY RUN THE REAL chunk_text()?
------------------------------
Our first dataset used hand-written, clean chunks. That measures the retriever
under ideal conditions it will never actually see. In production, documents are
split by chunk_text() into ~256-token windows with overlap, which cut across
sentences and headings. Evaluating on THOSE real chunks tells us how retrieval
behaves for real uploads — boundaries, noise, and all.

OUTPUT
------
Writes evals/corpus.json (a list of {"id": "c0", "text": ...}) and prints a
preview of every chunk with its id, so you can read them and decide which chunk
answers each question you'll write in stage 2.
"""

import json
import sys
from pathlib import Path

from app.services.chunk_service import chunk_text


def main():
    # The Windows console defaults to cp1252, which can't print arrows/emojis in
    # the chunk previews. Force UTF-8 so the preview never crashes on a symbol.
    sys.stdout.reconfigure(encoding="utf-8")

    # Default to the notes doc, but accept any path so you can re-run this on a
    # real study-material file later: `python -m evals.chunk_document my_notes.txt`
    doc_path = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/rag-async-notes.md")
    text = doc_path.read_text(encoding="utf-8")

    # The exact same call the app makes when you upload a document.
    chunks = chunk_text(text)
    corpus = [{"id": f"c{i}", "text": chunk} for i, chunk in enumerate(chunks)]

    out_path = Path("evals/corpus.json")
    out_path.write_text(json.dumps(corpus, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{len(corpus)} chunks from {doc_path}  ->  {out_path}\n")
    for item in corpus:
        # First ~110 chars, newlines flattened, so the list is skimmable.
        preview = " ".join(item["text"].split())[:110]
        print(f'{item["id"]:>4}: {preview}...')


if __name__ == "__main__":
    main()
