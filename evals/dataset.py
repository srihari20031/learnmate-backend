"""
The labeled evaluation set — now built on a REAL document.

Two pieces, same as before, but the corpus is no longer hand-written:

  CORPUS    — loaded from evals/corpus.json, which is produced by running the
              production chunk_text() over a real document:
                  python -m evals.chunk_document docs/rag-async-notes.md
              So the eval runs over the exact chunks the app would store and
              retrieve in production (real token boundaries, real overlap).

  QUESTIONS — for each question, the id of the chunk that actually answers it.
              These are paraphrased so they share few words with the gold chunk,
              and a couple have a planted distractor (a chunk that looks more
              similar than the right one) so the reranker has something to fix.

TO REBUILD ON YOUR OWN DOCUMENT
-------------------------------
1. Run chunk_document.py on a real study-material file -> regenerates corpus.json.
2. Read the printed chunk previews, then rewrite QUESTIONS below to point at the
   chunk ids that answer each question. That hand-labeling IS the eval.
"""

import json
from pathlib import Path

# Load the frozen, real-chunk corpus. (Regenerate with evals/chunk_document.py.)
_CORPUS_PATH = Path(__file__).parent / "corpus.json"
CORPUS = json.loads(_CORPUS_PATH.read_text(encoding="utf-8"))

# Each question names the chunk id that correctly answers it. The trailing
# comment notes the trap (a distractor chunk) where there is one.
QUESTIONS = [
    {"question": "Why did we shrink the document chunk size down to 256 tokens?", "relevant_ids": ["c1"]},
    {"question": "What async driver does the app use to talk to MongoDB?", "relevant_ids": ["c1"]},
    {"question": "When does the system skip the retrieval step completely?", "relevant_ids": ["c2"]},
    {"question": "What actually performs the batching, if it isn't asyncio itself?", "relevant_ids": ["c3"]},
    # Distractor: c10 mentions Node's "4 threads" libuv pool.
    {"question": "How many worker threads are in the default pool that to_thread draws from?", "relevant_ids": ["c4"]},
    {"question": "What everyday analogy explains why offloading to a thread keeps the server responsive?", "relevant_ids": ["c6"]},
    {"question": "How does a cross-encoder score relevance differently from the plain embedding model?", "relevant_ids": ["c6"]},
    {"question": "What are the two knobs that control the rerank step, and their default values?", "relevant_ids": ["c7"]},
    {"question": "If I put await in front of a synchronous function, will it stop blocking the event loop?", "relevant_ids": ["c8"]},
    {"question": "Which two synchronous Python clients did we have to wrap in to_thread?", "relevant_ids": ["c9"]},
    # Distractor: c4 mentions the "~40" thread pool.
    {"question": "How many threads are in Node's built-in libuv pool, and which built-ins use it?", "relevant_ids": ["c10"]},
    {"question": "What still needs to be fixed about the Groq LLM call?", "relevant_ids": ["c12"]},
    # --- added: cover c0 / c5 / c11 (previously unlabeled) + more distractors ---
    {"question": "Which vector database backs the pipeline, and what distance metric does it use?", "relevant_ids": ["c0"]},
    # Distractor: c1 also cites the 256-token limit, but only c0 gives the 384 dimensions.
    {"question": "How many dimensions does the embedding model output, and what is its max token limit?", "relevant_ids": ["c0"]},
    {"question": "In the two-user timeline, how long does the second user wait WITHOUT to_thread, and why?", "relevant_ids": ["c5"]},
    # Distractor: c4 sets up the 200ms timeline and c6 talks threads, but the "~200ms, both done" payoff is c5.
    {"question": "With to_thread, roughly how long until two concurrent 200ms encodes both finish?", "relevant_ids": ["c5"]},
    # Distractor: c8 also contrasts I/O vs CPU offloading.
    {"question": "What single distinction decides whether async work needs its own thread in any runtime?", "relevant_ids": ["c11"]},
    # Distractor: c10 has the Node mapping table; the 'three free / one needs a thread' pattern line is c11.
    {"question": "How many calls are 'free' in Node but need care in Python, and which one call needs a thread in both?", "relevant_ids": ["c11"]},
    # Distractor: c9 also names worker_threads for CPU work; the direct equivalence statement is in c10.
    {"question": "asyncio.to_thread is described as the rough equivalent of which Node.js primitive?", "relevant_ids": ["c10"]},
    {"question": "Besides reordering the chunks, what does the rerank debug logging print for each one?", "relevant_ids": ["c7"]},
]
