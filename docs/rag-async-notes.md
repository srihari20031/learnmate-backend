# RAG, Async & Concurrency — Engineering Notes

Reference notes for the LearnMate RAG pipeline: what we changed, and the
reasoning behind `asyncio`, batching, threads, reranking, and how the same
ideas map onto Node.js.

---

## 1. The RAG pipeline (current state)

```
Upload → extract text → chunk → embed → store (Qdrant + Mongo)        [indexing]
Message → embed query → vector search (top 20) → rerank (top 5) → LLM  [retrieval]
```

Components:

- **Embedding model**: `all-MiniLM-L6-v2` (bi-encoder, 384-dim, **max 256 tokens**).
- **Vector DB**: Qdrant (cosine similarity).
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **LLM**: Groq `llama-3.3-70b-versatile`.
- **Metadata / chunks / history**: MongoDB (via `motor`, async driver).

### Changes we made (in order)

1. **Chunk size 1000 → 256 tokens.** The embedding model silently truncates
   anything past 256 tokens, so 1000-token chunks were embedded using only
   their first ~25%. This was the single biggest retrieval-quality fix.
2. **Fixed `uuid` crash** in `chat.py` (`uuid.uuid4()` → `uuid4()` to match
   `from uuid import uuid4`). Only triggered on attachment uploads.
3. **Event-loop offload + batching** in indexing and retrieval (see §2–§4).
4. **Reranker** added: wide vector recall, then cross-encoder precision (§5).
5. **Gated retrieval**: skip the whole retrieval stack when the session has no
   indexed documents (cheap Mongo existence check).

---

## 2. Batching — what and why

**Batching = process many items in ONE model call.**

```python
# Slow: one model call per chunk
for chunk in chunks:
    emb = model.encode(chunk)          # 40 chunks → 40 calls

# Fast: one model call for all chunks
embeddings = model.encode(chunks)      # 40 chunks → 1 call (vectorized)
```

The neural net runs once over the whole list using vectorized matrix math, far
faster than a Python loop of single calls.

- Lives in **`index_document`** (indexing), where we have *many* chunks at once.
- **Not** in `get_claude_response` (retrieval), because a query is a *single*
  message — there is nothing to batch.

> Batching is a property of the **model call** (`encode([...])`).
> It has nothing to do with asyncio. asyncio never batches anything.

---

## 3. The async event loop & why we offload

FastAPI runs **every request on ONE single event-loop thread**, interleaving
them. The loop's superpower: when it `await`s genuinely async work, it parks
that request and serves others in the meantime.

The danger: a **blocking** call (synchronous CPU or synchronous I/O) run
directly on the loop freezes *every* request until it finishes.

```python
# Bad: blocks the one event-loop thread for the whole encode
query_embedding = generate_embeddings(user_message)

# Good: runs encode on a worker thread, loop stays free
query_embedding = await asyncio.to_thread(generate_embeddings, user_message)
```

`asyncio.to_thread(fn, *args)` takes **one** call, runs it on **one** worker
thread (from a pool of ~40), and returns an awaitable. It does **not** batch or
fan out — one call → one thread → one result.

---

## 4. "What's the use of threads if we don't parallelize one request?"

Within a single request, the three offloaded calls (embed → search → rerank)
run **sequentially** — each depends on the previous, so there's no parallelism
*inside* one request. The benefit shows up **across concurrent requests.**

### Timeline: 2 users send a message at once (encode = 200ms)

**Without `to_thread`** (encode on the event loop):

```
Event loop:  [A embed 200ms ████████][B embed 200ms ████████]
             loop BLOCKED during A → B can't even start → B waits 400ms
```

**With `to_thread`** (encode on worker threads):

```
Event loop:  hands A→thread1, instantly free, hands B→thread2
thread 1:    [A embed 200ms ████████]
thread 2:    [B embed 200ms ████████]   ← in parallel → both done ~200ms
```

So we **do** use multiple threads — not for one request's steps, but across
many concurrent requests. 10 users uploading → up to 10 worker threads running
10 embeddings at once, while the single event loop stays responsive.

**Analogy:** a cashier who personally bags each order stops the whole line
(blocking); a cashier who hands bagging to a bagger and starts the next
customer keeps the line moving (`to_thread`). Several baggers → several orders
finish at once.

---

## 5. Reranking — bi-encoder vs cross-encoder

| | Bi-encoder (embeddings) | Cross-encoder (reranker) |
|---|---|---|
| Input | query and chunk scored **separately** | `(query, chunk)` scored **together** |
| Speed | fast (vectors precomputed) | slow (one model run per pair) |
| Accuracy | approximate | high |
| Role | **recall** — grab top 20 | **precision** — reorder, keep top 5 |

We use the cheap bi-encoder to cast a wide net, then the expensive
cross-encoder to reorder only those 20 candidates. Knobs: `CANDIDATE_POOL`
(20), `FINAL_TOP_K` (5). Debug logging shows each surviving chunk's
before→after rank (e.g. a chunk vector-ranked #7 jumping to #0).

---

## 6. The core rule: `await` does NOT mean non-blocking

`await` behaves **identically** in Python and Node. It only means "pause this
function and let the loop work until what I'm waiting on is ready." Whether the
loop is actually freed depends on **what you await**:

- **Awaiting real async I/O** (handed to the OS) → loop freed. ✅
- **"Awaiting" a synchronous call** → it blocks *first*, then `await` wraps an
  already-finished value. Useless. ❌

Non-blocking comes from the work happening **elsewhere**:
- **I/O** → the OS kernel handles it off-thread.
- **CPU** → only a separate **thread** can free the loop. No `await` can,
  because the CPU genuinely has to do the work.

---

## 7. Node.js comparison

Node has the **same model**: a single-threaded event loop (libuv). Same
blocking danger. Two differences from Python:

### Difference 1 — I/O is non-blocking *by default* in Node

Node's I/O libraries (`fetch`, `fs.promises`, DB drivers) are **all** genuinely
async. `await fetch(...)` frees the loop automatically — no thread needed.

Python is a **mix**: `motor` (Mongo) is async, but `qdrant_client` and the Groq
client are **synchronous**. That's why we wrapped *those* in `to_thread`. Using
`AsyncQdrantClient` / an async HTTP client would remove the thread — just like
Node.

### Difference 2 — CPU work: NO difference, both need a thread

For CPU-bound work like `encode`, there is **no I/O to hand off**, so `await`
alone can't help in *either* language. Both need an explicit worker thread:

| | Blocks the loop | Frees the loop |
|---|---|---|
| **Node** | `await encode()` | `await worker.run(encode)` (worker_threads) |
| **Python** | `encode()` | `await asyncio.to_thread(encode)` |

`asyncio.to_thread` ≈ Node's `worker_threads`. Node also has a small libuv
threadpool (4 threads) used automatically by a few built-ins (`fs`, `crypto`,
`zlib`, DNS) — not general-purpose.

### Mapping our pipeline to Node

| Operation | Python (our code) | Node.js |
|---|---|---|
| Mongo query | `motor` async → fine | `await` — free |
| Qdrant search | sync client → needed `to_thread` | `await` — free |
| Groq LLM call | sync client → blocks (TODO) | `await fetch` — free |
| **Embedding `encode`** | CPU → needed `to_thread` | CPU → needs `worker_threads` |

The pattern: the three I/O calls are "free" in Node but need care in Python;
the one true CPU call (`encode`) needs a thread in **both**. I/O vs CPU is the
real dividing line in every async runtime.

---

## 8. Quick reference

- **Batching** = many items in one model call. Indexing only. Not asyncio.
- **`to_thread` / `worker_threads`** = move one blocking call off the loop.
  Never batches. Pays off under concurrent load.
- **`await`** = wait, *not* "make non-blocking". Freeing the loop comes from the
  work running elsewhere (OS for I/O, a thread for CPU).
- **I/O** = free in Node, depends on the library in Python.
- **CPU** = needs a thread in both.
- **Bi-encoder** = recall (cheap, wide). **Cross-encoder** = precision
  (expensive, narrow). Use both.

## 9. Retrieval eval (baseline)

Harness lives in `evals/` (`python -m evals.run_eval`). It chunks a real
document with the production `chunk_text()`, reproduces Qdrant's cosine search
in numpy, and runs the REAL embedding model + `rerank()`. It compares retrieval
with and without the reranker on hand-labeled questions.

Baseline on `docs/rag-async-notes.md` (13 chunks, 12 questions):

| metric | vector only | vector+rerank |
|--------|-------------|---------------|
| hit@1  | 0.750       | **1.000**     |
| hit@3  | 0.833       | **1.000**     |
| hit@5  | 1.000       | 1.000 (saturated — small corpus) |
| mrr    | 0.829       | **1.000**     |

Read: the reranker lifted the correct chunk to #1 on every question (one rescue
went from rank #5 to #1) with no regressions. hit@5 is saturated because top-5
covers ~38% of a 13-chunk corpus; hit@1 and MRR are the meaningful metrics here.
Numbers are directional (small, self-authored set) — grow the dataset with real
study material for production-grade confidence.

### Known TODOs

- Groq LLM call (`client.chat.completions.create`) is still synchronous and
  blocks the event loop — wrap in `to_thread` or use an async client.
- Re-index documents ingested before the 256-token chunk fix.
- Grow the eval set (real study material, 25-30 questions) for trustworthy numbers.
- Next measured feature: hybrid search (BM25 + vector) vs this baseline.

### Done

- ~~Move `ensure_collection()` to app startup~~ — done (runs once at startup).
- ~~Build a retrieval eval harness~~ — done (`evals/`, see §9).
- ~~Chunker stored lossy `tokenizer.decode()` text~~ — fixed: `chunk_text()` now
  slices the original string via offset mapping, preserving case/code/symbols.
