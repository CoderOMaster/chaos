# chaos-search

**On-device semantic search.** Embeddings and vector search that run entirely on
your machine — no server to host, no API keys, no data leaving the device, works
offline. A native C++ core (MiniLM + exact/HNSW search) with a small Python API.

Point it at some text, get semantic search in a few lines — nothing to deploy.

## Install

```bash
pip install chaos-search
```

Prebuilt wheels bundle ONNX Runtime; the MiniLM model downloads and caches
automatically on first use — no paths, no setup.

## Quickstart

```python
from chaos import Client, Document

client = Client()                       # MiniLM auto-downloaded + cached
index = client.open("notes")            # kind="flat" (default) or "hnsw"

added, updated = index.add([
    Document(id="1", text="Ship the on-device SDK by Friday."),
    Document(id="2", text="Follow up with the LiveKit team about latency."),
])
print(f"{added} added, {updated} updated, {index.count} total")

for m in index.search("what's due this week", top_k=3):
    print(m.id, m.score, m.text)         # m: Match(id, score, text, metadata)
```

### Async

`AsyncClient` mirrors the API for event-loop apps (FastAPI, agents); each call
runs in a worker thread, so concurrent searches overlap.

```python
from chaos import AsyncClient, Document

client = AsyncClient()
index = await client.open("notes")
await index.add([Document(id="1", text="Ship the SDK by Friday.")])
matches = await index.search("deadlines", top_k=3)
```

## Why chaos

- **Local-first & private** — runs in-process on CPU; documents and queries never
  leave the machine. No cloud, no accounts, no per-query network calls.
- **Zero infrastructure** — no vector DB to run, no embedding server. The model
  auto-downloads and caches; everything else is one native library.
- **Fast, native, single-process pipeline** — tokenize → embed → search all in
  C++ (the GIL is released), so there's no Python-side embedding and no
  cross-language overhead per query. A full query runs in **~3.6 ms median** on a
  laptop-class arm64 CPU; the vector search itself is **under a millisecond**.
- **Embeddable** — usable from Python *or* directly from C++.

## Indexes

- **`kind="flat"`** (default) — exact search, 100% recall, zero tuning. Best up
  to ~50k documents.
- **`kind="hnsw"`** — approximate graph index, sub-millisecond search into the
  millions. Tune recall/latency with `index.ef_search = 128`.

`add` upserts by id (re-adding an id updates it). Opened with a name, an index
saves its documents **and** its vectors/HNSW graph, so reopening by that name
**loads instantly without re-embedding** (a corpus that took minutes to build
reopens in a fraction of a second). Without a name it stays in memory.
`Document.metadata` is returned with matches (not searched).

## Models

`Client()` uses `all-MiniLM-L6-v2` by default. Pick another with
`Client(model="<hf-repo>")`, or use local files with
`Client(model_path=..., vocab_path=...)`. Pre-fetch for offline/first-run:

```bash
chaos download            # or: python -m chaos download
```

**Supported:** ONNX, WordPiece-tokenized, mean-pooled BERT-family encoders
(`all-MiniLM-L6-v2`, E5, most BERT `sentence-transformers`). **Not** supported:
SentencePiece/BPE models (Gemma, Jina, GTE, T5), CLS-pooled models, non-ONNX
formats, decoder LLMs.

## Performance

Measured on a laptop-class arm64 CPU (4 threads, 10k documents, real MiniLM):

| stage | median | p99 |
|---|---|---|
| end-to-end (embed + search) | ~3.6 ms | ~5.3 ms |
| vector search only (HNSW) | 0.05 ms | 0.07 ms |

Embedding is ~95% of the time — the vector search is effectively free. Numbers
scale with your CPU and model; full methodology and the flat/HNSW crossover are
in the docs.

## Requirements

- Python 3.9–3.13
- Prebuilt wheels: **macOS arm64**, **Linux x86_64 / aarch64**. Other platforms
  build from source (needs a C++17 toolchain + ONNX Runtime).

## Documentation & source

Full docs, C++ API, examples, and benchmarks:
**[github.com/CoderOMaster/chaos](https://github.com/CoderOMaster/chaos)**

MIT licensed.
