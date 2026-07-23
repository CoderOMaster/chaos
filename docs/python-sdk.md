# Python SDK reference

```python
from chaos import (
    Client, AsyncClient,        # entry points
    Index, AsyncIndex,          # index handles (returned by client.open)
    Document, Match,            # data types
    download_model,             # eager model fetch
    Engine, Hit,                # low-level native primitives
)
```

The whole pipeline (tokenize → MiniLM embed → vector search) runs in C++; the
GIL is released around native calls. There is no Python-side embedding.

---

## `Client`

Synchronous entry point.

```python
Client(
    model: str = "all-MiniLM-L6-v2",
    *,
    model_path: str | None = None,
    vocab_path: str | None = None,
    data_dir: str | None = "~/.chaos",
    threads: int = 4,
)
```

- **`model`** — friendly name or Hugging Face repo id. Resolved and cached on
  first use (see [Models](#models)).
- **`model_path` / `vocab_path`** — explicit local files; when both are given,
  no download happens (offline).
- **`data_dir`** — where named indexes persist. `None` disables persistence.
- **`threads`** — MiniLM intra-op threads.

### `Client.open`

```python
open(
    name: str | None = None,
    *,
    kind: str = "flat",          # "flat" (exact) or "hnsw" (approximate)
    threads: int | None = None,
    hnsw_m: int = 16,            # HNSW: neighbors per node
    ef_construction: int = 200,  # HNSW: build-time candidate breadth
    ef_search: int = 64,         # HNSW: query-time candidate breadth
) -> Index
```

Opens an index. With a `name` and a `data_dir`, documents persist to
`<data_dir>/<name>.jsonl` and reload (re-embedding) when reopened by that name.
The `hnsw_*` params are ignored for `kind="flat"`.

---

## `Index`

A handle to one index. Not constructed directly — returned by `Client.open`.

| Member | Description |
|---|---|
| `add(docs: Sequence[Document]) -> tuple[int, int]` | Embed + index (upsert by id). Returns `(added, updated)`. |
| `search(query: str, top_k: int = 10) -> list[Match]` | Top-k, most similar first. |
| `count -> int` / `len(index)` | Number of indexed documents. |
| `name -> str \| None` | The index name (or `None` if in-memory). |
| `ef_search = N` | Setter only: HNSW recall/latency knob (no-op for flat). |

```python
index = client.open("notes", kind="hnsw")
added, updated = index.add([Document(id="1", text="…")])
index.ef_search = 128                       # raise recall for HNSW
matches = index.search("query", top_k=5)
```

---

## `AsyncClient` / `AsyncIndex`

Same semantics as the sync pair; every call is offloaded to a worker thread
(the engine releases the GIL, so concurrent searches overlap).

```python
import asyncio
from chaos import AsyncClient, Document

async def main():
    client = AsyncClient()                    # same constructor args as Client
    index = await client.open("notes")        # -> AsyncIndex
    await index.add([Document(id="1", text="Ship the SDK by Friday.")])
    matches = await index.search("deadlines", top_k=3)
    print(index.count)                        # count/name are plain properties

asyncio.run(main())
```

> **Why async here is not about speed:** search is a local CPU matrix-multiply,
> so `await` alone yields no throughput. The benefit is (1) not blocking a host
> event loop during the ~5 ms embed and (2) overlapping many searches across
> threads. For a plain script, `Client` is simpler and has no thread-hop
> overhead. See [Concepts → Async](concepts.md#a-note-on-async).

---

## Data types

```python
@dataclass
class Document:
    id: str                       # stable, caller-supplied
    text: str
    metadata: dict = {}           # returned on matches, NOT searched

@dataclass
class Match:
    id: str
    score: float                  # cosine similarity in [-1, 1]; higher is better
    text: str
    metadata: dict = {}
```

`add` **upserts** by `id`: a new id is added, an existing id is updated (its
vector and text are replaced). `metadata` round-trips through search results.

---

## Models

`Client()` resolves `all-MiniLM-L6-v2` from the Hugging Face cache, downloading
it once (~86 MB) if absent. Cached files are used directly afterward — no
network round-trip, works offline.

```python
Client()                                        # default MiniLM
Client(model="sentence-transformers/all-MiniLM-L6-v2")   # any HF repo (ONNX export + vocab.txt)
Client(model_path="…/model.onnx", vocab_path="…/vocab.txt")  # explicit, offline
```

### Supported models

The embedding path fixes three things, so chaos supports a *family* of models,
not arbitrary ones:

| | Requirement |
|---|---|
| Format | **ONNX** (ONNX Runtime) — not raw PyTorch / GGUF / safetensors |
| Tokenizer | **BERT WordPiece, uncased** (`vocab.txt`, `[CLS]`/`[SEP]`) |
| Pooling | **mean-pool** over `last_hidden_state` + L2-normalize |

- ✅ ONNX BERT-family, WordPiece, mean-pooled sentence encoders: `all-MiniLM-L6-v2`
  / `L12`, the **E5** family, most BERT-based `sentence-transformers` models.
- ⚠️ CLS-pooled models (some **BGE** variants) tokenize fine but expect CLS
  pooling, not mean — results would be off.
- ❌ SentencePiece/BPE tokenizers (many GTE, T5/`sentence-t5`), non-ONNX formats,
  decoder/generative LLMs.

The C++ [`Embedder`](../include/chaos/embedder.hpp) interface is pluggable, so
configurable pooling, alternate tokenizers, or additional backends can be added.

### `download_model`

```python
from chaos import download_model
model_path, vocab_path = download_model("all-MiniLM-L6-v2")
```

Eagerly fetches a model into the cache and returns its local paths. Equivalent
CLI: `chaos download` (see the [CLI reference](cli.md)).

Known short names live in `chaos.models.KNOWN_MODELS`; any other string is
treated as a Hugging Face repo id (with conventional `onnx/model.onnx` +
`vocab.txt` filenames).

---

## Low-level: `Engine` / `Hit`

The native primitive the clients wrap. Use it if you want to manage ids,
persistence, and metadata yourself.

```python
from chaos import Engine

eng = Engine("model.onnx", "vocab.txt", index="flat", threads=4,
             hnsw_m=16, ef_construction=200, ef_search=64)
eng.upsert("1", "some text")                    # -> bool (True if it updated)
eng.upsert_many(["2", "3"], ["a", "b"])         # -> (added, updated)
hits = eng.search("query", 10)                  # -> list[Hit(id, score, text)]
eng.set_ef_search(128)
eng.dim            # 384 for MiniLM-L6
eng.is_hnsw        # bool
len(eng)           # doc count
```

`Engine` embeds with the real model and requires explicit `model.onnx` /
`vocab.txt` paths — the auto-download convenience lives in `Client`.
