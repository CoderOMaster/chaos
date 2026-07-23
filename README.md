# chaos

[![PyPI](https://img.shields.io/pypi/v/chaos-search)](https://pypi.org/project/chaos-search/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://pypi.org/project/chaos-search/)
[![CI](https://github.com/CoderOMaster/chaos/actions/workflows/ci.yml/badge.svg)](https://github.com/CoderOMaster/chaos/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A single-binary, CPU-only semantic search runtime for **edge devices**, targeting
**sub-10ms p99** end-to-end retrieval with a MiniLM embedding model. Written in
C++17 with no cross-language boundary on the hot path — everything from the
embedding forward pass to the vector search lives in one native process.

```bash
pip install chaos-search
```

```python
from chaos import Client, Document

client = Client()                       # MiniLM auto-downloaded + cached
index = client.open("notes")
index.add([Document(id="1", text="Ship the on-device SDK by Friday.")])
for m in index.search("deadlines", top_k=3):
    print(m.id, m.score, m.text)
```

**Documentation:** [docs/](docs/) — [getting started](docs/getting-started.md) ·
[concepts](docs/concepts.md) · [Python SDK](docs/python-sdk.md) ·
[C++ API](docs/cpp-api.md) · [CLI](docs/cli.md) · [benchmarks](docs/benchmarks.md)

## Why this design

The latency budget for one query is:

```
recv → tokenize → embed (MiniLM) → ANN search → gather/serialize
```

Embedding is ~70% of the budget; the index is the cheap part. So the two design
priorities are (1) keep the embedding pass fast and (2) make everything else
contribute **zero jitter** so p99 stays glued to p50.

### The p99 discipline (why C++, no GC)

p99 is a jitter problem, not a throughput problem. The knobs we control:

| Jitter source | Mitigation | Where |
|---|---|---|
| malloc/free on hot path | per-request bump **[Arena](include/chaos/arena.hpp)**, `reset()` not `free()` | queries allocate nothing |
| page faults on the index | 64-byte aligned contiguous store; `mlock`-ready | [flat_index.cpp](src/flat_index.cpp) |
| scheduler migration | thread pinning (`sched_setaffinity`) — planned in serving path | see Status |
| cold caches / first-run | explicit **warmup** pass before measuring | [bench_e2e.cpp](bench/bench_e2e.cpp) |
| scalar math | **SIMD dot** (NEON / AVX2 / scalar) | [distance.hpp](include/chaos/distance.hpp) |

Vectors are L2-normalized at insert time, so cosine similarity collapses to a
single dot product at query time.

## Index choice: exact flat scan first

For **edge-scale corpora** (on-device notes/docs — thousands to tens of
thousands of vectors) an exact SIMD brute-force scan beats a graph index: it is
exact (100% recall), has no build step, no tuning, and a perfectly predictable
tail. HNSW is only needed past the crossover point below.

## Measured results (all from the real MiniLM model)

Every number below is produced by `bench_e2e` running full-precision
`all-MiniLM-L6-v2` via ONNX Runtime — real tokenization, real forward pass, real
vectors in the index. There is no mock/simulated embedder anywhere in the
project. Machine: Apple M-class **arm64**, **NEON**, 4 intra-op threads, dim=384,
k=10. Reproduce with the [How to test](#how-to-test-reproduce) commands.

**Headline — fused end-to-end (embed + HNSW search in one pass), N=10k:**

| pipeline | p50 | p90 | p99 | p999 | max |
|---|---|---|---|---|---|
| **embed + HNSW (ef=64)** | 3.61 ms | 4.16 ms | **5.33 ms** | 7.22 ms | 7.88 ms |

That is a single measured loop — embed the query text, then search — not a sum of
separate numbers. **p99 = 5.33 ms, comfortably under the 10 ms target.**

**Per-stage breakdown (same run):**

| stage | p50 | p99 | p999 | note |
|---|---|---|---|---|
| embed (MiniLM) | 3.47 ms | 5.07 ms | 6.24 ms | ~95% of the budget |
| search: flat (exact) | 0.40 ms | 0.62 ms | 1.25 ms | 100% recall |
| search: HNSW ef=32 | 0.029 ms | 0.047 ms | 0.072 ms | recall@10 = 0.988 |
| search: HNSW ef=64 | 0.049 ms | 0.074 ms | 0.093 ms | recall@10 = 0.988 |
| search: HNSW ef=128 | 0.085 ms | 0.132 ms | 0.163 ms | recall@10 = 0.988 |

The embedding forward pass dominates (as predicted); search is a rounding error.
That's why the design spends its effort keeping embed fast and the tail jitter-free.

### Recall is verified, not assumed

HNSW is approximate, so [test_hnsw.cpp](tests/test_hnsw.cpp) measures recall@10
against the exact flat index over **real MiniLM embeddings** of a topical corpus
(N=5k). Recall is monotonic in `ef_search`, as it should be:

| ef_search | recall@10 |
|---|---|
| 16  | 0.926 |
| 64  | 0.987 |
| 200 | 1.000 |

### Index choice

- **≤ ~50k vectors:** `FlatIndex` — exact (100% recall), zero build, zero tuning,
  perfectly predictable tail. Search stays well under a millisecond per the
  breakdown above; the flat scan only becomes the bottleneck past ~100k.
- **≥ ~100k vectors:** `HnswIndex` — ~log(N) hops instead of a full scan, keeping
  search sub-millisecond into the millions. Tune `ef_search` for recall/latency.

Both implement the same [Index](include/chaos/index.hpp) interface, so the
embedder and callers don't change. Trade-off: HNSW build is a one-time cost
(off the query hot path) and uses more memory than the flat store.

<a name="how-to-test-reproduce"></a>
## How to test / reproduce

Every result above is reproducible from a clean checkout:

```bash
# 1. Core library + SIMD/top-k unit test — no model, no external deps
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/test_distance                       # -> "all tests passed"

# 2. Fetch the real model (~86 MB) — one time
bash scripts/fetch_model.sh                 # -> models/model.onnx, models/vocab.txt

# 3. Build the real MiniLM path (needs onnxruntime; brew install onnxruntime)
cmake -B build-onnx -S . -DCHAOS_ONNX=ON -DORT_ROOT=/opt/homebrew/opt/onnxruntime
cmake --build build-onnx -j

# 4. Correctness — real model
ctest --test-dir build-onnx --output-on-failure
#   test_tokenizer : WordPiece ids vs known bert-base-uncased values
#   test_hnsw      : recall@10 vs exact index over real MiniLM embeddings

# 5. The end-to-end benchmark that produced every number above
#    args: <model> <vocab> [N] [queries] [threads]
./build-onnx/bench_e2e models/model.onnx models/vocab.txt 10000 3000 4
```

To see the flat/HNSW crossover, raise `N` (e.g. `100000`) — note the corpus build
embeds N real sentences, so large N takes minutes (real inference, one-time).

## Python SDK

`chaos` ships a native Python module (pybind11). `tokenize → embed → search` all
run in C++; the GIL is released around native calls. There is no Python-side
embedding. **The MiniLM model downloads and caches automatically on first use —
no paths required.**

```bash
pip install chaos-search                        # published wheels bundle ONNX Runtime
```

Build from source (needs a C++17 toolchain + ONNX Runtime, e.g. `brew install onnxruntime`):

```bash
pip install .                                   # uses the CMake ORT default, or:
pip install . -C cmake.define.ORT_ROOT=/path/to/onnxruntime
```

### Synchronous (recommended for scripts/notebooks)

```python
from chaos import Client, Document

client = Client()                        # MiniLM auto-downloaded + cached on first use
index = client.open("notes")             # kind="flat" (default) or "hnsw"

# Add documents (embedded and indexed locally). Upsert by id.
added, updated = index.add([
    Document(id="1", text="Ship the on-device SDK by Friday."),
    Document(id="2", text="Follow up with the LiveKit team about latency."),
])
print(f"{added} added, {updated} updated, {index.count} total")

for m in index.search("what's due this week", top_k=3):
    print(f"{m.id} score={m.score:.3f} {m.text}")   # m: Match(id, score, text, metadata)
```

### Async (same semantics, for event-loop apps)

Use `AsyncClient` inside an event loop (FastAPI, agents). Each call offloads to a
worker thread; because the engine releases the GIL, concurrent searches overlap.

```python
import asyncio
from chaos import AsyncClient, Document

async def main():
    client = AsyncClient()
    index = await client.open("notes")
    await index.add([Document(id="1", text="Ship the SDK by Friday.")])
    for m in await index.search("deadlines", top_k=3):
        print(m.id, m.score, m.text)

asyncio.run(main())
```

> **Note on async:** it's here for event-loop friendliness, not speed. Search is
> a local CPU matrix-multiply, so `await` alone adds no throughput — the thread
> offload (GIL released) is what lets searches overlap. For a plain script, the
> synchronous `Client` is simpler and has no per-call thread-hop overhead.

### Models

No path needed — `Client()` resolves `all-MiniLM-L6-v2` from the Hugging Face
cache, downloading it once (~86 MB) if absent. Options:

```python
Client()                                            # default MiniLM, auto
Client(model="sentence-transformers/all-MiniLM-L6-v2")   # any HF repo (ONNX export)
Client(model_path="…/model.onnx", vocab_path="…/vocab.txt")  # explicit local files (offline)
```

Pre-fetch for offline/first-run-instant (e.g. in a Docker build):

```bash
python -m chaos download          # or: chaos download
```

#### Which models are supported

chaos runs a specific *family* of embedding models — not arbitrary ones. Three
things are fixed in the embedding path today:

| | Requirement |
|---|---|
| **Format** | **ONNX** (loaded via ONNX Runtime). Not raw PyTorch / GGUF / safetensors. |
| **Tokenizer** | **BERT WordPiece, uncased** (`vocab.txt`, `[CLS]`/`[SEP]`). |
| **Pooling** | **mean-pool** over `last_hidden_state` + L2-normalize. |

- ✅ **Works:** ONNX BERT-family, WordPiece, mean-pooled sentence encoders —
  `all-MiniLM-L6-v2` (default) and `L12`, the **E5** family, and most BERT-based
  `sentence-transformers` models. Point at any with `Client(model="<hf-repo>")`
  if the repo ships `onnx/model.onnx` + `vocab.txt`.
- ⚠️ **Wrong results:** CLS-pooled models (some **BGE** variants) — they tokenize
  fine but expect CLS pooling, not mean.
- ❌ **Not supported:** SentencePiece/BPE tokenizers (many GTE, T5/`sentence-t5`),
  non-ONNX formats, and decoder/generative LLMs.

The [`Embedder`](include/chaos/embedder.hpp) interface is pluggable, so
configurable pooling, alternate tokenizers, or other backends can be added — see
[docs/concepts.md](docs/concepts.md) and [docs/cpp-api.md](docs/cpp-api.md).

**Persistence.** Opened with a name, an index persists its documents to
`~/.chaos/<name>.jsonl` and reloads them (re-embedding) when reopened by that
name; unnamed indexes stay in-memory. `add` upserts by id (re-adding an id
updates it). `Document.metadata` is returned on matches but not searched.

Runnable examples in [`examples/`](examples/) (basics, HNSW, persistence,
metadata, async concurrency, low-level). The clients wrap the native C++
[`Engine`](include/chaos/engine.hpp) facade ([bindings](bindings/chaos_py.cpp));
`Engine`/`Hit` are also exposed for advanced low-level use.

## Build

The core library (index + SIMD kernels) and its unit test have **no external
dependencies**. The real MiniLM embedder, benchmarks, and recall test are gated
behind `-DCHAOS_ONNX=ON` and need ONNX Runtime + the model. See
[How to test / reproduce](#how-to-test-reproduce) for the full command sequence.

Cross-compiling for a specific edge target: `-DCHAOS_ARCH=cortex-a76` (ARM)
or `-DCHAOS_ARCH=x86-64-v3` (x86).

## Releasing to PyPI

Publishing a GitHub Release runs [`.github/workflows/publish.yml`](.github/workflows/publish.yml),
which builds wheels for macOS (arm64) and Linux (x86_64/aarch64) with
[`cibuildwheel`](https://cibuildwheel.pypa.io) and uploads them via PyPI
**Trusted Publishing** (OIDC — no API token). ONNX Runtime is downloaded per
platform ([scripts/ci_fetch_onnxruntime.sh](scripts/ci_fetch_onnxruntime.sh)) and
bundled into each wheel, so `pip install chaos-search` needs no system deps.

One-time setup: add a trusted publisher for this repo + `publish.yml` at
`https://pypi.org/manage/project/chaos-search/settings/publishing/`, then trigger
the workflow manually once (or point it at TestPyPI) to validate before a real
release.

## Layout

```
include/chaos/   distance.hpp  arena.hpp  timing.hpp  index.hpp  corpus.hpp
                 flat_index.hpp  hnsw_index.hpp  embedder.hpp  tokenizer.hpp
                 engine.hpp
src/             flat_index.cpp  hnsw_index.cpp
                 embedder_onnx.cpp  tokenizer.cpp  engine.cpp   (CHAOS_ONNX)
bindings/        chaos_py.cpp                                   (CHAOS_PYTHON)
python/chaos/    __init__.py  client.py  models.py  __main__.py  _core.pyi
bench/           bench_e2e.cpp                                  (CHAOS_ONNX)
tests/           test_distance.cpp                              (core, no model)
                 test_hnsw.cpp  test_tokenizer.cpp              (CHAOS_ONNX, real model)
examples/        quickstart.py
docs/            getting-started · concepts · python-sdk · cpp-api · cli · benchmarks
scripts/         fetch_model.sh  ci_fetch_onnxruntime.sh
.github/workflows/  ci.yml (build+test)  publish.yml (wheels → PyPI)
pyproject.toml   pip-installable Python package (scikit-build-core + cibuildwheel)
```

The only embedder is the real MiniLM model ([embedder_onnx.cpp](src/embedder_onnx.cpp));
[corpus.hpp](include/chaos/corpus.hpp) is a *text* generator for benchmarks (the
model embeds it), not a stand-in for the model. [engine.hpp](include/chaos/engine.hpp)
is the high-level facade that both the benchmark and the Python SDK use.

## Roadmap

- [x] SIMD distance (NEON / AVX2 / scalar), arena, timing
- [x] exact flat index + SIMD/top-k unit test
- [x] BERT WordPiece tokenizer (verified vs known ids) + ONNX Runtime MiniLM embedder
- [x] HNSW index + recall test on real embeddings + fused end-to-end benchmark
- [x] `Engine` facade + pybind11 module + Python SDK (sync `Client` + `AsyncClient`, sessions, upsert, local persistence)
- [ ] request server (io_uring/epoll)
- [ ] parallel/int8 HNSW build; multi-threaded sharded flat scan
- [ ] thread pinning + mlock wired into the serving path (warmup is in the bench today)
