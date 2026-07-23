# Getting started

## Install (Python)

Requires a C++17 toolchain, CMake ≥ 3.16, and ONNX Runtime (for the MiniLM
embedder).

```bash
# macOS
brew install onnxruntime

# from the repo root
pip install .
# ONNX Runtime installed somewhere non-standard? point the build at it:
pip install . -C cmake.define.ORT_ROOT=/path/to/onnxruntime
```

This builds the native extension and installs the `chaos` package plus a `chaos`
console script. Runtime dependency: `huggingface_hub` (for model download),
installed automatically.

## Your first program

No model path is needed — the MiniLM model is downloaded and cached on first use.

```python
from chaos import Client, Document

client = Client()
index = client.open("notes")

added, updated = index.add([
    Document(id="1", text="Ship the on-device SDK by Friday."),
    Document(id="2", text="Follow up with the LiveKit team about latency."),
    Document(id="3", text="Buy oat milk and coffee beans on the way home."),
])
print(f"{added} added, {updated} updated, {index.count} total")

for m in index.search("what's due this week", top_k=3):
    print(f"{m.id}  {m.score:.3f}  {m.text}")
```

First run downloads `all-MiniLM-L6-v2` (~86 MB) into the Hugging Face cache
(`~/.cache/huggingface`); later runs are instant and work offline.

## Models

```python
Client()                                    # default all-MiniLM-L6-v2, auto-downloaded
Client(model="some-org/minilm-onnx")        # any HF repo with an ONNX export + vocab.txt
Client(model_path="…/model.onnx",           # explicit local files (fully offline)
       vocab_path="…/vocab.txt")
```

Pre-fetch (offline prep, Docker builds, instant first run):

```bash
chaos download            # or: python -m chaos download
```

See **[Models](python-sdk.md#models)** and the **[CLI reference](cli.md)**.

## Choosing an index

- `client.open(name, kind="flat")` — **exact** search, zero tuning. Best up to
  ~50k documents. This is the default.
- `client.open(name, kind="hnsw")` — **approximate** graph index, sub-millisecond
  search into the millions. Tune recall/latency with `index.ef_search = N`.

See **[Concepts](concepts.md)** for the trade-off and the crossover point.

## Persistence

Open an index with a name and it persists its documents (`~/.chaos/<name>.jsonl`)
and its vectors + HNSW graph (`~/.chaos/<name>.idx`). Reopening by that name
**loads the vectors directly — no re-embedding** — so a corpus that took minutes
to build reopens in a fraction of a second. Unnamed indexes stay in-memory:

```python
index = client.open()          # ephemeral, in-memory
index = client.open("notes")   # persisted; reopens instantly (no re-embed)
```

## Building the C++ side

```bash
# core library + unit test (no model, no external deps)
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build

# with the real MiniLM embedder, benchmarks, and recall test
bash scripts/fetch_model.sh
cmake -B build-onnx -S . -DCHAOS_ONNX=ON -DORT_ROOT=/opt/homebrew/opt/onnxruntime
cmake --build build-onnx -j
ctest --test-dir build-onnx
```

See the **[C++ API](cpp-api.md)** for the native interfaces and build options.
