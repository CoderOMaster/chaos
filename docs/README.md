# chaos documentation

CPU-only, single-binary semantic search for edge devices — MiniLM embeddings +
exact/approximate vector search, with a native C++ core and a Python SDK.

## Contents

- **[Getting started](getting-started.md)** — install, first program, model setup
- **[Concepts](concepts.md)** — architecture, the latency budget, flat vs HNSW, the p99 discipline
- **[Python SDK](python-sdk.md)** — `Client` / `AsyncClient`, `Index`, `Document`, `Match`, models, low-level `Engine`
- **[C++ API](cpp-api.md)** — `Engine`, the `Index` interface, `FlatIndex` / `HnswIndex`, embedder, tokenizer
- **[CLI](cli.md)** — `chaos download` and offline/pre-fetch workflows
- **[Benchmarks](benchmarks.md)** — measured latency/recall and how to reproduce

## At a glance

```python
from chaos import Client, Document

client = Client()                       # MiniLM auto-downloaded + cached
index = client.open("notes")            # kind="flat" (default) or "hnsw"
index.add([Document(id="1", text="Ship the on-device SDK by Friday.")])
for m in index.search("deadlines", top_k=3):
    print(m.id, m.score, m.text)
```

```cpp
#include "chaos/engine.hpp"
chaos::Engine eng(model_path, vocab_path, "flat");
eng.upsert("1", "Ship the on-device SDK by Friday.");
for (const auto& d : eng.search("deadlines", 3))
    printf("%s %.3f %s\n", d.id.c_str(), d.score, d.text.c_str());
```

See the top-level [README](../README.md) for the project overview and design rationale.
