# C++ API reference

All types live in namespace `chaos`. Headers are under `include/chaos/`.

Two layers:

- **`Engine`** — the high-level facade (embed + index + document store). This is
  what the Python SDK binds to and what most C++ callers want.
- **`Index` / `FlatIndex` / `HnswIndex`** — the vector index primitives, plus the
  `Embedder` and tokenizer, if you want to compose them yourself.

---

## `Engine` — [engine.hpp](../include/chaos/engine.hpp)

```cpp
#include "chaos/engine.hpp"

chaos::Engine eng(model_path, vocab_path,
                  /*index_kind=*/"flat",   // or "hnsw"
                  /*threads=*/4,
                  /*hnsw_m=*/16, /*ef_construction=*/200, /*ef_search=*/64);

bool updated = eng.upsert("1", "Ship the SDK by Friday.");   // false = newly added
auto [added, upd] = eng.upsert_many(ids, texts);             // {added, updated}

for (const chaos::Doc& d : eng.search("deadlines", 10))
    printf("%s  %.3f  %s\n", d.id.c_str(), d.score, d.text.c_str());

eng.size();          // document count
eng.dim();           // 384 for MiniLM-L6
eng.is_hnsw();
eng.set_ef_search(128);
```

```cpp
struct Doc { std::string id; float score; std::string text; };
```

`upsert` embeds the text (MiniLM), then adds or replaces the vector by string
id, retaining the text so results carry it. Requires the ONNX build
(`-DCHAOS_ONNX=ON`).

---

## `Index` interface — [index.hpp](../include/chaos/index.hpp)

The abstraction both indexes implement; `Engine` and callers depend only on it.

```cpp
struct SearchHit { uint32_t id; float score; };  // id = insertion order

class Index {
 public:
  virtual size_t dim() const = 0;
  virtual size_t size() const = 0;
  virtual uint32_t add(const float* vec) = 0;            // returns id; vec is copied + L2-normalized
  virtual void update(uint32_t id, const float* vec) = 0;// in-place replace (upsert)
  virtual void search(const float* query, size_t k,
                      std::vector<SearchHit>& out,
                      void* scratch = nullptr,
                      size_t scratch_bytes = 0) const = 0;
};
```

Vectors are L2-normalized on insert, so cosine similarity is a plain dot product
at query time. `scratch` is optional caller-provided arena memory to keep the
hot path allocation-free.

### `FlatIndex` — [flat_index.hpp](../include/chaos/flat_index.hpp)

Exact brute-force scan with SIMD distance. No build step, no tuning, 100% recall,
perfectly predictable tail. Best for edge-scale corpora (up to ~50k vectors).

```cpp
chaos::FlatIndex idx(/*dim=*/384, /*reserve=*/0);
uint32_t id = idx.add(vec);       // vec: dim floats
idx.update(id, new_vec);          // exact in-place replacement
std::vector<chaos::SearchHit> hits;
idx.search(query, 10, hits);
```

### `HnswIndex` — [hnsw_index.hpp](../include/chaos/hnsw_index.hpp)

Hierarchical navigable small-world graph. Approximate; ~log(N) hops instead of a
full scan. For corpora past the flat-scan crossover (~100k+).

```cpp
chaos::HnswParams p;             // { M=16, ef_construction=200, ef_search=64, seed }
chaos::HnswIndex idx(384, p);
idx.add(vec);
idx.set_ef_search(128);          // recall/latency knob
idx.search(query, 10, hits);
```

`update` overwrites a node's vector in place without relinking the graph — exact
distances are recomputed on visit, so results stay correct for the node itself;
only its reachability can drift slightly. Rebuild for a fully re-optimized graph.

> **Threading:** search is single-threaded per instance (a mutable epoch-based
> visited set). Shard or lock for concurrent search.

---

## Embedder — [embedder.hpp](../include/chaos/embedder.hpp)

```cpp
class Embedder {
 public:
  virtual size_t dim() const = 0;
  virtual void embed(const char* text, float* out) const = 0;  // writes dim() floats
};

// Real MiniLM via ONNX Runtime (mean-pool + L2-normalize). CHAOS_ONNX only.
std::unique_ptr<Embedder> make_onnx_embedder(const char* model_path,
                                             const char* vocab_path,
                                             int intra_op_threads = 4);
```

The only embedder is the real model; there is no mock/stand-in.

## Tokenizer — [tokenizer.hpp](../include/chaos/tokenizer.hpp)

`WordPieceTokenizer` — BERT/MiniLM WordPiece (uncased). Loads a HuggingFace
`vocab.txt`, encodes to input ids + attention mask with `[CLS]`/`[SEP]` framing.
Used internally by the ONNX embedder.

## Distance kernels — [distance.hpp](../include/chaos/distance.hpp)

```cpp
float chaos::dot(const float* a, const float* b, size_t n);  // SIMD: NEON / AVX2 / scalar
void  chaos::l2_normalize(float* v, size_t n);
```

---

## Build options

| CMake option | Effect |
|---|---|
| *(default)* | Core library + `test_distance`. No external deps. |
| `-DCHAOS_ONNX=ON -DORT_ROOT=<path>` | Real MiniLM embedder, `Engine`, benchmarks, `test_tokenizer`/`test_hnsw`. |
| `-DCHAOS_PYTHON=ON` | The pybind11 module (implies `CHAOS_ONNX`). |
| `-DCHAOS_ARCH=<cpu>` | Target CPU for `-mcpu`/`-march` (e.g. `cortex-a76`, `x86-64-v3`). Default `native`. |

The core library builds `-fno-exceptions -fno-rtti` for its own translation
units; the ONNX embedder and the Python module re-enable both where required.

See the [C++ build section](getting-started.md#building-the-c-side) to get started.
