# Benchmarks

All numbers come from the **real MiniLM model** (`all-MiniLM-L6-v2`, ONNX
Runtime) via `bench_e2e` — real tokenization, real forward pass, real vectors in
the index. There is no mock/simulated embedder. Machine: Apple M-class
**arm64**, NEON, 4 intra-op threads, dim=384, k=10.

## Headline — fused end-to-end (N = 10k)

A single measured loop: embed the query text, then search (not a sum of separate
numbers).

| pipeline | p50 | p90 | p99 | p999 | max |
|---|---|---|---|---|---|
| **embed + HNSW (ef=64)** | 3.61 ms | 4.16 ms | **5.33 ms** | 7.22 ms | 7.88 ms |

**p99 = 5.33 ms — comfortably under the 10 ms target.**

## Per-stage breakdown (same run)

| stage | p50 | p99 | p999 | note |
|---|---|---|---|---|
| embed (MiniLM) | 3.47 ms | 5.07 ms | 6.24 ms | ~95% of the budget |
| search: flat (exact) | 0.40 ms | 0.62 ms | 1.25 ms | 100% recall |
| search: HNSW ef=32 | 0.029 ms | 0.047 ms | 0.072 ms | recall@10 = 0.988 |
| search: HNSW ef=64 | 0.049 ms | 0.074 ms | 0.093 ms | recall@10 = 0.988 |
| search: HNSW ef=128 | 0.085 ms | 0.132 ms | 0.163 ms | recall@10 = 0.988 |

The embedding pass dominates; search is a rounding error. That's the design
thesis in one table.

## Recall (verified, not assumed)

`test_hnsw` measures recall@10 against the exact flat index over real MiniLM
embeddings of a topical corpus (N=5k). Recall is monotonic in `ef_search`:

| ef_search | recall@10 |
|---|---|
| 16  | 0.926 |
| 64  | 0.987 |
| 200 | 1.000 |

## Reproduce

```bash
# core unit test — no model, no deps
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
./build/test_distance

# real model path
bash scripts/fetch_model.sh
cmake -B build-onnx -S . -DCHAOS_ONNX=ON -DORT_ROOT=/opt/homebrew/opt/onnxruntime
cmake --build build-onnx -j

ctest --test-dir build-onnx --output-on-failure     # tokenizer + recall (real model)

# the end-to-end benchmark that produced the tables above
#   args: <model> <vocab> [N] [queries] [threads]
./build-onnx/bench_e2e models/model.onnx models/vocab.txt 10000 3000 4
```

Raise `N` (e.g. `100000`) to see the flat→HNSW crossover. Note the corpus build
embeds `N` real sentences, so large `N` takes minutes (real inference, one-time).

## Interpretation

- **≤ ~50k vectors:** flat search is well under a millisecond; embed dominates,
  total p99 stays ~5 ms.
- **~100k+ vectors:** the exact scan alone approaches the budget; HNSW cuts
  search back to sub-millisecond at ≥98% recall, keeping end-to-end p99 low into
  the millions.

See [Concepts → Flat vs HNSW](concepts.md#flat-vs-hnsw) for the trade-off.
