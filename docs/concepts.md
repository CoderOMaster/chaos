# Concepts

## The pipeline

One query flows through:

```
recv → tokenize → embed (MiniLM) → ANN search → gather/serialize
```

Everything runs in one native process. There is **no cross-language boundary on
the hot path**: the Python SDK calls into C++ once per operation and the GIL is
released for the duration.

## The latency budget

Target: **sub-10 ms p99**, end to end, on CPU. Where it goes (arm64, MiniLM-L6):

| Stage | share | why |
|---|---|---|
| embed (MiniLM forward pass) | ~95% (~3–5 ms) | a transformer GEMM; the boss fight |
| ANN search | <1 ms | dot products over normalized vectors |
| tokenize / gather | negligible | |

So the design spends its effort on two things: keep the embedding pass fast, and
make everything else contribute **zero jitter** so p99 stays glued to p50.

## The p99 discipline

p99 is a *jitter* problem, not a throughput problem. In a no-GC language you can
drive the tail toward the median by removing jitter sources:

| Source | Mitigation |
|---|---|
| malloc/free on the hot path | per-request bump **arena**, reset (not freed) between queries |
| page faults on the index | 64-byte-aligned contiguous store, `mlock`-ready |
| scheduler migration | thread pinning (planned in the serving path) |
| cold caches / first call | explicit **warmup** before measuring |
| scalar math | **SIMD** dot product (NEON / AVX2 / scalar) |

Vectors are L2-normalized at insert time, so cosine similarity collapses to a
single dot product per candidate at query time.

## Flat vs HNSW

Two indexes behind one interface; pick per corpus size:

| | `FlatIndex` (exact) | `HnswIndex` (approximate) |
|---|---|---|
| Recall | 100% | tunable via `ef_search` (≈0.93–1.0) |
| Search cost | O(N) scan | ~O(log N) hops |
| Build | none | one-time graph build |
| Tuning | none | `M`, `ef_construction`, `ef_search` |
| Best for | ≤ ~50k vectors | ~100k → millions |

At edge scale (on-device notes/docs) the exact scan wins: it's simpler, exact,
and has a perfectly predictable tail. Past ~100k, the flat scan's cost exceeds
the budget and HNSW takes over, dropping search to sub-millisecond. Both hold
sub-10 ms end-to-end p99 at their respective scales — see [Benchmarks](benchmarks.md).

Python: `client.open(name, kind="flat" | "hnsw")`. C++: `FlatIndex` / `HnswIndex`.

## Why MiniLM stays full-precision

The model is **not** quantized — chaos supports bring-your-own embedding models,
so the speed comes from parallelism + SIMD on the GEMM (via ONNX Runtime), not
from shrinking the model. Vector-side quantization (compressing the *stored*
index) is a separate, orthogonal lever and doesn't touch the model.

## A note on async

The Python SDK offers `AsyncClient`, but async here is for **integration, not
speed**. Search is a local CPU matrix-multiply; `await`-ing it yields no
throughput on its own, because asyncio is single-threaded cooperative
concurrency and the coroutine doesn't yield mid-multiply. The async variant
offloads each call to a worker thread — and because the engine releases the GIL,
that's what lets concurrent searches actually overlap. It also keeps a host
event loop (FastAPI, an agent) responsive during the ~5 ms embed.

For a plain script, the synchronous `Client` is the honest choice: simpler, and
no per-call thread-hop overhead.
