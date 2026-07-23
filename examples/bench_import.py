"""Import a JSON corpus of [{id, text}, ...] and benchmark ingest + search.

    python examples/bench_import.py <docs.json> [limit] [kind]
      limit : max docs to load (default: all)
      kind  : "hnsw" (default, for large corpora) or "flat" (exact)

Ingest embeds every document with the real MiniLM model, so 100k docs takes a
few minutes (one-time). Search timing is per-query end-to-end (embed + search).
"""
import json
import sys
import time

from chaos import Client, Document

path = sys.argv[1]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
kind = sys.argv[3] if len(sys.argv) > 3 else "hnsw"

raw = json.load(open(path))
if limit:
    raw = raw[:limit]
docs = [Document(id=d["id"], text=d["text"]) for d in raw]
print(f"loaded {len(docs)} docs from {path}")

client = Client()                 # MiniLM auto-downloaded + cached
index = client.open(kind=kind)    # unnamed -> in-memory (isolates ingest timing)

# --- ingest: embed + index every doc ---
t0 = time.perf_counter()
added, updated = index.add(docs)
t_ingest = time.perf_counter() - t0
print(f"ingest ({kind}): {added} docs in {t_ingest:.1f}s  "
      f"= {len(docs) / t_ingest:.0f} docs/s ({t_ingest / len(docs) * 1e3:.2f} ms/doc)")

# --- search: per-query end-to-end (embed the query text, then search) ---
step = max(1, len(docs) // 300)
queries = [docs[i].text for i in range(0, len(docs), step)][:300]

for q in queries[:25]:            # warmup
    index.search(q, top_k=10)

lat_ms = []
for q in queries:
    t = time.perf_counter()
    index.search(q, top_k=10)
    lat_ms.append((time.perf_counter() - t) * 1e3)
lat_ms.sort()
pct = lambda p: lat_ms[int(p * (len(lat_ms) - 1))]
print(f"search over {len(lat_ms)} queries (top_k=10, end-to-end embed+search):")
print(f"  p50={pct(.50):.2f}ms  p90={pct(.90):.2f}ms  p99={pct(.99):.2f}ms  max={lat_ms[-1]:.2f}ms")
