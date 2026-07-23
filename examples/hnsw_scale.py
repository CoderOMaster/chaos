"""HNSW index for larger corpora, and the ef_search recall/latency knob.

    python examples/hnsw_scale.py
"""
import time

from chaos import Client, Document

client = Client()
index = client.open(kind="hnsw")     # approximate graph index

# A few hundred short, topical documents.
TOPICS = {
    "coffee": ["brew", "grind", "espresso", "beans", "pour-over", "descale"],
    "devops": ["deploy", "rollback", "latency", "cache", "pipeline", "incident"],
    "garden": ["prune", "water", "compost", "seedlings", "harvest", "mulch"],
    "music":  ["rehearse", "tempo", "melody", "score", "encore", "tuning"],
}
docs = []
i = 0
for topic, words in TOPICS.items():
    for a in words:
        for b in words:
            docs.append(Document(id=str(i), text=f"{topic}: remember to {a} the {b} today"))
            i += 1
index.add(docs)
print(f"indexed {index.count} docs into an HNSW index\n")

query = "notes about brewing coffee"
for ef in (16, 64, 128):
    index.ef_search = ef
    t0 = time.perf_counter()
    hits = index.search(query, top_k=3)
    dt = (time.perf_counter() - t0) * 1e3
    print(f"ef_search={ef:<4} {dt:.2f} ms")
    for m in hits:
        print(f"    {m.score:.3f}  {m.text}")
