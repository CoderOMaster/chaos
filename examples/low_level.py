"""Low-level native API: drive the C++ Engine directly (no Client/persistence).
Useful when you want to manage ids, storage, and metadata yourself.

    python examples/low_level.py
"""
from chaos import Engine, download_model

# Resolve (download + cache) the model, then hand the paths to the native Engine.
model_path, vocab_path = download_model("all-MiniLM-L6-v2")

eng = Engine(model_path, vocab_path, index="flat", threads=4)
print(f"engine dim={eng.dim} hnsw={eng.is_hnsw}")

updated = eng.upsert("1", "The bakery opens at six in the morning.")
added, upd = eng.upsert_many(["2", "3"],
                             ["Sourdough needs a long, slow proof.",
                              "The train departs from platform nine."])
print(f"upsert('1') updated={updated}; upsert_many -> {added} added, {upd} updated; len={len(eng)}")

for h in eng.search("fresh bread schedule", k=2):
    print(f"  {h.id}  {h.score:.3f}  {h.text}")   # h: Hit(id, score, text)
