// Index interface. Implementations: FlatIndex (exact, edge-scale default) and,
// later, HnswIndex for larger corpora. The embedder and server code depend only
// on this interface so the index is swappable.
#pragma once
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <vector>

namespace chaos {

struct SearchHit {
  uint32_t id;      // internal vector id (insertion order)
  float score;      // cosine similarity, higher is better
};

class Index {
 public:
  virtual ~Index() = default;
  virtual size_t dim() const = 0;
  virtual size_t size() const = 0;

  // Add a vector (copied, then L2-normalized internally). Returns its id.
  virtual uint32_t add(const float* vec) = 0;

  // Replace the vector at `id` in place (copied, then L2-normalized). Used for
  // upsert. Exact for the flat index; for HNSW the graph links are not rebuilt,
  // so an updated node's recall can drift slightly (fine for occasional edits).
  virtual void update(uint32_t id, const float* vec) = 0;

  // Top-k search. `query` must have length dim(). `out` is filled with up to k
  // hits sorted by descending score. `scratch`/`scratch_bytes` is optional
  // caller-provided arena memory to keep the hot path allocation-free.
  virtual void search(const float* query, size_t k, std::vector<SearchHit>& out,
                      void* scratch = nullptr, size_t scratch_bytes = 0) const = 0;

  // Binary (de)serialization of the built index so it can be reloaded without
  // re-embedding. `save` writes count + vectors (+ graph for HNSW); `load`
  // repopulates an index freshly constructed with the same dim. Format is
  // host-endian — a per-machine cache, not a portable interchange format.
  virtual void save(std::ostream& os) const = 0;
  virtual void load(std::istream& is) = 0;
};

}  // namespace chaos
