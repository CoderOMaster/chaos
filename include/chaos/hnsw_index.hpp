// HNSW index (Malkov & Yashunin, 2016) for corpora past the flat-scan
// crossover. A hierarchical navigable small-world graph reaches the top-k in
// ~log(N) hops instead of scanning all N vectors, trading exactness for a
// large latency win at scale. Search is approximate; recall is tunable via
// ef_search and verified in tests/test_hnsw.cpp.
//
// Vectors are L2-normalized on insert (like FlatIndex), so graph distance is
// 1 - dot and we navigate by maximizing cosine similarity directly.
//
// Build and search are single-threaded; the visited-set uses a mutable epoch
// buffer, so a given instance must not be searched from multiple threads at
// once (shard or lock for concurrency).
#pragma once
#include <cstdint>
#include <random>
#include <vector>

#include "chaos/index.hpp"

namespace chaos {

struct HnswParams {
  size_t M = 16;               // neighbors per node (base layer uses 2*M)
  size_t ef_construction = 200;  // candidate breadth while building
  size_t ef_search = 64;         // candidate breadth while querying (recall knob)
  uint64_t seed = 0x9e3779b9;    // deterministic level assignment
};

class HnswIndex : public Index {
 public:
  HnswIndex(size_t dim, HnswParams params = {}, size_t reserve = 0);
  ~HnswIndex() override;

  size_t dim() const override { return dim_; }
  size_t size() const override { return count_; }

  uint32_t add(const float* vec) override;
  void update(uint32_t id, const float* vec) override;
  void search(const float* query, size_t k, std::vector<SearchHit>& out,
              void* scratch, size_t scratch_bytes) const override;

  void set_ef_search(size_t ef) { params_.ef_search = ef; }

 private:
  using Cand = std::pair<float, uint32_t>;  // (similarity, node id)

  const float* row(uint32_t i) const { return data_ + size_t(i) * dim_; }
  float* row(uint32_t i) { return data_ + size_t(i) * dim_; }
  float sim(uint32_t a, uint32_t b) const;
  float sim_q(const float* q, uint32_t b) const;
  void grow_data(size_t new_cap);
  int random_level();

  // Greedy descent through the upper layers; returns the best entry node found
  // at `target_level` to seed the base-layer search.
  uint32_t greedy_descent(const float* q, uint32_t entry, int top, int target_level) const;
  // Beam search within one layer; fills `out` with up to ef (sim, id), min-heap.
  void search_layer(const float* q, uint32_t entry, size_t ef, int layer,
                    std::vector<Cand>& out) const;
  // Diversity heuristic (Malkov Alg. 4): pick <=M neighbors of `base`.
  void select_neighbors(uint32_t base, std::vector<Cand>& cand, size_t M,
                        std::vector<uint32_t>& out) const;

  size_t dim_;
  HnswParams params_;
  size_t count_ = 0, cap_ = 0;
  float* data_ = nullptr;

  std::vector<int> node_level_;                          // top layer per node
  std::vector<std::vector<std::vector<uint32_t>>> links_;  // [node][layer][nbrs]
  int max_level_ = -1;
  uint32_t entry_point_ = 0;
  double mL_;
  mutable std::mt19937_64 rng_;

  // Fast visited-set: node considered visited iff vis_[node] == vis_epoch_.
  mutable std::vector<uint32_t> vis_;
  mutable uint32_t vis_epoch_ = 0;
};

}  // namespace chaos
