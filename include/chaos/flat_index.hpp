// Exact brute-force index with SIMD scan. For edge-scale corpora (up to ~tens
// of thousands of on-device vectors) this beats a graph index: it is exact,
// has no build step, no tuning, perfectly predictable p99, and is cache-linear.
// Storage is one contiguous 64-byte-aligned block, `dim` floats per vector.
#pragma once
#include <cstdint>
#include <vector>

#include "chaos/index.hpp"

namespace chaos {

class FlatIndex : public Index {
 public:
  explicit FlatIndex(size_t dim, size_t reserve = 0);
  ~FlatIndex() override;

  size_t dim() const override { return dim_; }
  size_t size() const override { return count_; }

  uint32_t add(const float* vec) override;
  void update(uint32_t id, const float* vec) override;
  void search(const float* query, size_t k, std::vector<SearchHit>& out,
              void* scratch, size_t scratch_bytes) const override;
  void save(std::ostream& os) const override;
  void load(std::istream& is) override;

 private:
  const float* row(size_t i) const { return data_ + i * stride_; }
  float* row(size_t i) { return data_ + i * stride_; }
  void grow(size_t new_cap);

  size_t dim_;
  size_t stride_;     // == dim_ (kept explicit in case of future padding)
  size_t count_ = 0;
  size_t cap_ = 0;
  float* data_ = nullptr;  // aligned_alloc'd, mlock'able
};

}  // namespace chaos
