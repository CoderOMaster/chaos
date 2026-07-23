#include "chaos/flat_index.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <istream>
#include <ostream>

#include "chaos/distance.hpp"

namespace chaos {

FlatIndex::FlatIndex(size_t dim, size_t reserve) : dim_(dim), stride_(dim) {
  if (reserve) grow(reserve);
}

FlatIndex::~FlatIndex() { std::free(data_); }

void FlatIndex::grow(size_t new_cap) {
  if (new_cap <= cap_) return;
  size_t bytes = new_cap * stride_ * sizeof(float);
  // 64-byte aligned so every row starts on a cache line and SIMD loads are happy.
  float* next = static_cast<float*>(std::aligned_alloc(64, ((bytes + 63) & ~size_t(63))));
  if (data_) {
    std::memcpy(next, data_, count_ * stride_ * sizeof(float));
    std::free(data_);
  }
  data_ = next;
  cap_ = new_cap;
}

uint32_t FlatIndex::add(const float* vec) {
  if (count_ == cap_) grow(cap_ ? cap_ * 2 : 1024);
  float* dst = row(count_);
  std::memcpy(dst, vec, dim_ * sizeof(float));
  l2_normalize(dst, dim_);  // normalize once so search is a bare dot product
  return static_cast<uint32_t>(count_++);
}

void FlatIndex::update(uint32_t id, const float* vec) {
  float* dst = row(id);
  std::memcpy(dst, vec, dim_ * sizeof(float));
  l2_normalize(dst, dim_);  // exact replacement; search stays a bare dot product
}

void FlatIndex::save(std::ostream& os) const {
  uint64_t n = count_;
  os.write(reinterpret_cast<const char*>(&n), sizeof n);
  os.write(reinterpret_cast<const char*>(data_), n * stride_ * sizeof(float));
}

void FlatIndex::load(std::istream& is) {
  uint64_t n = 0;
  is.read(reinterpret_cast<char*>(&n), sizeof n);
  grow(n ? n : 1);
  is.read(reinterpret_cast<char*>(data_), n * stride_ * sizeof(float));
  count_ = n;
}

// Bounded top-k via a min-heap keyed on score. We keep the k best seen; the
// heap root is the current worst survivor, so a candidate only enters if it
// beats the root. k is small (typ. 10), so this is cheap relative to the scan.
void FlatIndex::search(const float* query, size_t k, std::vector<SearchHit>& out,
                       void* scratch, size_t scratch_bytes) const {
  out.clear();
  if (count_ == 0 || k == 0) return;
  k = std::min(k, count_);

  // Heap lives in caller scratch when provided, else in `out` itself.
  SearchHit* heap;
  bool use_scratch = scratch && scratch_bytes >= k * sizeof(SearchHit);
  if (use_scratch) {
    heap = static_cast<SearchHit*>(scratch);
  } else {
    out.resize(k);
    heap = out.data();
  }

  auto worse = [](const SearchHit& a, const SearchHit& b) { return a.score > b.score; };
  size_t filled = 0;

  for (size_t i = 0; i < count_; ++i) {
    float s = dot(query, row(i), dim_);
    if (filled < k) {
      heap[filled++] = {static_cast<uint32_t>(i), s};
      if (filled == k) std::make_heap(heap, heap + k, worse);
    } else if (s > heap[0].score) {
      std::pop_heap(heap, heap + k, worse);
      heap[k - 1] = {static_cast<uint32_t>(i), s};
      std::push_heap(heap, heap + k, worse);
    }
  }

  if (use_scratch) {
    out.assign(heap, heap + filled);
  }
  // Sort results by descending score for the caller.
  std::sort(out.begin(), out.end(),
            [](const SearchHit& a, const SearchHit& b) { return a.score > b.score; });
}

}  // namespace chaos
