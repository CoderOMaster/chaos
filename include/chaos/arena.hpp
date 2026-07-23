// Per-request bump allocator. Allocated once, reset (not freed) after each
// query, so the hot path never touches malloc/free and generates zero heap
// jitter -- the main lever for a tight p99 in a no-GC language.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>

namespace chaos {

class Arena {
 public:
  explicit Arena(size_t bytes) : cap_(bytes), off_(0) {
    base_ = static_cast<uint8_t*>(std::aligned_alloc(64, round_up(bytes, 64)));
  }
  ~Arena() { std::free(base_); }

  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;

  // Returns 64-byte aligned scratch, or nullptr if the arena is exhausted.
  void* alloc(size_t bytes) {
    size_t start = round_up(off_, 64);
    if (start + bytes > cap_) return nullptr;
    off_ = start + bytes;
    return base_ + start;
  }

  template <typename T>
  T* alloc_n(size_t count) {
    return static_cast<T*>(alloc(count * sizeof(T)));
  }

  // O(1) reset between requests -- reuse the same memory, no deallocation.
  void reset() { off_ = 0; }

  size_t used() const { return off_; }
  size_t capacity() const { return cap_; }

 private:
  static size_t round_up(size_t x, size_t a) { return (x + a - 1) & ~(a - 1); }
  uint8_t* base_;
  size_t cap_;
  size_t off_;
};

}  // namespace chaos
