// Correctness checks for the SIMD kernels and the flat index. Verifies the
// vectorized dot product matches a scalar reference and that top-k search
// agrees with an exhaustive brute-force ranking.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include "chaos/distance.hpp"
#include "chaos/flat_index.hpp"

using namespace chaos;

static int failures = 0;
#define CHECK(cond, msg)                              \
  do {                                                \
    if (!(cond)) {                                    \
      std::printf("FAIL: %s\n", msg);                 \
      ++failures;                                     \
    }                                                 \
  } while (0)

static float scalar_dot(const float* a, const float* b, size_t n) {
  float s = 0;
  for (size_t i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}

int main() {
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);

  // 1. SIMD dot matches scalar across several lengths (incl. non-multiples).
  for (size_t n : {384u, 383u, 385u, 16u, 7u, 1u}) {
    std::vector<float> a(n), b(n);
    for (size_t i = 0; i < n; ++i) { a[i] = dist(rng); b[i] = dist(rng); }
    float got = dot(a.data(), b.data(), n);
    float ref = scalar_dot(a.data(), b.data(), n);
    CHECK(std::fabs(got - ref) < 1e-3f * (1 + std::fabs(ref)), "dot mismatch");
  }

  // 2. Normalization yields unit length.
  {
    std::vector<float> v(384);
    for (auto& x : v) x = dist(rng);
    l2_normalize(v.data(), v.size());
    float nrm = std::sqrt(dot(v.data(), v.data(), v.size()));
    CHECK(std::fabs(nrm - 1.f) < 1e-4f, "normalize not unit length");
  }

  // 3. Flat index top-k matches exhaustive ranking. Vectors here are random
  //    numeric fixtures -- this test exercises the SIMD math and top-k logic,
  //    not embedding quality, so it needs no model.
  {
    const size_t dim = 384, N = 2000, K = 10;
    FlatIndex idx(dim, N);
    std::vector<std::vector<float>> stored(N, std::vector<float>(dim));
    for (size_t i = 0; i < N; ++i) {
      for (auto& x : stored[i]) x = dist(rng);
      idx.add(stored[i].data());
    }

    std::vector<float> q(dim);
    for (auto& x : q) x = dist(rng);
    l2_normalize(q.data(), dim);

    std::vector<SearchHit> hits;
    std::vector<unsigned char> scratch(K * sizeof(SearchHit));
    idx.search(q.data(), K, hits, scratch.data(), scratch.size());

    // Brute-force reference (index stores normalized vectors, so recompute).
    std::vector<float> ref_scores(N);
    for (size_t i = 0; i < N; ++i) {
      std::vector<float> nv = stored[i];
      l2_normalize(nv.data(), dim);
      ref_scores[i] = dot(q.data(), nv.data(), dim);
    }
    std::sort(ref_scores.begin(), ref_scores.end(), std::greater<float>());

    CHECK(hits.size() == K, "wrong hit count");
    for (size_t i = 0; i < hits.size(); ++i)
      CHECK(std::fabs(hits[i].score - ref_scores[i]) < 1e-4f, "top-k score mismatch");
    for (size_t i = 1; i < hits.size(); ++i)
      CHECK(hits[i - 1].score >= hits[i].score, "results not sorted desc");
  }

  if (failures == 0) std::printf("all tests passed\n");
  return failures ? 1 : 0;
}
