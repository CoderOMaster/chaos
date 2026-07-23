// HNSW correctness on REAL MiniLM embeddings: recall@10 vs the exact flat
// index. Embeds a topical text corpus with the real model, so recall reflects
// the actual embedding distribution, not synthetic data.
//   ./test_hnsw <model.onnx> <vocab.txt>
#include <cstdio>
#include <string>
#include <unordered_set>
#include <vector>

#include "chaos/corpus.hpp"
#include "chaos/distance.hpp"
#include "chaos/embedder.hpp"
#include "chaos/flat_index.hpp"
#include "chaos/hnsw_index.hpp"

using namespace chaos;

static int failures = 0;
static void check(bool ok, const char* msg) {
  if (!ok) { std::printf("FAIL: %s\n", msg); ++failures; }
}

static double recall_at_k(const Index& approx, const FlatIndex& exact,
                          const std::vector<std::vector<float>>& queries, size_t k) {
  size_t hit = 0, total = 0;
  std::vector<SearchHit> a, e;
  for (const auto& q : queries) {
    approx.search(q.data(), k, a, nullptr, 0);
    exact.search(q.data(), k, e, nullptr, 0);
    std::unordered_set<uint32_t> truth;
    for (const auto& h : e) truth.insert(h.id);
    for (const auto& h : a) if (truth.count(h.id)) ++hit;
    total += e.size();
  }
  return double(hit) / double(total);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <model.onnx> <vocab.txt>\n", argv[0]);
    return 2;
  }
  auto emb = make_onnx_embedder(argv[1], argv[2], 4);
  const size_t dim = emb->dim(), N = 5000, K = 10;

  CorpusGen corpus;
  FlatIndex flat(dim, N);
  HnswIndex hnsw(dim, HnswParams{}, N);
  std::vector<float> v(dim);
  for (size_t i = 0; i < N; ++i) {
    emb->embed(corpus.sentence(i).c_str(), v.data());
    flat.add(v.data());
    hnsw.add(v.data());
  }
  check(hnsw.size() == N, "hnsw indexed all vectors");

  // Queries: fresh sentences from the same topics (real embeddings).
  std::vector<std::vector<float>> queries;
  for (size_t i = 0; i < 300; ++i) {
    std::vector<float> q(dim);
    emb->embed(corpus.sentence(N + i).c_str(), q.data());
    l2_normalize(q.data(), dim);
    queries.push_back(std::move(q));
  }

  hnsw.set_ef_search(64);
  double r64 = recall_at_k(hnsw, flat, queries, K);
  std::printf("recall@%zu (ef=64):  %.4f\n", K, r64);
  check(r64 > 0.90, "recall@10 > 0.90 at ef=64");

  hnsw.set_ef_search(16);
  std::printf("recall@%zu (ef=16):  %.4f\n", K, recall_at_k(hnsw, flat, queries, K));

  hnsw.set_ef_search(200);
  double r200 = recall_at_k(hnsw, flat, queries, K);
  std::printf("recall@%zu (ef=200): %.4f\n", K, r200);
  check(r200 >= r64 - 1e-9, "higher ef_search does not reduce recall");

  if (failures == 0) std::printf("all hnsw tests passed\n");
  return failures ? 1 : 0;
}
