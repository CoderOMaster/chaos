// End-to-end benchmark on the REAL MiniLM model. Builds a corpus by embedding
// topical text through MiniLM into both indexes, then measures every stage with
// real vectors: embedding, flat search, HNSW search (+ recall vs exact), and a
// genuinely fused embed+search end-to-end pass.
//
//   ./bench_e2e <model.onnx> <vocab.txt> [N] [queries] [threads]
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_set>
#include <vector>

#include "chaos/corpus.hpp"
#include "chaos/distance.hpp"
#include "chaos/embedder.hpp"
#include "chaos/flat_index.hpp"
#include "chaos/hnsw_index.hpp"
#include "chaos/timing.hpp"

using namespace chaos;

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <model.onnx> <vocab.txt> [N] [queries] [threads]\n", argv[0]);
    return 2;
  }
  size_t N = argc > 3 ? std::strtoul(argv[3], nullptr, 10) : 10000;
  size_t Q = argc > 4 ? std::strtoul(argv[4], nullptr, 10) : 2000;
  int threads = argc > 5 ? std::atoi(argv[5]) : 4;

  auto emb = make_onnx_embedder(argv[1], argv[2], threads);
  const size_t dim = emb->dim(), K = 10;
  CorpusGen corpus;
  std::printf("chaos end-to-end (REAL MiniLM)  N=%zu  dim=%zu  k=%zu  queries=%zu  threads=%d\n\n",
              N, dim, K, Q, threads);

  // Build both indexes from real embeddings of the corpus.
  FlatIndex flat(dim, N);
  HnswIndex hnsw(dim, HnswParams{}, N);
  std::vector<float> v(dim);
  uint64_t t0 = now_ns();
  for (size_t i = 0; i < N; ++i) {
    emb->embed(corpus.sentence(i).c_str(), v.data());
    flat.add(v.data());
    hnsw.add(v.data());
  }
  double build_s = (now_ns() - t0) / 1e9;
  std::printf("corpus build: %.1f s  (%.0f docs/s, real embed+index into both)\n", build_s, N / build_s);

  // Semantic sanity check: results should share the query's topic.
  {
    std::string qtext = corpus.sentence(N + 1);
    std::vector<float> q(dim);
    emb->embed(qtext.c_str(), q.data());
    std::vector<SearchHit> hits;
    flat.search(q.data(), 3, hits, nullptr, 0);
    std::printf("\nquery: \"%s\"\n", qtext.c_str());
    for (auto& h : hits)
      std::printf("  %.3f  %s\n", h.score, corpus.sentence(h.id).c_str());
    std::printf("\n");
  }

  // Precompute real query embeddings for the search-only measurements.
  std::vector<std::vector<float>> q(Q, std::vector<float>(dim));
  for (size_t i = 0; i < Q; ++i) {
    emb->embed(corpus.sentence(N + i).c_str(), q[i].data());
    l2_normalize(q[i].data(), dim);
  }

  std::vector<SearchHit> hits, truth;

  // 1. Embedding latency (real forward pass).
  {
    for (size_t i = 0; i < 200; ++i) emb->embed(corpus.sentence(i).c_str(), v.data());  // warm
    std::vector<uint64_t> s; s.reserve(Q);
    for (size_t i = 0; i < Q; ++i) {
      const std::string txt = corpus.sentence(N + i);
      uint64_t a = now_ns();
      emb->embed(txt.c_str(), v.data());
      s.push_back(now_ns() - a);
    }
    print_stats("embed", summarize(std::move(s)));
  }

  // 2. Search-only latency, per index (over real precomputed query vectors).
  auto search_bench = [&](const Index& idx, const char* label) {
    for (size_t i = 0; i < 500; ++i) idx.search(q[i % Q].data(), K, hits, nullptr, 0);  // warm
    std::vector<uint64_t> s; s.reserve(Q);
    for (size_t i = 0; i < Q; ++i) {
      uint64_t a = now_ns();
      idx.search(q[i].data(), K, hits, nullptr, 0);
      s.push_back(now_ns() - a);
    }
    print_stats(label, summarize(std::move(s)));
  };

  search_bench(flat, "search: flat (exact)");
  for (size_t ef : {32u, 64u, 128u}) {
    hnsw.set_ef_search(ef);
    size_t hit = 0, total = 0;
    for (size_t i = 0; i < Q; ++i) {
      hnsw.search(q[i].data(), K, hits, nullptr, 0);
      flat.search(q[i].data(), K, truth, nullptr, 0);
      std::unordered_set<uint32_t> gt;
      for (auto& h : truth) gt.insert(h.id);
      for (auto& h : hits) if (gt.count(h.id)) ++hit;
      total += truth.size();
    }
    char label[64];
    std::snprintf(label, sizeof label, "search: hnsw ef=%zu (r@10=%.3f)", ef, double(hit) / total);
    search_bench(hnsw, label);
  }

  // 3. Fused end-to-end: real embed + real HNSW search in one loop (ef=64).
  hnsw.set_ef_search(64);
  {
    std::vector<uint64_t> s; s.reserve(Q);
    for (size_t i = 0; i < Q; ++i) {
      const std::string txt = corpus.sentence(N + i);
      uint64_t a = now_ns();
      emb->embed(txt.c_str(), v.data());
      l2_normalize(v.data(), dim);
      hnsw.search(v.data(), K, hits, nullptr, 0);
      s.push_back(now_ns() - a);
    }
    std::printf("\n");
    print_stats("END-TO-END embed+hnsw", summarize(std::move(s)));
  }
  return 0;
}
