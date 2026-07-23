// High-level facade: text in, ranked documents out, keyed by caller-supplied
// string ids with upsert semantics. Owns the MiniLM embedder, an index (flat or
// HNSW), and the original document text. This is the surface the Python SDK
// binds to.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "chaos/index.hpp"

namespace chaos {

class Embedder;  // real MiniLM embedder (ONNX)

struct Doc {
  std::string id;    // caller-supplied id
  float score;       // cosine similarity, higher is better
  std::string text;  // the original document
};

class Engine {
 public:
  // index_kind: "flat" (exact) or "hnsw" (approximate, for large corpora).
  Engine(const std::string& model_path, const std::string& vocab_path,
         const std::string& index_kind = "flat", int threads = 4,
         size_t hnsw_m = 16, size_t ef_construction = 200, size_t ef_search = 64);
  ~Engine();

  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // Insert or replace one document by id. Returns true if it replaced an
  // existing id (update), false if it was newly inserted (add).
  bool upsert(const std::string& id, const std::string& text);

  // Batch upsert. Returns {added, updated} counts.
  std::pair<size_t, size_t> upsert_many(const std::vector<std::string>& ids,
                                        const std::vector<std::string>& texts);

  // Embed the query and return up to k documents, most similar first.
  std::vector<Doc> search(const std::string& query, size_t k = 10) const;

  size_t size() const;
  size_t dim() const;
  bool is_hnsw() const { return is_hnsw_; }
  void set_ef_search(size_t ef);  // HNSW recall/latency knob; no-op for flat

  // Persist the built index (ids + text + vectors, plus the HNSW graph) so it
  // can be reopened without re-embedding. `load` repopulates from such a file;
  // the query embedder is still needed to embed queries afterward.
  void save(const std::string& path) const;
  void load(const std::string& path);

 private:
  std::unique_ptr<Embedder> emb_;
  std::unique_ptr<Index> index_;
  std::vector<std::string> ids_;             // internal id -> string id
  std::vector<std::string> docs_;            // internal id -> text
  std::unordered_map<std::string, uint32_t> id_map_;  // string id -> internal id
  size_t dim_;
  bool is_hnsw_;
};

}  // namespace chaos
