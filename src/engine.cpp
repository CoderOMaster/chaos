#include "chaos/engine.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>

#include "chaos/distance.hpp"
#include "chaos/embedder.hpp"
#include "chaos/flat_index.hpp"
#include "chaos/hnsw_index.hpp"

namespace chaos {

namespace {
constexpr char kMagic[8] = {'C', 'H', 'A', 'O', 'S', 'I', 'X', '1'};

void write_str(std::ostream& os, const std::string& s) {
  uint32_t n = static_cast<uint32_t>(s.size());
  os.write(reinterpret_cast<const char*>(&n), sizeof n);
  os.write(s.data(), n);
}
std::string read_str(std::istream& is) {
  uint32_t n = 0;
  is.read(reinterpret_cast<char*>(&n), sizeof n);
  std::string s(n, '\0');
  if (n) is.read(&s[0], n);
  return s;
}
}  // namespace

Engine::Engine(const std::string& model_path, const std::string& vocab_path,
               const std::string& index_kind, int threads, size_t hnsw_m,
               size_t ef_construction, size_t ef_search)
    : is_hnsw_(index_kind == "hnsw") {
  emb_ = make_onnx_embedder(model_path.c_str(), vocab_path.c_str(), threads);
  dim_ = emb_->dim();
  if (index_kind == "flat") {
    index_.reset(new FlatIndex(dim_));
  } else if (index_kind == "hnsw") {
    HnswParams p;
    p.M = hnsw_m;
    p.ef_construction = ef_construction;
    p.ef_search = ef_search;
    index_.reset(new HnswIndex(dim_, p));
  } else {
    throw std::invalid_argument("index must be 'flat' or 'hnsw'");
  }
}

Engine::~Engine() = default;

bool Engine::upsert(const std::string& id, const std::string& text) {
  std::vector<float> v(dim_);
  emb_->embed(text.c_str(), v.data());  // index normalizes on add/update
  auto it = id_map_.find(id);
  if (it != id_map_.end()) {
    uint32_t internal = it->second;
    index_->update(internal, v.data());
    docs_[internal] = text;
    return true;  // updated
  }
  uint32_t internal = index_->add(v.data());
  id_map_.emplace(id, internal);
  ids_.push_back(id);   // internal == ids_.size() - 1
  docs_.push_back(text);
  return false;  // added
}

std::pair<size_t, size_t> Engine::upsert_many(const std::vector<std::string>& ids,
                                              const std::vector<std::string>& texts) {
  if (ids.size() != texts.size())
    throw std::invalid_argument("ids and texts must have the same length");
  size_t added = 0, updated = 0;
  for (size_t i = 0; i < ids.size(); ++i) {
    if (upsert(ids[i], texts[i])) ++updated;
    else ++added;
  }
  return {added, updated};
}

std::vector<Doc> Engine::search(const std::string& query, size_t k) const {
  std::vector<float> q(dim_);
  emb_->embed(query.c_str(), q.data());
  l2_normalize(q.data(), dim_);  // queries aren't inserted, so normalize here

  std::vector<SearchHit> hits;
  index_->search(q.data(), k, hits, nullptr, 0);

  std::vector<Doc> out;
  out.reserve(hits.size());
  for (const auto& h : hits) out.push_back({ids_[h.id], h.score, docs_[h.id]});
  return out;
}

size_t Engine::size() const { return index_->size(); }
size_t Engine::dim() const { return dim_; }

void Engine::set_ef_search(size_t ef) {
  if (is_hnsw_) static_cast<HnswIndex*>(index_.get())->set_ef_search(ef);
}

void Engine::save(const std::string& path) const {
  std::ofstream os(path, std::ios::binary);
  if (!os) throw std::runtime_error("chaos: cannot write index file: " + path);
  os.write(kMagic, sizeof kMagic);
  uint64_t dim = dim_, cnt = index_->size();
  uint8_t hn = is_hnsw_ ? 1 : 0;
  os.write(reinterpret_cast<const char*>(&dim), sizeof dim);
  os.write(reinterpret_cast<const char*>(&hn), sizeof hn);
  os.write(reinterpret_cast<const char*>(&cnt), sizeof cnt);
  for (uint64_t i = 0; i < cnt; ++i) {
    write_str(os, ids_[i]);
    write_str(os, docs_[i]);
  }
  index_->save(os);
}

void Engine::load(const std::string& path) {
  std::ifstream is(path, std::ios::binary);
  if (!is) throw std::runtime_error("chaos: cannot read index file: " + path);
  char magic[8];
  is.read(magic, sizeof magic);
  if (std::memcmp(magic, kMagic, sizeof kMagic) != 0)
    throw std::runtime_error("chaos: not a chaos index file: " + path);
  uint64_t dim = 0, cnt = 0;
  uint8_t hn = 0;
  is.read(reinterpret_cast<char*>(&dim), sizeof dim);
  is.read(reinterpret_cast<char*>(&hn), sizeof hn);
  is.read(reinterpret_cast<char*>(&cnt), sizeof cnt);
  if (dim != dim_)
    throw std::runtime_error("chaos: index dim mismatch (model changed?)");

  is_hnsw_ = hn != 0;
  if (is_hnsw_) index_.reset(new HnswIndex(dim_, HnswParams{}));
  else index_.reset(new FlatIndex(dim_));

  ids_.clear();
  docs_.clear();
  id_map_.clear();
  ids_.reserve(cnt);
  docs_.reserve(cnt);
  for (uint64_t i = 0; i < cnt; ++i) {
    std::string id = read_str(is);
    std::string text = read_str(is);
    id_map_.emplace(id, static_cast<uint32_t>(i));
    ids_.push_back(std::move(id));
    docs_.push_back(std::move(text));
  }
  index_->load(is);
}

}  // namespace chaos
