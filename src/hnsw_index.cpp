#include "chaos/hnsw_index.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <istream>
#include <ostream>
#include <queue>

#include "chaos/distance.hpp"

namespace chaos {

HnswIndex::HnswIndex(size_t dim, HnswParams params, size_t reserve)
    : dim_(dim), params_(params), mL_(1.0 / std::log(double(params.M))),
      rng_(params.seed) {
  if (reserve) grow_data(reserve);
}

HnswIndex::~HnswIndex() { std::free(data_); }

float HnswIndex::sim(uint32_t a, uint32_t b) const { return dot(row(a), row(b), dim_); }
float HnswIndex::sim_q(const float* q, uint32_t b) const { return dot(q, row(b), dim_); }

void HnswIndex::grow_data(size_t new_cap) {
  if (new_cap <= cap_) return;
  size_t bytes = new_cap * dim_ * sizeof(float);
  float* next = static_cast<float*>(std::aligned_alloc(64, (bytes + 63) & ~size_t(63)));
  if (data_) { std::memcpy(next, data_, count_ * dim_ * sizeof(float)); std::free(data_); }
  data_ = next;
  cap_ = new_cap;
}

int HnswIndex::random_level() {
  std::uniform_real_distribution<double> u(0.0, 1.0);
  double r = u(rng_);
  if (r < 1e-12) r = 1e-12;
  return int(-std::log(r) * mL_);
}

// Descend greedily from the top layer to just above the target, taking the
// single best (most similar) neighbor at each step -- the ef=1 phase.
uint32_t HnswIndex::greedy_descent(const float* q, uint32_t entry, int top,
                                   int target_level) const {
  uint32_t cur = entry;
  float cur_sim = sim_q(q, cur);
  for (int lc = top; lc > target_level; --lc) {
    bool improved = true;
    while (improved) {
      improved = false;
      for (uint32_t nb : links_[cur][lc]) {
        float s = sim_q(q, nb);
        if (s > cur_sim) { cur_sim = s; cur = nb; improved = true; }
      }
    }
  }
  return cur;
}

// Beam search in one layer. `cand` is a max-heap on similarity (expand nearest
// first); `res` is a min-heap on similarity (holds the ef best, evict worst).
void HnswIndex::search_layer(const float* q, uint32_t entry, size_t ef, int layer,
                             std::vector<Cand>& out) const {
  ++vis_epoch_;
  if (vis_epoch_ == 0) { std::fill(vis_.begin(), vis_.end(), 0); vis_epoch_ = 1; }

  std::priority_queue<Cand> cand;                                       // max-heap
  std::priority_queue<Cand, std::vector<Cand>, std::greater<Cand>> res;  // min-heap

  float s0 = sim_q(q, entry);
  cand.push({s0, entry});
  res.push({s0, entry});
  vis_[entry] = vis_epoch_;

  while (!cand.empty()) {
    Cand c = cand.top();
    cand.pop();
    if (c.first < res.top().first && res.size() >= ef) break;  // worst-in-beam gate
    for (uint32_t nb : links_[c.second][layer]) {
      if (vis_[nb] == vis_epoch_) continue;
      vis_[nb] = vis_epoch_;
      float s = sim_q(q, nb);
      if (res.size() < ef || s > res.top().first) {
        cand.push({s, nb});
        res.push({s, nb});
        if (res.size() > ef) res.pop();
      }
    }
  }

  out.clear();
  out.reserve(res.size());
  while (!res.empty()) { out.push_back(res.top()); res.pop(); }  // ascending sim
}

// Keep a neighbor e only if it is more similar to `base` than to any neighbor
// already chosen -- spreads links out so the graph stays navigable.
void HnswIndex::select_neighbors(uint32_t base, std::vector<Cand>& cand, size_t M,
                                 std::vector<uint32_t>& out) const {
  (void)base;
  std::sort(cand.begin(), cand.end(), [](const Cand& a, const Cand& b) {
    return a.first > b.first;  // most similar to base first
  });
  out.clear();
  for (const auto& e : cand) {
    if (out.size() >= M) break;
    bool keep = true;
    for (uint32_t r : out) {
      if (sim(e.second, r) > e.first) { keep = false; break; }  // closer to r than to base
    }
    if (keep) out.push_back(e.second);
  }
}

uint32_t HnswIndex::add(const float* vec) {
  if (count_ == cap_) grow_data(cap_ ? cap_ * 2 : 1024);
  const uint32_t id = uint32_t(count_);
  float* dst = row(id);
  std::memcpy(dst, vec, dim_ * sizeof(float));
  l2_normalize(dst, dim_);

  const int level = random_level();
  node_level_.push_back(level);
  links_.emplace_back(level + 1);  // layers 0..level
  ++count_;
  if (vis_.size() < count_) vis_.resize(count_, 0);

  if (max_level_ < 0) {  // first node becomes the entry point
    entry_point_ = id;
    max_level_ = level;
    return id;
  }

  const size_t M0 = params_.M * 2;
  uint32_t entry = entry_point_;
  // Phase 1: descend from the top down to the node's own top layer.
  if (level < max_level_)
    entry = greedy_descent(dst, entry, max_level_, level);

  // Phase 2: connect at every layer from min(level, max_level_) down to 0.
  std::vector<Cand> found;
  std::vector<uint32_t> selected;
  int start = std::min(level, max_level_);
  for (int lc = start; lc >= 0; --lc) {
    search_layer(dst, entry, params_.ef_construction, lc, found);
    if (!found.empty()) entry = found.back().second;  // best (highest sim) seeds next layer

    const size_t Mmax = (lc == 0) ? M0 : params_.M;
    select_neighbors(id, found, params_.M, selected);

    links_[id][lc] = selected;
    for (uint32_t nb : selected) {
      auto& nlinks = links_[nb][lc];
      nlinks.push_back(id);
      if (nlinks.size() > Mmax) {  // prune the neighbor back down to Mmax
        std::vector<Cand> nc;
        nc.reserve(nlinks.size());
        for (uint32_t x : nlinks) nc.push_back({sim(nb, x), x});
        std::vector<uint32_t> pruned;
        select_neighbors(nb, nc, Mmax, pruned);
        nlinks.swap(pruned);
      }
    }
  }

  if (level > max_level_) { max_level_ = level; entry_point_ = id; }
  return id;
}

void HnswIndex::update(uint32_t id, const float* vec) {
  // In-place vector replacement; graph links are kept. Exact distances to this
  // node are recomputed on every visit, so results stay correct for the node
  // itself -- only its reachability can drift slightly since edges aren't
  // relinked. Fine for occasional edits; rebuild for a fully re-optimized graph.
  float* dst = row(id);
  std::memcpy(dst, vec, dim_ * sizeof(float));
  l2_normalize(dst, dim_);
}

void HnswIndex::save(std::ostream& os) const {
  auto w = [&](const void* p, size_t n) { os.write(static_cast<const char*>(p), n); };
  uint64_t n = count_;
  w(&n, sizeof n);
  w(data_, n * dim_ * sizeof(float));
  int32_t maxl = max_level_;
  uint32_t ep = entry_point_, efs = static_cast<uint32_t>(params_.ef_search);
  w(&maxl, sizeof maxl);
  w(&ep, sizeof ep);
  w(&efs, sizeof efs);
  for (uint64_t i = 0; i < n; ++i) {
    int32_t lv = node_level_[i];
    w(&lv, sizeof lv);
  }
  for (uint64_t i = 0; i < n; ++i) {
    for (int lc = 0; lc <= node_level_[i]; ++lc) {
      const auto& nb = links_[i][lc];
      uint32_t cnt = static_cast<uint32_t>(nb.size());
      w(&cnt, sizeof cnt);
      if (cnt) w(nb.data(), cnt * sizeof(uint32_t));
    }
  }
}

void HnswIndex::load(std::istream& is) {
  auto r = [&](void* p, size_t n) { is.read(static_cast<char*>(p), n); };
  uint64_t n = 0;
  r(&n, sizeof n);
  grow_data(n ? n : 1);
  r(data_, n * dim_ * sizeof(float));
  count_ = n;
  int32_t maxl = -1;
  uint32_t ep = 0, efs = 0;
  r(&maxl, sizeof maxl);
  r(&ep, sizeof ep);
  r(&efs, sizeof efs);
  max_level_ = maxl;
  entry_point_ = ep;
  params_.ef_search = efs;
  node_level_.resize(n);
  for (uint64_t i = 0; i < n; ++i) {
    int32_t lv = 0;
    r(&lv, sizeof lv);
    node_level_[i] = lv;
  }
  links_.assign(n, {});
  for (uint64_t i = 0; i < n; ++i) {
    links_[i].resize(node_level_[i] + 1);
    for (int lc = 0; lc <= node_level_[i]; ++lc) {
      uint32_t cnt = 0;
      r(&cnt, sizeof cnt);
      links_[i][lc].resize(cnt);
      if (cnt) r(links_[i][lc].data(), cnt * sizeof(uint32_t));
    }
  }
  vis_.assign(n, 0);
  vis_epoch_ = 0;
}

void HnswIndex::search(const float* query, size_t k, std::vector<SearchHit>& out,
                       void* /*scratch*/, size_t /*scratch_bytes*/) const {
  out.clear();
  if (count_ == 0 || k == 0) return;

  uint32_t entry = greedy_descent(query, entry_point_, max_level_, 0);
  std::vector<Cand> found;
  const size_t ef = std::max(params_.ef_search, k);
  search_layer(query, entry, ef, 0, found);

  // found is ascending by similarity; take the top-k from the end.
  std::sort(found.begin(), found.end(), [](const Cand& a, const Cand& b) {
    return a.first > b.first;
  });
  const size_t n = std::min(k, found.size());
  out.resize(n);
  for (size_t i = 0; i < n; ++i) out[i] = {found[i].second, found[i].first};
}

}  // namespace chaos
