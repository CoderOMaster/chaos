// Latency measurement: monotonic nanosecond clock + percentile summary.
// p99/p999 are what matter for this project, so we keep every sample and
// report the tail explicitly rather than just a mean.
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace chaos {

inline uint64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

struct LatencyStats {
  double p50_ms, p90_ms, p99_ms, p999_ms, max_ms, mean_ms;
  size_t n;
};

// samples in nanoseconds.
inline LatencyStats summarize(std::vector<uint64_t> samples) {
  LatencyStats s{};
  s.n = samples.size();
  if (samples.empty()) return s;
  std::sort(samples.begin(), samples.end());
  auto pct = [&](double p) {
    size_t idx = static_cast<size_t>(p * (samples.size() - 1));
    return samples[idx] / 1e6;  // ns -> ms
  };
  double sum = 0;
  for (uint64_t v : samples) sum += v;
  s.mean_ms = (sum / samples.size()) / 1e6;
  s.p50_ms = pct(0.50);
  s.p90_ms = pct(0.90);
  s.p99_ms = pct(0.99);
  s.p999_ms = pct(0.999);
  s.max_ms = samples.back() / 1e6;
  return s;
}

inline void print_stats(const char* label, const LatencyStats& s) {
  std::printf("%-22s n=%-7zu  p50=%.3f  p90=%.3f  p99=%.3f  p999=%.3f  max=%.3f  mean=%.3f  (ms)\n",
              label, s.n, s.p50_ms, s.p90_ms, s.p99_ms, s.p999_ms, s.max_ms, s.mean_ms);
}

}  // namespace chaos
