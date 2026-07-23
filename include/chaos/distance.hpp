// SIMD distance kernels for the search hot path.
//
// Vectors are L2-normalized at insert time, so cosine similarity reduces to a
// plain dot product. We dispatch at compile time to the widest ISA available
// (ARM NEON on edge, AVX2 on x86) and fall back to an auto-vectorizable scalar
// loop otherwise.
#pragma once
#include <cstddef>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  #include <arm_neon.h>
  #define CHAOS_SIMD_NEON 1
#elif defined(__AVX2__)
  #include <immintrin.h>
  #define CHAOS_SIMD_AVX2 1
#endif

namespace chaos {

// Dot product of two float vectors of length n. With normalized inputs this is
// cosine similarity in [-1, 1]; larger is more similar.
inline float dot(const float* __restrict a, const float* __restrict b, size_t n) {
#if defined(CHAOS_SIMD_NEON)
  // Four independent accumulators hide FMA latency; 384 dims = 24 full lanes.
  float32x4_t acc0 = vdupq_n_f32(0.f), acc1 = vdupq_n_f32(0.f);
  float32x4_t acc2 = vdupq_n_f32(0.f), acc3 = vdupq_n_f32(0.f);
  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(a + i),      vld1q_f32(b + i));
    acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4),  vld1q_f32(b + i + 4));
    acc2 = vfmaq_f32(acc2, vld1q_f32(a + i + 8),  vld1q_f32(b + i + 8));
    acc3 = vfmaq_f32(acc3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
  }
  float32x4_t acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
  for (; i + 4 <= n; i += 4)
    acc = vfmaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
  float s = vaddvq_f32(acc);
  for (; i < n; ++i) s += a[i] * b[i];
  return s;
#elif defined(CHAOS_SIMD_AVX2)
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),     _mm256_loadu_ps(b + i),     acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
  }
  __m256 acc = _mm256_add_ps(acc0, acc1);
  __m128 lo = _mm256_castps256_ps128(acc);
  __m128 hi = _mm256_extractf128_ps(acc, 1);
  lo = _mm_add_ps(lo, hi);
  lo = _mm_hadd_ps(lo, lo);
  lo = _mm_hadd_ps(lo, lo);
  float s = _mm_cvtss_f32(lo);
  for (; i < n; ++i) s += a[i] * b[i];
  return s;
#else
  float s = 0.f;
  for (size_t i = 0; i < n; ++i) s += a[i] * b[i];  // auto-vectorized at -O3
  return s;
#endif
}

// In-place L2 normalization. Called once per vector at insert time so queries
// stay a single dot product.
inline void l2_normalize(float* v, size_t n) {
  float s = dot(v, v, n);
  if (s <= 0.f) return;
  float inv = 1.f / __builtin_sqrtf(s);
  for (size_t i = 0; i < n; ++i) v[i] *= inv;
}

}  // namespace chaos
