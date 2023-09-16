#pragma once


#include <immintrin.h>
#include <math.h>
#include <memory.h>
#include <nmmintrin.h>
#include <smmintrin.h>
#include <stdio.h>

namespace fastnum {
namespace cpu {
namespace kernel {

inline float reduce_add_ps(const __m256& x) {
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    return _mm_cvtss_f32(x32);
}

inline float reduce_max_ps(const __m256& x) {
    const __m128 x128 = _mm_max_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    const __m128 x64 = _mm_max_ps(x128, _mm_movehl_ps(x128, x128));
    const __m128 x32 = _mm_max_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    return _mm_cvtss_f32(x32);
}

inline float reduce_min_ps(const __m256& x) {
    const __m128 x128 = _mm_min_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    const __m128 x64 = _mm_min_ps(x128, _mm_movehl_ps(x128, x128));
    const __m128 x32 = _mm_min_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    return _mm_cvtss_f32(x32);
}

inline static float reduce_add_ps(__m128 x128) {
	const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
	const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
	return _mm_cvtss_f32(x32);
}


}  // namespace kernel
}  // namespace cpu
}  // namespace fastnum