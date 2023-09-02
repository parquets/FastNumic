#pragma once

#include <memory.h>
#include <stdio.h>
#include <algorithm>
#include <cmath>

#include <immintrin.h>
#include <nmmintrin.h>
#include <smmintrin.h>

#include "vectorize_x86.hpp"

namespace fastnum {
namespace cpu {
namespace kernel {

inline void compare_max(float* C, const float* A, const float* B, int size) {
    int i = 0;
    for (i = 0; i < size - 31; i += 32) {
        __m256 _v_a0 = _mm256_loadu_ps(A + 0 * 8);
        __m256 _v_a1 = _mm256_loadu_ps(A + 1 * 8);
        __m256 _v_a2 = _mm256_loadu_ps(A + 2 * 8);
        __m256 _v_a3 = _mm256_loadu_ps(A + 3 * 8);

        __m256 _v_b0 = _mm256_loadu_ps(B + 0 * 8);
        __m256 _v_b1 = _mm256_loadu_ps(B + 1 * 8);
        __m256 _v_b2 = _mm256_loadu_ps(B + 2 * 8);
        __m256 _v_b3 = _mm256_loadu_ps(B + 3 * 8);

        __m256 _v_c0 = _mm256_max_ps(_v_a0, _v_b0);
        __m256 _v_c1 = _mm256_max_ps(_v_a1, _v_b1);
        __m256 _v_c2 = _mm256_max_ps(_v_a2, _v_b2);
        __m256 _v_c3 = _mm256_max_ps(_v_a3, _v_b3);

        _mm256_storeu_ps(C + 0 * 8, _v_c0);
        _mm256_storeu_ps(C + 1 * 8, _v_c1);
        _mm256_storeu_ps(C + 2 * 8, _v_c2);
        _mm256_storeu_ps(C + 3 * 8, _v_c3);

        A += 32;
        B += 32;
        C += 32;
    }
    for (; i < size - 15; i += 16) {
        __m256 _v_a0 = _mm256_loadu_ps(A + 0 * 8);
        __m256 _v_a1 = _mm256_loadu_ps(A + 1 * 8);

        __m256 _v_b0 = _mm256_loadu_ps(B + 0 * 8);
        __m256 _v_b1 = _mm256_loadu_ps(B + 1 * 8);

        __m256 _v_c0 = _mm256_max_ps(_v_a0, _v_b0);
        __m256 _v_c1 = _mm256_max_ps(_v_a1, _v_b1);

        _mm256_storeu_ps(C + 0 * 8, _v_c0);
        _mm256_storeu_ps(C + 1 * 8, _v_c1);

        A += 16;
        B += 16;
        C += 16;
    }
    for (; i < size - 7; i += 8) {
        __m256 _v_a0 = _mm256_loadu_ps(A + 0 * 8);
        __m256 _v_b0 = _mm256_loadu_ps(B + 0 * 8);
        __m256 _v_c0 = _mm256_max_ps(_v_a0, _v_b0);
        _mm256_storeu_ps(C + 0 * 8, _v_c0);

        A += 8;
        B += 8;
        C += 8;
    }

    for (; i < size; ++i) {
        *C = std::max(*A, *B);
        ++C;
        ++A;
        ++B;
    }
}

inline void compare_min(float* C, const float* A, const float* B, int size) {
    int i = 0;
    for (i = 0; i < size - 31; i += 32) {
        __m256 _v_a0 = _mm256_loadu_ps(A + 0 * 8);
        __m256 _v_a1 = _mm256_loadu_ps(A + 1 * 8);
        __m256 _v_a2 = _mm256_loadu_ps(A + 2 * 8);
        __m256 _v_a3 = _mm256_loadu_ps(A + 3 * 8);

        __m256 _v_b0 = _mm256_loadu_ps(B + 0 * 8);
        __m256 _v_b1 = _mm256_loadu_ps(B + 1 * 8);
        __m256 _v_b2 = _mm256_loadu_ps(B + 2 * 8);
        __m256 _v_b3 = _mm256_loadu_ps(B + 3 * 8);

        __m256 _v_c0 = _mm256_min_ps(_v_a0, _v_b0);
        __m256 _v_c1 = _mm256_min_ps(_v_a1, _v_b1);
        __m256 _v_c2 = _mm256_min_ps(_v_a2, _v_b2);
        __m256 _v_c3 = _mm256_min_ps(_v_a3, _v_b3);

        _mm256_storeu_ps(C + 0 * 8, _v_c0);
        _mm256_storeu_ps(C + 1 * 8, _v_c1);
        _mm256_storeu_ps(C + 2 * 8, _v_c2);
        _mm256_storeu_ps(C + 3 * 8, _v_c3);

        A += 32;
        B += 32;
        C += 32;
    }
    for (; i < size - 15; i += 16) {
        __m256 _v_a0 = _mm256_loadu_ps(A + 0 * 8);
        __m256 _v_a1 = _mm256_loadu_ps(A + 1 * 8);
        __m256 _v_b0 = _mm256_loadu_ps(B + 0 * 8);
        __m256 _v_b1 = _mm256_loadu_ps(B + 1 * 8);

        __m256 _v_c0 = _mm256_min_ps(_v_a0, _v_b0);
        __m256 _v_c1 = _mm256_min_ps(_v_a1, _v_b1);

        _mm256_storeu_ps(C + 0 * 8, _v_c0);
        _mm256_storeu_ps(C + 1 * 8, _v_c1);

        A += 16;
        B += 16;
        C += 16;
    }
    for (; i < size - 7; i += 8) {
        __m256 _v_a0 = _mm256_loadu_ps(A + 0 * 8);
        __m256 _v_b0 = _mm256_loadu_ps(B + 0 * 8);
        __m256 _v_c0 = _mm256_min_ps(_v_a0, _v_b0);
        _mm256_storeu_ps(C + 0 * 8, _v_c0);

        A += 8;
        B += 8;
        C += 8;
    }

    for (; i < size; ++i) {
        *C = std::min(*A, *B);
        ++C;
        ++A;
        ++B;
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace fastnum