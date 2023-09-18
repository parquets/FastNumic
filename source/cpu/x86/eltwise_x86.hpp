#pragma once

#include <memory.h>
#include <stdio.h>
#include <cmath>

#include <immintrin.h>
#include <nmmintrin.h>
#include <smmintrin.h>

#include "vectorize_x86.hpp"

namespace fastnum {
namespace cpu {
namespace kernel {

inline void scale(float *C, float alpha, const float *A, float beta, int size) {
    int i = 0;

    __m256 _v_alpha = _mm256_broadcast_ss(&alpha);
    __m256 _v_beta = _mm256_broadcast_ss(&beta);

    for (i = 0; i < size - 7; i += 8) {
        __m256 _v_a = _mm256_loadu_ps(A);
        __m256 _v_c = _mm256_fmadd_ps(_v_alpha, _v_a, _v_beta);

        _mm256_storeu_ps(C, _v_c);

        A += 8;
        C += 8;
    }
    for (; i < size; ++i) {
        C[i] = alpha * A[i] + beta;
    }
}

inline void log10(float *A, const float *B, int size) {
    int i = 0;
    for (i = 0; i < size - 7; i += 8) {
        __m256 _v_b = _mm256_loadu_ps(B + i);
        __m256 _v_r = _mm256_log10_ps(_v_b);
        _mm256_storeu_ps(A + i, _v_r);
    }
    for (; i < size; ++i) {
        A[i] = std::log10(B[i]);
    }
}

inline void log2(float *A, const float *B, int size) {
    int i = 0;
    for (i = 0; i < size - 7; i += 8) {
        __m256 _v_b = _mm256_loadu_ps(B + i);
        __m256 _v_r = _mm256_log2_ps(_v_b);
        _mm256_storeu_ps(A + i, _v_r);
    }
    for (; i < size; ++i) {
        A[i] = std::log2(B[i]);
    }
}

inline void log(float *A, const float *B, int size) {
    int i = 0;
    for (i = 0; i < size - 7; i += 8) {
        __m256 _v_b = _mm256_loadu_ps(B + i);
        __m256 _v_r = _mm256_log_ps(_v_b);
        _mm256_storeu_ps(A + i, _v_r);
    }
    for (; i < size; ++i) {
        A[i] = std::log(B[i]);
    }
}

inline void exp10(float *A, const float *B, int size) {
    int i = 0;
    for (i = 0; i < size - 7; i += 8) {
        __m256 _v_b = _mm256_loadu_ps(B + i);
        __m256 _v_r = _mm256_exp10_ps(_v_b);
        _mm256_storeu_ps(A + i, _v_r);
    }
    for (; i < size; ++i) {
        A[i] = std::pow(10, B[i]);
    }
}

inline void exp2(float *A, const float *B, int size) {
    int i = 0;
    for (i = 0; i < size - 7; i += 8) {
        __m256 _v_b = _mm256_loadu_ps(B + i);
        __m256 _v_r = _mm256_exp2_ps(_v_b);
        _mm256_storeu_ps(A + i, _v_r);
    }
    for (; i < size; ++i) {
        A[i] = std::pow(2, B[i]);
    }
}

inline void exp(float *A, const float *B, int size) {
    int i = 0;
    for (i = 0; i < size - 7; i += 8) {
        __m256 _v_b = _mm256_loadu_ps(B + i);
        __m256 _v_r = _mm256_exp_ps(_v_b);
        _mm256_storeu_ps(A + i, _v_r);
    }
    for (; i < size; ++i) {
        A[i] = std::exp(B[i]);
    }
}

inline void relu(float *A, const float *B, int size) {
    int i;
    for (i = 0; i < size - 7; i += 8) {
        __m256 x = _mm256_loadu_ps(B);
        x = _mm256_max_ps(x, _mm256_set1_ps(0.0));
        _mm256_storeu_ps(A, x);
        A += 8;
        B += 8;
    }
    for (; i < size; ++i) {
        *A = *B > 0 ? *B : 0;
        ++A;
        ++B;
    }
}

inline void sigmoid(float *A, const float *B, int size) {
    int i;
    for (i = 0; i < size - 7; i += 8) {
        __m256 x = _mm256_loadu_ps(B);
        __m256 v = _mm256_add_ps(_mm256_set1_ps(1.0), _mm256_exp_ps(_mm256_mul_ps(x, _mm256_set1_ps(-1.0f))));
        x = _mm256_rcp_ps(v);
        _mm256_storeu_ps(A, x);
        A += 8;
        B += 8;
    }
    for (; i < size; ++i) {
        *A = 1.0f / (1.0f + std::exp(-1 * (B[0])));
        ++A;
        ++B;
    }
}

// inline void threshold_less(float* A, const float* B, int size) {

// }

}  // namespace kernel
}  // namespace cpu
}  // namespace fastnum