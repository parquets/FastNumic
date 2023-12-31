#pragma once

#include <math.h>
#include <memory.h>
#include <stdio.h>

#include <immintrin.h>
#include <nmmintrin.h>
#include <smmintrin.h>

#include "vectorize_x86.hpp"

namespace fastnum {
namespace cpu {
namespace kernel {

inline void smma_mxn(int M, int N, const float* packed_A, const float* packed_B, int K, float* C, int ldc) {
    for (int k = 0; k < K; ++k) {
        for (int m = 0; m < M; ++m) {
            float tmp = packed_A[m];
            float* c_ptr = C + m * ldc;
            int n = 0;
            for (; n < N; ++n) {
                c_ptr[n] += tmp * packed_B[n];
            }
        }
        packed_A += M;
        packed_B += N;
    }
}

inline void smma_4x4(const float* packed_A, const float* packed_B, int K, float* C, int ldc) {
    float* c_ptr0 = C + 0 * ldc;
    float* c_ptr1 = C + 1 * ldc;
    float* c_ptr2 = C + 2 * ldc;
    float* c_ptr3 = C + 3 * ldc;

    __m128 _v_c0 = _mm_loadu_ps(c_ptr0);
    __m128 _v_c1 = _mm_loadu_ps(c_ptr1);
    __m128 _v_c2 = _mm_loadu_ps(c_ptr2);
    __m128 _v_c3 = _mm_loadu_ps(c_ptr3);

    for (int k = 0; k < K; ++k) {
        __m128 _v_a0 = _mm_broadcast_ss(packed_A + 0);
        __m128 _v_a1 = _mm_broadcast_ss(packed_A + 1);
        __m128 _v_a2 = _mm_broadcast_ss(packed_A + 2);
        __m128 _v_a3 = _mm_broadcast_ss(packed_A + 3);

        __m128 _v_b = _mm_load_ps(packed_B);

        _v_c0 = _mm_fmadd_ps(_v_a0, _v_b, _v_c0);
        _v_c1 = _mm_fmadd_ps(_v_a1, _v_b, _v_c1);
        _v_c2 = _mm_fmadd_ps(_v_a2, _v_b, _v_c2);
        _v_c3 = _mm_fmadd_ps(_v_a3, _v_b, _v_c3);

        packed_A += 4;
        packed_B += 4;
    }

    _mm_storeu_ps(c_ptr0, _v_c0);
    _mm_storeu_ps(c_ptr1, _v_c1);
    _mm_storeu_ps(c_ptr2, _v_c2);
    _mm_storeu_ps(c_ptr3, _v_c3);
}

inline void smma_8x1(const float* packed_A, const float* packed_B, int K, float* C, int ldc) {
    __m256 _v_c0 = _mm256_setzero_ps();
    __m256 _v_c1 = _mm256_setzero_ps();
    __m256 _v_c2 = _mm256_setzero_ps();
    __m256 _v_c3 = _mm256_setzero_ps();

    int k = 0;

    for (k = 0; k < K - 3; k += 4) {
        __m256 _v_a0 = _mm256_load_ps(packed_A + 8 * 0);
        __m256 _v_a1 = _mm256_load_ps(packed_A + 8 * 1);
        __m256 _v_a2 = _mm256_load_ps(packed_A + 8 * 2);
        __m256 _v_a3 = _mm256_load_ps(packed_A + 8 * 3);

        __m256 _v_b0 = _mm256_broadcast_ss(packed_B + 0);
        __m256 _v_b1 = _mm256_broadcast_ss(packed_B + 1);
        __m256 _v_b2 = _mm256_broadcast_ss(packed_B + 2);
        __m256 _v_b3 = _mm256_broadcast_ss(packed_B + 3);

        _v_c0 = _mm256_fmadd_ps(_v_a0, _v_b0, _v_c0);
        _v_c1 = _mm256_fmadd_ps(_v_a1, _v_b1, _v_c1);
        _v_c2 = _mm256_fmadd_ps(_v_a2, _v_b2, _v_c2);
        _v_c3 = _mm256_fmadd_ps(_v_a3, _v_b3, _v_c3);

        packed_A += 32;
        packed_B += 4;
    }

    __m256 _v_c = _mm256_add_ps(_mm256_add_ps(_v_c0, _v_c1), _mm256_add_ps(_v_c2, _v_c3));

    for (; k < K; ++k) {
        __m256 _v_a = _mm256_load_ps(packed_A);
        __m256 _v_b = _mm256_broadcast_ss(packed_B);
        _v_c = _mm256_fmadd_ps(_v_a, _v_b, _v_c);
        packed_A += 8;
        packed_B += 1;
    }

    float c_data[8];
    _mm256_storeu_ps(c_data, _v_c);

    *(C + 0 * ldc) += c_data[0];
    *(C + 1 * ldc) += c_data[1];
    *(C + 2 * ldc) += c_data[2];
    *(C + 3 * ldc) += c_data[3];
    *(C + 4 * ldc) += c_data[4];
    *(C + 5 * ldc) += c_data[5];
    *(C + 6 * ldc) += c_data[6];
    *(C + 7 * ldc) += c_data[7];
}

inline void smma_4x1(const float* packed_A, const float* packed_B, int K, float* C, int ldc) {
    __m128 _v_c0 = _mm_setzero_ps();
    __m128 _v_c1 = _mm_setzero_ps();
    __m128 _v_c2 = _mm_setzero_ps();
    __m128 _v_c3 = _mm_setzero_ps();

    int k = 0;

    for (k = 0; k < K - 3; k += 4) {
        __m128 _v_a0 = _mm_load_ps(packed_A + 4 * 0);
        __m128 _v_a1 = _mm_load_ps(packed_A + 4 * 1);
        __m128 _v_a2 = _mm_load_ps(packed_A + 4 * 2);
        __m128 _v_a3 = _mm_load_ps(packed_A + 4 * 3);

        __m128 _v_b0 = _mm_broadcast_ss(packed_B + 0);
        __m128 _v_b1 = _mm_broadcast_ss(packed_B + 1);
        __m128 _v_b2 = _mm_broadcast_ss(packed_B + 2);
        __m128 _v_b3 = _mm_broadcast_ss(packed_B + 3);

        _v_c0 = _mm_fmadd_ps(_v_a0, _v_b0, _v_c0);
        _v_c1 = _mm_fmadd_ps(_v_a1, _v_b1, _v_c1);
        _v_c2 = _mm_fmadd_ps(_v_a2, _v_b2, _v_c2);
        _v_c3 = _mm_fmadd_ps(_v_a3, _v_b3, _v_c3);

        packed_A += 16;
        packed_B += 4;
    }

    __m128 _v_c = _mm_add_ps(_mm_add_ps(_v_c0, _v_c1), _mm_add_ps(_v_c2, _v_c3));

    for (; k < K; ++k) {
        __m128 _v_a = _mm_load_ps(packed_A);
        __m128 _v_b = _mm_broadcast_ss(packed_B);
        _v_c = _mm_fmadd_ps(_v_a, _v_b, _v_c);
        packed_A += 4;
        packed_B += 1;
    }

    float c_data[4];
    _mm_storeu_ps(c_data, _v_c);

    *(C + 0 * ldc) += c_data[0];
    *(C + 1 * ldc) += c_data[1];
    *(C + 2 * ldc) += c_data[2];
    *(C + 3 * ldc) += c_data[3];
}

inline void smma_8x8(const float* packed_A, const float* packed_B, int K, float* C, int ldc) {
    float* c0_ptr = C + 0 * ldc;
    float* c1_ptr = C + 1 * ldc;
    float* c2_ptr = C + 2 * ldc;
    float* c3_ptr = C + 3 * ldc;
    float* c4_ptr = C + 4 * ldc;
    float* c5_ptr = C + 5 * ldc;
    float* c6_ptr = C + 6 * ldc;
    float* c7_ptr = C + 7 * ldc;

    __m256 _vc0 = _mm256_loadu_ps(c0_ptr);
    __m256 _vc1 = _mm256_loadu_ps(c1_ptr);
    __m256 _vc2 = _mm256_loadu_ps(c2_ptr);
    __m256 _vc3 = _mm256_loadu_ps(c3_ptr);
    __m256 _vc4 = _mm256_loadu_ps(c4_ptr);
    __m256 _vc5 = _mm256_loadu_ps(c5_ptr);
    __m256 _vc6 = _mm256_loadu_ps(c6_ptr);
    __m256 _vc7 = _mm256_loadu_ps(c7_ptr);

    __m256 _vb;
    __m256 _va0, _va1, _va2, _va3;

    for (int k = 0; k < K; ++k) {
        _vb = _mm256_load_ps(packed_B);

        _va0 = _mm256_broadcast_ss(packed_A + 0);
        _va1 = _mm256_broadcast_ss(packed_A + 1);
        _va2 = _mm256_broadcast_ss(packed_A + 2);
        _va3 = _mm256_broadcast_ss(packed_A + 3);

        _vc0 = _mm256_fmadd_ps(_va0, _vb, _vc0);
        _vc1 = _mm256_fmadd_ps(_va1, _vb, _vc1);
        _vc2 = _mm256_fmadd_ps(_va2, _vb, _vc2);
        _vc3 = _mm256_fmadd_ps(_va3, _vb, _vc3);

        _va0 = _mm256_broadcast_ss(packed_A + 4);
        _va1 = _mm256_broadcast_ss(packed_A + 5);
        _va2 = _mm256_broadcast_ss(packed_A + 6);
        _va3 = _mm256_broadcast_ss(packed_A + 7);

        _vc4 = _mm256_fmadd_ps(_va0, _vb, _vc4);
        _vc5 = _mm256_fmadd_ps(_va1, _vb, _vc5);
        _vc6 = _mm256_fmadd_ps(_va2, _vb, _vc6);
        _vc7 = _mm256_fmadd_ps(_va3, _vb, _vc7);

        packed_A += 8;
        packed_B += 8;
    }

    _mm256_storeu_ps(c0_ptr, _vc0);
    _mm256_storeu_ps(c1_ptr, _vc1);
    _mm256_storeu_ps(c2_ptr, _vc2);
    _mm256_storeu_ps(c3_ptr, _vc3);
    _mm256_storeu_ps(c4_ptr, _vc4);
    _mm256_storeu_ps(c5_ptr, _vc5);
    _mm256_storeu_ps(c6_ptr, _vc6);
    _mm256_storeu_ps(c7_ptr, _vc7);
}

inline void smma_6x16(const float* packed_A, const float* packed_B, int K, float* C, int ldc) {
    __m256 _v_c00 = _mm256_loadu_ps(C);
    __m256 _v_c01 = _mm256_loadu_ps(C + 8);
    __m256 _v_c10 = _mm256_loadu_ps(C + ldc);
    __m256 _v_c11 = _mm256_loadu_ps(C + ldc + 8);
    __m256 _v_c20 = _mm256_loadu_ps(C + 2 * ldc);
    __m256 _v_c21 = _mm256_loadu_ps(C + 2 * ldc + 8);
    __m256 _v_c30 = _mm256_loadu_ps(C + 3 * ldc);
    __m256 _v_c31 = _mm256_loadu_ps(C + 3 * ldc + 8);
    __m256 _v_c40 = _mm256_loadu_ps(C + 4 * ldc);
    __m256 _v_c41 = _mm256_loadu_ps(C + 4 * ldc + 8);
    __m256 _v_c50 = _mm256_loadu_ps(C + 5 * ldc);
    __m256 _v_c51 = _mm256_loadu_ps(C + 5 * ldc + 8);

    __m256 _v_b0, _v_b1, _v_a;

    for (int i = 0; i < K; ++i) {
        _v_b0 = _mm256_load_ps(packed_B);
        _v_b1 = _mm256_load_ps(packed_B + 8);

        _v_a = _mm256_broadcast_ss(packed_A + 0);
        _v_c00 = _mm256_fmadd_ps(_v_a, _v_b0, _v_c00);
        _v_c01 = _mm256_fmadd_ps(_v_a, _v_b1, _v_c01);

        _v_a = _mm256_broadcast_ss(packed_A + 1);
        _v_c10 = _mm256_fmadd_ps(_v_a, _v_b0, _v_c10);
        _v_c11 = _mm256_fmadd_ps(_v_a, _v_b1, _v_c11);

        _v_a = _mm256_broadcast_ss(packed_A + 2);
        _v_c20 = _mm256_fmadd_ps(_v_a, _v_b0, _v_c20);
        _v_c21 = _mm256_fmadd_ps(_v_a, _v_b1, _v_c21);

        _v_a = _mm256_broadcast_ss(packed_A + 3);
        _v_c30 = _mm256_fmadd_ps(_v_a, _v_b0, _v_c30);
        _v_c31 = _mm256_fmadd_ps(_v_a, _v_b1, _v_c31);

        _v_a = _mm256_broadcast_ss(packed_A + 4);
        _v_c40 = _mm256_fmadd_ps(_v_a, _v_b0, _v_c40);
        _v_c41 = _mm256_fmadd_ps(_v_a, _v_b1, _v_c41);

        _v_a = _mm256_broadcast_ss(packed_A + 5);
        _v_c50 = _mm256_fmadd_ps(_v_a, _v_b0, _v_c50);
        _v_c51 = _mm256_fmadd_ps(_v_a, _v_b1, _v_c51);

        packed_A += 6;
        packed_B += 16;
    }

    _mm256_storeu_ps(C, _v_c00);
    _mm256_storeu_ps(C + 8, _v_c01);
    _mm256_storeu_ps(C + ldc, _v_c10);
    _mm256_storeu_ps(C + ldc + 8, _v_c11);
    _mm256_storeu_ps(C + 2 * ldc, _v_c20);
    _mm256_storeu_ps(C + 2 * ldc + 8, _v_c21);
    _mm256_storeu_ps(C + 3 * ldc, _v_c30);
    _mm256_storeu_ps(C + 3 * ldc + 8, _v_c31);
    _mm256_storeu_ps(C + 4 * ldc, _v_c40);
    _mm256_storeu_ps(C + 4 * ldc + 8, _v_c41);
    _mm256_storeu_ps(C + 5 * ldc, _v_c50);
    _mm256_storeu_ps(C + 5 * ldc + 8, _v_c51);
}

inline void smma_6x8(const float* packed_A, const float* packed_B, int K, float* C, int ldc) {
    __m256 _v_c0 = _mm256_loadu_ps(C + 0 * ldc);
    __m256 _v_c1 = _mm256_loadu_ps(C + 1 * ldc);
    __m256 _v_c2 = _mm256_loadu_ps(C + 2 * ldc);
    __m256 _v_c3 = _mm256_loadu_ps(C + 3 * ldc);
    __m256 _v_c4 = _mm256_loadu_ps(C + 4 * ldc);
    __m256 _v_c5 = _mm256_loadu_ps(C + 5 * ldc);

    for (int i = 0; i < K; ++i) {
        __m256 _v_b0 = _mm256_load_ps(packed_B);

        __m256 _v_a0 = _mm256_broadcast_ss(packed_A + 0);
        __m256 _v_a1 = _mm256_broadcast_ss(packed_A + 1);
        __m256 _v_a2 = _mm256_broadcast_ss(packed_A + 2);
        __m256 _v_a3 = _mm256_broadcast_ss(packed_A + 3);
        __m256 _v_a4 = _mm256_broadcast_ss(packed_A + 4);
        __m256 _v_a5 = _mm256_broadcast_ss(packed_A + 5);

        _v_c0 = _mm256_fmadd_ps(_v_a0, _v_b0, _v_c0);
        _v_c1 = _mm256_fmadd_ps(_v_a1, _v_b0, _v_c1);
        _v_c2 = _mm256_fmadd_ps(_v_a2, _v_b0, _v_c2);
        _v_c3 = _mm256_fmadd_ps(_v_a3, _v_b0, _v_c3);
        _v_c4 = _mm256_fmadd_ps(_v_a4, _v_b0, _v_c4);
        _v_c5 = _mm256_fmadd_ps(_v_a5, _v_b0, _v_c5);

        packed_A += 6;
        packed_B += 8;
    }

    _mm256_storeu_ps(C + 0 * ldc, _v_c0);
    _mm256_storeu_ps(C + 1 * ldc, _v_c1);
    _mm256_storeu_ps(C + 2 * ldc, _v_c2);
    _mm256_storeu_ps(C + 3 * ldc, _v_c3);
    _mm256_storeu_ps(C + 4 * ldc, _v_c4);
    _mm256_storeu_ps(C + 5 * ldc, _v_c5);
}

inline void smma_6x4(const float* packed_A, const float* packed_B, int K, float* C, int ldc) {
    __m128 _v_c0 = _mm_loadu_ps(C + 0 * ldc);
    __m128 _v_c1 = _mm_loadu_ps(C + 1 * ldc);
    __m128 _v_c2 = _mm_loadu_ps(C + 2 * ldc);
    __m128 _v_c3 = _mm_loadu_ps(C + 3 * ldc);
    __m128 _v_c4 = _mm_loadu_ps(C + 4 * ldc);
    __m128 _v_c5 = _mm_loadu_ps(C + 5 * ldc);

    for (int i = 0; i < K; ++i) {
        __m128 _v_b0 = _mm_load_ps(packed_B);

        __m128 _v_a0 = _mm_broadcast_ss(packed_A + 0);
        __m128 _v_a1 = _mm_broadcast_ss(packed_A + 1);
        __m128 _v_a2 = _mm_broadcast_ss(packed_A + 2);
        __m128 _v_a3 = _mm_broadcast_ss(packed_A + 3);
        __m128 _v_a4 = _mm_broadcast_ss(packed_A + 4);
        __m128 _v_a5 = _mm_broadcast_ss(packed_A + 5);

        _v_c0 = _mm_fmadd_ps(_v_a0, _v_b0, _v_c0);
        _v_c1 = _mm_fmadd_ps(_v_a1, _v_b0, _v_c1);
        _v_c2 = _mm_fmadd_ps(_v_a2, _v_b0, _v_c2);
        _v_c3 = _mm_fmadd_ps(_v_a3, _v_b0, _v_c3);
        _v_c4 = _mm_fmadd_ps(_v_a4, _v_b0, _v_c4);
        _v_c5 = _mm_fmadd_ps(_v_a5, _v_b0, _v_c5);

        packed_A += 6;
        packed_B += 4;
    }

    _mm_storeu_ps(C + 0 * ldc, _v_c0);
    _mm_storeu_ps(C + 1 * ldc, _v_c1);
    _mm_storeu_ps(C + 2 * ldc, _v_c2);
    _mm_storeu_ps(C + 3 * ldc, _v_c3);
    _mm_storeu_ps(C + 4 * ldc, _v_c4);
    _mm_storeu_ps(C + 5 * ldc, _v_c5);
}

inline void smma_6x1(const float* packed_A, const float* packed_B, int K, float* C, int ldc) {
    float c0 = 0;
    float c1 = 0;
    float c2 = 0;
    float c3 = 0;
    float c4 = 0;
    float c5 = 0;
    float c6 = 0;

    for (int k = 0; k < K; ++k) {
        float a0 = packed_A[0];
        float a1 = packed_A[1];
        float a2 = packed_A[2];
        float a3 = packed_A[3];
        float a4 = packed_A[4];
        float a5 = packed_A[5];

        float b = packed_B[0];

        c0 += a0 * b;
        c1 += a1 * b;
        c2 += a2 * b;
        c3 += a3 * b;
        c4 += a4 * b;
        c5 += a5 * b;

        packed_A += 6;
        packed_B += 1;
    }
    *(C + 0 * ldc) += c0;
    *(C + 1 * ldc) += c1;
    *(C + 2 * ldc) += c2;
    *(C + 3 * ldc) += c3;
    *(C + 4 * ldc) += c4;
    *(C + 5 * ldc) += c5;
}


inline void smma_4x16(const float* packed_A, const float* packed_B, int K, float* C, int ldc) {
    __m256 _v_c00 = _mm256_loadu_ps(C + 0 * ldc + 0);
    __m256 _v_c01 = _mm256_loadu_ps(C + 0 * ldc + 8);
    __m256 _v_c10 = _mm256_loadu_ps(C + 1 * ldc + 0);
    __m256 _v_c11 = _mm256_loadu_ps(C + 1 * ldc + 8);
    __m256 _v_c20 = _mm256_loadu_ps(C + 2 * ldc + 0);
    __m256 _v_c21 = _mm256_loadu_ps(C + 2 * ldc + 8);
    __m256 _v_c30 = _mm256_loadu_ps(C + 3 * ldc + 0);
    __m256 _v_c31 = _mm256_loadu_ps(C + 3 * ldc + 8);

    for (int k = 0; k < K; ++k) {
        __m256 _v_b0 = _mm256_load_ps(packed_B + 0);
        __m256 _v_b1 = _mm256_load_ps(packed_B + 8);

        __m256 _v_a0 = _mm256_broadcast_ss(packed_A + 0);
        __m256 _v_a1 = _mm256_broadcast_ss(packed_A + 1);
        __m256 _v_a2 = _mm256_broadcast_ss(packed_A + 2);
        __m256 _v_a3 = _mm256_broadcast_ss(packed_A + 3);

        _v_c00 = _mm256_fmadd_ps(_v_a0, _v_b0, _v_c00);
        _v_c01 = _mm256_fmadd_ps(_v_a0, _v_b1, _v_c01);
        _v_c10 = _mm256_fmadd_ps(_v_a1, _v_b0, _v_c10);
        _v_c11 = _mm256_fmadd_ps(_v_a1, _v_b1, _v_c11);
        _v_c20 = _mm256_fmadd_ps(_v_a2, _v_b0, _v_c20);
        _v_c21 = _mm256_fmadd_ps(_v_a2, _v_b1, _v_c21);
        _v_c30 = _mm256_fmadd_ps(_v_a3, _v_b0, _v_c30);
        _v_c31 = _mm256_fmadd_ps(_v_a3, _v_b1, _v_c31);

        packed_A += 4;
        packed_B += 16;
    }

    _mm256_storeu_ps(C + 0 * ldc + 0, _v_c00);
    _mm256_storeu_ps(C + 0 * ldc + 8, _v_c01);
    _mm256_storeu_ps(C + 1 * ldc + 0, _v_c10);
    _mm256_storeu_ps(C + 1 * ldc + 8, _v_c11);
    _mm256_storeu_ps(C + 2 * ldc + 0, _v_c20);
    _mm256_storeu_ps(C + 2 * ldc + 8, _v_c21);
    _mm256_storeu_ps(C + 3 * ldc + 0, _v_c30);
    _mm256_storeu_ps(C + 3 * ldc + 8, _v_c31);
}

inline void smma_1x16(const float* packed_A, const float* packed_B, int K, float* C, int ldc) {
    int k = 0;

    __m256 _v_c00 = _mm256_loadu_ps(C + 0);
    __m256 _v_c01 = _mm256_loadu_ps(C + 8);

    __m256 _v_c10 = _mm256_setzero_ps();
    __m256 _v_c11 = _mm256_setzero_ps();

    for (k = 0; k < K - 1; k += 2) {
        __m256 _v_a0 = _mm256_broadcast_ss(packed_A + 0);
        __m256 _v_a1 = _mm256_broadcast_ss(packed_A + 1);

        __m256 _v_b00 = _mm256_load_ps(packed_B + 16 * 0 + 0);
        __m256 _v_b01 = _mm256_load_ps(packed_B + 16 * 0 + 8);
        __m256 _v_b10 = _mm256_load_ps(packed_B + 16 * 1 + 0);
        __m256 _v_b11 = _mm256_load_ps(packed_B + 16 * 1 + 8);

        _v_c00 = _mm256_fmadd_ps(_v_a0, _v_b00, _v_c00);
        _v_c01 = _mm256_fmadd_ps(_v_a0, _v_b01, _v_c01);
        _v_c10 = _mm256_fmadd_ps(_v_a1, _v_b10, _v_c10);
        _v_c11 = _mm256_fmadd_ps(_v_a1, _v_b11, _v_c11);

        packed_A += 2;
        packed_B += 32;
    }

    __m256 _v_c0 = _mm256_add_ps(_v_c00, _v_c10);
    __m256 _v_c1 = _mm256_add_ps(_v_c01, _v_c11);

    for (; k < K; ++k) {
        __m256 _v_a0 = _mm256_broadcast_ss(packed_A + 0);
        __m256 _v_b0 = _mm256_load_ps(packed_B + 16 * 0 + 0);
        __m256 _v_b1 = _mm256_load_ps(packed_B + 16 * 0 + 8);

        _v_c0 = _mm256_fmadd_ps(_v_a0, _v_b0, _v_c0);
        _v_c1 = _mm256_fmadd_ps(_v_a0, _v_b1, _v_c1);

        ++packed_A;
        packed_B += 16;
    }
    _mm256_storeu_ps(C + 0, _v_c0);
    _mm256_storeu_ps(C + 8, _v_c1);

}

inline void smma_4x8(const float* packed_A, const float* packed_B, int K, float* C, int ldc) {
    __m256 _v_c0 = _mm256_loadu_ps(C + 0 * ldc);
    __m256 _v_c1 = _mm256_loadu_ps(C + 1 * ldc);
    __m256 _v_c2 = _mm256_loadu_ps(C + 2 * ldc);
    __m256 _v_c3 = _mm256_loadu_ps(C + 3 * ldc);

    for (int i = 0; i < K; ++i) {
        __m256 _v_b0 = _mm256_load_ps(packed_B);

        __m256 _v_a0 = _mm256_broadcast_ss(packed_A + 0);
        __m256 _v_a1 = _mm256_broadcast_ss(packed_A + 1);
        __m256 _v_a2 = _mm256_broadcast_ss(packed_A + 2);
        __m256 _v_a3 = _mm256_broadcast_ss(packed_A + 3);

        _v_c0 = _mm256_fmadd_ps(_v_a0, _v_b0, _v_c0);
        _v_c1 = _mm256_fmadd_ps(_v_a1, _v_b0, _v_c1);
        _v_c2 = _mm256_fmadd_ps(_v_a2, _v_b0, _v_c2);
        _v_c3 = _mm256_fmadd_ps(_v_a3, _v_b0, _v_c3);

        packed_A += 4;
        packed_B += 8;
    }

    _mm256_storeu_ps(C + 0 * ldc, _v_c0);
    _mm256_storeu_ps(C + 1 * ldc, _v_c1);
    _mm256_storeu_ps(C + 2 * ldc, _v_c2);
    _mm256_storeu_ps(C + 3 * ldc, _v_c3);
}

inline void smma_16x6(const float* packed_A, const float* packed_B, int K, float* C, int ldc) {
    __m256 _v_c00 = _mm256_loadu_ps(C);
    __m256 _v_c01 = _mm256_loadu_ps(C + 8);
    __m256 _v_c10 = _mm256_loadu_ps(C + ldc);
    __m256 _v_c11 = _mm256_loadu_ps(C + ldc + 8);
    __m256 _v_c20 = _mm256_loadu_ps(C + 2 * ldc);
    __m256 _v_c21 = _mm256_loadu_ps(C + 2 * ldc + 8);
    __m256 _v_c30 = _mm256_loadu_ps(C + 3 * ldc);
    __m256 _v_c31 = _mm256_loadu_ps(C + 3 * ldc + 8);
    __m256 _v_c40 = _mm256_loadu_ps(C + 4 * ldc);
    __m256 _v_c41 = _mm256_loadu_ps(C + 4 * ldc + 8);
    __m256 _v_c50 = _mm256_loadu_ps(C + 5 * ldc);
    __m256 _v_c51 = _mm256_loadu_ps(C + 5 * ldc + 8);

    __m256 _v_a0, _v_a1, _v_b;

    for (int k = 0; k < K; ++k) {
        _v_a0 = _mm256_load_ps(packed_A);
        _v_a1 = _mm256_load_ps(packed_A + 8);

        _v_b = _mm256_broadcast_ss(packed_B + 0);
        _v_c00 = _mm256_fmadd_ps(_v_b, _v_a0, _v_c00);
        _v_c01 = _mm256_fmadd_ps(_v_b, _v_a1, _v_c01);

        _v_b = _mm256_broadcast_ss(packed_B + 1);
        _v_c10 = _mm256_fmadd_ps(_v_b, _v_a0, _v_c10);
        _v_c11 = _mm256_fmadd_ps(_v_b, _v_a1, _v_c11);

        _v_b = _mm256_broadcast_ss(packed_B + 2);
        _v_c20 = _mm256_fmadd_ps(_v_b, _v_a0, _v_c20);
        _v_c21 = _mm256_fmadd_ps(_v_b, _v_a1, _v_c21);

        _v_b = _mm256_broadcast_ss(packed_B + 3);
        _v_c30 = _mm256_fmadd_ps(_v_b, _v_a0, _v_c30);
        _v_c31 = _mm256_fmadd_ps(_v_b, _v_a1, _v_c31);

        _v_b = _mm256_broadcast_ss(packed_B + 4);
        _v_c40 = _mm256_fmadd_ps(_v_b, _v_a0, _v_c40);
        _v_c41 = _mm256_fmadd_ps(_v_b, _v_a1, _v_c41);

        _v_b = _mm256_broadcast_ss(packed_B + 5);
        _v_c50 = _mm256_fmadd_ps(_v_b, _v_a0, _v_c50);
        _v_c51 = _mm256_fmadd_ps(_v_b, _v_a1, _v_c51);

        packed_A += 16;
        packed_B += 6;
    }

    _mm256_storeu_ps(C, _v_c00);
    _mm256_storeu_ps(C + 8, _v_c01);
    _mm256_storeu_ps(C + ldc, _v_c10);
    _mm256_storeu_ps(C + ldc + 8, _v_c11);
    _mm256_storeu_ps(C + 2 * ldc, _v_c20);
    _mm256_storeu_ps(C + 2 * ldc + 8, _v_c21);
    _mm256_storeu_ps(C + 3 * ldc, _v_c30);
    _mm256_storeu_ps(C + 3 * ldc + 8, _v_c31);
    _mm256_storeu_ps(C + 4 * ldc, _v_c40);
    _mm256_storeu_ps(C + 4 * ldc + 8, _v_c41);
    _mm256_storeu_ps(C + 5 * ldc, _v_c40);
    _mm256_storeu_ps(C + 5 * ldc + 8, _v_c41);
}

inline void smma_1x8(const float* packed_A, const float* packed_B, int K, float* C, int ldc) {
    int k = 0;

    __m256 _v_c0 = _mm256_loadu_ps(C);
    __m256 _v_c1 = _mm256_setzero_ps();
    __m256 _v_c2 = _mm256_setzero_ps();
    __m256 _v_c3 = _mm256_setzero_ps();

    for (k = 0; k < K - 3; k += 4) {
        __m256 _v_a0 = _mm256_broadcast_ss(packed_A + 0);
        __m256 _v_a1 = _mm256_broadcast_ss(packed_A + 1);
        __m256 _v_a2 = _mm256_broadcast_ss(packed_A + 2);
        __m256 _v_a3 = _mm256_broadcast_ss(packed_A + 3);

        __m256 _v_b0 = _mm256_load_ps(packed_B + 8 * 0);
        __m256 _v_b1 = _mm256_load_ps(packed_B + 8 * 1);
        __m256 _v_b2 = _mm256_load_ps(packed_B + 8 * 2);
        __m256 _v_b3 = _mm256_load_ps(packed_B + 8 * 3);

        _v_c0 = _mm256_fmadd_ps(_v_a0, _v_b0, _v_c0);
        _v_c1 = _mm256_fmadd_ps(_v_a1, _v_b1, _v_c1);
        _v_c2 = _mm256_fmadd_ps(_v_a2, _v_b2, _v_c2);
        _v_c3 = _mm256_fmadd_ps(_v_a3, _v_b3, _v_c3);

        packed_A += 4;
        packed_B += 32;
    }

    __m256 _v_c = _mm256_add_ps(_mm256_add_ps(_v_c0, _v_c1), _mm256_add_ps(_v_c2, _v_c3));

    for (; k < K; ++k) {
        __m256 _v_a = _mm256_broadcast_ss(packed_A + 0);
        __m256 _v_b = _mm256_load_ps(packed_B + 8 * 0);

        _v_c = _mm256_fmadd_ps(_v_a, _v_b, _v_c);

        ++packed_A;
        packed_B += 8;
    }
    _mm256_storeu_ps(C, _v_c);
}

inline void smma_1x4(const float* packed_A, const float* packed_B, int K, float* C, int ldc) {
    int k = 0;

    __m128 _v_c0 = _mm_loadu_ps(C);
    __m128 _v_c1 = _mm_setzero_ps();
    __m128 _v_c2 = _mm_setzero_ps();
    __m128 _v_c3 = _mm_setzero_ps();

    for (k = 0; k < K - 3; k += 4) {
        __m128 _v_a0 = _mm_broadcast_ss(packed_A + 0);
        __m128 _v_a1 = _mm_broadcast_ss(packed_A + 1);
        __m128 _v_a2 = _mm_broadcast_ss(packed_A + 2);
        __m128 _v_a3 = _mm_broadcast_ss(packed_A + 3);

        __m128 _v_b0 = _mm_load_ps(packed_B + 4 * 0);
        __m128 _v_b1 = _mm_load_ps(packed_B + 4 * 1);
        __m128 _v_b2 = _mm_load_ps(packed_B + 4 * 2);
        __m128 _v_b3 = _mm_load_ps(packed_B + 4 * 3);

        _v_c0 = _mm_fmadd_ps(_v_a0, _v_b0, _v_c0);
        _v_c1 = _mm_fmadd_ps(_v_a1, _v_b1, _v_c1);
        _v_c2 = _mm_fmadd_ps(_v_a2, _v_b2, _v_c2);
        _v_c3 = _mm_fmadd_ps(_v_a3, _v_b3, _v_c3);

        packed_A += 4;
        packed_B += 16;
    }

    __m128 _v_c = _mm_add_ps(_mm_add_ps(_v_c0, _v_c1), _mm_add_ps(_v_c2, _v_c3));

    for (; k < K; ++k) {
        __m128 _v_a = _mm_broadcast_ss(packed_A);
        __m128 _v_b = _mm_load_ps(packed_B);
        _v_c = _mm_fmadd_ps(_v_a, _v_b, _v_c);
        packed_A += 1;
        packed_B += 4;
    }
    _mm_storeu_ps(C, _v_c);
}

inline void smma_1x1(const float* packed_A, const float* packed_B, int K, float* C, int ldc) {
    __m256 _v_c0 = _mm256_setzero_ps();
    __m256 _v_c1 = _mm256_setzero_ps();
    __m256 _v_c2 = _mm256_setzero_ps();
    __m256 _v_c3 = _mm256_setzero_ps();

    int k = 0;
    for (k = 0; k < K - 31; k += 32) {
        __m256 _v_a0 = _mm256_load_ps(packed_A + 0 * 8);
        __m256 _v_a1 = _mm256_load_ps(packed_A + 1 * 8);
        __m256 _v_a2 = _mm256_load_ps(packed_A + 2 * 8);
        __m256 _v_a3 = _mm256_load_ps(packed_A + 3 * 8);

        __m256 _v_b0 = _mm256_load_ps(packed_B + 0 * 8);
        __m256 _v_b1 = _mm256_load_ps(packed_B + 1 * 8);
        __m256 _v_b2 = _mm256_load_ps(packed_B + 2 * 8);
        __m256 _v_b3 = _mm256_load_ps(packed_B + 3 * 8);

        _v_c0 = _mm256_fmadd_ps(_v_a0, _v_b0, _v_c0);
        _v_c1 = _mm256_fmadd_ps(_v_a1, _v_b1, _v_c1);
        _v_c2 = _mm256_fmadd_ps(_v_a2, _v_b2, _v_c2);
        _v_c3 = _mm256_fmadd_ps(_v_a3, _v_b3, _v_c3);

        packed_A += 32;
        packed_B += 32;
    }

    _v_c0 = _mm256_add_ps(_v_c0, _v_c1);
    _v_c2 = _mm256_add_ps(_v_c2, _v_c3);

    for (; k < K - 15; k += 16) {
        __m256 _v_a0 = _mm256_load_ps(packed_A + 0 * 8);
        __m256 _v_a1 = _mm256_load_ps(packed_A + 1 * 8);
        __m256 _v_b0 = _mm256_load_ps(packed_B + 0 * 8);
        __m256 _v_b1 = _mm256_load_ps(packed_B + 1 * 8);

        _v_c0 = _mm256_fmadd_ps(_v_a0, _v_b0, _v_c0);
        _v_c2 = _mm256_fmadd_ps(_v_a1, _v_b1, _v_c2);

        packed_A += 16;
        packed_B += 16;
    }

    _v_c0 = _mm256_add_ps(_v_c0, _v_c2);
    float sum = reduce_add_ps(_v_c0);

    for (; k < K; ++k) {
        sum += (*packed_A++) * (*packed_B++);
    }

    *C += sum;
}


}  // namespace kernel
}  // namespace cpu
}  // namespace fastnum