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

inline void mma_mxn(int M, int N, const float* packed_A, const float* packed_B, int K, float* C, int ldc) {
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

inline void mma_6xn(int N, const float* packed_A, const float* packed_B, int K, float* C, int ldc) {
    float* c0_ptr = C + 0 * ldc;
    float* c1_ptr = C + 1 * ldc;
    float* c2_ptr = C + 2 * ldc;
    float* c3_ptr = C + 3 * ldc;
    float* c4_ptr = C + 4 * ldc;
    float* c5_ptr = C + 5 * ldc;

    for (int k = 0; k < K; ++k) {

        float tmp0 = packed_A[0];
        float tmp1 = packed_A[1];
        float tmp2 = packed_A[2];
        float tmp3 = packed_A[3];
        float tmp4 = packed_A[4];
        float tmp5 = packed_A[5];

        for (int n = 0; n < N; ++n) {
            c0_ptr[n] += tmp0 * packed_B[n];
            c1_ptr[n] += tmp1 * packed_B[n];
            c2_ptr[n] += tmp2 * packed_B[n];
            c3_ptr[n] += tmp3 * packed_B[n];
            c4_ptr[n] += tmp4 * packed_B[n];
            c5_ptr[n] += tmp5 * packed_B[n];
        }

        packed_A += 6;
        packed_B += N;
    }
}

inline void mma_8xn(int N, const float* packed_A, const float* packed_B, int K, float* C, int ldc) {
    float* c0_ptr = C + 0 * ldc;
    float* c1_ptr = C + 1 * ldc;
    float* c2_ptr = C + 2 * ldc;
    float* c3_ptr = C + 3 * ldc;
    float* c4_ptr = C + 4 * ldc;
    float* c5_ptr = C + 5 * ldc;
    float* c6_ptr = C + 6 * ldc;
    float* c7_ptr = C + 7 * ldc;

    for (int k = 0; k < K; ++k) {

        float tmp0 = packed_A[0];
        float tmp1 = packed_A[1];
        float tmp2 = packed_A[2];
        float tmp3 = packed_A[3];
        float tmp4 = packed_A[4];
        float tmp5 = packed_A[5];
        float tmp6 = packed_A[6];
        float tmp7 = packed_A[7];

        for (int n = 0; n < N; ++n) {
            c0_ptr[n] += tmp0 * packed_B[n];
            c1_ptr[n] += tmp1 * packed_B[n];
            c2_ptr[n] += tmp2 * packed_B[n];
            c3_ptr[n] += tmp3 * packed_B[n];
            c4_ptr[n] += tmp4 * packed_B[n];
            c5_ptr[n] += tmp5 * packed_B[n];
            c6_ptr[n] += tmp6 * packed_B[n];
            c7_ptr[n] += tmp7 * packed_B[n];
        }

        packed_A += 8;
        packed_B += N;
    }
}

inline void mma_16xn(int N, const float* packed_A, const float* packed_B, int K, float* C, int ldc) {
    float* c0_ptr = C + 0 * ldc;
    float* c1_ptr = C + 1 * ldc;
    float* c2_ptr = C + 2 * ldc;
    float* c3_ptr = C + 3 * ldc;
    float* c4_ptr = C + 4 * ldc;
    float* c5_ptr = C + 5 * ldc;
    float* c6_ptr = C + 6 * ldc;
    float* c7_ptr = C + 7 * ldc;
    float* c8_ptr = C + 8 * ldc;
    float* c9_ptr = C + 9 * ldc;
    float* c10_ptr = C + 10 * ldc;
    float* c11_ptr = C + 11 * ldc;
    float* c12_ptr = C + 12 * ldc;
    float* c13_ptr = C + 13 * ldc;
    float* c14_ptr = C + 14 * ldc;
    float* c15_ptr = C + 15 * ldc;

    for (int k = 0; k < K; ++k) {
        float tmp0 = packed_A[0];
        float tmp1 = packed_A[1];
        float tmp2 = packed_A[2];
        float tmp3 = packed_A[3];
        float tmp4 = packed_A[4];
        float tmp5 = packed_A[5];
        float tmp6 = packed_A[6];
        float tmp7 = packed_A[7];
        float tmp8 = packed_A[8];
        float tmp9 = packed_A[9];
        float tmp10 = packed_A[10];
        float tmp11 = packed_A[11];
        float tmp12 = packed_A[12];
        float tmp13 = packed_A[13];
        float tmp14 = packed_A[14];
        float tmp15 = packed_A[15];

        for (int n = 0; n < N; ++n) {
            c0_ptr[n] += tmp0 * packed_B[n];
            c1_ptr[n] += tmp1 * packed_B[n];
            c2_ptr[n] += tmp2 * packed_B[n];
            c3_ptr[n] += tmp3 * packed_B[n];
            c4_ptr[n] += tmp4 * packed_B[n];
            c5_ptr[n] += tmp5 * packed_B[n];
            c6_ptr[n] += tmp6 * packed_B[n];
            c7_ptr[n] += tmp7 * packed_B[n];
            c8_ptr[n] += tmp8 * packed_B[n];
            c9_ptr[n] += tmp9 * packed_B[n];
            c10_ptr[n] += tmp10 * packed_B[n];
            c11_ptr[n] += tmp11 * packed_B[n];
            c12_ptr[n] += tmp12 * packed_B[n];
            c13_ptr[n] += tmp13 * packed_B[n];
            c14_ptr[n] += tmp14 * packed_B[n];
            c15_ptr[n] += tmp15 * packed_B[n];
        }

        packed_A += 16;
        packed_B += N;
    }
}

inline void mma_mx8(int M, const float* packed_A, const float* packed_B, int K, float* C, int ldc) {

    for (int k = 0; k < K; ++k) {
        __m256 _v_B = _mm256_loadu_ps(packed_B);
        int m = 0;
        for (m = 0; m < M - 3; m += 4) {
            __m256 _v_C0 = _mm256_loadu_ps(C + (m + 0) * ldc);
            __m256 _v_C1 = _mm256_loadu_ps(C + (m + 1) * ldc);
            __m256 _v_C2 = _mm256_loadu_ps(C + (m + 2) * ldc);
            __m256 _v_C3 = _mm256_loadu_ps(C + (m + 3) * ldc);

            __m256 _v_A0 = _mm256_broadcast_ss(packed_A + m + 0);
            __m256 _v_A1 = _mm256_broadcast_ss(packed_A + m + 1);
            __m256 _v_A2 = _mm256_broadcast_ss(packed_A + m + 2);
            __m256 _v_A3 = _mm256_broadcast_ss(packed_A + m + 3);

            _v_C0 = _mm256_fmadd_ps(_v_A0, _v_B, _v_C0);
            _v_C1 = _mm256_fmadd_ps(_v_A1, _v_B, _v_C1);
            _v_C2 = _mm256_fmadd_ps(_v_A2, _v_B, _v_C2);
            _v_C3 = _mm256_fmadd_ps(_v_A3, _v_B, _v_C3);

            _mm256_storeu_ps(C + (m + 0) * ldc, _v_C0);
            _mm256_storeu_ps(C + (m + 1) * ldc, _v_C1);
            _mm256_storeu_ps(C + (m + 2) * ldc, _v_C2);
            _mm256_storeu_ps(C + (m + 3) * ldc, _v_C3);
        }
        for (; m < M; ++m) {
            __m256 _v_C = _mm256_loadu_ps(C + m * ldc);
            __m256 _v_A = _mm256_broadcast_ss(packed_A + m);
            _v_C = _mm256_fmadd_ps(_v_A, _v_B, _v_C);
            _mm256_storeu_ps(C + m * ldc, _v_C);
        }
        packed_A += M;
        packed_B += 8;
    }
}

inline void mma_mx6(int M, const float* packed_A, const float* packed_B, int K, float* C, int ldc) {
    for (int k = 0; k < K; ++k) {
        float b0 = *packed_B++;
        float b1 = *packed_B++;
        float b2 = *packed_B++;
        float b3 = *packed_B++;
        float b4 = *packed_B++;
        float b5 = *packed_B++;

        for (int m = 0; m < M; ++m) {
            float a = packed_A[m];
            float c0 = a * b0;
            float c1 = a * b1;
            float c2 = a * b2;
            float c3 = a * b3;
            float c4 = a * b4;
            float c5 = a * b5;
            float* c_ptr = C + m * ldc;
            c_ptr[0] += c0;
            c_ptr[1] += c1;
            c_ptr[2] += c2;
            c_ptr[3] += c3;
            c_ptr[4] += c4;
            c_ptr[5] += c5;
        }

        packed_A += M;
    }
}

inline void mma_mx1(int M, const float* packed_A, const float* packed_B, int K, float* C, int ldc) {
    for (int k = 0; k < K; ++k) {
        float b = *packed_B++;
        for (int m = 0; m < M; ++m) {
            float a = packed_A[m];
            *(C + m * ldc) += a * b;
        }
        packed_A += M;
    }
}

inline void mma_mx16(int M, const float* packed_A, const float* packed_B, int K, float* C, int ldc) {
    for (int k = 0; k < K; ++k) {
        __m256 _v_b0 = _mm256_loadu_ps(packed_B + 0);
        __m256 _v_b1 = _mm256_loadu_ps(packed_B + 8);

        for (int m = 0; m < M; ++m) {

            __m256 _v_c0 = _mm256_loadu_ps(C + m * ldc + 0);
            __m256 _v_c1 = _mm256_loadu_ps(C + m * ldc + 8);
            __m256 _v_a = _mm256_broadcast_ss(packed_A + m);

            _v_c0 = _mm256_fmadd_ps(_v_a, _v_b0, _v_c0);
            _v_c1 = _mm256_fmadd_ps(_v_a, _v_b1, _v_c1);

            _mm256_storeu_ps(C + m * ldc + 0, _v_c0);
            _mm256_storeu_ps(C + m * ldc + 8, _v_c1);
        }
        packed_A += M;
        packed_B += 16;
    }
}

inline void mma_8x1(const float* packed_A, const float* packed_B, int K, float* C, int ldc) {
    __m256 _v_c0 = _mm256_setzero_ps();
    __m256 _v_c1 = _mm256_setzero_ps();
    __m256 _v_c2 = _mm256_setzero_ps();
    __m256 _v_c3 = _mm256_setzero_ps();

    int k = 0;

    for (k = 0; k < K - 3; k += 4) {
        __m256 _v_a0 = _mm256_loadu_ps(packed_A + 8 * 0);
        __m256 _v_a1 = _mm256_loadu_ps(packed_A + 8 * 1);
        __m256 _v_a2 = _mm256_loadu_ps(packed_A + 8 * 2);
        __m256 _v_a3 = _mm256_loadu_ps(packed_A + 8 * 3);

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
        __m256 _v_a = _mm256_loadu_ps(packed_A);
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

inline void mma_4x1(const float* packed_A, const float* packed_B, int K, float* C, int ldc) {
    __m128 _v_c0 = _mm_setzero_ps();
    __m128 _v_c1 = _mm_setzero_ps();
    __m128 _v_c2 = _mm_setzero_ps();
    __m128 _v_c3 = _mm_setzero_ps();

    int k = 0;

    for (k = 0; k < K - 3; k += 4) {
        __m128 _v_a0 = _mm_loadu_ps(packed_A + 4 * 0);
        __m128 _v_a1 = _mm_loadu_ps(packed_A + 4 * 1);
        __m128 _v_a2 = _mm_loadu_ps(packed_A + 4 * 2);
        __m128 _v_a3 = _mm_loadu_ps(packed_A + 4 * 3);

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
        __m128 _v_a = _mm_loadu_ps(packed_A);
        __m128 _v_b = _mm_broadcast_ss(packed_B);
        _v_c = _mm_fmadd_ps(_v_a, _v_b, _v_c);
        packed_A += 8;
        packed_B += 1;
    }

    float c_data[4];
    _mm_storeu_ps(c_data, _v_c);

    *(C + 0 * ldc) += c_data[0];
    *(C + 1 * ldc) += c_data[1];
    *(C + 2 * ldc) += c_data[2];
    *(C + 3 * ldc) += c_data[3];
}

inline void mma_8x8(const float* packed_A, const float* packed_B, int K, float* C, int ldc) {
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

        _vb = _mm256_loadu_ps(packed_B);

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

inline void mma_6x16(const float* packed_A, const float* packed_B, int K, float* C, int ldc) {
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

        _v_b0 = _mm256_loadu_ps(packed_B);
        _v_b1 = _mm256_loadu_ps(packed_B + 8);

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

inline void mma_6x8(const float* packed_A, const float* packed_B, int K, float* C, int ldc) {
    __m256 _v_c0 = _mm256_loadu_ps(C + 0 * ldc);
    __m256 _v_c1 = _mm256_loadu_ps(C + 2 * ldc);
    __m256 _v_c2 = _mm256_loadu_ps(C + 2 * ldc);
    __m256 _v_c3 = _mm256_loadu_ps(C + 3 * ldc);
    __m256 _v_c4 = _mm256_loadu_ps(C + 4 * ldc);
    __m256 _v_c5 = _mm256_loadu_ps(C + 5 * ldc);

    for (int i = 0; i < K; ++i) {

        __m256 _v_b0 = _mm256_loadu_ps(packed_B);

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

inline void mma_16x6(const float* packed_A, const float* packed_B, int K, float* C, int ldc) {
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
        _v_a0 = _mm256_loadu_ps(packed_A);
        _v_a1 = _mm256_loadu_ps(packed_A + 8);

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

inline void mma_1x8(const float* packed_A, const float* packed_B, int K, float* C, int ldc) {
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

        __m256 _v_b0 = _mm256_loadu_ps(packed_B + 8 * 0);
        __m256 _v_b1 = _mm256_loadu_ps(packed_B + 8 * 1);
        __m256 _v_b2 = _mm256_loadu_ps(packed_B + 8 * 2);
        __m256 _v_b3 = _mm256_loadu_ps(packed_B + 8 * 3);

        _v_c0 = _mm256_fmadd_ps(_v_a0, _v_b0, _v_c0);
        _v_c1 = _mm256_fmadd_ps(_v_a1, _v_b1, _v_c1);
        _v_c2 = _mm256_fmadd_ps(_v_a2, _v_b2, _v_c2);
        _v_c3 = _mm256_fmadd_ps(_v_a3, _v_b3, _v_c3);

        packed_A += 4;
        packed_B += 32;
    }

    __m256 _v_c = _mm256_add_ps(_mm256_add_ps(_v_c0, _v_c1), _mm256_add_ps(_v_c2, _v_c3));
    _mm256_storeu_ps(C, _v_c);
}

inline void mma_1x4(const float* packed_A, const float* packed_B, int K, float* C, int ldc) {
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

        __m128 _v_b0 = _mm_loadu_ps(packed_B + 4 * 0);
        __m128 _v_b1 = _mm_loadu_ps(packed_B + 4 * 1);
        __m128 _v_b2 = _mm_loadu_ps(packed_B + 4 * 2);
        __m128 _v_b3 = _mm_loadu_ps(packed_B + 4 * 3);

        _v_c0 = _mm_fmadd_ps(_v_a0, _v_b0, _v_c0);
        _v_c1 = _mm_fmadd_ps(_v_a1, _v_b1, _v_c1);
        _v_c2 = _mm_fmadd_ps(_v_a2, _v_b2, _v_c2);
        _v_c3 = _mm_fmadd_ps(_v_a3, _v_b3, _v_c3);

        packed_A += 4;
        packed_B += 32;
    }

    __m128 _v_c = _mm_add_ps(_mm_add_ps(_v_c0, _v_c1), _mm_add_ps(_v_c2, _v_c3));

    for (; k < K; ++k) {
        __m128 _v_a = _mm_broadcast_ss(packed_A);
        __m128 _v_b = _mm_loadu_ps(packed_B);
        _v_c = _mm_fmadd_ps(_v_a, _v_b, _v_c);
    }
    _mm_storeu_ps(C, _v_c);
}

inline void mma_1x1(const float* packed_A, const float* packed_B, int K, float* C, int ldc) {
    __m256 _v_c0 = _mm256_setzero_ps();
    __m256 _v_c1 = _mm256_setzero_ps();
    __m256 _v_c2 = _mm256_setzero_ps();
    __m256 _v_c3 = _mm256_setzero_ps();

    int k = 0;
    for (k = 0; k < K - 31; k += 32) {
        __m256 _v_a0 = _mm256_loadu_ps(packed_A + 0 * 8);
        __m256 _v_a1 = _mm256_loadu_ps(packed_A + 1 * 8);
        __m256 _v_a2 = _mm256_loadu_ps(packed_A + 2 * 8);
        __m256 _v_a3 = _mm256_loadu_ps(packed_A + 3 * 8);

        __m256 _v_b0 = _mm256_loadu_ps(packed_B + 0 * 8);
        __m256 _v_b1 = _mm256_loadu_ps(packed_B + 1 * 8);
        __m256 _v_b2 = _mm256_loadu_ps(packed_B + 2 * 8);
        __m256 _v_b3 = _mm256_loadu_ps(packed_B + 3 * 8);

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
        __m256 _v_a0 = _mm256_loadu_ps(packed_A + 0 * 8);
        __m256 _v_a1 = _mm256_loadu_ps(packed_A + 1 * 8);
        __m256 _v_b0 = _mm256_loadu_ps(packed_B + 0 * 8);
        __m256 _v_b1 = _mm256_loadu_ps(packed_B + 1 * 8);

        _v_c0 = _mm256_fmadd_ps(_v_a0, _v_b0, _v_c0);
        _v_c2 = _mm256_fmadd_ps(_v_a1, _v_b1, _v_c2);
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