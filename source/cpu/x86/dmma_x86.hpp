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

inline void print_avx(__m256d& v) {
    double d[4];
    _mm256_storeu_pd(d, v);
    printf("%.5f %.5f %.5f %.5f\n", d[0], d[1], d[2], d[3]);
}

inline void dmma_6x4(const double* packed_A, const double* packed_B, int K, double* C, int ldc) {
    double* c_ptr0 = C + 0 * ldc;
    double* c_ptr1 = C + 1 * ldc;
    double* c_ptr2 = C + 2 * ldc;
    double* c_ptr3 = C + 3 * ldc;
    double* c_ptr4 = C + 4 * ldc;
    double* c_ptr5 = C + 5 * ldc;

    __m256d _v_c0 = _mm256_loadu_pd(c_ptr0);
    __m256d _v_c1 = _mm256_loadu_pd(c_ptr1);
    __m256d _v_c2 = _mm256_loadu_pd(c_ptr2);
    __m256d _v_c3 = _mm256_loadu_pd(c_ptr3);
    __m256d _v_c4 = _mm256_loadu_pd(c_ptr4);
    __m256d _v_c5 = _mm256_loadu_pd(c_ptr5);

    for (int k = 0; k < K; ++k) {
        __m256d _v_a0 = _mm256_broadcast_sd(packed_A + 0);
        __m256d _v_a1 = _mm256_broadcast_sd(packed_A + 1);
        __m256d _v_a2 = _mm256_broadcast_sd(packed_A + 2);
        __m256d _v_a3 = _mm256_broadcast_sd(packed_A + 3);
        __m256d _v_a4 = _mm256_broadcast_sd(packed_A + 4);
        __m256d _v_a5 = _mm256_broadcast_sd(packed_A + 5);

        __m256d _v_b = _mm256_loadu_pd(packed_B);

        _v_c0 = _mm256_fmadd_pd(_v_a0, _v_b, _v_c0);
        _v_c1 = _mm256_fmadd_pd(_v_a1, _v_b, _v_c1);
        _v_c2 = _mm256_fmadd_pd(_v_a2, _v_b, _v_c2);
        _v_c3 = _mm256_fmadd_pd(_v_a3, _v_b, _v_c3);
        _v_c4 = _mm256_fmadd_pd(_v_a4, _v_b, _v_c4);
        _v_c5 = _mm256_fmadd_pd(_v_a5, _v_b, _v_c5);

        packed_A += 6;
        packed_B += 4;
    }

    _mm256_storeu_pd(c_ptr0, _v_c0);
    _mm256_storeu_pd(c_ptr1, _v_c1);
    _mm256_storeu_pd(c_ptr2, _v_c2);
    _mm256_storeu_pd(c_ptr3, _v_c3);
    _mm256_storeu_pd(c_ptr4, _v_c4);
    _mm256_storeu_pd(c_ptr5, _v_c5);
}

inline void dmma_6x1(const double* packed_A, const double* packed_B, int K, double* C, int ldc) {
    double c0 = 0;
    double c1 = 0;
    double c2 = 0;
    double c3 = 0;
    double c4 = 0;
    double c5 = 0;
    double c6 = 0;

    for (int k = 0; k < K; ++k) {
        double a0 = packed_A[0];
        double a1 = packed_A[1];
        double a2 = packed_A[2];
        double a3 = packed_A[3];
        double a4 = packed_A[4];
        double a5 = packed_A[5];

        double b = packed_B[0];

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

inline void dmma_4x8(const double* packed_A, const double* packed_B, int K, double* C, int ldc) {
    double* c_ptr0 = C + 0 * ldc;
    double* c_ptr1 = C + 1 * ldc;
    double* c_ptr2 = C + 2 * ldc;
    double* c_ptr3 = C + 3 * ldc;

    __m256d _v_c00 = _mm256_loadu_pd(c_ptr0 + 0);
    __m256d _v_c01 = _mm256_loadu_pd(c_ptr0 + 4);
    __m256d _v_c10 = _mm256_loadu_pd(c_ptr1 + 0);
    __m256d _v_c11 = _mm256_loadu_pd(c_ptr1 + 4);
    __m256d _v_c20 = _mm256_loadu_pd(c_ptr2 + 0);
    __m256d _v_c21 = _mm256_loadu_pd(c_ptr2 + 4);
    __m256d _v_c30 = _mm256_loadu_pd(c_ptr3 + 0);
    __m256d _v_c31 = _mm256_loadu_pd(c_ptr3 + 4);

    for (int k = 0; k < K; ++k) {
        __m256d _v_a0 = _mm256_broadcast_sd(packed_A + 0);
        __m256d _v_a1 = _mm256_broadcast_sd(packed_A + 1);
        __m256d _v_a2 = _mm256_broadcast_sd(packed_A + 2);
        __m256d _v_a3 = _mm256_broadcast_sd(packed_A + 3);

        __m256d _v_b0 = _mm256_loadu_pd(packed_B + 0);
        __m256d _v_b1 = _mm256_loadu_pd(packed_B + 4);

        _v_c00 = _mm256_fmadd_pd(_v_a0, _v_b0, _v_c00);
        _v_c10 = _mm256_fmadd_pd(_v_a1, _v_b0, _v_c10);
        _v_c20 = _mm256_fmadd_pd(_v_a2, _v_b0, _v_c20);
        _v_c30 = _mm256_fmadd_pd(_v_a3, _v_b0, _v_c30);

        _v_c01 = _mm256_fmadd_pd(_v_a0, _v_b1, _v_c01);
        _v_c11 = _mm256_fmadd_pd(_v_a1, _v_b1, _v_c11);
        _v_c21 = _mm256_fmadd_pd(_v_a2, _v_b1, _v_c21);
        _v_c31 = _mm256_fmadd_pd(_v_a3, _v_b1, _v_c31);

        packed_A += 4;
        packed_B += 8;
    }

    _mm256_storeu_pd(c_ptr0 + 0, _v_c00);
    _mm256_storeu_pd(c_ptr0 + 4, _v_c01);
    _mm256_storeu_pd(c_ptr1 + 0, _v_c10);
    _mm256_storeu_pd(c_ptr1 + 4, _v_c11);
    _mm256_storeu_pd(c_ptr2 + 0, _v_c20);
    _mm256_storeu_pd(c_ptr2 + 4, _v_c21);
    _mm256_storeu_pd(c_ptr3 + 0, _v_c30);
    _mm256_storeu_pd(c_ptr3 + 4, _v_c31);
}

inline void dmma_4x4(const double* packed_A, const double* packed_B, int K, double* C, int ldc) {
    double* c_ptr0 = C + 0 * ldc;
    double* c_ptr1 = C + 1 * ldc;
    double* c_ptr2 = C + 2 * ldc;
    double* c_ptr3 = C + 3 * ldc;

    __m256d _v_c0 = _mm256_loadu_pd(c_ptr0);
    __m256d _v_c1 = _mm256_loadu_pd(c_ptr1);
    __m256d _v_c2 = _mm256_loadu_pd(c_ptr2);
    __m256d _v_c3 = _mm256_loadu_pd(c_ptr3);

    for (int k = 0; k < K; ++k) {
        __m256d _v_a0 = _mm256_broadcast_sd(packed_A + 0);
        __m256d _v_a1 = _mm256_broadcast_sd(packed_A + 1);
        __m256d _v_a2 = _mm256_broadcast_sd(packed_A + 2);
        __m256d _v_a3 = _mm256_broadcast_sd(packed_A + 3);


        __m256d _v_b = _mm256_loadu_pd(packed_B);


        _v_c0 = _mm256_fmadd_pd(_v_a0, _v_b, _v_c0);
        _v_c1 = _mm256_fmadd_pd(_v_a1, _v_b, _v_c1);
        _v_c2 = _mm256_fmadd_pd(_v_a2, _v_b, _v_c2);
        _v_c3 = _mm256_fmadd_pd(_v_a3, _v_b, _v_c3);

        packed_A += 4;
        packed_B += 4;
    }

    _mm256_storeu_pd(c_ptr0, _v_c0);
    _mm256_storeu_pd(c_ptr1, _v_c1);
    _mm256_storeu_pd(c_ptr2, _v_c2);
    _mm256_storeu_pd(c_ptr3, _v_c3);
}

inline void dmma_4x1(const double* packed_A, const double* packed_B, int K, double* C, int ldc) {
    __m256d _v_c0 = _mm256_setzero_pd();
    __m256d _v_c1 = _mm256_setzero_pd();
    __m256d _v_c2 = _mm256_setzero_pd();
    __m256d _v_c3 = _mm256_setzero_pd();

    int k = 0;

    for (k = 0; k < K - 3; k += 4) {
        __m256d _v_a0 = _mm256_loadu_pd(packed_A + 4 * 0);
        __m256d _v_a1 = _mm256_loadu_pd(packed_A + 4 * 1);
        __m256d _v_a2 = _mm256_loadu_pd(packed_A + 4 * 2);
        __m256d _v_a3 = _mm256_loadu_pd(packed_A + 4 * 3);

        __m256d _v_b0 = _mm256_broadcast_sd(packed_B + 0);
        __m256d _v_b1 = _mm256_broadcast_sd(packed_B + 1);
        __m256d _v_b2 = _mm256_broadcast_sd(packed_B + 2);
        __m256d _v_b3 = _mm256_broadcast_sd(packed_B + 3);

        _v_c0 = _mm256_fmadd_pd(_v_a0, _v_b0, _v_c0);
        _v_c1 = _mm256_fmadd_pd(_v_a1, _v_b1, _v_c1);
        _v_c2 = _mm256_fmadd_pd(_v_a2, _v_b2, _v_c2);
        _v_c3 = _mm256_fmadd_pd(_v_a3, _v_b3, _v_c3);

        packed_A += 16;
        packed_B += 4;
    }

    __m256d _v_c = _mm256_add_pd(_mm256_add_pd(_v_c0, _v_c1), _mm256_add_pd(_v_c2, _v_c3));

    for (; k < K; ++k) {
        __m256d _v_a = _mm256_loadu_pd(packed_A);
        __m256d _v_b = _mm256_broadcast_sd(packed_B);
        _v_c = _mm256_fmadd_pd(_v_a, _v_b, _v_c);
        packed_A += 4;
        packed_B += 1;
    }

    double c_data[4];
    _mm256_storeu_pd(c_data, _v_c);

    *(C + 0 * ldc) += c_data[0];
    *(C + 1 * ldc) += c_data[1];
    *(C + 2 * ldc) += c_data[2];
    *(C + 3 * ldc) += c_data[3];
}

inline void dmma_1x8(const double* packed_A, const double* packed_B, int K, double* C, int ldc) {
    double* c_ptr0 = C + 0 * ldc;

    __m256d _v_c00 = _mm256_loadu_pd(c_ptr0 + 0);
    __m256d _v_c01 = _mm256_loadu_pd(c_ptr0 + 4);

    for (int k = 0; k < K; ++k) {
        __m256d _v_a0 = _mm256_broadcast_sd(packed_A + 0);

        __m256d _v_b0 = _mm256_loadu_pd(packed_B + 0);
        __m256d _v_b1 = _mm256_loadu_pd(packed_B + 4);

        _v_c00 = _mm256_fmadd_pd(_v_a0, _v_b0, _v_c00);
        _v_c01 = _mm256_fmadd_pd(_v_a0, _v_b1, _v_c01);

        packed_A += 1;
        packed_B += 8;
    }

    _mm256_storeu_pd(c_ptr0 + 0, _v_c00);
    _mm256_storeu_pd(c_ptr0 + 4, _v_c01);
}

inline void dmma_1x4(const double* packed_A, const double* packed_B, int K, double* C, int ldc) {
    int k = 0;

    __m256d _v_c0 = _mm256_loadu_pd(C);
    __m256d _v_c1 = _mm256_setzero_pd();
    __m256d _v_c2 = _mm256_setzero_pd();
    __m256d _v_c3 = _mm256_setzero_pd();

    for (k = 0; k < K - 3; k += 4) {
        __m256d _v_a0 = _mm256_broadcast_sd(packed_A + 0);
        __m256d _v_a1 = _mm256_broadcast_sd(packed_A + 1);
        __m256d _v_a2 = _mm256_broadcast_sd(packed_A + 2);
        __m256d _v_a3 = _mm256_broadcast_sd(packed_A + 3);

        __m256d _v_b0 = _mm256_loadu_pd(packed_B + 4 * 0);
        __m256d _v_b1 = _mm256_loadu_pd(packed_B + 4 * 1);
        __m256d _v_b2 = _mm256_loadu_pd(packed_B + 4 * 2);
        __m256d _v_b3 = _mm256_loadu_pd(packed_B + 4 * 3);

        _v_c0 = _mm256_fmadd_pd(_v_a0, _v_b0, _v_c0);
        _v_c1 = _mm256_fmadd_pd(_v_a1, _v_b1, _v_c1);
        _v_c2 = _mm256_fmadd_pd(_v_a2, _v_b2, _v_c2);
        _v_c3 = _mm256_fmadd_pd(_v_a3, _v_b3, _v_c3);

        packed_A += 4;
        packed_B += 16;
    }

    __m256d _v_c = _mm256_add_pd(_mm256_add_pd(_v_c0, _v_c1), _mm256_add_pd(_v_c2, _v_c3));

    for (; k < K; ++k) {
        __m256d _v_a = _mm256_broadcast_sd(packed_A);
        __m256d _v_b = _mm256_loadu_pd(packed_B);
        _v_c = _mm256_fmadd_pd(_v_a, _v_b, _v_c);
        packed_A += 1;
        packed_B += 4;
    }
    _mm256_storeu_pd(C, _v_c);
}

inline void dmma_1x1(const double* packed_A, const double* packed_B, int K, double* C, int ldc) {
    __m256d _v_c0 = _mm256_setzero_pd();
    __m256d _v_c1 = _mm256_setzero_pd();
    __m256d _v_c2 = _mm256_setzero_pd();
    __m256d _v_c3 = _mm256_setzero_pd();

    int k = 0;
    for (k = 0; k < K - 15; k += 16) {
        __m256d _v_a0 = _mm256_loadu_pd(packed_A + 0 * 4);
        __m256d _v_a1 = _mm256_loadu_pd(packed_A + 1 * 4);
        __m256d _v_a2 = _mm256_loadu_pd(packed_A + 2 * 4);
        __m256d _v_a3 = _mm256_loadu_pd(packed_A + 3 * 4);

        __m256d _v_b0 = _mm256_loadu_pd(packed_B + 0 * 4);
        __m256d _v_b1 = _mm256_loadu_pd(packed_B + 1 * 4);
        __m256d _v_b2 = _mm256_loadu_pd(packed_B + 2 * 4);
        __m256d _v_b3 = _mm256_loadu_pd(packed_B + 3 * 4);

        _v_c0 = _mm256_fmadd_pd(_v_a0, _v_b0, _v_c0);
        _v_c1 = _mm256_fmadd_pd(_v_a1, _v_b1, _v_c1);
        _v_c2 = _mm256_fmadd_pd(_v_a2, _v_b2, _v_c2);
        _v_c3 = _mm256_fmadd_pd(_v_a3, _v_b3, _v_c3);

        packed_A += 16;
        packed_B += 16;
    }

    _v_c0 = _mm256_add_pd(_v_c0, _v_c1);
    _v_c2 = _mm256_add_pd(_v_c2, _v_c3);

    for (; k < K - 7; k += 8) {
        __m256d _v_a0 = _mm256_loadu_pd(packed_A + 0 * 4);
        __m256d _v_a1 = _mm256_loadu_pd(packed_A + 1 * 4);
        __m256d _v_b0 = _mm256_loadu_pd(packed_B + 0 * 4);
        __m256d _v_b1 = _mm256_loadu_pd(packed_B + 1 * 4);

        _v_c0 = _mm256_fmadd_pd(_v_a0, _v_b0, _v_c0);
        _v_c2 = _mm256_fmadd_pd(_v_a1, _v_b1, _v_c2);

        packed_A += 8;
        packed_B += 8;
    }

    _v_c0 = _mm256_add_pd(_v_c0, _v_c2);
    double sum = reduce_add_pd(_v_c0);

    for (; k < K; ++k) {
        sum += (*packed_A++) * (*packed_B++);
    }

    *C += sum;
}

}  // namespace kernel
}  // namespace cpu
}  // namespace fastnum