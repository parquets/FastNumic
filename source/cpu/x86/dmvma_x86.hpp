#pragma once

#include <math.h>
#include <memory.h>
#include <stdio.h>

#include <immintrin.h>
#include <nmmintrin.h>
#include <smmintrin.h>

#include "dtranspose_x86.hpp"
#include "vectorize_x86.hpp"

namespace fastnum {
namespace cpu {
namespace kernel {

inline void dmvma_nr8(int K, double alpha, const double* A, int lda, const double* B, double beta, double* C) {
    const double* a_ptr0 = A + 0 * lda;
    const double* a_ptr1 = A + 1 * lda;
    const double* a_ptr2 = A + 2 * lda;
    const double* a_ptr3 = A + 3 * lda;
    const double* a_ptr4 = A + 4 * lda;
    const double* a_ptr5 = A + 5 * lda;
    const double* a_ptr6 = A + 6 * lda;
    const double* a_ptr7 = A + 7 * lda;

    __m256d _v_c0 = _mm256_setzero_pd();
    __m256d _v_c1 = _mm256_setzero_pd();
    __m256d _v_c2 = _mm256_setzero_pd();
    __m256d _v_c3 = _mm256_setzero_pd();
    __m256d _v_c4 = _mm256_setzero_pd();
    __m256d _v_c5 = _mm256_setzero_pd();
    __m256d _v_c6 = _mm256_setzero_pd();
    __m256d _v_c7 = _mm256_setzero_pd();

    __m256d _v_alpha = _mm256_broadcast_sd(&alpha);

    int k = 0;
    for (k = 0; k < K - 3; k += 4) {
        __m256d _v_a0 = _mm256_loadu_pd(a_ptr0);
        __m256d _v_a1 = _mm256_loadu_pd(a_ptr1);
        __m256d _v_a2 = _mm256_loadu_pd(a_ptr2);
        __m256d _v_a3 = _mm256_loadu_pd(a_ptr3);
        __m256d _v_a4 = _mm256_loadu_pd(a_ptr4);
        __m256d _v_a5 = _mm256_loadu_pd(a_ptr5);
        __m256d _v_a6 = _mm256_loadu_pd(a_ptr6);
        __m256d _v_a7 = _mm256_loadu_pd(a_ptr7);

        __m256d _v_b0 = _mm256_loadu_pd(B);
        _v_b0 = _mm256_mul_pd(_v_b0, _v_alpha);

        _v_c0 = _mm256_fmadd_pd(_v_a0, _v_b0, _v_c0);
        _v_c1 = _mm256_fmadd_pd(_v_a1, _v_b0, _v_c1);
        _v_c2 = _mm256_fmadd_pd(_v_a2, _v_b0, _v_c2);
        _v_c3 = _mm256_fmadd_pd(_v_a3, _v_b0, _v_c3);
        _v_c4 = _mm256_fmadd_pd(_v_a4, _v_b0, _v_c4);
        _v_c5 = _mm256_fmadd_pd(_v_a5, _v_b0, _v_c5);
        _v_c6 = _mm256_fmadd_pd(_v_a6, _v_b0, _v_c6);
        _v_c7 = _mm256_fmadd_pd(_v_a7, _v_b0, _v_c7);

        a_ptr0 += 4;
        a_ptr1 += 4;
        a_ptr2 += 4;
        a_ptr3 += 4;
        a_ptr4 += 4;
        a_ptr5 += 4;
        a_ptr6 += 4;
        a_ptr7 += 4;
        B += 4;
    }

    double c0 = reduce_add_pd(_v_c0);
    double c1 = reduce_add_pd(_v_c1);
    double c2 = reduce_add_pd(_v_c2);
    double c3 = reduce_add_pd(_v_c3);
    double c4 = reduce_add_pd(_v_c4);
    double c5 = reduce_add_pd(_v_c5);
    double c6 = reduce_add_pd(_v_c6);
    double c7 = reduce_add_pd(_v_c7);

    for (; k < K; ++k) {
        c0 += (*a_ptr0++) * (*B) * alpha;
        c1 += (*a_ptr1++) * (*B) * alpha;
        c2 += (*a_ptr2++) * (*B) * alpha;
        c3 += (*a_ptr3++) * (*B) * alpha;
        c4 += (*a_ptr4++) * (*B) * alpha;
        c5 += (*a_ptr5++) * (*B) * alpha;
        c6 += (*a_ptr6++) * (*B) * alpha;
        c7 += (*a_ptr7++) * (*B) * alpha;

        ++B;
    }

    C[0] += c0;
    C[1] += c1;
    C[2] += c2;
    C[3] += c3;
    C[4] += c4;
    C[5] += c5;
    C[6] += c6;
    C[7] += c7;
}

inline void dmvma_nr6(int K, double alpha, const double* A, int lda, const double* B, double beta, double* C) {
    const double* a_ptr0 = A + 0 * lda;
    const double* a_ptr1 = A + 1 * lda;
    const double* a_ptr2 = A + 2 * lda;
    const double* a_ptr3 = A + 3 * lda;
    const double* a_ptr4 = A + 4 * lda;
    const double* a_ptr5 = A + 5 * lda;

    __m256d _v_c0 = _mm256_setzero_pd();
    __m256d _v_c1 = _mm256_setzero_pd();
    __m256d _v_c2 = _mm256_setzero_pd();
    __m256d _v_c3 = _mm256_setzero_pd();
    __m256d _v_c4 = _mm256_setzero_pd();
    __m256d _v_c5 = _mm256_setzero_pd();

    __m256d _v_alpha = _mm256_broadcast_sd(&alpha);

    int k = 0;
    for (k = 0; k < K - 3; k += 4) {
        __m256d _v_a0 = _mm256_loadu_pd(a_ptr0);
        __m256d _v_a1 = _mm256_loadu_pd(a_ptr1);
        __m256d _v_a2 = _mm256_loadu_pd(a_ptr2);
        __m256d _v_a3 = _mm256_loadu_pd(a_ptr3);
        __m256d _v_a4 = _mm256_loadu_pd(a_ptr4);
        __m256d _v_a5 = _mm256_loadu_pd(a_ptr5);

        __m256d _v_b0 = _mm256_loadu_pd(B + 0);
        _v_b0 = _mm256_mul_pd(_v_b0, _v_alpha);

        _v_c0 = _mm256_fmadd_pd(_v_a0, _v_b0, _v_c0);
        _v_c1 = _mm256_fmadd_pd(_v_a1, _v_b0, _v_c1);
        _v_c2 = _mm256_fmadd_pd(_v_a2, _v_b0, _v_c2);
        _v_c3 = _mm256_fmadd_pd(_v_a3, _v_b0, _v_c3);
        _v_c4 = _mm256_fmadd_pd(_v_a4, _v_b0, _v_c4);
        _v_c5 = _mm256_fmadd_pd(_v_a5, _v_b0, _v_c5);

        a_ptr0 += 4;
        a_ptr1 += 4;
        a_ptr2 += 4;
        a_ptr3 += 4;
        a_ptr4 += 4;
        a_ptr5 += 4;
        B += 4;
    }

    double c0 = reduce_add_pd(_v_c0);
    double c1 = reduce_add_pd(_v_c1);
    double c2 = reduce_add_pd(_v_c2);
    double c3 = reduce_add_pd(_v_c3);
    double c4 = reduce_add_pd(_v_c4);
    double c5 = reduce_add_pd(_v_c5);

    for (; k < K; ++k) {
        c0 += (*a_ptr0++) * B[0] * alpha;
        c1 += (*a_ptr1++) * B[0] * alpha;
        c2 += (*a_ptr2++) * B[0] * alpha;
        c3 += (*a_ptr3++) * B[0] * alpha;
        c4 += (*a_ptr4++) * B[0] * alpha;
        c5 += (*a_ptr5++) * B[0] * alpha;
        ++B;
    }
    C[0] += c0;
    C[1] += c1;
    C[2] += c2;
    C[3] += c3;
    C[4] += c4;
    C[5] += c5;
}

inline void dmvma_nr4(int K, double alpha, const double* A, int lda, const double* B, double beta, double* C) {
    const double* a_ptr0 = A + 0 * lda;
    const double* a_ptr1 = A + 1 * lda;
    const double* a_ptr2 = A + 2 * lda;
    const double* a_ptr3 = A + 3 * lda;

    __m256d _v_c0 = _mm256_setzero_pd();
    __m256d _v_c1 = _mm256_setzero_pd();
    __m256d _v_c2 = _mm256_setzero_pd();
    __m256d _v_c3 = _mm256_setzero_pd();

    __m256d _v_alpha = _mm256_broadcast_sd(&alpha);

    int k = 0;
    for (k = 0; k < K - 3; k += 4) {
        __m256d _v_a0 = _mm256_loadu_pd(a_ptr0);
        __m256d _v_a1 = _mm256_loadu_pd(a_ptr1);
        __m256d _v_a2 = _mm256_loadu_pd(a_ptr2);
        __m256d _v_a3 = _mm256_loadu_pd(a_ptr3);

        __m256d _v_b0 = _mm256_loadu_pd(B + 0);
        _v_b0 = _mm256_mul_pd(_v_b0, _v_alpha);

        _v_c0 = _mm256_fmadd_pd(_v_a0, _v_b0, _v_c0);
        _v_c1 = _mm256_fmadd_pd(_v_a1, _v_b0, _v_c1);
        _v_c2 = _mm256_fmadd_pd(_v_a2, _v_b0, _v_c2);
        _v_c3 = _mm256_fmadd_pd(_v_a3, _v_b0, _v_c3);

        a_ptr0 += 4;
        a_ptr1 += 4;
        a_ptr2 += 4;
        a_ptr3 += 4;

        B += 4;
    }

    double c0 = reduce_add_pd(_v_c0);
    double c1 = reduce_add_pd(_v_c1);
    double c2 = reduce_add_pd(_v_c2);
    double c3 = reduce_add_pd(_v_c3);

    for (; k < K; ++k) {
        c0 += (*a_ptr0++) * B[0] * alpha;
        c1 += (*a_ptr1++) * B[0] * alpha;
        c2 += (*a_ptr2++) * B[0] * alpha;
        c3 += (*a_ptr3++) * B[0] * alpha;
        ++B;
    }
    C[0] += c0;
    C[1] += c1;
    C[2] += c2;
    C[3] += c3;
}

inline void dmvma_nr1(int K, double alpha, const double* A, int lda, const double* B, double beta, double* C) {
    const double* a_ptr0 = A + 0 * lda;

    __m256d _v_c0 = _mm256_setzero_pd();
    __m256d _v_alpha = _mm256_broadcast_sd(&alpha);

    int k = 0;
    for (k = 0; k < K - 3; k += 4) {
        __m256d _v_a0 = _mm256_loadu_pd(a_ptr0);
        __m256d _v_b0 = _mm256_loadu_pd(B + 0);
        _v_b0 = _mm256_mul_pd(_v_b0, _v_alpha);
        _v_c0 = _mm256_fmadd_pd(_v_a0, _v_b0, _v_c0);

        a_ptr0 += 4;
        B += 4;
    }

    double c0 = reduce_add_pd(_v_c0);

    for (; k < K; ++k) {
        c0 += (*a_ptr0++) * B[0] * alpha;
        ++B;
    }
    C[0] += c0;
}

inline void dmvma_tr6(int K, double alpha, const double* A, int lda, const double* B, double beta, double* C) {
    const double* a_ptr0 = A + 0 * lda;
    const double* a_ptr1 = A + 1 * lda;
    const double* a_ptr2 = A + 2 * lda;
    const double* a_ptr3 = A + 3 * lda;
    const double* a_ptr4 = A + 4 * lda;
    const double* a_ptr5 = A + 5 * lda;

    __m256d _v_alpha = _mm256_broadcast_sd(&alpha);

    __m256d _v_b0 = _mm256_broadcast_sd(B + 0);
    __m256d _v_b1 = _mm256_broadcast_sd(B + 1);
    __m256d _v_b2 = _mm256_broadcast_sd(B + 2);
    __m256d _v_b3 = _mm256_broadcast_sd(B + 3);
    __m256d _v_b4 = _mm256_broadcast_sd(B + 4);
    __m256d _v_b5 = _mm256_broadcast_sd(B + 5);

    _v_b0 = _mm256_mul_pd(_v_b0, _v_alpha);
    _v_b1 = _mm256_mul_pd(_v_b1, _v_alpha);
    _v_b2 = _mm256_mul_pd(_v_b2, _v_alpha);
    _v_b3 = _mm256_mul_pd(_v_b3, _v_alpha);
    _v_b4 = _mm256_mul_pd(_v_b4, _v_alpha);
    _v_b5 = _mm256_mul_pd(_v_b5, _v_alpha);

    int k = 0;
    for (k = 0; k < K - 3; k += 4) {
        __m256d _v_c = _mm256_loadu_pd(C);

        __m256d _v_a0 = _mm256_loadu_pd(a_ptr0);
        __m256d _v_a1 = _mm256_loadu_pd(a_ptr1);
        __m256d _v_a2 = _mm256_loadu_pd(a_ptr2);
        __m256d _v_a3 = _mm256_loadu_pd(a_ptr3);
        __m256d _v_a4 = _mm256_loadu_pd(a_ptr4);
        __m256d _v_a5 = _mm256_loadu_pd(a_ptr5);

        __m256d _v_t0 = _mm256_fmadd_pd(_v_a0, _v_b0, _mm256_mul_pd(_v_a1, _v_b1));
        __m256d _v_t1 = _mm256_fmadd_pd(_v_a2, _v_b2, _mm256_mul_pd(_v_a3, _v_b3));
        __m256d _v_t2 = _mm256_fmadd_pd(_v_a4, _v_b4, _mm256_mul_pd(_v_a5, _v_b5));

        __m256d _v_t = _mm256_add_pd(_v_t0, _mm256_add_pd(_v_t1, _v_t2));
        _v_c = _mm256_add_pd(_v_t, _v_c);
        _mm256_storeu_pd(C, _v_c);

        a_ptr0 += 4;
        a_ptr1 += 4;
        a_ptr2 += 4;
        a_ptr3 += 4;
        a_ptr4 += 4;
        a_ptr5 += 4;
        C += 4;
    }

    for (; k < K; ++k) {
        C[0] += (*a_ptr0++) * B[0] * alpha;
        C[0] += (*a_ptr1++) * B[1] * alpha;
        C[0] += (*a_ptr2++) * B[2] * alpha;
        C[0] += (*a_ptr3++) * B[3] * alpha;
        C[0] += (*a_ptr4++) * B[4] * alpha;
        C[0] += (*a_ptr5++) * B[5] * alpha;
        C += 1;
    }
}

inline void dmvma_tr4(int K, double alpha, const double* A, int lda, const double* B, double beta, double* C) {
    const double* a_ptr0 = A + 0 * lda;
    const double* a_ptr1 = A + 1 * lda;
    const double* a_ptr2 = A + 2 * lda;
    const double* a_ptr3 = A + 3 * lda;
    __m256d _v_alpha = _mm256_broadcast_sd(&alpha);
    __m256d _v_b0 = _mm256_broadcast_sd(B + 0);
    __m256d _v_b1 = _mm256_broadcast_sd(B + 1);
    __m256d _v_b2 = _mm256_broadcast_sd(B + 2);
    __m256d _v_b3 = _mm256_broadcast_sd(B + 3);

    _v_b0 = _mm256_mul_pd(_v_b0, _v_alpha);
    _v_b1 = _mm256_mul_pd(_v_b1, _v_alpha);
    _v_b2 = _mm256_mul_pd(_v_b2, _v_alpha);
    _v_b3 = _mm256_mul_pd(_v_b3, _v_alpha);

    int k = 0;
    for (k = 0; k < K - 3; k += 4) {
        __m256d _v_c = _mm256_loadu_pd(C);

        __m256d _v_a0 = _mm256_loadu_pd(a_ptr0);
        __m256d _v_a1 = _mm256_loadu_pd(a_ptr1);
        __m256d _v_a2 = _mm256_loadu_pd(a_ptr2);
        __m256d _v_a3 = _mm256_loadu_pd(a_ptr3);

        __m256d _v_t0 = _mm256_fmadd_pd(_v_a0, _v_b0, _mm256_mul_pd(_v_a1, _v_b1));
        __m256d _v_t1 = _mm256_fmadd_pd(_v_a2, _v_b2, _mm256_mul_pd(_v_a3, _v_b3));

        __m256d _v_t = _mm256_add_pd(_v_t0, _v_t1);
        _v_c = _mm256_add_pd(_v_t, _v_c);
        _mm256_storeu_pd(C, _v_c);

        a_ptr0 += 4;
        a_ptr1 += 4;
        a_ptr2 += 4;
        a_ptr3 += 4;
        C += 4;
    }

    for (; k < K; ++k) {
        C[0] += (*a_ptr0++) * B[0] * alpha;
        C[0] += (*a_ptr1++) * B[1] * alpha;
        C[0] += (*a_ptr2++) * B[2] * alpha;
        C[0] += (*a_ptr3++) * B[3] * alpha;
        C += 1;
    }
}

inline void dmvma_tr1(int K, double alpha, const double* A, int lda, const double* B, double beta, double* C) {
    for (int k = 0; k < K; ++k) {
        C[k] += A[k] * B[0] * alpha;
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace fastnum