#pragma once

#include <math.h>
#include <memory.h>
#include <stdio.h>

#include <immintrin.h>
#include <nmmintrin.h>
#include <smmintrin.h>

#include "stranspose_x86.hpp"
#include "vectorize_x86.hpp"

namespace fastnum {
namespace cpu {
namespace kernel {

inline void smvma_nr8(int K, const float* A, int lda, const float* B, float* C) {
    const float* a_ptr0 = A + 0 * lda;
    const float* a_ptr1 = A + 1 * lda;
    const float* a_ptr2 = A + 2 * lda;
    const float* a_ptr3 = A + 3 * lda;
    const float* a_ptr4 = A + 4 * lda;
    const float* a_ptr5 = A + 5 * lda;
    const float* a_ptr6 = A + 6 * lda;
    const float* a_ptr7 = A + 7 * lda;

    __m256 _v_c0 = _mm256_setzero_ps();
    __m256 _v_c1 = _mm256_setzero_ps();
    __m256 _v_c2 = _mm256_setzero_ps();
    __m256 _v_c3 = _mm256_setzero_ps();
    __m256 _v_c4 = _mm256_setzero_ps();
    __m256 _v_c5 = _mm256_setzero_ps();
    __m256 _v_c6 = _mm256_setzero_ps();
    __m256 _v_c7 = _mm256_setzero_ps();

    int k = 0;
    for (k = 0; k < K - 7; k += 8) {
        __m256 _v_a0 = _mm256_loadu_ps(a_ptr0);
        __m256 _v_a1 = _mm256_loadu_ps(a_ptr1);
        __m256 _v_a2 = _mm256_loadu_ps(a_ptr2);
        __m256 _v_a3 = _mm256_loadu_ps(a_ptr3);
        __m256 _v_a4 = _mm256_loadu_ps(a_ptr4);
        __m256 _v_a5 = _mm256_loadu_ps(a_ptr5);
        __m256 _v_a6 = _mm256_loadu_ps(a_ptr6);
        __m256 _v_a7 = _mm256_loadu_ps(a_ptr7);

        __m256 _v_b0 = _mm256_loadu_ps(B);

        _v_c0 = _mm256_fmadd_ps(_v_a0, _v_b0, _v_c0);
        _v_c1 = _mm256_fmadd_ps(_v_a1, _v_b0, _v_c1);
        _v_c2 = _mm256_fmadd_ps(_v_a2, _v_b0, _v_c2);
        _v_c3 = _mm256_fmadd_ps(_v_a3, _v_b0, _v_c3);
        _v_c4 = _mm256_fmadd_ps(_v_a4, _v_b0, _v_c4);
        _v_c5 = _mm256_fmadd_ps(_v_a5, _v_b0, _v_c5);
        _v_c6 = _mm256_fmadd_ps(_v_a6, _v_b0, _v_c6);
        _v_c7 = _mm256_fmadd_ps(_v_a7, _v_b0, _v_c7);

        a_ptr0 += 8;
        a_ptr1 += 8;
        a_ptr2 += 8;
        a_ptr3 += 8;
        a_ptr4 += 8;
        a_ptr5 += 8;
        a_ptr6 += 8;
        a_ptr7 += 8;
        B += 8;
    }

    float c0 = reduce_add_ps(_v_c0);
    float c1 = reduce_add_ps(_v_c1);
    float c2 = reduce_add_ps(_v_c2);
    float c3 = reduce_add_ps(_v_c3);
    float c4 = reduce_add_ps(_v_c4);
    float c5 = reduce_add_ps(_v_c5);
    float c6 = reduce_add_ps(_v_c6);
    float c7 = reduce_add_ps(_v_c7);

    for (; k < K; ++k) {
        c0 += (*a_ptr0++) * *B;
        c1 += (*a_ptr1++) * *B;
        c2 += (*a_ptr2++) * *B;
        c3 += (*a_ptr3++) * *B;
        c4 += (*a_ptr4++) * *B;
        c5 += (*a_ptr5++) * *B;
        c6 += (*a_ptr6++) * *B;
        c7 += (*a_ptr7++) * *B;

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


inline void smvma_nr6(int K, const float* A, int lda, const float* B, float* C) {
    const float* a_ptr0 = A + 0 * lda;
    const float* a_ptr1 = A + 1 * lda;
    const float* a_ptr2 = A + 2 * lda;
    const float* a_ptr3 = A + 3 * lda;
    const float* a_ptr4 = A + 4 * lda;
    const float* a_ptr5 = A + 5 * lda;

    __m256 _v_c0 = _mm256_setzero_ps();
    __m256 _v_c1 = _mm256_setzero_ps();
    __m256 _v_c2 = _mm256_setzero_ps();
    __m256 _v_c3 = _mm256_setzero_ps();
    __m256 _v_c4 = _mm256_setzero_ps();
    __m256 _v_c5 = _mm256_setzero_ps();

    int k = 0;
    for (k = 0; k < K - 15; k += 16) {
        __m256 _v_a0 = _mm256_loadu_ps(a_ptr0);
        __m256 _v_a1 = _mm256_loadu_ps(a_ptr1);
        __m256 _v_a2 = _mm256_loadu_ps(a_ptr2);
        __m256 _v_a3 = _mm256_loadu_ps(a_ptr3);
        __m256 _v_a4 = _mm256_loadu_ps(a_ptr4);
        __m256 _v_a5 = _mm256_loadu_ps(a_ptr5);

        __m256 _v_b0 = _mm256_loadu_ps(B);
        __m256 _v_b1 = _mm256_loadu_ps(B + 8);

        _v_c0 = _mm256_fmadd_ps(_v_a0, _v_b0, _v_c0);
        _v_c1 = _mm256_fmadd_ps(_v_a1, _v_b0, _v_c1);
        _v_c2 = _mm256_fmadd_ps(_v_a2, _v_b0, _v_c2);
        _v_c3 = _mm256_fmadd_ps(_v_a3, _v_b0, _v_c3);
        _v_c4 = _mm256_fmadd_ps(_v_a4, _v_b0, _v_c4);
        _v_c5 = _mm256_fmadd_ps(_v_a5, _v_b0, _v_c5);

        _v_a0 = _mm256_loadu_ps(a_ptr0 + 8);
        _v_a1 = _mm256_loadu_ps(a_ptr1 + 8);
        _v_a2 = _mm256_loadu_ps(a_ptr2 + 8);
        _v_a3 = _mm256_loadu_ps(a_ptr3 + 8);
        _v_a4 = _mm256_loadu_ps(a_ptr4 + 8);
        _v_a5 = _mm256_loadu_ps(a_ptr5 + 8);

        _v_c0 = _mm256_fmadd_ps(_v_a0, _v_b1, _v_c0);
        _v_c1 = _mm256_fmadd_ps(_v_a1, _v_b1, _v_c1);
        _v_c2 = _mm256_fmadd_ps(_v_a2, _v_b1, _v_c2);
        _v_c3 = _mm256_fmadd_ps(_v_a3, _v_b1, _v_c3);
        _v_c4 = _mm256_fmadd_ps(_v_a4, _v_b1, _v_c4);
        _v_c5 = _mm256_fmadd_ps(_v_a5, _v_b1, _v_c5);

        a_ptr0 += 16;
        a_ptr1 += 16;
        a_ptr2 += 16;
        a_ptr3 += 16;
        a_ptr4 += 16;
        a_ptr5 += 16;
        B += 16;
    }

    for (; k < K - 7; k += 8) {
        __m256 _v_a0 = _mm256_loadu_ps(a_ptr0);
        __m256 _v_a1 = _mm256_loadu_ps(a_ptr1);
        __m256 _v_a2 = _mm256_loadu_ps(a_ptr2);
        __m256 _v_a3 = _mm256_loadu_ps(a_ptr3);
        __m256 _v_a4 = _mm256_loadu_ps(a_ptr4);
        __m256 _v_a5 = _mm256_loadu_ps(a_ptr5);

        __m256 _v_b0 = _mm256_loadu_ps(B);

        _v_c0 = _mm256_fmadd_ps(_v_a0, _v_b0, _v_c0);
        _v_c1 = _mm256_fmadd_ps(_v_a1, _v_b0, _v_c1);
        _v_c2 = _mm256_fmadd_ps(_v_a2, _v_b0, _v_c2);
        _v_c3 = _mm256_fmadd_ps(_v_a3, _v_b0, _v_c3);
        _v_c4 = _mm256_fmadd_ps(_v_a4, _v_b0, _v_c4);
        _v_c5 = _mm256_fmadd_ps(_v_a5, _v_b0, _v_c5);

        a_ptr0 += 8;
        a_ptr1 += 8;
        a_ptr2 += 8;
        a_ptr3 += 8;
        a_ptr4 += 8;
        a_ptr5 += 8;
        B += 8;
    }

    float c0 = reduce_add_ps(_v_c0);
    float c1 = reduce_add_ps(_v_c1);
    float c2 = reduce_add_ps(_v_c2);
    float c3 = reduce_add_ps(_v_c3);
    float c4 = reduce_add_ps(_v_c4);
    float c5 = reduce_add_ps(_v_c5);

    for (; k < K; ++k) {
        c0 += (*a_ptr0++) * *B;
        c1 += (*a_ptr1++) * *B;
        c2 += (*a_ptr2++) * *B;
        c3 += (*a_ptr3++) * *B;
        c4 += (*a_ptr4++) * *B;
        c5 += (*a_ptr5++) * *B;

        ++B;
    }

    C[0] += c0;
    C[1] += c1;
    C[2] += c2;
    C[3] += c3;
    C[4] += c4;
    C[5] += c5;
}

inline void smvma_nr4(int K, const float* A, int lda, const float* B, float* C) {
    const float* a_ptr0 = A + 0 * lda;
    const float* a_ptr1 = A + 1 * lda;
    const float* a_ptr2 = A + 2 * lda;
    const float* a_ptr3 = A + 3 * lda;

    __m256 _v_c0 = _mm256_setzero_ps();
    __m256 _v_c1 = _mm256_setzero_ps();
    __m256 _v_c2 = _mm256_setzero_ps();
    __m256 _v_c3 = _mm256_setzero_ps();

    int k = 0;
    for (k = 0; k < K - 15; k += 16) {
        __m256 _v_a0 = _mm256_loadu_ps(a_ptr0);
        __m256 _v_a1 = _mm256_loadu_ps(a_ptr1);
        __m256 _v_a2 = _mm256_loadu_ps(a_ptr2);
        __m256 _v_a3 = _mm256_loadu_ps(a_ptr3);

        __m256 _v_b0 = _mm256_loadu_ps(B);
        __m256 _v_b1 = _mm256_loadu_ps(B + 8);

        _v_c0 = _mm256_fmadd_ps(_v_a0, _v_b0, _v_c0);
        _v_c1 = _mm256_fmadd_ps(_v_a1, _v_b0, _v_c1);
        _v_c2 = _mm256_fmadd_ps(_v_a2, _v_b0, _v_c2);
        _v_c3 = _mm256_fmadd_ps(_v_a3, _v_b0, _v_c3);

        _v_a0 = _mm256_loadu_ps(a_ptr0 + 8);
        _v_a1 = _mm256_loadu_ps(a_ptr1 + 8);
        _v_a2 = _mm256_loadu_ps(a_ptr2 + 8);
        _v_a3 = _mm256_loadu_ps(a_ptr3 + 8);

        _v_c0 = _mm256_fmadd_ps(_v_a0, _v_b1, _v_c0);
        _v_c1 = _mm256_fmadd_ps(_v_a1, _v_b1, _v_c1);
        _v_c2 = _mm256_fmadd_ps(_v_a2, _v_b1, _v_c2);
        _v_c3 = _mm256_fmadd_ps(_v_a3, _v_b1, _v_c3);

        a_ptr0 += 16;
        a_ptr1 += 16;
        a_ptr2 += 16;
        a_ptr3 += 16;
        B += 16;
    }

    for (; k < K - 7; k += 8) {
        __m256 _v_a0 = _mm256_loadu_ps(a_ptr0);
        __m256 _v_a1 = _mm256_loadu_ps(a_ptr1);
        __m256 _v_a2 = _mm256_loadu_ps(a_ptr2);
        __m256 _v_a3 = _mm256_loadu_ps(a_ptr3);

        __m256 _v_b0 = _mm256_loadu_ps(B);

        _v_c0 = _mm256_fmadd_ps(_v_a0, _v_b0, _v_c0);
        _v_c1 = _mm256_fmadd_ps(_v_a1, _v_b0, _v_c1);
        _v_c2 = _mm256_fmadd_ps(_v_a2, _v_b0, _v_c2);
        _v_c3 = _mm256_fmadd_ps(_v_a3, _v_b0, _v_c3);

        a_ptr0 += 8;
        a_ptr1 += 8;
        a_ptr2 += 8;
        a_ptr3 += 8;
        B += 8;
    }

    float c0 = reduce_add_ps(_v_c0);
    float c1 = reduce_add_ps(_v_c1);
    float c2 = reduce_add_ps(_v_c2);
    float c3 = reduce_add_ps(_v_c3);

    for (; k < K; ++k) {
        c0 += (*a_ptr0++) * *B;
        c1 += (*a_ptr1++) * *B;
        c2 += (*a_ptr2++) * *B;
        c3 += (*a_ptr3++) * *B;

        ++B;
    }

    C[0] += c0;
    C[1] += c1;
    C[2] += c2;
    C[3] += c3;
}

inline void smvma_nr1(int K, const float* A, int lda, const float* B, float* C) {
    const float* a_ptr0 = A + 0 * lda;

    __m256 _v_c0 = _mm256_setzero_ps();

    int k = 0;
    for (k = 0; k < K - 15; k += 16) {
        __m256 _v_a0 = _mm256_loadu_ps(a_ptr0);
        __m256 _v_b0 = _mm256_loadu_ps(B);
        __m256 _v_b1 = _mm256_loadu_ps(B + 8);
        _v_c0 = _mm256_fmadd_ps(_v_a0, _v_b0, _v_c0);
        _v_a0 = _mm256_loadu_ps(a_ptr0 + 8);
        _v_c0 = _mm256_fmadd_ps(_v_a0, _v_b1, _v_c0);
        a_ptr0 += 16;
        B += 16;
    }

    for (; k < K - 7; k += 8) {
        __m256 _v_a0 = _mm256_loadu_ps(a_ptr0);
        __m256 _v_b0 = _mm256_loadu_ps(B);
        _v_c0 = _mm256_fmadd_ps(_v_a0, _v_b0, _v_c0);
        a_ptr0 += 8;
        B += 8;
    }

    float c0 = reduce_add_ps(_v_c0);
    for (; k < K; ++k) {
        c0 += (*a_ptr0++) * *B;
        ++B;
    }
    C[0] += c0;
}

inline void smvma_tr6(int K, const float* A, int lda, const float* B, float* C) {
    const float* a_ptr0 = A + 0 * lda;
    const float* a_ptr1 = A + 1 * lda;
    const float* a_ptr2 = A + 2 * lda;
    const float* a_ptr3 = A + 3 * lda;
    const float* a_ptr4 = A + 4 * lda;
    const float* a_ptr5 = A + 5 * lda;

    __m256 _v_w0 = _mm256_broadcast_ss(B + 0);
    __m256 _v_w1 = _mm256_broadcast_ss(B + 1);
    __m256 _v_w2 = _mm256_broadcast_ss(B + 2);
    __m256 _v_w3 = _mm256_broadcast_ss(B + 3);
    __m256 _v_w4 = _mm256_broadcast_ss(B + 4);
    __m256 _v_w5 = _mm256_broadcast_ss(B + 5);

    int k = 0;
    for (k = 0; k < K - 7; k += 8) {
        __m256 _v_c = _mm256_loadu_ps(C);

        __m256 _v_a0 = _mm256_loadu_ps(a_ptr0);
        __m256 _v_a1 = _mm256_loadu_ps(a_ptr1);
        __m256 _v_a2 = _mm256_loadu_ps(a_ptr2);
        __m256 _v_a3 = _mm256_loadu_ps(a_ptr3);
        __m256 _v_a4 = _mm256_loadu_ps(a_ptr4);
        __m256 _v_a5 = _mm256_loadu_ps(a_ptr5);

        __m256 _v_t0 = _mm256_fmadd_ps(_v_a1, _v_w1, _mm256_mul_ps(_v_a0, _v_w0));
        __m256 _v_t1 = _mm256_fmadd_ps(_v_a3, _v_w3, _mm256_mul_ps(_v_a2, _v_w2));
        __m256 _v_t2 = _mm256_fmadd_ps(_v_a5, _v_w5, _mm256_mul_ps(_v_a4, _v_w4));

        __m256 _v_t = _mm256_add_ps(_mm256_add_ps(_v_t0, _v_t1), _v_t2);
        _v_c = _mm256_add_ps(_v_c, _v_t);
        _mm256_storeu_ps(C, _v_c);

        a_ptr0 += 8;
        a_ptr1 += 8;
        a_ptr2 += 8;
        a_ptr3 += 8;
        a_ptr4 += 8;
        a_ptr5 += 8;
        C += 8;
    }

    for (; k < K; ++k) {
        *C += (*a_ptr0++) * B[0];
        *C += (*a_ptr1++) * B[1];
        *C += (*a_ptr2++) * B[2];
        *C += (*a_ptr3++) * B[3];
        *C += (*a_ptr4++) * B[4];
        *C += (*a_ptr5++) * B[5];
        ++C;
    }
}

inline void smvma_tr4(int K, const float* A, int lda, const float* B, float* C) {
    const float* a_ptr0 = A + 0 * lda;
    const float* a_ptr1 = A + 1 * lda;
    const float* a_ptr2 = A + 2 * lda;
    const float* a_ptr3 = A + 3 * lda;

    __m256 _v_w0 = _mm256_broadcast_ss(B + 0);
    __m256 _v_w1 = _mm256_broadcast_ss(B + 1);
    __m256 _v_w2 = _mm256_broadcast_ss(B + 2);
    __m256 _v_w3 = _mm256_broadcast_ss(B + 3);


    int k = 0;
    for (k = 0; k < K - 7; k += 8) {
        __m256 _v_c = _mm256_loadu_ps(C);

        __m256 _v_a0 = _mm256_loadu_ps(a_ptr0);
        __m256 _v_a1 = _mm256_loadu_ps(a_ptr1);
        __m256 _v_a2 = _mm256_loadu_ps(a_ptr2);
        __m256 _v_a3 = _mm256_loadu_ps(a_ptr3);

        __m256 _v_t0 = _mm256_fmadd_ps(_v_a1, _v_w1, _mm256_mul_ps(_v_a0, _v_w0));
        __m256 _v_t1 = _mm256_fmadd_ps(_v_a3, _v_w3, _mm256_mul_ps(_v_a2, _v_w2));

        __m256 _v_t = _mm256_add_ps(_v_t0, _v_t1);
        _v_c = _mm256_add_ps(_v_c, _v_t);
        _mm256_storeu_ps(C, _v_c);

        a_ptr0 += 8;
        a_ptr1 += 8;
        a_ptr2 += 8;
        a_ptr3 += 8;
        C += 8;
    }

    for (; k < K; ++k) {
        *C += (*a_ptr0++) * B[0];
        *C += (*a_ptr1++) * B[1];
        *C += (*a_ptr2++) * B[2];
        *C += (*a_ptr3++) * B[3];
        ++C;
    }
}

inline void smvma_tr1(int K, const float* A, int lda, const float* B, float* C) {
    const float* a_ptr0 = A + 0 * lda;
    __m256 _v_w0 = _mm256_broadcast_ss(B + 0);

    int k = 0;
    for (k = 0; k < K - 7; k += 8) {
        __m256 _v_c = _mm256_loadu_ps(C);
        __m256 _v_a0 = _mm256_loadu_ps(a_ptr0);
        _v_c = _mm256_fmadd_ps(_v_a0, _v_w0, _v_c);
        _mm256_storeu_ps(C, _v_c);
        a_ptr0 += 8;
        C += 8;
    }
    for (; k < K; ++k) {
        *C += (*a_ptr0++) * B[0];
        ++C;
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace fastnum