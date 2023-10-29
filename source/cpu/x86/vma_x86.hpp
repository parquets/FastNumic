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


inline void vma6(int K, const float* A, int lda, const float* B, float* C) {
    const float* a_ptr0 = A + 0*lda;
    const float* a_ptr1 = A + 1*lda;
    const float* a_ptr2 = A + 2*lda;
    const float* a_ptr3 = A + 3*lda;
    const float* a_ptr4 = A + 4*lda;
    const float* a_ptr5 = A + 5*lda;

    __m256 _v_c0 = _mm256_setzero_ps();
    __m256 _v_c1 = _mm256_setzero_ps();
    __m256 _v_c2 = _mm256_setzero_ps();
    __m256 _v_c3 = _mm256_setzero_ps();
    __m256 _v_c4 = _mm256_setzero_ps();
    __m256 _v_c5 = _mm256_setzero_ps();

    int k=0;
    for(k=0; k<K-15; k+=16) {
        __m256 _v_a0 = _mm256_loadu_ps(a_ptr0);
        __m256 _v_a1 = _mm256_loadu_ps(a_ptr1);
        __m256 _v_a2 = _mm256_loadu_ps(a_ptr2);
        __m256 _v_a3 = _mm256_loadu_ps(a_ptr3);
        __m256 _v_a4 = _mm256_loadu_ps(a_ptr4);
        __m256 _v_a5 = _mm256_loadu_ps(a_ptr5);

        __m256 _v_b0 = _mm256_loadu_ps(B);
        __m256 _v_b1 = _mm256_loadu_ps(B+8);

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

    for(; k<K-7; k+=8) {
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

    for(; k<K; ++k) {
        c0 += (*a_ptr0++) * *B;
        c1 += (*a_ptr1++) * *B;
        c2 += (*a_ptr2++) * *B;
        c3 += (*a_ptr3++) * *B;
        c4 += (*a_ptr4++) * *B;
        c5 += (*a_ptr5++) * *B;

        ++B;
    }

    C[0] = c0;
    C[1] = c1;
    C[2] = c2;
    C[3] = c3;
    C[4] = c4;
    C[5] = c5;

}

inline void vma4(int K, const float* A, int lda, const float* B, float* C) {
    const float* a_ptr0 = A + 0*lda;
    const float* a_ptr1 = A + 1*lda;
    const float* a_ptr2 = A + 2*lda;
    const float* a_ptr3 = A + 3*lda;

    __m256 _v_c0 = _mm256_setzero_ps();
    __m256 _v_c1 = _mm256_setzero_ps();
    __m256 _v_c2 = _mm256_setzero_ps();
    __m256 _v_c3 = _mm256_setzero_ps();

    int k=0;
    for(k=0; k<K-15; k+=16) {
        __m256 _v_a0 = _mm256_loadu_ps(a_ptr0);
        __m256 _v_a1 = _mm256_loadu_ps(a_ptr1);
        __m256 _v_a2 = _mm256_loadu_ps(a_ptr2);
        __m256 _v_a3 = _mm256_loadu_ps(a_ptr3);

        __m256 _v_b0 = _mm256_loadu_ps(B);
        __m256 _v_b1 = _mm256_loadu_ps(B+8);

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

    for(; k<K-7; k+=8) {
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

    for(; k<K; ++k) {
        c0 += (*a_ptr0++) * *B;
        c1 += (*a_ptr1++) * *B;
        c2 += (*a_ptr2++) * *B;
        c3 += (*a_ptr3++) * *B;

        ++B;
    }

    C[0] = c0;
    C[1] = c1;
    C[2] = c2;
    C[3] = c3;
}

inline void vma1(int K, const float* A, int lda, const float* B, float* C) {
    const float* a_ptr0 = A + 0*lda;

    __m256 _v_c0 = _mm256_setzero_ps();

    int k=0;
    for(k=0; k<K-15; k+=16) {
        __m256 _v_a0 = _mm256_loadu_ps(a_ptr0);
        __m256 _v_b0 = _mm256_loadu_ps(B);
        __m256 _v_b1 = _mm256_loadu_ps(B+8);
        _v_c0 = _mm256_fmadd_ps(_v_a0, _v_b0, _v_c0);
        _v_a0 = _mm256_loadu_ps(a_ptr0 + 8);
        _v_c0 = _mm256_fmadd_ps(_v_a0, _v_b1, _v_c0);
        a_ptr0 += 16;
        B += 16;
    }

    for(; k<K-7; k+=8) {
        __m256 _v_a0 = _mm256_loadu_ps(a_ptr0);
        __m256 _v_b0 = _mm256_loadu_ps(B);
        _v_c0 = _mm256_fmadd_ps(_v_a0, _v_b0, _v_c0);
        a_ptr0 += 8;
        B += 8;
    }

    float c0 = reduce_add_ps(_v_c0);
    for(; k<K; ++k) {
        c0 += (*a_ptr0++) * *B;
        ++B;
    }
    C[0] = c0;
}


}
}
}