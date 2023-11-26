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

inline void add(float* C, const float* A, const float* B, int size) {
    int i = 0;
    for (i = 0; i < size - 7; i += 8) {
        __m256 _v_a = _mm256_loadu_ps(A);
        __m256 _v_b = _mm256_loadu_ps(B);
        __m256 _v_c = _mm256_add_ps(_v_a, _v_b);
        _mm256_storeu_ps(C, _v_c);

        A += 8;
        B += 8;
        C += 8;
    }
    for (; i < size; ++i) {
        *C++ = *A++ + *B++;
    }
}

inline void sub(float* C, const float* A, const float* B, int size) {
    int i = 0;
    for (i = 0; i < size - 7; i += 8) {
        __m256 _v_a = _mm256_loadu_ps(A);
        __m256 _v_b = _mm256_loadu_ps(B);
        __m256 _v_c = _mm256_sub_ps(_v_a, _v_b);
        _mm256_storeu_ps(C, _v_c);

        A += 8;
        B += 8;
        C += 8;
    }
    for (; i < size; ++i) {
        *C++ = (*A++) - (*B++);
    }
}


inline void mul(float* C, const float* A, const float* B, int size) {
    int i = 0;
    for (i = 0; i < size - 15; i += 16) {
        __m256 _v_a0 = _mm256_loadu_ps(A+0);
        __m256 _v_a1 = _mm256_loadu_ps(A+8);
        __m256 _v_b0 = _mm256_loadu_ps(B+0);
        __m256 _v_b1 = _mm256_loadu_ps(B+8);
        __m256 _v_c0 = _mm256_mul_ps(_v_a0, _v_b0);
        __m256 _v_c1 = _mm256_mul_ps(_v_a1, _v_b1);
        _mm256_storeu_ps(C + 0, _v_c0);
        _mm256_storeu_ps(C + 8, _v_c1);


        A += 16;
        B += 16;
        C += 16;
    }
    for (; i < size - 7; i += 8) {
        __m256 _v_a = _mm256_loadu_ps(A);
        __m256 _v_b = _mm256_loadu_ps(B);
        __m256 _v_c = _mm256_mul_ps(_v_a, _v_b);
        _mm256_storeu_ps(C, _v_c);

        A += 8;
        B += 8;
        C += 8;
    }
    for (; i < size; ++i) {
        *C++ = (*A++) * (*B++);
    }
}

inline void div(float* C, const float* A, const float* B, int size) {
    int i = 0;
    for (i = 0; i < size - 7; i += 8) {
        __m256 _v_a = _mm256_loadu_ps(A);
        __m256 _v_b = _mm256_loadu_ps(B);

        __m256 _v_c = _mm256_div_ps(_v_a, _v_b);
        _mm256_storeu_ps(C, _v_c);

        A += 8;
        B += 8;
        C += 8;
    }
    for (; i < size; ++i) {
        *C++ = (*A++) / (*B++);
    }
}

inline void fma(float* D, const float* A, const float* B, const float* C, int size) {
    int i = 0;
    for (i = 0; i < size - 7; i += 8) {
        __m256 _v_a = _mm256_loadu_ps(A);
        __m256 _v_b = _mm256_loadu_ps(B);
        __m256 _v_c = _mm256_loadu_ps(C);

        __m256 _v_d = _mm256_fmadd_ps(_v_a, _v_b, _v_c);

        _mm256_storeu_ps(D, _v_d);

        A += 8;
        B += 8;
        C += 8;
        D += 8;
    }
    for (; i < size; ++i) {
        *D++ = (*A++) * (*B++) + (*C++);
    }
}


inline void fma(float* D, const float* A, const float* B, const float C, int size) {
    int i = 0;
    __m256 _v_c = _mm256_broadcast_ss(&C);
    for (i = 0; i < size - 7; i += 8) {
        __m256 _v_a = _mm256_loadu_ps(A);
        __m256 _v_b = _mm256_loadu_ps(B);
        __m256 _v_d = _mm256_fmadd_ps(_v_a, _v_b, _v_c);

        _mm256_storeu_ps(D, _v_d);

        A += 8;
        B += 8;
        D += 8;
    }
    for (; i < size; ++i) {
        *D++ = (*A++) * (*B++) + C;
    }
}

inline void add_scaler(float* C, const float* A, const float B, int size) {
    __m256 _v_b = _mm256_broadcast_ss(&B);
    int i = 0;

    for (i = 0; i < size - 7; ++i) {
        __m256 _v_a = _mm256_loadu_ps(A + i);
        _v_a = _mm256_add_ps(_v_a, _v_b);
        _mm256_storeu_ps(C + i, _v_a);
    }
    for (; i < size; ++i) {
        *C++ = (*A++) + B;
    }
}

inline void sub_scaler(float* C, const float* A, const float B, int size) {
    __m256 _v_b = _mm256_broadcast_ss(&B);
    int i = 0;

    for (i = 0; i < size - 7; ++i) {
        __m256 _v_a = _mm256_loadu_ps(A + i);
        _v_a = _mm256_sub_ps(_v_a, _v_b);
        _mm256_storeu_ps(C + i, _v_a);
    }
    for (; i < size; ++i) {
        *C++ = (*A++) - B;
    }
}

inline void mul_scaler(float* C, const float* A, const float B, int size) {
    __m256 _v_b = _mm256_broadcast_ss(&B);
    int i = 0;

    for (i = 0; i < size - 15; i+=16) {
        __m256 _v_a0 = _mm256_loadu_ps(A + 0);
        __m256 _v_a1 = _mm256_loadu_ps(A + 8);
        _v_a0 = _mm256_mul_ps(_v_a0, _v_b);
        _v_a1 = _mm256_mul_ps(_v_a1, _v_b);
        _mm256_storeu_ps(C + 0, _v_a0);
        _mm256_storeu_ps(C + 8, _v_a1);
        A += 16;
        C += 16;
    }
    for (; i < size; ++i) {
        *C++ = (*A++) * B;
    }
}

inline void div_scaler(float* C, const float* A, const float B, int size) {
    float inv_B = static_cast<float>(1.0 / B);
    __m256 _v_b = _mm256_broadcast_ss(&inv_B);
    int i = 0;

    for (i = 0; i < size - 7; ++i) {
        __m256 _v_a = _mm256_loadu_ps(A + i);
        _v_a = _mm256_mul_ps(_v_a, _v_b);
        _mm256_storeu_ps(C + i, _v_a);
    }
    for (; i < size; ++i) {
        *C++ = (*A++) * inv_B;
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace fastnum