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

inline void print_avx_pack(__m256d& v) {
    double d[4];
    _mm256_storeu_pd(d, v);
    printf("%.5f %.5f %.5f %.5f\n", d[0], d[1], d[2], d[3]);
}

inline void dpack_h8(double* packed, const double* X, int ldx, int n_pack, double scale=1.0) {
    const double* x0_ptr = X + 0 * ldx;
    const double* x1_ptr = X + 1 * ldx;
    const double* x2_ptr = X + 2 * ldx;
    const double* x3_ptr = X + 3 * ldx;
    const double* x4_ptr = X + 4 * ldx;
    const double* x5_ptr = X + 5 * ldx;
    const double* x6_ptr = X + 6 * ldx;
    const double* x7_ptr = X + 7 * ldx;

    __m256d _v_scale = _mm256_broadcast_sd(&scale);

    int i = 0;
    for (i = 0; i < n_pack - 3; i += 4) {
        __m256d r0 = _mm256_loadu_pd(x0_ptr);
        __m256d r1 = _mm256_loadu_pd(x1_ptr);
        __m256d r2 = _mm256_loadu_pd(x2_ptr);
        __m256d r3 = _mm256_loadu_pd(x3_ptr);
        __m256d r4 = _mm256_loadu_pd(x4_ptr);
        __m256d r5 = _mm256_loadu_pd(x5_ptr);
        __m256d r6 = _mm256_loadu_pd(x6_ptr);
        __m256d r7 = _mm256_loadu_pd(x7_ptr);

        transpose_4x4(r0, r1, r2, r3);
        transpose_4x4(r4, r5, r6, r7);

        r0 = _mm256_mul_pd(r0, _v_scale);
        r1 = _mm256_mul_pd(r1, _v_scale);
        r2 = _mm256_mul_pd(r2, _v_scale);
        r3 = _mm256_mul_pd(r3, _v_scale);
        r4 = _mm256_mul_pd(r4, _v_scale);
        r5 = _mm256_mul_pd(r5, _v_scale);
        r6 = _mm256_mul_pd(r6, _v_scale);
        r7 = _mm256_mul_pd(r7, _v_scale);

        _mm256_store_pd(packed + 0, r0);
        _mm256_store_pd(packed + 4, r4);
        _mm256_store_pd(packed + 8, r1);
        _mm256_store_pd(packed + 12, r5);
        _mm256_store_pd(packed + 16, r2);
        _mm256_store_pd(packed + 20, r6);
        _mm256_store_pd(packed + 24, r3);
        _mm256_store_pd(packed + 28, r7);

        x0_ptr += 4;
        x1_ptr += 4;
        x2_ptr += 4;
        x3_ptr += 4;
        x4_ptr += 4;
        x5_ptr += 4;
        x6_ptr += 4;
        x7_ptr += 4;
        packed += 32;
    }

    for (; i < n_pack; ++i) {
        *packed++ = (*x0_ptr++)*scale;
        *packed++ = (*x1_ptr++)*scale;
        *packed++ = (*x2_ptr++)*scale;
        *packed++ = (*x3_ptr++)*scale;
        *packed++ = (*x4_ptr++)*scale;
        *packed++ = (*x5_ptr++)*scale;
        *packed++ = (*x6_ptr++)*scale;
        *packed++ = (*x7_ptr++)*scale;
    }
}

inline void dpack_h6(double* packed, const double* X, int ldx, int n_pack, double scale=1.0) {
    const double* x0_ptr = X + 0 * ldx;
    const double* x1_ptr = X + 1 * ldx;
    const double* x2_ptr = X + 2 * ldx;
    const double* x3_ptr = X + 3 * ldx;
    const double* x4_ptr = X + 4 * ldx;
    const double* x5_ptr = X + 5 * ldx;

    __m256d _v_scale = _mm256_broadcast_sd(&scale);

    int i;
    for (i = 0; i < n_pack - 3; i += 4) {
        __m256d r0 = _mm256_loadu_pd(x0_ptr);
        __m256d r1 = _mm256_loadu_pd(x1_ptr);
        __m256d r2 = _mm256_loadu_pd(x2_ptr);
        __m256d r3 = _mm256_loadu_pd(x3_ptr);
        __m256d r4 = _mm256_loadu_pd(x4_ptr);
        __m256d r5 = _mm256_loadu_pd(x5_ptr);

        transpose_6x4(r0, r1, r2, r3, r4, r5);

        r0 = _mm256_mul_pd(r0, _v_scale);
        r1 = _mm256_mul_pd(r1, _v_scale);
        r2 = _mm256_mul_pd(r2, _v_scale);
        r3 = _mm256_mul_pd(r3, _v_scale);
        r4 = _mm256_mul_pd(r4, _v_scale);
        r5 = _mm256_mul_pd(r5, _v_scale);

        _mm256_store_pd(packed + 0, r0);
        _mm256_store_pd(packed + 4, r1);
        _mm256_store_pd(packed + 8, r2);
        _mm256_store_pd(packed + 12, r3);
        _mm256_store_pd(packed + 16, r4);
        _mm256_store_pd(packed + 20, r5);

        x0_ptr += 4;
        x1_ptr += 4;
        x2_ptr += 4;
        x3_ptr += 4;
        x4_ptr += 4;
        x5_ptr += 4;
        packed += 24;
    }

    for (; i < n_pack; ++i) {
        *packed++ = (*x0_ptr++) * scale;
        *packed++ = (*x1_ptr++) * scale;
        *packed++ = (*x2_ptr++) * scale;
        *packed++ = (*x3_ptr++) * scale;
        *packed++ = (*x4_ptr++) * scale;
        *packed++ = (*x5_ptr++) * scale;
    }
}

inline void dpack_h4(double* packed, const double* X, int ldx, int n_pack, double scale=1.0) {
    const double* x0_ptr = X + 0 * ldx;
    const double* x1_ptr = X + 1 * ldx;
    const double* x2_ptr = X + 2 * ldx;
    const double* x3_ptr = X + 3 * ldx;

    __m256d _v_scale = _mm256_broadcast_sd(&scale);

    int i;
    for (i = 0; i < n_pack - 3; i += 4) {
        __m256d r0 = _mm256_loadu_pd(x0_ptr);  // a0,a1,a2,a3,a4,a5,a6,a7
        __m256d r1 = _mm256_loadu_pd(x1_ptr);  // b0,b1,b2,b3,b4,b5,n6,n7
        __m256d r2 = _mm256_loadu_pd(x2_ptr);  // c0,c1,c2,c3,c4,c5,c6,c7
        __m256d r3 = _mm256_loadu_pd(x3_ptr);  // d0,d1,d2,d3,d4,d5,d6,d7

        transpose_4x4(r0, r1, r2, r3);

        r0 = _mm256_mul_pd(r0, _v_scale);
        r1 = _mm256_mul_pd(r1, _v_scale);
        r2 = _mm256_mul_pd(r2, _v_scale);
        r3 = _mm256_mul_pd(r3, _v_scale);

        _mm256_store_pd(packed + 0, r0);
        _mm256_store_pd(packed + 4, r1);
        _mm256_store_pd(packed + 8, r2);
        _mm256_store_pd(packed + 12, r3);

        x0_ptr += 4;
        x1_ptr += 4;
        x2_ptr += 4;
        x3_ptr += 4;
        packed += 16;
    }

    for (; i < n_pack; ++i) {
        *packed++ = (*x0_ptr++)*scale;
        *packed++ = (*x1_ptr++)*scale;
        *packed++ = (*x2_ptr++)*scale;
        *packed++ = (*x3_ptr++)*scale;
    }
}

inline void dpack_v8(double* packed, const double* X, int ldx, int n_pack, double scale=1.0) {
    __m256d _v_scale = _mm256_broadcast_sd(&scale);

    int i;
    for (i = 0; i < n_pack - 3; i += 4) {
        __m256d r00 = _mm256_loadu_pd(X + 0 * ldx + 0);
        __m256d r01 = _mm256_loadu_pd(X + 0 * ldx + 4);
        __m256d r10 = _mm256_loadu_pd(X + 1 * ldx + 0);
        __m256d r11 = _mm256_loadu_pd(X + 1 * ldx + 4);
        __m256d r20 = _mm256_loadu_pd(X + 2 * ldx + 0);
        __m256d r21 = _mm256_loadu_pd(X + 2 * ldx + 4);
        __m256d r30 = _mm256_loadu_pd(X + 3 * ldx + 0);
        __m256d r31 = _mm256_loadu_pd(X + 3 * ldx + 4);

        r00 = _mm256_mul_pd(r00, _v_scale);
        r01 = _mm256_mul_pd(r01, _v_scale);
        r10 = _mm256_mul_pd(r10, _v_scale);
        r11 = _mm256_mul_pd(r11, _v_scale);
        r20 = _mm256_mul_pd(r20, _v_scale);
        r21 = _mm256_mul_pd(r21, _v_scale);
        r30 = _mm256_mul_pd(r30, _v_scale);
        r31 = _mm256_mul_pd(r31, _v_scale);

        _mm256_store_pd(packed + 0, r00);
        _mm256_store_pd(packed + 4, r01);
        _mm256_store_pd(packed + 8, r10);
        _mm256_store_pd(packed + 12, r11);
        _mm256_store_pd(packed + 16, r20);
        _mm256_store_pd(packed + 20, r21);
        _mm256_store_pd(packed + 24, r30);
        _mm256_store_pd(packed + 28, r31);

        X += 4 * ldx;
        packed += 32;
    }

    for (; i < n_pack; ++i) {
        __m256d d0 = _mm256_loadu_pd(X);
        __m256d d1 = _mm256_loadu_pd(X + 4);
        d0 = _mm256_mul_pd(d0, _v_scale);
        d1 = _mm256_mul_pd(d1, _v_scale);
        _mm256_store_pd(packed + 0, d0);
        _mm256_store_pd(packed + 4, d1);
        X += ldx;
        packed += 8;
    }
}

inline void dpack_v6(double* packed, const double* X, int ldx, int n_pack, double scale=1.0) {
    int i = 0;
    for (i = 0; i < n_pack - 3; i += 4) {
        const double* x0_ptr = X + 0 * ldx;
        const double* x1_ptr = X + 1 * ldx;
        const double* x2_ptr = X + 2 * ldx;
        const double* x3_ptr = X + 3 * ldx;

        packed[0] = x0_ptr[0] * scale;
        packed[1] = x0_ptr[1] * scale;
        packed[2] = x0_ptr[2] * scale;
        packed[3] = x0_ptr[3] * scale;
        packed[4] = x0_ptr[4] * scale;
        packed[5] = x0_ptr[5] * scale;
        packed += 6;

        packed[0] = x1_ptr[0] * scale;
        packed[1] = x1_ptr[1] * scale;
        packed[2] = x1_ptr[2] * scale;
        packed[3] = x1_ptr[3] * scale;
        packed[4] = x1_ptr[4] * scale;
        packed[5] = x1_ptr[5] * scale;
        packed += 6;

        packed[0] = x2_ptr[0] * scale;
        packed[1] = x2_ptr[1] * scale;
        packed[2] = x2_ptr[2] * scale;
        packed[3] = x2_ptr[3] * scale;
        packed[4] = x2_ptr[4] * scale;
        packed[5] = x2_ptr[5] * scale;
        packed += 6;

        packed[0] = x3_ptr[0] * scale;
        packed[1] = x3_ptr[1] * scale;
        packed[2] = x3_ptr[2] * scale;
        packed[3] = x3_ptr[3] * scale;
        packed[4] = x3_ptr[4] * scale;
        packed[5] = x3_ptr[5] * scale;
        packed += 6;

        X += 4 * ldx;
    }

    for (; i < n_pack; ++i) {
        packed[0] = X[0] * scale;
        packed[1] = X[1] * scale;
        packed[2] = X[2] * scale;
        packed[3] = X[3] * scale;
        packed[4] = X[4] * scale;
        packed[5] = X[5] * scale;

        packed += 6;
        X += ldx;
    }
}

inline void dpack_v4(double* packed, const double* X, int ldx, int n_pack, double scale=1.0) {
    __m256d _v_scale = _mm256_broadcast_sd(&scale);
    int i = 0;
    for (i = 0; i < n_pack - 3; i += 4) {
        
        __m256d r0 = _mm256_loadu_pd(X + 0 * ldx);
        __m256d r1 = _mm256_loadu_pd(X + 1 * ldx);
        __m256d r2 = _mm256_loadu_pd(X + 2 * ldx);
        __m256d r3 = _mm256_loadu_pd(X + 3 * ldx);

        r0 = _mm256_mul_pd(r0, _v_scale);
        r1 = _mm256_mul_pd(r1, _v_scale);
        r2 = _mm256_mul_pd(r2, _v_scale);
        r3 = _mm256_mul_pd(r3, _v_scale);


        _mm256_store_pd(packed + 0, r0);
        _mm256_store_pd(packed + 4, r1);
        _mm256_store_pd(packed + 8, r2);
        _mm256_store_pd(packed + 12, r3);

        X += 4 * ldx;
        packed += 16;
    }

    for (; i < n_pack; ++i) {
        __m256d d0 = _mm256_loadu_pd(X);
        d0 = _mm256_mul_pd(d0, _v_scale);
        _mm256_store_pd(packed, d0);
        X += ldx;
        packed += 4;
    }
}

inline void dpack_h1(double* packed, const double* X, int ldx, int n_pack, double scale=1.0) {
    // memcpy(packed, X, sizeof(double) * n_pack);
    for(int i=0; i<n_pack; ++i) {
        packed[i] = X[i] * scale;
    }
    packed += n_pack;
}

inline void dpack_v1(double* packed, const double* X, int ldx, int n_pack, double scale=1.0) {
    for (int i = 0; i < n_pack; ++i) {
        *packed++ = *(X + i * ldx) * scale;
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace fastnum