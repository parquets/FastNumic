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

inline void spack_h8(float* packed, const float* X, int ldx, int n_pack) {
    /* 水平方向的 pack 操作*/
    const float* x0_ptr = X + 0 * ldx;
    const float* x1_ptr = X + 1 * ldx;
    const float* x2_ptr = X + 2 * ldx;
    const float* x3_ptr = X + 3 * ldx;
    const float* x4_ptr = X + 4 * ldx;
    const float* x5_ptr = X + 5 * ldx;
    const float* x6_ptr = X + 6 * ldx;
    const float* x7_ptr = X + 7 * ldx;

    int i;
    for (i = 0; i < n_pack - 7; i += 8) {
        __m256 _v_a0 = _mm256_loadu_ps(x0_ptr);  // a0,a1,a2,a3,a4,a5,a6,a7
        __m256 _v_a1 = _mm256_loadu_ps(x1_ptr);  // b0,b1,b2,b3,b4,b5,b6,b7
        __m256 _v_a2 = _mm256_loadu_ps(x2_ptr);  // c0,c1,c2,c3,c4,c5,c6,c7
        __m256 _v_a3 = _mm256_loadu_ps(x3_ptr);  // d0,d1,d2,d3,d4,d5,d6,d7
        __m256 _v_a4 = _mm256_loadu_ps(x4_ptr);  // e0,e1,e2,e3,e4,e5,e6,e7
        __m256 _v_a5 = _mm256_loadu_ps(x5_ptr);  // f0,f1,f2,f3,f4,f5,f6,f7
        __m256 _v_a6 = _mm256_loadu_ps(x6_ptr);  // g0,g1,g2,g3,g4,g5,g6,g7
        __m256 _v_a7 = _mm256_loadu_ps(x7_ptr);  // h0,d1,h2,h3,h4,h5,h6,h7

        transpose_8x8(_v_a0, _v_a1, _v_a2, _v_a3, _v_a4, _v_a5, _v_a6, _v_a7);

        _mm256_store_ps(packed + 0, _v_a0);
        _mm256_store_ps(packed + 8, _v_a1);
        _mm256_store_ps(packed + 16, _v_a2);
        _mm256_store_ps(packed + 24, _v_a3);
        _mm256_store_ps(packed + 32, _v_a4);
        _mm256_store_ps(packed + 40, _v_a5);
        _mm256_store_ps(packed + 48, _v_a6);
        _mm256_store_ps(packed + 56, _v_a7);

        x0_ptr += 8;
        x1_ptr += 8;
        x2_ptr += 8;
        x3_ptr += 8;
        x4_ptr += 8;
        x5_ptr += 8;
        x6_ptr += 8;
        x7_ptr += 8;
        packed += 64;
    }

    for (; i < n_pack; ++i) {
        *packed++ = *x0_ptr++;
        *packed++ = *x1_ptr++;
        *packed++ = *x2_ptr++;
        *packed++ = *x3_ptr++;
        *packed++ = *x4_ptr++;
        *packed++ = *x5_ptr++;
        *packed++ = *x6_ptr++;
        *packed++ = *x7_ptr++;
    }

}

inline void spack_v8(float* packed, const float* X, int ldx, int n_pack) {
    /* 垂直方向的 pack 操作*/
    int i = 0;

    for (i = 0; i < n_pack - 7; i += 8) {
        __m256 tmp0 = _mm256_loadu_ps(X + 0 * ldx);
        __m256 tmp1 = _mm256_loadu_ps(X + 1 * ldx);
        __m256 tmp2 = _mm256_loadu_ps(X + 2 * ldx);
        __m256 tmp3 = _mm256_loadu_ps(X + 3 * ldx);
        __m256 tmp4 = _mm256_loadu_ps(X + 4 * ldx);
        __m256 tmp5 = _mm256_loadu_ps(X + 5 * ldx);
        __m256 tmp6 = _mm256_loadu_ps(X + 6 * ldx);
        __m256 tmp7 = _mm256_loadu_ps(X + 7 * ldx);

        _mm256_store_ps(packed + 0, tmp0);
        _mm256_store_ps(packed + 8, tmp1);
        _mm256_store_ps(packed + 16, tmp2);
        _mm256_store_ps(packed + 24, tmp3);
        _mm256_store_ps(packed + 32, tmp4);
        _mm256_store_ps(packed + 40, tmp5);
        _mm256_store_ps(packed + 48, tmp6);
        _mm256_store_ps(packed + 56, tmp7);

        X += 8 * ldx;
        packed += 64;
    }

    for (; i < n_pack; ++i) {
        __m256 tmp = _mm256_loadu_ps(X);
        _mm256_storeu_ps(packed, tmp);
        X += ldx;
        packed += 8;
    }
}

inline void spack_h6(float* packed, const float* X, int ldx, int n_pack) {
    const float* x0_ptr = X + 0 * ldx;
    const float* x1_ptr = X + 1 * ldx;
    const float* x2_ptr = X + 2 * ldx;
    const float* x3_ptr = X + 3 * ldx;
    const float* x4_ptr = X + 4 * ldx;
    const float* x5_ptr = X + 5 * ldx;

    int i;
    for (i = 0; i < n_pack - 7; i += 8) {
        __m256 r0 = _mm256_loadu_ps(x0_ptr);  // a0,a1,a2,a3,a4,a5,a6,a7
        __m256 r1 = _mm256_loadu_ps(x1_ptr);  // b0,b1,b2,b3,b4,b5,n6,n7
        __m256 r2 = _mm256_loadu_ps(x2_ptr);  // c0,c1,c2,c3,c4,c5,c6,c7
        __m256 r3 = _mm256_loadu_ps(x3_ptr);  // d0,d1,d2,d3,d4,d5,d6,d7
        __m256 r4 = _mm256_loadu_ps(x4_ptr);  // e0,e1,e2,e3,e4,e5,e6,e7
        __m256 r5 = _mm256_loadu_ps(x5_ptr);  // f0,f1,f2,f3,f4,f5,f6,f7

        _mm_prefetch((char*)(x0_ptr + 8), _MM_HINT_T0);
        _mm_prefetch((char*)(x1_ptr + 8), _MM_HINT_T0);
        _mm_prefetch((char*)(x2_ptr + 8), _MM_HINT_T0);
        _mm_prefetch((char*)(x4_ptr + 8), _MM_HINT_T0);
        _mm_prefetch((char*)(x5_ptr + 8), _MM_HINT_T0);

        transpose_6x8(r0, r1, r2, r3, r4, r5);

        _mm256_store_ps(packed + 0, r0);
        _mm256_store_ps(packed + 8, r1);
        _mm256_store_ps(packed + 16, r2);
        _mm256_store_ps(packed + 24, r3);
        _mm256_store_ps(packed + 32, r4);
        _mm256_store_ps(packed + 40, r5);

        x0_ptr += 8;
        x1_ptr += 8;
        x2_ptr += 8;
        x3_ptr += 8;
        x4_ptr += 8;
        x5_ptr += 8;
        packed += 48;
    }

    for (; i < n_pack; ++i) {
        *packed++ = *x0_ptr++;
        *packed++ = *x1_ptr++;
        *packed++ = *x2_ptr++;
        *packed++ = *x3_ptr++;
        *packed++ = *x4_ptr++;
        *packed++ = *x5_ptr++;
    }
}

inline void spack_v6(float* packed, const float* X, int ldx, int n_pack) {
    int i = 0;
    for (i = 0; i < n_pack - 3; i += 4) {
        const float* x0_ptr = X + 0 * ldx;
        const float* x1_ptr = X + 1 * ldx;
        const float* x2_ptr = X + 2 * ldx;
        const float* x3_ptr = X + 3 * ldx;

        packed[0] = x0_ptr[0];
        packed[1] = x0_ptr[1];
        packed[2] = x0_ptr[2];
        packed[3] = x0_ptr[3];
        packed[4] = x0_ptr[4];
        packed[5] = x0_ptr[5];
        packed += 6;

        packed[0] = x1_ptr[0];
        packed[1] = x1_ptr[1];
        packed[2] = x1_ptr[2];
        packed[3] = x1_ptr[3];
        packed[4] = x1_ptr[4];
        packed[5] = x1_ptr[5];
        packed += 6;

        packed[0] = x2_ptr[0];
        packed[1] = x2_ptr[1];
        packed[2] = x2_ptr[2];
        packed[3] = x2_ptr[3];
        packed[4] = x2_ptr[4];
        packed[5] = x2_ptr[5];
        packed += 6;

        packed[0] = x3_ptr[0];
        packed[1] = x3_ptr[1];
        packed[2] = x3_ptr[2];
        packed[3] = x3_ptr[3];
        packed[4] = x3_ptr[4];
        packed[5] = x3_ptr[5];
        packed += 6;

        X += 4 * ldx;
    }

    for (; i < n_pack; ++i) {
        packed[0] = X[0];
        packed[1] = X[1];
        packed[2] = X[2];
        packed[3] = X[3];
        packed[4] = X[4];
        packed[5] = X[5];

        packed += 6;
        X += ldx;
    }
}

inline void spack_h4(float* packed, const float* X, int ldx, int n_pack) {
    /* 水平方向的 pack 操作*/
    const float* x0_ptr = X + 0 * ldx;
    const float* x1_ptr = X + 1 * ldx;
    const float* x2_ptr = X + 2 * ldx;
    const float* x3_ptr = X + 3 * ldx;

    int i;
    for (i = 0; i < n_pack - 3; i += 4) {
        __m128 _v_a0 = _mm_loadu_ps(x0_ptr);  // a0,a1,a2,a3
        __m128 _v_a1 = _mm_loadu_ps(x1_ptr);  // b0,b1,b2,b3
        __m128 _v_a2 = _mm_loadu_ps(x2_ptr);  // c0,c1,c2,c3
        __m128 _v_a3 = _mm_loadu_ps(x3_ptr);  // d0,d1,d2,d3

        transpose_4x4(_v_a0, _v_a1, _v_a2, _v_a3);

        _mm_store_ps(packed + 0, _v_a0);
        _mm_store_ps(packed + 4, _v_a1);
        _mm_store_ps(packed + 8, _v_a2);
        _mm_store_ps(packed + 12, _v_a3);

        x0_ptr += 4;
        x1_ptr += 4;
        x2_ptr += 4;
        x3_ptr += 4;
        packed += 16;
    }

    for (; i < n_pack; ++i) {
        *packed++ = *x0_ptr++;
        *packed++ = *x1_ptr++;
        *packed++ = *x2_ptr++;
        *packed++ = *x3_ptr++;
    }

}

inline void spack_v4(float* packed, const float* X, int ldx, int n_pack) {
    int i = 0;
    for (i = 0; i < n_pack - 3; i += 4) {
        __m128 _v_x0 = _mm_loadu_ps(X + 0 * ldx);
        __m128 _v_x1 = _mm_loadu_ps(X + 1 * ldx);
        __m128 _v_x2 = _mm_loadu_ps(X + 2 * ldx);
        __m128 _v_x3 = _mm_loadu_ps(X + 3 * ldx);

        _mm_storeu_ps(packed, _v_x0);
        _mm_storeu_ps(packed + 4, _v_x1);
        _mm_storeu_ps(packed + 8, _v_x2);
        _mm_storeu_ps(packed + 12, _v_x3);

        packed += 16;
        X += 4 * ldx;
    }
    for (; i < n_pack; ++i) {
        *packed++ = X[0];
        *packed++ = X[1];
        *packed++ = X[2];
        *packed++ = X[3];
        X += ldx;
    }

}

inline void spack_h16(float* packed, const float* X, int ldx, int n_pack) {
    const float* x0_ptr = X + 0 * ldx;
    const float* x1_ptr = X + 1 * ldx;
    const float* x2_ptr = X + 2 * ldx;
    const float* x3_ptr = X + 3 * ldx;
    const float* x4_ptr = X + 4 * ldx;
    const float* x5_ptr = X + 5 * ldx;
    const float* x6_ptr = X + 6 * ldx;
    const float* x7_ptr = X + 7 * ldx;

    const float* x8_ptr = X + 8 * ldx;
    const float* x9_ptr = X + 9 * ldx;
    const float* x10_ptr = X + 10 * ldx;
    const float* x11_ptr = X + 11 * ldx;
    const float* x12_ptr = X + 12 * ldx;
    const float* x13_ptr = X + 13 * ldx;
    const float* x14_ptr = X + 14 * ldx;
    const float* x15_ptr = X + 15 * ldx;

    int i;
    for (i = 0; i < n_pack - 7; i += 8) {
        __m256 _v_a0 = _mm256_loadu_ps(x0_ptr);
        __m256 _v_a1 = _mm256_loadu_ps(x1_ptr);
        __m256 _v_a2 = _mm256_loadu_ps(x2_ptr);
        __m256 _v_a3 = _mm256_loadu_ps(x3_ptr);
        __m256 _v_a4 = _mm256_loadu_ps(x4_ptr);
        __m256 _v_a5 = _mm256_loadu_ps(x5_ptr);
        __m256 _v_a6 = _mm256_loadu_ps(x6_ptr);
        __m256 _v_a7 = _mm256_loadu_ps(x7_ptr);

        transpose_8x8(_v_a0, _v_a1, _v_a2, _v_a3, _v_a4, _v_a5, _v_a6, _v_a7);

        __m256 _v_a8 = _mm256_loadu_ps(x8_ptr);
        __m256 _v_a9 = _mm256_loadu_ps(x9_ptr);
        __m256 _v_a10 = _mm256_loadu_ps(x10_ptr);
        __m256 _v_a11 = _mm256_loadu_ps(x11_ptr);
        __m256 _v_a12 = _mm256_loadu_ps(x12_ptr);
        __m256 _v_a13 = _mm256_loadu_ps(x13_ptr);
        __m256 _v_a14 = _mm256_loadu_ps(x14_ptr);
        __m256 _v_a15 = _mm256_loadu_ps(x15_ptr);

        transpose_8x8(_v_a8, _v_a9, _v_a10, _v_a11, _v_a12, _v_a13, _v_a14, _v_a15);

        _mm256_store_ps(packed, _v_a0);
        _mm256_store_ps(packed + 8, _v_a8);
        _mm256_store_ps(packed + 16, _v_a1);
        _mm256_store_ps(packed + 24, _v_a9);
        _mm256_store_ps(packed + 32, _v_a2);
        _mm256_store_ps(packed + 40, _v_a10);
        _mm256_store_ps(packed + 48, _v_a3);
        _mm256_store_ps(packed + 56, _v_a11);
        _mm256_store_ps(packed + 64, _v_a4);
        _mm256_store_ps(packed + 72, _v_a12);
        _mm256_store_ps(packed + 80, _v_a5);
        _mm256_store_ps(packed + 88, _v_a13);
        _mm256_store_ps(packed + 96, _v_a6);
        _mm256_store_ps(packed + 104, _v_a14);
        _mm256_store_ps(packed + 112, _v_a7);
        _mm256_store_ps(packed + 120, _v_a15);

        packed += 128;
        x0_ptr += 8;
        x1_ptr += 8;
        x2_ptr += 8;
        x3_ptr += 8;
        x4_ptr += 8;
        x5_ptr += 8;
        x6_ptr += 8;
        x7_ptr += 8;
        x8_ptr += 8;
        x9_ptr += 8;
        x10_ptr += 8;
        x11_ptr += 8;
        x12_ptr += 8;
        x13_ptr += 8;
        x14_ptr += 8;
        x15_ptr += 8;
    }

    for (; i < n_pack; ++i) {
        *packed++ = *x0_ptr++;
        *packed++ = *x1_ptr++;
        *packed++ = *x2_ptr++;
        *packed++ = *x3_ptr++;
        *packed++ = *x4_ptr++;
        *packed++ = *x5_ptr++;
        *packed++ = *x6_ptr++;
        *packed++ = *x7_ptr++;
        *packed++ = *x8_ptr++;
        *packed++ = *x9_ptr++;
        *packed++ = *x10_ptr++;
        *packed++ = *x11_ptr++;
        *packed++ = *x12_ptr++;
        *packed++ = *x13_ptr++;
        *packed++ = *x14_ptr++;
        *packed++ = *x15_ptr++;
    }
}

inline void spack_v16(float* packed, const float* X, int ldx, int n_pack) {
    int i = 0;
    for (i = 0; i < n_pack - 3; i += 4) {
        __m256 d0 = _mm256_loadu_ps(X);
        __m256 d1 = _mm256_loadu_ps(X + 8);
        __m256 d2 = _mm256_loadu_ps(X + ldx);
        __m256 d3 = _mm256_loadu_ps(X + 8 + ldx);
        __m256 d4 = _mm256_loadu_ps(X + 2 * ldx);
        __m256 d5 = _mm256_loadu_ps(X + 8 + 2 * ldx);
        __m256 d6 = _mm256_loadu_ps(X + 3 * ldx);
        __m256 d7 = _mm256_loadu_ps(X + 8 + 3 * ldx);

        _mm256_store_ps(packed, d0);
        _mm256_store_ps(packed + 8, d1);
        _mm256_store_ps(packed + 16, d2);
        _mm256_store_ps(packed + 24, d3);
        _mm256_store_ps(packed + 32, d4);
        _mm256_store_ps(packed + 40, d5);
        _mm256_store_ps(packed + 48, d6);
        _mm256_store_ps(packed + 56, d7);

        X += 4 * ldx;
        packed += 64;
    }

    for (; i < n_pack; ++i) {
        __m256 d0 = _mm256_loadu_ps(X);
        __m256 d1 = _mm256_loadu_ps(X + 8);
        _mm256_store_ps(packed, d0);
        _mm256_store_ps(packed + 8, d1);
        X += ldx;
        packed += 16;
    }
}



inline void spack_h1(float* packed, const float* X, int ldx, int n_pack) {
    memcpy(packed, X, sizeof(float) * n_pack);
    packed += n_pack;
}


inline void spack_v1(float* packed, const float* X, int ldx, int n_pack) {
    for (int i = 0; i < n_pack; ++i) {
        *packed++ = *(X + i * ldx);
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace fastnum