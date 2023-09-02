#pragma once

#include <immintrin.h>
#include <math.h>
#include <memory.h>
#include <nmmintrin.h>
#include <smmintrin.h>
#include <stdio.h>

namespace fastnum {
namespace cpu {
namespace kernel {

inline void transpose_8x8(__m256& _v_a0, __m256& _v_a1, __m256& _v_a2, __m256& _v_a3, __m256& _v_a4, __m256& _v_a5,
                          __m256& _v_a6, __m256& _v_a7) {
    __m256 _v_t0 = _mm256_unpacklo_ps(_v_a0, _v_a1);  // a0,b0,a1,b1,a4,b4,a5,b5
    __m256 _v_t1 = _mm256_unpackhi_ps(_v_a0, _v_a1);  // a2,b2,a3,b3,a6,b6,a7,b7
    __m256 _v_t2 = _mm256_unpacklo_ps(_v_a2, _v_a3);  // c0,d0,c1,d1,c4,d4,c5,d5
    __m256 _v_t3 = _mm256_unpackhi_ps(_v_a2, _v_a3);  // c2,d2,c3,d3,c6,d6,c7,d7
    __m256 _v_t4 = _mm256_unpacklo_ps(_v_a4, _v_a5);  // e0,f0,e1,f1,e4,f4,e5,f5
    __m256 _v_t5 = _mm256_unpackhi_ps(_v_a4, _v_a5);  // e2,f2,e3,f3,e6,f6,e7,f7
    __m256 _v_t6 = _mm256_unpacklo_ps(_v_a6, _v_a7);  // g0,h0,g1,h1,g4,h4,g5,h5
    __m256 _v_t7 = _mm256_unpackhi_ps(_v_a6, _v_a7);  // g2,h2,g3,h3,g6,h6,g7,h7

    __m256 _v_u0 = _mm256_shuffle_ps(_v_t0, _v_t2, _MM_SHUFFLE(1, 0, 1, 0));  // a0,b0,c0,d0,a4,b4,c4,d4
    __m256 _v_u1 = _mm256_shuffle_ps(_v_t0, _v_t2, _MM_SHUFFLE(3, 2, 3, 2));  // a1,b1,c1,d1,a5,b5,c5,d5
    __m256 _v_u2 = _mm256_shuffle_ps(_v_t1, _v_t3, _MM_SHUFFLE(1, 0, 1, 0));  // a2,b2,c2,d2,a6,b6,c6,d6
    __m256 _v_u3 = _mm256_shuffle_ps(_v_t1, _v_t3, _MM_SHUFFLE(3, 2, 3, 2));  // a3,b3,c3,d3,a7,b7,c7,d7
    __m256 _v_u4 = _mm256_shuffle_ps(_v_t4, _v_t6, _MM_SHUFFLE(1, 0, 1, 0));  // e0,f0,g0,h0,e4,f4,g4,h4
    __m256 _v_u5 = _mm256_shuffle_ps(_v_t4, _v_t6, _MM_SHUFFLE(3, 2, 3, 2));  // e1,f1,g1,h1,e5,f5,g5,h5
    __m256 _v_u6 = _mm256_shuffle_ps(_v_t5, _v_t7, _MM_SHUFFLE(1, 0, 1, 0));  // e2,f2,g2,h2,e6,f6,g6,h6
    __m256 _v_u7 = _mm256_shuffle_ps(_v_t5, _v_t7, _MM_SHUFFLE(3, 2, 3, 2));  // e3,f3,g3,h3,e7,f7,g7,h7

    _v_a0 = _mm256_permute2f128_ps(_v_u0, _v_u4, _MM_SHUFFLE(0, 2, 0, 0));  // a0,b0,c0,d0,e0,f0,g0,h0
    _v_a1 = _mm256_permute2f128_ps(_v_u1, _v_u5, _MM_SHUFFLE(0, 2, 0, 0));  // a1,b1,c1,d1,e1,f1,g1,h1
    _v_a2 = _mm256_permute2f128_ps(_v_u2, _v_u6, _MM_SHUFFLE(0, 2, 0, 0));  // a2,b2,c2,d2,e2,f2,g2,h2
    _v_a3 = _mm256_permute2f128_ps(_v_u3, _v_u7, _MM_SHUFFLE(0, 2, 0, 0));  // a3,b3,c3,d3,e3,f3,g3,h3
    _v_a4 = _mm256_permute2f128_ps(_v_u0, _v_u4, _MM_SHUFFLE(0, 3, 0, 1));  // a4,b4,c4,d4,e4,f4,g4,h4
    _v_a5 = _mm256_permute2f128_ps(_v_u1, _v_u5, _MM_SHUFFLE(0, 3, 0, 1));  // a5,b5,c5,d5,e5,f5,g5,h5
    _v_a6 = _mm256_permute2f128_ps(_v_u2, _v_u6, _MM_SHUFFLE(0, 3, 0, 1));  // a6,b6,c6,d6,e6,f6,g6,h6
    _v_a7 = _mm256_permute2f128_ps(_v_u3, _v_u7, _MM_SHUFFLE(0, 3, 0, 1));  // a7,b7,c7,d7,e7,f7,g7,h7
}

inline void transpose_6x8(__m256& _v_a0, __m256& _v_a1, __m256& _v_a2, __m256& _v_a3, __m256& _v_a4, __m256& _v_a5) {
    __m256 _v_t0 = _mm256_unpacklo_ps(_v_a0, _v_a1);  // a0,b0,a1,b1,a4,b4,a5,b5
    __m256 _v_t1 = _mm256_unpackhi_ps(_v_a0, _v_a1);  // a2,b2,a3,b3,a6,b6,a7,b7
    __m256 _v_t2 = _mm256_unpacklo_ps(_v_a2, _v_a3);  // c0,d0,c1,d1,c4,d4,c5,d5
    __m256 _v_t3 = _mm256_unpackhi_ps(_v_a2, _v_a3);  // c2,d2,c3,d3,c6,d6,c7,d7
    __m256 _v_t4 = _mm256_unpacklo_ps(_v_a4, _v_a5);  // e0,f0,e1,f1,e4,f4,e5,f5
    __m256 _v_t5 = _mm256_unpackhi_ps(_v_a4, _v_a5);  // e2,f2,e3,f3,e6,f6,e7,f7

    __m256 _v_u0 = _mm256_shuffle_ps(_v_t0, _v_t2, _MM_SHUFFLE(1, 0, 1, 0));  // a0,b0,c0,d0,a4,b4,c4,d4
    __m256 _v_u1 = _mm256_shuffle_ps(_v_t4, _v_t0, _MM_SHUFFLE(3, 2, 1, 0));  // e0,f0,a1,b1,e4,f4,a5,b5
    __m256 _v_u2 = _mm256_shuffle_ps(_v_t1, _v_t3, _MM_SHUFFLE(1, 0, 1, 0));  // a2,b2,c2,d2,a6,b6,c6,d6
    __m256 _v_u3 = _mm256_shuffle_ps(_v_t5, _v_t1, _MM_SHUFFLE(3, 2, 1, 0));  // e2,f2,a3,b3,e6,f6,a7,b7
    __m256 _v_u4 = _mm256_shuffle_ps(_v_t2, _v_t4, _MM_SHUFFLE(3, 2, 3, 2));  // c1,d1,e1,f1,c5,d5,e5,f5
    __m256 _v_u5 = _mm256_shuffle_ps(_v_t3, _v_t5, _MM_SHUFFLE(3, 2, 3, 2));  // c3,d3,e3,f3,c7,d7,e7,f7

    _v_a0 = _mm256_permute2f128_ps(_v_u0, _v_u1, _MM_SHUFFLE(0, 2, 0, 0));  // a0,b0,c0,d0,e0,f0,a1,b1
    _v_a1 = _mm256_permute2f128_ps(_v_u4, _v_u2, _MM_SHUFFLE(0, 2, 0, 0));  // c1,d1,e1,f1,a2,b2,c2,d2
    _v_a2 = _mm256_permute2f128_ps(_v_u3, _v_u5, _MM_SHUFFLE(0, 2, 0, 0));  // e2,f2,a3,b3,c3,d3,e3,f3
    _v_a3 = _mm256_permute2f128_ps(_v_u0, _v_u1, _MM_SHUFFLE(0, 3, 0, 1));  // a4,b4,c4,d4,e4,f4,a5,b5
    _v_a4 = _mm256_permute2f128_ps(_v_u4, _v_u2, _MM_SHUFFLE(0, 3, 0, 1));  // c5,d5,e5,f5,a6,b6,c6,d6
    _v_a5 = _mm256_permute2f128_ps(_v_u3, _v_u5, _MM_SHUFFLE(0, 3, 0, 1));  // e6,f6,a7,b7,c7,d7,e7,f7
}

inline void transpose_4x4(__m128& _v_a0, __m128& _v_a1, __m128& _v_a2, __m128& _v_a3) {
    __m128 tmp3, tmp2, tmp1, tmp0;
    tmp0 = _mm_unpacklo_ps(_v_a0, _v_a1);
    tmp2 = _mm_unpacklo_ps(_v_a2, _v_a3);
    tmp1 = _mm_unpackhi_ps(_v_a0, _v_a1);
    tmp3 = _mm_unpackhi_ps(_v_a2, _v_a3);
    _v_a0 = _mm_movelh_ps(tmp0, tmp2);
    _v_a1 = _mm_movehl_ps(tmp2, tmp0);
    _v_a2 = _mm_movelh_ps(tmp1, tmp3);
    _v_a3 = _mm_movehl_ps(tmp3, tmp1);
}

inline void transpose(const float* A, int lda, float* B, int ldb) {
    __m256 r0, r1, r2, r3, r4, r5, r6, r7;

    r0 = _mm256_loadu_ps(A + 0 * lda);
    r1 = _mm256_loadu_ps(A + 1 * lda);
    r2 = _mm256_loadu_ps(A + 2 * lda);
    r3 = _mm256_loadu_ps(A + 3 * lda);
    r4 = _mm256_loadu_ps(A + 4 * lda);
    r5 = _mm256_loadu_ps(A + 5 * lda);
    r6 = _mm256_loadu_ps(A + 6 * lda);
    r7 = _mm256_loadu_ps(A + 7 * lda);

    transpose_8x8(r0, r1, r2, r3, r4, r5, r6, r7);

    _mm256_storeu_ps(B + 0 * ldb, r0);
    _mm256_storeu_ps(B + 1 * ldb, r1);
    _mm256_storeu_ps(B + 2 * ldb, r2);
    _mm256_storeu_ps(B + 3 * ldb, r3);
    _mm256_storeu_ps(B + 4 * ldb, r4);
    _mm256_storeu_ps(B + 5 * ldb, r5);
    _mm256_storeu_ps(B + 6 * ldb, r6);
    _mm256_storeu_ps(B + 7 * ldb, r7);
}

}  // namespace kernel
}  // namespace cpu
}  // namespace fastnum