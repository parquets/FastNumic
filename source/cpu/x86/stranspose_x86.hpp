#pragma once

#include <immintrin.h>
#include <math.h>
#include <memory.h>
#include <nmmintrin.h>
#include <smmintrin.h>
#include <stdio.h>
#include <stdlib.h>

namespace fastnum {
namespace cpu {
namespace kernel {


inline void
transpose_8x8(__m256& _v_a0, __m256& _v_a1, __m256& _v_a2, __m256& _v_a3, __m256& _v_a4, __m256& _v_a5, __m256& _v_a6, __m256& _v_a7) {
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

inline void transpose12x8(__m256& _r0,
                          __m256& _r1,
                          __m256& _r2,
                          __m256& _r3,
                          __m256& _r4,
                          __m256& _r5,
                          __m256& _r6,
                          __m256& _r7,
                          __m256& _r8,
                          __m256& _r9,
                          __m256& _ra,
                          __m256& _rb) {
    __m256 _tmp0 = _mm256_unpacklo_ps(_r0, _r1);
    __m256 _tmp1 = _mm256_unpackhi_ps(_r0, _r1);
    __m256 _tmp2 = _mm256_unpacklo_ps(_r2, _r3);
    __m256 _tmp3 = _mm256_unpackhi_ps(_r2, _r3);
    __m256 _tmp4 = _mm256_unpacklo_ps(_r4, _r5);
    __m256 _tmp5 = _mm256_unpackhi_ps(_r4, _r5);
    __m256 _tmp6 = _mm256_unpacklo_ps(_r6, _r7);
    __m256 _tmp7 = _mm256_unpackhi_ps(_r6, _r7);
    __m256 _tmp8 = _mm256_unpacklo_ps(_r8, _r9);
    __m256 _tmp9 = _mm256_unpackhi_ps(_r8, _r9);
    __m256 _tmpa = _mm256_unpacklo_ps(_ra, _rb);
    __m256 _tmpb = _mm256_unpackhi_ps(_ra, _rb);

    __m256 _tmpc = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpd = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmpe = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpf = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmpg = _mm256_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmph = _mm256_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmpi = _mm256_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpj = _mm256_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmpk = _mm256_shuffle_ps(_tmp8, _tmpa, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpl = _mm256_shuffle_ps(_tmp8, _tmpa, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmpm = _mm256_shuffle_ps(_tmp9, _tmpb, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpn = _mm256_shuffle_ps(_tmp9, _tmpb, _MM_SHUFFLE(3, 2, 3, 2));

    _r0 = _mm256_permute2f128_ps(_tmpc, _tmpg, _MM_SHUFFLE(0, 2, 0, 0));
    _r1 = _mm256_permute2f128_ps(_tmpk, _tmpd, _MM_SHUFFLE(0, 2, 0, 0));
    _r2 = _mm256_permute2f128_ps(_tmph, _tmpl, _MM_SHUFFLE(0, 2, 0, 0));
    _r3 = _mm256_permute2f128_ps(_tmpe, _tmpi, _MM_SHUFFLE(0, 2, 0, 0));
    _r4 = _mm256_permute2f128_ps(_tmpm, _tmpf, _MM_SHUFFLE(0, 2, 0, 0));
    _r5 = _mm256_permute2f128_ps(_tmpj, _tmpn, _MM_SHUFFLE(0, 2, 0, 0));
    _r6 = _mm256_permute2f128_ps(_tmpc, _tmpg, _MM_SHUFFLE(0, 3, 0, 1));
    _r7 = _mm256_permute2f128_ps(_tmpk, _tmpd, _MM_SHUFFLE(0, 3, 0, 1));
    _r8 = _mm256_permute2f128_ps(_tmph, _tmpl, _MM_SHUFFLE(0, 3, 0, 1));
    _r9 = _mm256_permute2f128_ps(_tmpe, _tmpi, _MM_SHUFFLE(0, 3, 0, 1));
    _ra = _mm256_permute2f128_ps(_tmpm, _tmpf, _MM_SHUFFLE(0, 3, 0, 1));
    _rb = _mm256_permute2f128_ps(_tmpj, _tmpn, _MM_SHUFFLE(0, 3, 0, 1));
}

inline void transpose_4x8(__m256& _v_r0, __m256& _v_r1, __m256& _v_r2, __m256& _v_r3) {
    __m256 _tmp0 = _mm256_unpacklo_ps(_v_r0, _v_r1);
    __m256 _tmp1 = _mm256_unpackhi_ps(_v_r0, _v_r1);
    __m256 _tmp2 = _mm256_unpacklo_ps(_v_r2, _v_r3);
    __m256 _tmp3 = _mm256_unpackhi_ps(_v_r2, _v_r3);

    __m256 _tmp4 = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmp5 = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmp6 = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmp7 = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(3, 2, 3, 2));

    _v_r0 = _mm256_permute2f128_ps(_tmp4, _tmp5, _MM_SHUFFLE(0, 2, 0, 0));
    _v_r1 = _mm256_permute2f128_ps(_tmp6, _tmp7, _MM_SHUFFLE(0, 2, 0, 0));
    _v_r2 = _mm256_permute2f128_ps(_tmp4, _tmp5, _MM_SHUFFLE(0, 3, 0, 1));
    _v_r3 = _mm256_permute2f128_ps(_tmp6, _tmp7, _MM_SHUFFLE(0, 3, 0, 1));
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

inline void transpose_16x8(__m256& _r0, __m256& _r1, __m256& _r2, __m256& _r3, __m256& _r4, __m256& _r5, __m256& _r6, __m256& _r7,
                          __m256& _r8, __m256& _r9, __m256& _ra, __m256& _rb, __m256& _rc, __m256& _rd, __m256& _re, __m256& _rf) {
    __m256 _tmp0 = _mm256_unpacklo_ps(_r0, _r1);
    __m256 _tmp1 = _mm256_unpackhi_ps(_r0, _r1);
    __m256 _tmp2 = _mm256_unpacklo_ps(_r2, _r3);
    __m256 _tmp3 = _mm256_unpackhi_ps(_r2, _r3);
    __m256 _tmp4 = _mm256_unpacklo_ps(_r4, _r5);
    __m256 _tmp5 = _mm256_unpackhi_ps(_r4, _r5);
    __m256 _tmp6 = _mm256_unpacklo_ps(_r6, _r7);
    __m256 _tmp7 = _mm256_unpackhi_ps(_r6, _r7);
    __m256 _tmp8 = _mm256_unpacklo_ps(_r8, _r9);
    __m256 _tmp9 = _mm256_unpackhi_ps(_r8, _r9);
    __m256 _tmpa = _mm256_unpacklo_ps(_ra, _rb);
    __m256 _tmpb = _mm256_unpackhi_ps(_ra, _rb);
    __m256 _tmpc = _mm256_unpacklo_ps(_rc, _rd);
    __m256 _tmpd = _mm256_unpackhi_ps(_rc, _rd);
    __m256 _tmpe = _mm256_unpacklo_ps(_re, _rf);
    __m256 _tmpf = _mm256_unpackhi_ps(_re, _rf);

    __m256 _tmpg = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmph = _mm256_shuffle_ps(_tmp0, _tmp2, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmpi = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpj = _mm256_shuffle_ps(_tmp1, _tmp3, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmpk = _mm256_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpl = _mm256_shuffle_ps(_tmp4, _tmp6, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmpm = _mm256_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpn = _mm256_shuffle_ps(_tmp5, _tmp7, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmpo = _mm256_shuffle_ps(_tmp8, _tmpa, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpp = _mm256_shuffle_ps(_tmp8, _tmpa, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmpq = _mm256_shuffle_ps(_tmp9, _tmpb, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpr = _mm256_shuffle_ps(_tmp9, _tmpb, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmps = _mm256_shuffle_ps(_tmpc, _tmpe, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpt = _mm256_shuffle_ps(_tmpc, _tmpe, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 _tmpu = _mm256_shuffle_ps(_tmpd, _tmpf, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 _tmpv = _mm256_shuffle_ps(_tmpd, _tmpf, _MM_SHUFFLE(3, 2, 3, 2));

    _r0 = _mm256_permute2f128_ps(_tmpg, _tmpk, _MM_SHUFFLE(0, 2, 0, 0));
    _r1 = _mm256_permute2f128_ps(_tmpo, _tmps, _MM_SHUFFLE(0, 2, 0, 0));
    _r2 = _mm256_permute2f128_ps(_tmph, _tmpl, _MM_SHUFFLE(0, 2, 0, 0));
    _r3 = _mm256_permute2f128_ps(_tmpp, _tmpt, _MM_SHUFFLE(0, 2, 0, 0));
    _r4 = _mm256_permute2f128_ps(_tmpi, _tmpm, _MM_SHUFFLE(0, 2, 0, 0));
    _r5 = _mm256_permute2f128_ps(_tmpq, _tmpu, _MM_SHUFFLE(0, 2, 0, 0));
    _r6 = _mm256_permute2f128_ps(_tmpj, _tmpn, _MM_SHUFFLE(0, 2, 0, 0));
    _r7 = _mm256_permute2f128_ps(_tmpr, _tmpv, _MM_SHUFFLE(0, 2, 0, 0));
    _r8 = _mm256_permute2f128_ps(_tmpg, _tmpk, _MM_SHUFFLE(0, 3, 0, 1));
    _r9 = _mm256_permute2f128_ps(_tmpo, _tmps, _MM_SHUFFLE(0, 3, 0, 1));
    _ra = _mm256_permute2f128_ps(_tmph, _tmpl, _MM_SHUFFLE(0, 3, 0, 1));
    _rb = _mm256_permute2f128_ps(_tmpp, _tmpt, _MM_SHUFFLE(0, 3, 0, 1));
    _rc = _mm256_permute2f128_ps(_tmpi, _tmpm, _MM_SHUFFLE(0, 3, 0, 1));
    _rd = _mm256_permute2f128_ps(_tmpq, _tmpu, _MM_SHUFFLE(0, 3, 0, 1));
    _re = _mm256_permute2f128_ps(_tmpj, _tmpn, _MM_SHUFFLE(0, 3, 0, 1));
    _rf = _mm256_permute2f128_ps(_tmpr, _tmpv, _MM_SHUFFLE(0, 3, 0, 1));
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


inline void swap(__m128& _v_r0, __m128& _v_r1) {
    __m128 _v_t = _v_r0;
    _v_r0 = _v_r1;
    _v_r1 = _v_t;
}



inline void
transpose_8x4(__m128& _v_a0, __m128& _v_a1, __m128& _v_a2, __m128& _v_a3, __m128& _v_a4, __m128& _v_a5, __m128& _v_a6, __m128& _v_a7) {
    transpose_4x4(_v_a0, _v_a1, _v_a2, _v_a3);
    transpose_4x4(_v_a4, _v_a5, _v_a6, _v_a7);

    swap(_v_a1, _v_a4);
    swap(_v_a4, _v_a2);
    swap(_v_a3, _v_a5);
    swap(_v_a5, _v_a6);
}

inline void transpose_8x8(const float* A, int lda, float* B, int ldb) {
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

inline void transpose_6x8(const float* A, int lda, float* B, int ldb) {
    __m256 r0, r1, r2, r3, r4, r5;

    r0 = _mm256_loadu_ps(A + 0 * lda);
    r1 = _mm256_loadu_ps(A + 1 * lda);
    r2 = _mm256_loadu_ps(A + 2 * lda);
    r3 = _mm256_loadu_ps(A + 3 * lda);
    r4 = _mm256_loadu_ps(A + 4 * lda);
    r5 = _mm256_loadu_ps(A + 5 * lda);

    transpose_6x8(r0, r1, r2, r3, r4, r5);

    _mm256_storeu_ps(B + 0 * ldb, r0);
    _mm256_storeu_ps(B + 1 * ldb, r1);
    _mm256_storeu_ps(B + 2 * ldb, r2);
    _mm256_storeu_ps(B + 3 * ldb, r3);
    _mm256_storeu_ps(B + 4 * ldb, r4);
    _mm256_storeu_ps(B + 5 * ldb, r5);
}



}  // namespace kernel
}  // namespace cpu
}  // namespace fastnum