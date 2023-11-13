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

static void print_avx_trans(__m256d x) {
    double a[4];
    _mm256_storeu_pd(a, x);
    printf("%.5lf, %.5lf, %.5lf, %.5lf\n", a[0], a[1], a[2], a[3]);
}

inline void swap(__m256d& _v_r0, __m256d& _v_r1) {
    __m256d _v_t = _v_r0;
    _v_r0 = _v_r1;
    _v_r1 = _v_t;
}

inline void transpose_4x4(__m256d& _v_a0, __m256d& _v_a1, __m256d& _v_a2, __m256d& _v_a3) {
    __m256d tmp3, tmp2, tmp1, tmp0;
    tmp0 = _mm256_unpacklo_pd(_v_a0, _v_a1);  // a0, b0, a2, b2
    tmp1 = _mm256_unpacklo_pd(_v_a2, _v_a3);  // c0, d0, c2, d2
    tmp2 = _mm256_unpackhi_pd(_v_a0, _v_a1);  // a1, b1, a3, b3
    tmp3 = _mm256_unpackhi_pd(_v_a2, _v_a3);  // c1, d1, c3, d3
    _v_a0 = _mm256_permute2f128_pd(tmp0, tmp1, _MM_SHUFFLE(0, 2, 0, 0));
    _v_a1 = _mm256_permute2f128_pd(tmp2, tmp3, _MM_SHUFFLE(0, 2, 0, 0));
    _v_a2 = _mm256_permute2f128_pd(tmp0, tmp1, _MM_SHUFFLE(0, 3, 0, 1));
    _v_a3 = _mm256_permute2f128_pd(tmp2, tmp3, _MM_SHUFFLE(0, 3, 0, 1));
}

inline void transpose_6x4(__m256d& _v_a0, __m256d& _v_a1, __m256d& _v_a2, __m256d& _v_a3, __m256d& _v_a4, __m256d& _v_a5) {
    // a0, a1, a2, a3
    // b0, b1, b2, b3
    // c0, c1, c2, c3
    // d0, d1, d2, d3
    // e0, e1, e2, e3
    // f0, f1, f2, f3
    __m256d _v_t0 = _mm256_unpacklo_pd(_v_a0, _v_a1); // a0, b0, a2, b2
    __m256d _v_t1 = _mm256_unpackhi_pd(_v_a0, _v_a1); // a1, b1, a3, b3
    __m256d _v_t2 = _mm256_unpacklo_pd(_v_a2, _v_a3); // c0, d0, c2, d2
    __m256d _v_t3 = _mm256_unpackhi_pd(_v_a2, _v_a3); // c1, d1, c3, d3
    __m256d _v_t4 = _mm256_unpacklo_pd(_v_a4, _v_a5); // e0, f0, e2, f2
    __m256d _v_t5 = _mm256_unpackhi_pd(_v_a4, _v_a5); // e1, f1, e3, f3


    _v_a0 = _mm256_permute2f128_pd(_v_t0, _v_t2, 0b00100000); // a0, b0, c0, d0
    _v_a1 = _mm256_permute2f128_pd(_v_t4, _v_t1, 0b00100000); // e0, f0, a1, b1
    _v_a2 = _mm256_permute2f128_pd(_v_t3, _v_t5, 0b00100000); // c1, d1, e1, f1
    _v_a3 = _mm256_permute2f128_pd(_v_t0, _v_t2, 0b00110001); // a2, b2, c2, d2
    _v_a4 = _mm256_permute2f128_pd(_v_t4, _v_t1, 0b00110001); // e2, f2, a3, b3
    _v_a5 = _mm256_permute2f128_pd(_v_t3, _v_t5, 0b00110001); // c3, d3, e3, f3
}

inline void transpose_8x4(__m256d& _v_a0,
                          __m256d& _v_a1,
                          __m256d& _v_a2,
                          __m256d& _v_a3,
                          __m256d& _v_a4,
                          __m256d& _v_a5,
                          __m256d& _v_a6,
                          __m256d& _v_a7) {
    transpose_4x4(_v_a0, _v_a1, _v_a2, _v_a3);
    transpose_4x4(_v_a4, _v_a5, _v_a6, _v_a7);

    swap(_v_a1, _v_a4);
    swap(_v_a4, _v_a2);
    swap(_v_a3, _v_a5);
    swap(_v_a5, _v_a6);
}

inline void transpose_8x8(const double* A, int lda, double* B, int ldb) {
    __m256d r0 = _mm256_loadu_pd(A + 0 * lda);
    __m256d r1 = _mm256_loadu_pd(A + 1 * lda);
    __m256d r2 = _mm256_loadu_pd(A + 2 * lda);
    __m256d r3 = _mm256_loadu_pd(A + 3 * lda);

    transpose_4x4(r0, r1, r2, r3);

    _mm256_storeu_pd(B + 0 * ldb, r0);
    _mm256_storeu_pd(B + 1 * ldb, r1);
    _mm256_storeu_pd(B + 2 * ldb, r2);
    _mm256_storeu_pd(B + 3 * ldb, r3);

    r0 = _mm256_loadu_pd(A + 0 * lda + 4);
    r1 = _mm256_loadu_pd(A + 1 * lda + 4);
    r2 = _mm256_loadu_pd(A + 2 * lda + 4);
    r3 = _mm256_loadu_pd(A + 3 * lda + 4);

    transpose_4x4(r0, r1, r2, r3);

    _mm256_storeu_pd(B + 4 * ldb, r0);
    _mm256_storeu_pd(B + 5 * ldb, r1);
    _mm256_storeu_pd(B + 6 * ldb, r2);
    _mm256_storeu_pd(B + 7 * ldb, r3);

    __m256d r4 = _mm256_loadu_pd(A + 4 * lda);
    __m256d r5 = _mm256_loadu_pd(A + 5 * lda);
    __m256d r6 = _mm256_loadu_pd(A + 6 * lda);
    __m256d r7 = _mm256_loadu_pd(A + 7 * lda);

    transpose_4x4(r4, r5, r6, r7);

    _mm256_storeu_pd(B + 0 * ldb + 4, r4);
    _mm256_storeu_pd(B + 1 * ldb + 4, r5);
    _mm256_storeu_pd(B + 2 * ldb + 4, r6);
    _mm256_storeu_pd(B + 3 * ldb + 4, r7);

    r4 = _mm256_loadu_pd(A + 4 * lda + 4);
    r5 = _mm256_loadu_pd(A + 5 * lda + 4);
    r6 = _mm256_loadu_pd(A + 6 * lda + 4);
    r7 = _mm256_loadu_pd(A + 7 * lda + 4);

    transpose_4x4(r4, r5, r6, r7);

    _mm256_storeu_pd(B + 4 * ldb + 4, r4);
    _mm256_storeu_pd(B + 5 * ldb + 4, r5);
    _mm256_storeu_pd(B + 6 * ldb + 4, r6);
    _mm256_storeu_pd(B + 7 * ldb + 4, r7);
}

inline void transpose_4x4(const double* A, int lda, double* B, int ldb) {
    __m256d r0, r1, r2, r3;

    r0 = _mm256_loadu_pd(A + 0 * lda);
    r1 = _mm256_loadu_pd(A + 1 * lda);
    r2 = _mm256_loadu_pd(A + 2 * lda);
    r3 = _mm256_loadu_pd(A + 3 * lda);

    transpose_4x4(r0, r1, r2, r3);

    _mm256_storeu_pd(B + 0 * ldb, r0);
    _mm256_storeu_pd(B + 1 * ldb, r1);
    _mm256_storeu_pd(B + 2 * ldb, r2);
    _mm256_storeu_pd(B + 3 * ldb, r3);
}

}  // namespace kernel
}  // namespace cpu
}  // namespace fastnum