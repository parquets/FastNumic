#pragma once

#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>

#include <immintrin.h>
#include <nmmintrin.h>
#include <smmintrin.h>

#include "stranspose_x86.hpp"
#include "vectorize_x86.hpp"

namespace fastnum {
namespace cpu {
namespace kernel {

template <int TileSize, int KernelSize, int StrideSize, int NPack>
struct WinogradDataTransformKernel {};

inline void print_sse(__m128 x) {
    float d[4];
    _mm_storeu_ps(d, x);
    printf("%.5f %.5f %.5f %.5f\n", d[0], d[1], d[2], d[3]);
}

inline void print_avx(__m256 x) {
    float d[8];
    _mm256_storeu_ps(d, x);
    printf("%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n", d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
}

static void dataTransLeftUnit4K3S1Pack1(float* dest, int stride, const float* source) {
    for (int i = 0; i < 4; ++i) {
        float r0 = source[0 * 4 + i];
        float r1 = source[1 * 4 + i];
        float r2 = source[2 * 4 + i];
        float r3 = source[3 * 4 + i];

        float t0 = r0 - r2;
        float t1 = r1 + r2;
        float t2 = -r1 + r2;
        float t3 = r3 - r1;

        dest[0 * stride + i] = t0;
        dest[1 * stride + i] = t1;
        dest[2 * stride + i] = t2;
        dest[3 * stride + i] = t3;

    }
}

static void dataTransLeftUnit4K3S1Pack4(float* dest, int stride, const float* source) {
    for (int i = 0; i < 4; ++i) {
        __m128 _v_r0 = _mm_loadu_ps(source + 0 * 4 * 4 + i * 4);
        __m128 _v_r1 = _mm_loadu_ps(source + 1 * 4 * 4 + i * 4);
        __m128 _v_r2 = _mm_loadu_ps(source + 2 * 4 * 4 + i * 4);
        __m128 _v_r3 = _mm_loadu_ps(source + 3 * 4 * 4 + i * 4);

        __m128 _v_t0 = _mm_sub_ps(_v_r0, _v_r2);
        __m128 _v_t1 = _mm_add_ps(_v_r1, _v_r2);
        __m128 _v_t2 = _mm_sub_ps(_v_r2, _v_r1);
        __m128 _v_t3 = _mm_sub_ps(_v_r3, _v_r1);

        _mm_storeu_ps(dest + 0 * stride + i * 4, _v_t0);
        _mm_storeu_ps(dest + 1 * stride + i * 4, _v_t1);
        _mm_storeu_ps(dest + 2 * stride + i * 4, _v_t2);
        _mm_storeu_ps(dest + 3 * stride + i * 4, _v_t3);
    }
}

static void dataTransLeftUnit4K3S1Pack8(float* dest, int stride, const float* source) {
    for (int i = 0; i < 4; ++i) {
        __m256 _v_r0 = _mm256_loadu_ps(source + 0 * 4 * 8 + i * 8);
        __m256 _v_r1 = _mm256_loadu_ps(source + 1 * 4 * 8 + i * 8);
        __m256 _v_r2 = _mm256_loadu_ps(source + 2 * 4 * 8 + i * 8);
        __m256 _v_r3 = _mm256_loadu_ps(source + 3 * 4 * 8 + i * 8);

        __m256 _v_t0 = _mm256_sub_ps(_v_r0, _v_r2);
        __m256 _v_t1 = _mm256_add_ps(_v_r1, _v_r2);
        __m256 _v_t2 = _mm256_sub_ps(_v_r2, _v_r1);
        __m256 _v_t3 = _mm256_sub_ps(_v_r3, _v_r1);

        _mm256_storeu_ps(dest + 0 * stride + i * 8, _v_t0);
        _mm256_storeu_ps(dest + 1 * stride + i * 8, _v_t1);
        _mm256_storeu_ps(dest + 2 * stride + i * 8, _v_t2);
        _mm256_storeu_ps(dest + 3 * stride + i * 8, _v_t3);
    }
}

static void dataTransRightUnit4K3S1Pack1(float* dest, const float* source, int ldh, int ldc) {
    for (int i = 0; i < 4; ++i) {
        float r0 = source[i * ldh + 0];
        float r1 = source[i * ldh + 1];
        float r2 = source[i * ldh + 2];
        float r3 = source[i * ldh + 3];


        float t0 = r0 - r2;
        float t1 = r1 + r2;
        float t2 = -r1 + r2;
        float t3 = r3 - r1;

        dest[i * 4 + 0] = t0;
        dest[i * 4 + 1] = t1;
        dest[i * 4 + 2] = t2;
        dest[i * 4 + 3] = t3;
    }
}

static void dataTransRightUnit4K3S1Pack4(float* dest, const float* source, int ldh, int ldc) {
    for (int i = 0; i < 4; ++i) {
        __m128 _v_r0 = _mm_loadu_ps(source + i * ldh + 0 * ldc);
        __m128 _v_r1 = _mm_loadu_ps(source + i * ldh + 1 * ldc);
        __m128 _v_r2 = _mm_loadu_ps(source + i * ldh + 2 * ldc);
        __m128 _v_r3 = _mm_loadu_ps(source + i * ldh + 3 * ldc);

        transpose_4x4(_v_r0, _v_r1, _v_r2, _v_r3);

        __m128 _v_t0 = _mm_sub_ps(_v_r0, _v_r2);
        __m128 _v_t1 = _mm_add_ps(_v_r1, _v_r2);
        __m128 _v_t2 = _mm_sub_ps(_v_r2, _v_r1);
        __m128 _v_t3 = _mm_sub_ps(_v_r3, _v_r1);

        _mm_storeu_ps(dest + i * 4 * 4 + 0 * 4, _v_t0);
        _mm_storeu_ps(dest + i * 4 * 4 + 1 * 4, _v_t1);
        _mm_storeu_ps(dest + i * 4 * 4 + 2 * 4, _v_t2);
        _mm_storeu_ps(dest + i * 4 * 4 + 3 * 4, _v_t3);
    }
}

static void dataTransRightUnit4K3S1Pack8(float* dest, const float* source, int ldh, int ldc) {
    for (int i = 0; i < 4; ++i) {
        __m128 _v_tr0 = _mm_loadu_ps(source + i * ldh + 0 * ldc);
        __m128 _v_tr1 = _mm_loadu_ps(source + i * ldh + 1 * ldc);
        __m128 _v_tr2 = _mm_loadu_ps(source + i * ldh + 2 * ldc);
        __m128 _v_tr3 = _mm_loadu_ps(source + i * ldh + 3 * ldc);
        __m128 _v_tr4 = _mm_loadu_ps(source + i * ldh + 4 * ldc);
        __m128 _v_tr5 = _mm_loadu_ps(source + i * ldh + 5 * ldc);
        __m128 _v_tr6 = _mm_loadu_ps(source + i * ldh + 6 * ldc);
        __m128 _v_tr7 = _mm_loadu_ps(source + i * ldh + 7 * ldc);

        transpose_8x4(_v_tr0, _v_tr1, _v_tr2, _v_tr3, _v_tr4, _v_tr5, _v_tr6, _v_tr7);

        __m256 _v_r0 = _mm256_set_m128(_v_tr1, _v_tr0);
        __m256 _v_r1 = _mm256_set_m128(_v_tr3, _v_tr2);
        __m256 _v_r2 = _mm256_set_m128(_v_tr5, _v_tr4);
        __m256 _v_r3 = _mm256_set_m128(_v_tr7, _v_tr6);

        __m256 _v_t0 = _mm256_sub_ps(_v_r0, _v_r2);
        __m256 _v_t1 = _mm256_add_ps(_v_r1, _v_r2);
        __m256 _v_t2 = _mm256_sub_ps(_v_r2, _v_r1);
        __m256 _v_t3 = _mm256_sub_ps(_v_r3, _v_r1);

        _mm256_storeu_ps(dest + i * 4 * 8 + 0 * 8, _v_t0);
        _mm256_storeu_ps(dest + i * 4 * 8 + 1 * 8, _v_t1);
        _mm256_storeu_ps(dest + i * 4 * 8 + 2 * 8, _v_t2);
        _mm256_storeu_ps(dest + i * 4 * 8 + 3 * 8, _v_t3);
    }
}

inline void winogradDataTransUnit4K3S1Pack1(float* dest, int ldest, const float* source, int ldh, int ldc) {
    float dB[4 * 4 * 1] = {0};
    dataTransRightUnit4K3S1Pack1(dB, source, ldh, ldc);
    dataTransLeftUnit4K3S1Pack1(dest, ldest, dB);
}

inline void winogradDataTransUnit4K3S1Pack4(float* dest, int ldest, const float* source, int ldh, int ldc) {
    float dB[4 * 4 * 4] = {0};
    dataTransRightUnit4K3S1Pack4(dB, source, ldh, ldc);
    dataTransLeftUnit4K3S1Pack4(dest, ldest, dB);
}

inline void winogradDataTransUnit4K3S1Pack8(float* dest, int ldest, const float* source, int ldh, int ldc) {
    float dB[4 * 4 * 8] = {0};
    dataTransRightUnit4K3S1Pack8(dB, source, ldh, ldc);
    dataTransLeftUnit4K3S1Pack8(dest, ldest, dB);
}


static void dataTransLeftUnit8K3S1Pack1(float* dest, int stride, const float* source) {
    for (int i = 0; i < 8; ++i) {
        float r0 = source[0 * 8 + i];
        float r1 = source[1 * 8 + i];
        float r2 = source[2 * 8 + i];
        float r3 = source[3 * 8 + i];
        float r4 = source[4 * 8 + i];
        float r5 = source[5 * 8 + i];
        float r6 = source[6 * 8 + i];
        float r7 = source[7 * 8 + i];

        float t12a = static_cast<float>(-4.25 * r4 + (r2 + r6));
        float t12b = static_cast<float>(-4.25 * r3 + (r1 + r5));
        float t34a = static_cast<float>(-1.25 * r4 + (0.25 * r2 + r6));
        float t34b = static_cast<float>(2 * r5 + (-2.5 * r3 + (0.5 * r1)));
        float t56a = static_cast<float>(4 * (-1.25 * r4 + r2) + r6);
        float t56b = static_cast<float>(0.5 * r5 + (-2.5 * r3 + (2 * r1)));

        float t0 = static_cast<float>((5.25 * (r4 - r2) + r0) - r6);
        float t1 = static_cast<float>(t12a + t12b);
        float t2 = static_cast<float>(t12a - t12b);
        float t3 = static_cast<float>(t34a + t34b);
        float t4 = static_cast<float>(t34a - t34b);
        float t5 = static_cast<float>(t56a + t56b);
        float t6 = static_cast<float>(t56a - t56b);
        float t7 = static_cast<float>(5.25 * (r3 - r5) + (r7 - r1));

        dest[0 * stride + i] = t0;
        dest[1 * stride + i] = t1;
        dest[2 * stride + i] = t2;
        dest[3 * stride + i] = t3;
        dest[4 * stride + i] = t4;
        dest[5 * stride + i] = t5;
        dest[6 * stride + i] = t6;
        dest[7 * stride + i] = t7;
    }
}

static void dataTransLeftUnit8K3S1Pack4(float* dest, int stride, const float* source) {
    __m128 _v5_25 = _mm_set1_ps(5.25f);
    __m128 _vm4_25 = _mm_set1_ps(-4.25f);
    __m128 _vm1_25 = _mm_set1_ps(-1.25f);
    __m128 _v0_25 = _mm_set1_ps(0.25f);
    __m128 _vm2_5 = _mm_set1_ps(-2.5f);
    __m128 _v0_5 = _mm_set1_ps(0.5f);
    __m128 _v2 = _mm_set1_ps(2.f);
    __m128 _v4 = _mm_set1_ps(4.f);

    for (int i = 0; i < 8; ++i) {
        __m128 _v_r0 = _mm_loadu_ps(source + 0 * 8 * 4 + i * 4);
        __m128 _v_r1 = _mm_loadu_ps(source + 1 * 8 * 4 + i * 4);
        __m128 _v_r2 = _mm_loadu_ps(source + 2 * 8 * 4 + i * 4);
        __m128 _v_r3 = _mm_loadu_ps(source + 3 * 8 * 4 + i * 4);
        __m128 _v_r4 = _mm_loadu_ps(source + 4 * 8 * 4 + i * 4);
        __m128 _v_r5 = _mm_loadu_ps(source + 5 * 8 * 4 + i * 4);
        __m128 _v_r6 = _mm_loadu_ps(source + 6 * 8 * 4 + i * 4);
        __m128 _v_r7 = _mm_loadu_ps(source + 7 * 8 * 4 + i * 4);

        __m128 _v_t12a = _mm_fmadd_ps(_vm4_25, _v_r4, _mm_add_ps(_v_r2, _v_r6));
        __m128 _v_t12b = _mm_fmadd_ps(_vm4_25, _v_r3, _mm_add_ps(_v_r1, _v_r5));
        __m128 _v_t34a = _mm_fmadd_ps(_vm1_25, _v_r4, _mm_fmadd_ps(_v0_25, _v_r2, _v_r6));
        __m128 _v_t34b = _mm_fmadd_ps(_v2, _v_r5, _mm_fmadd_ps(_vm2_5, _v_r3, _mm_mul_ps(_v_r1, _v0_5)));
        __m128 _v_t56a = _mm_fmadd_ps(_v4, _mm_fmadd_ps(_vm1_25, _v_r4, _v_r2), _v_r6);
        __m128 _v_t56b = _mm_fmadd_ps(_v0_5, _v_r5, _mm_fmadd_ps(_vm2_5, _v_r3, _mm_mul_ps(_v_r1, _v2)));

        __m128 _v_t0 = _mm_sub_ps(_mm_fmadd_ps(_v5_25, _mm_sub_ps(_v_r4, _v_r2), _v_r0), _v_r6);
        __m128 _v_t1 = _mm_add_ps(_v_t12a, _v_t12b);
        __m128 _v_t2 = _mm_sub_ps(_v_t12a, _v_t12b);
        __m128 _v_t3 = _mm_add_ps(_v_t34a, _v_t34b);
        __m128 _v_t4 = _mm_sub_ps(_v_t34a, _v_t34b);
        __m128 _v_t5 = _mm_add_ps(_v_t56a, _v_t56b);
        __m128 _v_t6 = _mm_sub_ps(_v_t56a, _v_t56b);
        __m128 _v_t7 = _mm_fmadd_ps(_v5_25, _mm_sub_ps(_v_r3, _v_r5), _mm_sub_ps(_v_r7, _v_r1));

        _mm_storeu_ps(dest + 0 * stride + i * 4, _v_t0);
        _mm_storeu_ps(dest + 1 * stride + i * 4, _v_t1);
        _mm_storeu_ps(dest + 2 * stride + i * 4, _v_t2);
        _mm_storeu_ps(dest + 3 * stride + i * 4, _v_t3);
        _mm_storeu_ps(dest + 4 * stride + i * 4, _v_t4);
        _mm_storeu_ps(dest + 5 * stride + i * 4, _v_t5);
        _mm_storeu_ps(dest + 6 * stride + i * 4, _v_t6);
        _mm_storeu_ps(dest + 7 * stride + i * 4, _v_t7);
    }
}

static void dataTransLeftUnit8K3S1Pack8(float* dest, int stride, const float* source) {
    __m256 _v5_25 = _mm256_set1_ps(5.25f);
    __m256 _vm4_25 = _mm256_set1_ps(-4.25f);
    __m256 _vm1_25 = _mm256_set1_ps(-1.25f);
    __m256 _v0_25 = _mm256_set1_ps(0.25f);
    __m256 _vm2_5 = _mm256_set1_ps(-2.5f);
    __m256 _v0_5 = _mm256_set1_ps(0.5f);
    __m256 _v2 = _mm256_set1_ps(2.f);
    __m256 _v4 = _mm256_set1_ps(4.f);

    for (int i = 0; i < 8; ++i) {
        __m256 _v_r0 = _mm256_loadu_ps(source + 0 * 8 * 8 + i * 8);
        __m256 _v_r1 = _mm256_loadu_ps(source + 1 * 8 * 8 + i * 8);
        __m256 _v_r2 = _mm256_loadu_ps(source + 2 * 8 * 8 + i * 8);
        __m256 _v_r3 = _mm256_loadu_ps(source + 3 * 8 * 8 + i * 8);
        __m256 _v_r4 = _mm256_loadu_ps(source + 4 * 8 * 8 + i * 8);
        __m256 _v_r5 = _mm256_loadu_ps(source + 5 * 8 * 8 + i * 8);
        __m256 _v_r6 = _mm256_loadu_ps(source + 6 * 8 * 8 + i * 8);
        __m256 _v_r7 = _mm256_loadu_ps(source + 7 * 8 * 8 + i * 8);

        __m256 _v_t12a = _mm256_fmadd_ps(_vm4_25, _v_r4, _mm256_add_ps(_v_r2, _v_r6));
        __m256 _v_t12b = _mm256_fmadd_ps(_vm4_25, _v_r3, _mm256_add_ps(_v_r1, _v_r5));
        __m256 _v_t34a = _mm256_fmadd_ps(_vm1_25, _v_r4, _mm256_fmadd_ps(_v0_25, _v_r2, _v_r6));
        __m256 _v_t34b = _mm256_fmadd_ps(_v2, _v_r5, _mm256_fmadd_ps(_vm2_5, _v_r3, _mm256_mul_ps(_v_r1, _v0_5)));
        __m256 _v_t56a = _mm256_fmadd_ps(_v4, _mm256_fmadd_ps(_vm1_25, _v_r4, _v_r2), _v_r6);
        __m256 _v_t56b = _mm256_fmadd_ps(_v0_5, _v_r5, _mm256_fmadd_ps(_vm2_5, _v_r3, _mm256_mul_ps(_v_r1, _v2)));

        __m256 _v_t0 = _mm256_sub_ps(_mm256_fmadd_ps(_v5_25, _mm256_sub_ps(_v_r4, _v_r2), _v_r0), _v_r6);
        __m256 _v_t1 = _mm256_add_ps(_v_t12a, _v_t12b);
        __m256 _v_t2 = _mm256_sub_ps(_v_t12a, _v_t12b);
        __m256 _v_t3 = _mm256_add_ps(_v_t34a, _v_t34b);
        __m256 _v_t4 = _mm256_sub_ps(_v_t34a, _v_t34b);
        __m256 _v_t5 = _mm256_add_ps(_v_t56a, _v_t56b);
        __m256 _v_t6 = _mm256_sub_ps(_v_t56a, _v_t56b);
        __m256 _v_t7 = _mm256_fmadd_ps(_v5_25, _mm256_sub_ps(_v_r3, _v_r5), _mm256_sub_ps(_v_r7, _v_r1));

        _mm256_storeu_ps(dest + 0 * stride + i * 8, _v_t0);
        _mm256_storeu_ps(dest + 1 * stride + i * 8, _v_t1);
        _mm256_storeu_ps(dest + 2 * stride + i * 8, _v_t2);
        _mm256_storeu_ps(dest + 3 * stride + i * 8, _v_t3);
        _mm256_storeu_ps(dest + 4 * stride + i * 8, _v_t4);
        _mm256_storeu_ps(dest + 5 * stride + i * 8, _v_t5);
        _mm256_storeu_ps(dest + 6 * stride + i * 8, _v_t6);
        _mm256_storeu_ps(dest + 7 * stride + i * 8, _v_t7);
    }
}

static void dataTransRightUnit8K3S1Pack1(float* dest, const float* source, int ldh, int ldce) {
    for (int i = 0; i < 8; ++i) {
        float r0 = source[i * ldh + 0];
        float r1 = source[i * ldh + 1];
        float r2 = source[i * ldh + 2];
        float r3 = source[i * ldh + 3];
        float r4 = source[i * ldh + 4];
        float r5 = source[i * ldh + 5];
        float r6 = source[i * ldh + 6];
        float r7 = source[i * ldh + 7];

        float t12a = static_cast<float>(-4.25 * r4 + (r2 + r6));
        float t12b = static_cast<float>(-4.25 * r3 + (r1 + r5));
        float t34a = static_cast<float>(-1.25 * r4 + (0.25 * r2 + r6));
        float t34b = static_cast<float>(2 * r5 + (-2.5 * r3 + (0.5 * r1)));
        float t56a = static_cast<float>(4 * (-1.25 * r4 + r2) + r6);
        float t56b = static_cast<float>(0.5 * r5 + (-2.5 * r3 + (2 * r1)));

        float t0 = static_cast<float>((5.25 * (r4 - r2) + r0) - r6);
        float t1 = static_cast<float>(t12a + t12b);
        float t2 = static_cast<float>(t12a - t12b);
        float t3 = static_cast<float>(t34a + t34b);
        float t4 = static_cast<float>(t34a - t34b);
        float t5 = static_cast<float>(t56a + t56b);
        float t6 = static_cast<float>(t56a - t56b);
        float t7 = static_cast<float>(5.25 * (r3 - r5) + (r7 - r1));

        dest[i * 8 + 0] = t0;
        dest[i * 8 + 1] = t1;
        dest[i * 8 + 2] = t2;
        dest[i * 8 + 3] = t3;
        dest[i * 8 + 4] = t4;
        dest[i * 8 + 5] = t5;
        dest[i * 8 + 6] = t6;
        dest[i * 8 + 7] = t7;
    }
}

static void dataTransRightUnit8K3S1Pack4(float* dest, const float* source, int ldh, int ldc) {
    __m128 _v5_25 = _mm_set1_ps(5.25f);
    __m128 _vm4_25 = _mm_set1_ps(-4.25f);
    __m128 _vm1_25 = _mm_set1_ps(-1.25f);
    __m128 _v0_25 = _mm_set1_ps(0.25f);
    __m128 _vm2_5 = _mm_set1_ps(-2.5f);
    __m128 _v0_5 = _mm_set1_ps(0.5f);
    __m128 _v2 = _mm_set1_ps(2.f);
    __m128 _v4 = _mm_set1_ps(4.f);

    for (int i = 0; i < 8; ++i) {
        __m256 _v_tr0 = _mm256_loadu_ps(source + i * ldh + 0 * ldc);
        __m256 _v_tr1 = _mm256_loadu_ps(source + i * ldh + 1 * ldc);
        __m256 _v_tr2 = _mm256_loadu_ps(source + i * ldh + 2 * ldc);
        __m256 _v_tr3 = _mm256_loadu_ps(source + i * ldh + 3 * ldc);

        transpose_4x8(_v_tr0, _v_tr1, _v_tr2, _v_tr3);

        __m128 _v_r0 = _mm256_extractf128_ps(_v_tr0, 0);
        __m128 _v_r1 = _mm256_extractf128_ps(_v_tr0, 1);
        __m128 _v_r2 = _mm256_extractf128_ps(_v_tr1, 0);
        __m128 _v_r3 = _mm256_extractf128_ps(_v_tr1, 1);
        __m128 _v_r4 = _mm256_extractf128_ps(_v_tr2, 0);
        __m128 _v_r5 = _mm256_extractf128_ps(_v_tr2, 1);
        __m128 _v_r6 = _mm256_extractf128_ps(_v_tr3, 0);
        __m128 _v_r7 = _mm256_extractf128_ps(_v_tr3, 1);

        __m128 _v_t12a = _mm_fmadd_ps(_vm4_25, _v_r4, _mm_add_ps(_v_r2, _v_r6));
        __m128 _v_t12b = _mm_fmadd_ps(_vm4_25, _v_r3, _mm_add_ps(_v_r1, _v_r5));
        __m128 _v_t34a = _mm_fmadd_ps(_vm1_25, _v_r4, _mm_fmadd_ps(_v0_25, _v_r2, _v_r6));
        __m128 _v_t34b = _mm_fmadd_ps(_v2, _v_r5, _mm_fmadd_ps(_vm2_5, _v_r3, _mm_mul_ps(_v_r1, _v0_5)));
        __m128 _v_t56a = _mm_fmadd_ps(_v4, _mm_fmadd_ps(_vm1_25, _v_r4, _v_r2), _v_r6);
        __m128 _v_t56b = _mm_fmadd_ps(_v0_5, _v_r5, _mm_fmadd_ps(_vm2_5, _v_r3, _mm_mul_ps(_v_r1, _v2)));

        __m128 _v_t0 = _mm_sub_ps(_mm_fmadd_ps(_v5_25, _mm_sub_ps(_v_r4, _v_r2), _v_r0), _v_r6);
        __m128 _v_t1 = _mm_add_ps(_v_t12a, _v_t12b);
        __m128 _v_t2 = _mm_sub_ps(_v_t12a, _v_t12b);
        __m128 _v_t3 = _mm_add_ps(_v_t34a, _v_t34b);
        __m128 _v_t4 = _mm_sub_ps(_v_t34a, _v_t34b);
        __m128 _v_t5 = _mm_add_ps(_v_t56a, _v_t56b);
        __m128 _v_t6 = _mm_sub_ps(_v_t56a, _v_t56b);
        __m128 _v_t7 = _mm_fmadd_ps(_v5_25, _mm_sub_ps(_v_r3, _v_r5), _mm_sub_ps(_v_r7, _v_r1));

        _mm_storeu_ps(dest + i * 8 * 4 + 0 * 4, _v_t0);
        _mm_storeu_ps(dest + i * 8 * 4 + 1 * 4, _v_t1);
        _mm_storeu_ps(dest + i * 8 * 4 + 2 * 4, _v_t2);
        _mm_storeu_ps(dest + i * 8 * 4 + 3 * 4, _v_t3);
        _mm_storeu_ps(dest + i * 8 * 4 + 4 * 4, _v_t4);
        _mm_storeu_ps(dest + i * 8 * 4 + 5 * 4, _v_t5);
        _mm_storeu_ps(dest + i * 8 * 4 + 6 * 4, _v_t6);
        _mm_storeu_ps(dest + i * 8 * 4 + 7 * 4, _v_t7);
    }
}

static void dataTransRightUnit8K3S1Pack8(float* dest, const float* source, int ldh, int ldc) {
    __m256 _v5_25 = _mm256_set1_ps(5.25f);
    __m256 _vm4_25 = _mm256_set1_ps(-4.25f);
    __m256 _vm1_25 = _mm256_set1_ps(-1.25f);
    __m256 _v0_25 = _mm256_set1_ps(0.25f);
    __m256 _vm2_5 = _mm256_set1_ps(-2.5f);
    __m256 _v0_5 = _mm256_set1_ps(0.5f);
    __m256 _v2 = _mm256_set1_ps(2.f);
    __m256 _v4 = _mm256_set1_ps(4.f);

    for (int i = 0; i < 8; ++i) {
        __m256 _v_r0 = _mm256_loadu_ps(source + i * ldh + 0 * ldc);
        __m256 _v_r1 = _mm256_loadu_ps(source + i * ldh + 1 * ldc);
        __m256 _v_r2 = _mm256_loadu_ps(source + i * ldh + 2 * ldc);
        __m256 _v_r3 = _mm256_loadu_ps(source + i * ldh + 3 * ldc);
        __m256 _v_r4 = _mm256_loadu_ps(source + i * ldh + 4 * ldc);
        __m256 _v_r5 = _mm256_loadu_ps(source + i * ldh + 5 * ldc);
        __m256 _v_r6 = _mm256_loadu_ps(source + i * ldh + 6 * ldc);
        __m256 _v_r7 = _mm256_loadu_ps(source + i * ldh + 7 * ldc);

        transpose_8x8(_v_r0, _v_r1, _v_r2, _v_r3, _v_r4, _v_r5, _v_r6, _v_r7);

        __m256 _v_t12a = _mm256_fmadd_ps(_vm4_25, _v_r4, _mm256_add_ps(_v_r2, _v_r6));
        __m256 _v_t12b = _mm256_fmadd_ps(_vm4_25, _v_r3, _mm256_add_ps(_v_r1, _v_r5));
        __m256 _v_t34a = _mm256_fmadd_ps(_vm1_25, _v_r4, _mm256_fmadd_ps(_v0_25, _v_r2, _v_r6));
        __m256 _v_t34b = _mm256_fmadd_ps(_v2, _v_r5, _mm256_fmadd_ps(_vm2_5, _v_r3, _mm256_mul_ps(_v_r1, _v0_5)));
        __m256 _v_t56a = _mm256_fmadd_ps(_v4, _mm256_fmadd_ps(_vm1_25, _v_r4, _v_r2), _v_r6);
        __m256 _v_t56b = _mm256_fmadd_ps(_v0_5, _v_r5, _mm256_fmadd_ps(_vm2_5, _v_r3, _mm256_mul_ps(_v_r1, _v2)));

        __m256 _v_t0 = _mm256_sub_ps(_mm256_fmadd_ps(_v5_25, _mm256_sub_ps(_v_r4, _v_r2), _v_r0), _v_r6);
        __m256 _v_t1 = _mm256_add_ps(_v_t12a, _v_t12b);
        __m256 _v_t2 = _mm256_sub_ps(_v_t12a, _v_t12b);
        __m256 _v_t3 = _mm256_add_ps(_v_t34a, _v_t34b);
        __m256 _v_t4 = _mm256_sub_ps(_v_t34a, _v_t34b);
        __m256 _v_t5 = _mm256_add_ps(_v_t56a, _v_t56b);
        __m256 _v_t6 = _mm256_sub_ps(_v_t56a, _v_t56b);
        __m256 _v_t7 = _mm256_fmadd_ps(_v5_25, _mm256_sub_ps(_v_r3, _v_r5), _mm256_sub_ps(_v_r7, _v_r1));

        _mm256_storeu_ps(dest + i * 8 * 8 + 0 * 8, _v_t0);
        _mm256_storeu_ps(dest + i * 8 * 8 + 1 * 8, _v_t1);
        _mm256_storeu_ps(dest + i * 8 * 8 + 2 * 8, _v_t2);
        _mm256_storeu_ps(dest + i * 8 * 8 + 3 * 8, _v_t3);
        _mm256_storeu_ps(dest + i * 8 * 8 + 4 * 8, _v_t4);
        _mm256_storeu_ps(dest + i * 8 * 8 + 5 * 8, _v_t5);
        _mm256_storeu_ps(dest + i * 8 * 8 + 6 * 8, _v_t6);
        _mm256_storeu_ps(dest + i * 8 * 8 + 7 * 8, _v_t7);
    }
}

inline void winogradDataTransUnit8K3S1Pack1(float* dest, int ldest, const float* source, int ldh, int ldc) {
    float dB[8 * 8 * 1] = {0};
    dataTransRightUnit8K3S1Pack1(dB, source, ldh, ldc);
    dataTransLeftUnit8K3S1Pack1(dest, ldest, dB);
}

inline void winogradDataTransUnit8K3S1Pack4(float* dest, int ldest, const float* source, int ldh, int ldc) {
    float dB[8 * 8 * 4] = {0};
    dataTransRightUnit8K3S1Pack4(dB, source, ldh, ldc);
    dataTransLeftUnit8K3S1Pack4(dest, ldest, dB);
}

inline void winogradDataTransUnit8K3S1Pack8(float* dest, int ldest, const float* source, int ldh, int ldc) {
    float dB[8 * 8 * 8] = {0};
    dataTransRightUnit8K3S1Pack8(dB, source, ldh, ldc);
    dataTransLeftUnit8K3S1Pack8(dest, ldest, dB);
}


}  // namespace kernel
}  // namespace cpu
}  // namespace fastnum