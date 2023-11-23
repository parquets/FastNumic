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


static void uvTransRightUnit4K3S1Pack1(float* dest, int stride, const float* source) {
    dest[0] += source[0 * 4 + 0] + source[0 * 4 + 1] + source[0 * 4 + 2];
    dest[1] += source[0 * 4 + 1] - source[0 * 4 + 2] + source[0 * 4 + 3];
    dest += stride;
    dest[0] += source[1 * 4 + 0] + source[1 * 4 + 1] + source[1 * 4 + 2];
    dest[1] += source[1 * 4 + 1] - source[1 * 4 + 2] + source[1 * 4 + 3];
}

static void uvTransLeftUnit4K3S1Pack1(float* dest, const float* source_v,  const float* source_u, int stride) {
    dest[0] = source_v[0 * stride + 0] * source_u[0 * stride + 0] + source_v[1 * stride + 0] * source_u[1 * stride + 0] + source_v[2 * stride + 0] * source_u[2 * stride + 0];
    dest[1] = source_v[0 * stride + 1] * source_u[0 * stride + 1] + source_v[1 * stride + 1] * source_u[1 * stride + 1] + source_v[2 * stride + 1] * source_u[2 * stride + 1];
    dest[2] = source_v[0 * stride + 2] * source_u[0 * stride + 2] + source_v[1 * stride + 2] * source_u[1 * stride + 2] + source_v[2 * stride + 2] * source_u[2 * stride + 2];
    dest[3] = source_v[0 * stride + 3] * source_u[0 * stride + 3] + source_v[1 * stride + 3] * source_u[1 * stride + 3] + source_v[2 * stride + 3] * source_u[2 * stride + 3];

    dest[4] = source_v[1 * stride + 0] * source_u[1 * stride + 0] - source_v[2 * stride + 0] * source_u[2 * stride + 0] + source_v[3 * stride + 0] * source_u[3 * stride + 0];
    dest[5] = source_v[1 * stride + 1] * source_u[1 * stride + 1] - source_v[2 * stride + 1] * source_u[2 * stride + 1] + source_v[3 * stride + 1] * source_u[3 * stride + 1];
    dest[6] = source_v[1 * stride + 2] * source_u[1 * stride + 2] - source_v[2 * stride + 2] * source_u[2 * stride + 2] + source_v[3 * stride + 2] * source_u[3 * stride + 2];
    dest[7] = source_v[1 * stride + 3] * source_u[1 * stride + 3] - source_v[2 * stride + 3] * source_u[2 * stride + 3] + source_v[3 * stride + 3] * source_u[3 * stride + 3];
}


static void uvTransLeftUnit4K3S1Pack4(float* dest, const float* source_v, const float* source_u, int stride) {
    for (int i = 0; i < 4; ++i) {
        int ind0 = 0 * stride + i * 4;
        int ind1 = 1 * stride + i * 4;
        int ind2 = 2 * stride + i * 4;
        int ind3 = 3 * stride + i * 4;
        __m128 _v_r0 = _mm_mul_ps(_mm_loadu_ps(source_v + ind0), _mm_loadu_ps(source_u + ind0));
        __m128 _v_r1 = _mm_mul_ps(_mm_loadu_ps(source_v + ind1), _mm_loadu_ps(source_u + ind1));
        __m128 _v_r2 = _mm_mul_ps(_mm_loadu_ps(source_v + ind2), _mm_loadu_ps(source_u + ind2));
        __m128 _v_r3 = _mm_mul_ps(_mm_loadu_ps(source_v + ind3), _mm_loadu_ps(source_u + ind3));

        __m128 _v_t0 = _mm_add_ps(_v_r0, _mm_add_ps(_v_r1, _v_r2));
        __m128 _v_t1 = _mm_add_ps(_mm_sub_ps(_v_r1, _v_r2), _v_r3);

        _mm_storeu_ps(dest + 0 * 4 * 4 + i * 4, _v_t0);
        _mm_storeu_ps(dest + 1 * 4 * 4 + i * 4, _v_t1);
    }
}

static void uvTransRightUnit4K3S1Pack4(float* dest, int stride, const float* source) {
    for (int i = 0; i < 2; ++i) {
        __m128 _v_r0 = _mm_loadu_ps(source + i * 4 * 4 + 0 * 4);
        __m128 _v_r1 = _mm_loadu_ps(source + i * 4 * 4 + 1 * 4);
        __m128 _v_r2 = _mm_loadu_ps(source + i * 4 * 4 + 2 * 4);
        __m128 _v_r3 = _mm_loadu_ps(source + i * 4 * 4 + 3 * 4);

        __m128 _v_t0 = _mm_add_ps(_v_r0, _mm_add_ps(_v_r1, _v_r2));
        __m128 _v_t1 = _mm_add_ps(_mm_sub_ps(_v_r1, _v_r2), _v_r3);

        dest[i * stride + 0] += reduce_add_ps(_v_t0);
        dest[i * stride + 1] += reduce_add_ps(_v_t1);
    }
}

static void uvTransLeftUnit4K3S1Pack8(float* dest, const float* source_v, const float* source_u, int stride) {
    for (int i = 0; i < 4; ++i) {
        int ind0 = 0 * stride + i * 8;
        int ind1 = 1 * stride + i * 8;
        int ind2 = 2 * stride + i * 8;
        int ind3 = 3 * stride + i * 8;
        __m256 _v_r0 = _mm256_mul_ps(_mm256_loadu_ps(source_v + ind0), _mm256_loadu_ps(source_u + ind0));
        __m256 _v_r1 = _mm256_mul_ps(_mm256_loadu_ps(source_v + ind1), _mm256_loadu_ps(source_u + ind1));
        __m256 _v_r2 = _mm256_mul_ps(_mm256_loadu_ps(source_v + ind2), _mm256_loadu_ps(source_u + ind2));
        __m256 _v_r3 = _mm256_mul_ps(_mm256_loadu_ps(source_v + ind3), _mm256_loadu_ps(source_u + ind3));

        __m256 _v_t0 = _mm256_add_ps(_v_r0, _mm256_add_ps(_v_r1, _v_r2));
        __m256 _v_t1 = _mm256_add_ps(_mm256_sub_ps(_v_r1, _v_r2), _v_r3);

        _mm256_storeu_ps(dest + 0 * 4 * 8 + i * 8, _v_t0);
        _mm256_storeu_ps(dest + 1 * 4 * 8 + i * 8, _v_t1);
     }
}

static void uvTransRightUnit4K3S1Pack8(float* dest, int stride, const float* source) {
    for (int i = 0; i < 2; ++i) {
        __m256 _v_r0 = _mm256_loadu_ps(source + i * 4 * 8 + 0 * 8);
        __m256 _v_r1 = _mm256_loadu_ps(source + i * 4 * 8 + 1 * 8);
        __m256 _v_r2 = _mm256_loadu_ps(source + i * 4 * 8 + 2 * 8);
        __m256 _v_r3 = _mm256_loadu_ps(source + i * 4 * 8 + 3 * 8);

        __m256 _v_t0 = _mm256_add_ps(_v_r0, _mm256_add_ps(_v_r1, _v_r2));
        __m256 _v_t1 = _mm256_add_ps(_mm256_sub_ps(_v_r1, _v_r2), _v_r3);

        dest[i * stride + 0] += reduce_add_ps(_v_t0);
        dest[i * stride + 1] += reduce_add_ps(_v_t1);
    }
}

inline void winogradUVTransUnit4K3S1Pack1(float* dest, int ldd, const float* source_v, const float* source_u, int lds) {
    float ATUV[2 * 4 * 1] = {0};
    uvTransLeftUnit4K3S1Pack1(ATUV, source_v, source_u, lds);
    uvTransRightUnit4K3S1Pack1(dest, ldd, ATUV);
}

inline void winogradUVTransUnit4K3S1Pack4(float* dest, int ldd, const float* source_v, const float* source_u, int lds) {
    float ATUV[2 * 4 * 4] = {0};
    uvTransLeftUnit4K3S1Pack4(ATUV, source_v, source_u, lds);
    uvTransRightUnit4K3S1Pack4(dest, ldd, ATUV);
}

inline void winogradUVTransUnit4K3S1Pack8(float* dest, int ldd, const float* source_v, const float* source_u, int lds) {
    float ATUV[2 * 4 * 8] = {0};
    uvTransLeftUnit4K3S1Pack8(ATUV, source_v, source_u, lds);
    uvTransRightUnit4K3S1Pack8(dest, ldd, ATUV);
}

static void uvTransLeftUnit8K3S1Pack1(float* dest, const float* source_v, const float* source_u, int stride) {
    for (int i = 0; i < 8; ++i) {
        float r0 = source_v[0 * stride + i] * source_u[0 * stride + i];
        float r1 = source_v[1 * stride + i] * source_u[1 * stride + i];
        float r2 = source_v[2 * stride + i] * source_u[2 * stride + i];
        float r3 = source_v[3 * stride + i] * source_u[3 * stride + i];
        float r4 = source_v[4 * stride + i] * source_u[4 * stride + i];
        float r5 = source_v[5 * stride + i] * source_u[5 * stride + i];
        float r6 = source_v[6 * stride + i] * source_u[6 * stride + i];
        float r7 = source_v[7 * stride + i] * source_u[7 * stride + i];

        float t024a = r1 + r2;
        float t135a = r1 - r2;
        float t024b = r3 + r4;
        float t135b = r3 - r4;
        float t024c = r5 + r6;
        float t135c = r5 - r6;

        float t0 = (r0 + t024a) + (32 * t024c + t024b);
        float t1 = 16 * t135c + (2 * t135b + t135a);
        float t2 = 8 * t024c + (4 * t024b + t024a);
        float t3 = 4 * t135c + (8 * t135b + t135a);
        float t4 = 2 * t024c + (16 * t024b + t024a);
        float t5 = (r7 + t135a) + (32 * t135b + t135c);

        dest[0 * 8 + i] = t0;
        dest[1 * 8 + i] = t1;
        dest[2 * 8 + i] = t2;
        dest[3 * 8 + i] = t3;
        dest[4 * 8 + i] = t4;
        dest[5 * 8 + i] = t5;
    }
}

static void uvTransLeftUnit8K3S1Pack4(float* dest, const float* source_v, const float* source_u, int stride) {
    __m128 _v32 = _mm_set1_ps(32.0f);
    __m128 _v16 = _mm_set1_ps(16.0f);
    __m128 _v8 = _mm_set1_ps(8.f);
    __m128 _v4 = _mm_set1_ps(4.f);
    __m128 _v2 = _mm_set1_ps(2.f);

    for (int i = 0; i < 8; ++i) {
        int ind0 = 0 * stride + i * 4;
        int ind1 = 1 * stride + i * 4;
        int ind2 = 2 * stride + i * 4;
        int ind3 = 3 * stride + i * 4;
        int ind4 = 4 * stride + i * 4;
        int ind5 = 5 * stride + i * 4;
        int ind6 = 6 * stride + i * 4;
        int ind7 = 7 * stride + i * 4;

        __m128 _v_r0 = _mm_mul_ps(_mm_loadu_ps(source_v + ind0), _mm_loadu_ps(source_u + ind0));
        __m128 _v_r1 = _mm_mul_ps(_mm_loadu_ps(source_v + ind1), _mm_loadu_ps(source_u + ind1));
        __m128 _v_r2 = _mm_mul_ps(_mm_loadu_ps(source_v + ind2), _mm_loadu_ps(source_u + ind2));
        __m128 _v_r3 = _mm_mul_ps(_mm_loadu_ps(source_v + ind3), _mm_loadu_ps(source_u + ind3));
        __m128 _v_r4 = _mm_mul_ps(_mm_loadu_ps(source_v + ind4), _mm_loadu_ps(source_u + ind4));
        __m128 _v_r5 = _mm_mul_ps(_mm_loadu_ps(source_v + ind5), _mm_loadu_ps(source_u + ind5));
        __m128 _v_r6 = _mm_mul_ps(_mm_loadu_ps(source_v + ind6), _mm_loadu_ps(source_u + ind6));
        __m128 _v_r7 = _mm_mul_ps(_mm_loadu_ps(source_v + ind7), _mm_loadu_ps(source_u + ind7));

        __m128 _v_t024a = _mm_add_ps(_v_r1, _v_r2);
        __m128 _v_t135a = _mm_sub_ps(_v_r1, _v_r2);
        __m128 _v_t024b = _mm_add_ps(_v_r3, _v_r4);
        __m128 _v_t135b = _mm_sub_ps(_v_r3, _v_r4);
        __m128 _v_t024c = _mm_add_ps(_v_r5, _v_r6);
        __m128 _v_t135c = _mm_sub_ps(_v_r5, _v_r6);

        __m128 _v_t0 = _mm_add_ps(_mm_add_ps(_v_r0, _v_t024a), _mm_fmadd_ps(_v32, _v_t024c, _v_t024b));
        __m128 _v_t1 = _mm_fmadd_ps(_v16, _v_t135c, _mm_fmadd_ps(_v2, _v_t135b, _v_t135a));
        __m128 _v_t2 = _mm_fmadd_ps(_v8, _v_t024c, _mm_fmadd_ps(_v4, _v_t024b, _v_t024a));
        __m128 _v_t3 = _mm_fmadd_ps(_v4, _v_t135c, _mm_fmadd_ps(_v8, _v_t135b, _v_t135a));
        __m128 _v_t4 = _mm_fmadd_ps(_v2, _v_t024c, _mm_fmadd_ps(_v16, _v_t024b, _v_t024a));
        __m128 _v_t5 = _mm_add_ps(_mm_add_ps(_v_r7, _v_t135a), _mm_fmadd_ps(_v32, _v_t135b, _v_t135c));

        _mm_storeu_ps(dest + 0 * 8 * 4 + i * 4, _v_t0);
        _mm_storeu_ps(dest + 1 * 8 * 4 + i * 4, _v_t1);
        _mm_storeu_ps(dest + 2 * 8 * 4 + i * 4, _v_t2);
        _mm_storeu_ps(dest + 3 * 8 * 4 + i * 4, _v_t3);
        _mm_storeu_ps(dest + 4 * 8 * 4 + i * 4, _v_t4);
        _mm_storeu_ps(dest + 5 * 8 * 4 + i * 4, _v_t5);
    }
}

static void uvTransLeftUnit8K3S1Pack8(float* dest, const float* source_v, const float* source_u, int stride) {
    __m256 _v32 = _mm256_set1_ps(32.0f);
    __m256 _v16 = _mm256_set1_ps(16.0f);
    __m256 _v8 = _mm256_set1_ps(8.f);
    __m256 _v4 = _mm256_set1_ps(4.f);
    __m256 _v2 = _mm256_set1_ps(2.f);

    for (int i = 0; i < 8; ++i) {
        int ind0  = 0 * stride + i * 8;
        int ind1  = 1 * stride + i * 8;
        int ind2  = 2 * stride + i * 8;
        int ind3  = 3 * stride + i * 8;
        int ind4  = 4 * stride + i * 8;
        int ind5  = 5 * stride + i * 8;
        int ind6  = 6 * stride + i * 8;
        int ind7  = 7 * stride + i * 8;

        __m256 _v_r0 = _mm256_mul_ps(_mm256_loadu_ps(source_v + ind0), _mm256_loadu_ps(source_u + ind0));
        __m256 _v_r1 = _mm256_mul_ps(_mm256_loadu_ps(source_v + ind1), _mm256_loadu_ps(source_u + ind1));
        __m256 _v_r2 = _mm256_mul_ps(_mm256_loadu_ps(source_v + ind2), _mm256_loadu_ps(source_u + ind2));
        __m256 _v_r3 = _mm256_mul_ps(_mm256_loadu_ps(source_v + ind3), _mm256_loadu_ps(source_u + ind3));
        __m256 _v_r4 = _mm256_mul_ps(_mm256_loadu_ps(source_v + ind4), _mm256_loadu_ps(source_u + ind4));
        __m256 _v_r5 = _mm256_mul_ps(_mm256_loadu_ps(source_v + ind5), _mm256_loadu_ps(source_u + ind5));
        __m256 _v_r6 = _mm256_mul_ps(_mm256_loadu_ps(source_v + ind6), _mm256_loadu_ps(source_u + ind6));
        __m256 _v_r7 = _mm256_mul_ps(_mm256_loadu_ps(source_v + ind7), _mm256_loadu_ps(source_u + ind7));

        __m256 _v_t024a = _mm256_add_ps(_v_r1, _v_r2);
        __m256 _v_t135a = _mm256_sub_ps(_v_r1, _v_r2);
        __m256 _v_t024b = _mm256_add_ps(_v_r3, _v_r4);
        __m256 _v_t135b = _mm256_sub_ps(_v_r3, _v_r4);
        __m256 _v_t024c = _mm256_add_ps(_v_r5, _v_r6);
        __m256 _v_t135c = _mm256_sub_ps(_v_r5, _v_r6);

        __m256 _v_t0 = _mm256_add_ps(_mm256_add_ps(_v_r0, _v_t024a), _mm256_fmadd_ps(_v32, _v_t024c, _v_t024b));
        __m256 _v_t1 = _mm256_fmadd_ps(_v16, _v_t135c, _mm256_fmadd_ps(_v2, _v_t135b, _v_t135a));
        __m256 _v_t2 = _mm256_fmadd_ps(_v8, _v_t024c, _mm256_fmadd_ps(_v4, _v_t024b, _v_t024a));
        __m256 _v_t3 = _mm256_fmadd_ps(_v4, _v_t135c, _mm256_fmadd_ps(_v8, _v_t135b, _v_t135a));
        __m256 _v_t4 = _mm256_fmadd_ps(_v2, _v_t024c, _mm256_fmadd_ps(_v16, _v_t024b, _v_t024a));
        __m256 _v_t5 = _mm256_add_ps(_mm256_add_ps(_v_r7, _v_t135a), _mm256_fmadd_ps(_v32, _v_t135b, _v_t135c));

        _mm256_storeu_ps(dest + 0 * 8 * 8 + i * 8, _v_t0);
        _mm256_storeu_ps(dest + 1 * 8 * 8 + i * 8, _v_t1);
        _mm256_storeu_ps(dest + 2 * 8 * 8 + i * 8, _v_t2);
        _mm256_storeu_ps(dest + 3 * 8 * 8 + i * 8, _v_t3);
        _mm256_storeu_ps(dest + 4 * 8 * 8 + i * 8, _v_t4);
        _mm256_storeu_ps(dest + 5 * 8 * 8 + i * 8, _v_t5);
    }
}


static void uvTransRightUnit8K3S1Pack1(float* dest, int stride, const float* source) {
    for (int i = 0; i < 6; ++i) {
        float r0 = source[i * 8 + 0];
        float r1 = source[i * 8 + 1];
        float r2 = source[i * 8 + 2];
        float r3 = source[i * 8 + 3];
        float r4 = source[i * 8 + 4];
        float r5 = source[i * 8 + 5];
        float r6 = source[i * 8 + 6];
        float r7 = source[i * 8 + 7];

        float t024a = r1 + r2;
        float t135a = r1 - r2;
        float t024b = r3 + r4;
        float t135b = r3 - r4;
        float t024c = r5 + r6;
        float t135c = r5 - r6;

        float t0 = (r0 + t024a) + (32 * t024c + t024b);
        float t1 = 16 * t135c + (2 * t135b + t135a);
        float t2 = 8 * t024c + (4 * t024b + t024a);
        float t3 = 4 * t135c + (8 * t135b + t135a);
        float t4 = 2 * t024c + (16 * t024b + t024a);
        float t5 = (r7 + t135a) + (32 * t135b + t135c);

        dest[0] += t0;
        dest[1] += t1;
        dest[2] += t2;
        dest[3] += t3;
        dest[4] += t4;
        dest[5] += t5;

        dest += stride;
    }
}

static void uvTransRightUnit8K3S1Pack4(float* dest, int stride, const float* source) {
    __m128 _v32 = _mm_set1_ps(32.0f);
    __m128 _v16 = _mm_set1_ps(16.0f);
    __m128 _v8 = _mm_set1_ps(8.f);
    __m128 _v4 = _mm_set1_ps(4.f);
    __m128 _v2 = _mm_set1_ps(2.f);

    for (int i = 0; i < 6; ++i) {
        __m128 _v_r0 = _mm_loadu_ps(source + i * 8 * 4 + 0 * 4);
        __m128 _v_r1 = _mm_loadu_ps(source + i * 8 * 4 + 1 * 4);
        __m128 _v_r2 = _mm_loadu_ps(source + i * 8 * 4 + 2 * 4);
        __m128 _v_r3 = _mm_loadu_ps(source + i * 8 * 4 + 3 * 4);
        __m128 _v_r4 = _mm_loadu_ps(source + i * 8 * 4 + 4 * 4);
        __m128 _v_r5 = _mm_loadu_ps(source + i * 8 * 4 + 5 * 4);
        __m128 _v_r6 = _mm_loadu_ps(source + i * 8 * 4 + 6 * 4);
        __m128 _v_r7 = _mm_loadu_ps(source + i * 8 * 4 + 7 * 4);

        __m128 _v_t024a = _mm_add_ps(_v_r1, _v_r2);
        __m128 _v_t135a = _mm_sub_ps(_v_r1, _v_r2);
        __m128 _v_t024b = _mm_add_ps(_v_r3, _v_r4);
        __m128 _v_t135b = _mm_sub_ps(_v_r3, _v_r4);
        __m128 _v_t024c = _mm_add_ps(_v_r5, _v_r6);
        __m128 _v_t135c = _mm_sub_ps(_v_r5, _v_r6);

        __m128 _v_t0 = _mm_add_ps(_mm_add_ps(_v_r0, _v_t024a), _mm_fmadd_ps(_v32, _v_t024c, _v_t024b));
        __m128 _v_t1 = _mm_fmadd_ps(_v16, _v_t135c, _mm_fmadd_ps(_v2, _v_t135b, _v_t135a));
        __m128 _v_t2 = _mm_fmadd_ps(_v8, _v_t024c, _mm_fmadd_ps(_v4, _v_t024b, _v_t024a));
        __m128 _v_t3 = _mm_fmadd_ps(_v4, _v_t135c, _mm_fmadd_ps(_v8, _v_t135b, _v_t135a));
        __m128 _v_t4 = _mm_fmadd_ps(_v2, _v_t024c, _mm_fmadd_ps(_v16, _v_t024b, _v_t024a));
        __m128 _v_t5 = _mm_add_ps(_mm_add_ps(_v_r7, _v_t135a), _mm_fmadd_ps(_v32, _v_t135b, _v_t135c));

        dest[0] += reduce_add_ps(_v_t0);
        dest[1] += reduce_add_ps(_v_t1);
        dest[2] += reduce_add_ps(_v_t2);
        dest[3] += reduce_add_ps(_v_t3);
        dest[4] += reduce_add_ps(_v_t4);
        dest[5] += reduce_add_ps(_v_t5);
        dest += stride;
    }
}

static void uvTransRightUnit8K3S1Pack8(float* dest, int stride, const float* source)  {
    __m256 _v32 = _mm256_set1_ps(32.0f);
    __m256 _v16 = _mm256_set1_ps(16.0f);
    __m256 _v8 = _mm256_set1_ps(8.f);
    __m256 _v4 = _mm256_set1_ps(4.f);
    __m256 _v2 = _mm256_set1_ps(2.f);

    for (int i = 0; i < 6; ++i) {
        __m256 _v_r0 = _mm256_loadu_ps(source + i * 8 * 8 + 0 * 8);
        __m256 _v_r1 = _mm256_loadu_ps(source + i * 8 * 8 + 1 * 8);
        __m256 _v_r2 = _mm256_loadu_ps(source + i * 8 * 8 + 2 * 8);
        __m256 _v_r3 = _mm256_loadu_ps(source + i * 8 * 8 + 3 * 8);
        __m256 _v_r4 = _mm256_loadu_ps(source + i * 8 * 8 + 4 * 8);
        __m256 _v_r5 = _mm256_loadu_ps(source + i * 8 * 8 + 5 * 8);
        __m256 _v_r6 = _mm256_loadu_ps(source + i * 8 * 8 + 6 * 8);
        __m256 _v_r7 = _mm256_loadu_ps(source + i * 8 * 8 + 7 * 8);

        __m256 _v_t024a = _mm256_add_ps(_v_r1, _v_r2);
        __m256 _v_t135a = _mm256_sub_ps(_v_r1, _v_r2);
        __m256 _v_t024b = _mm256_add_ps(_v_r3, _v_r4);
        __m256 _v_t135b = _mm256_sub_ps(_v_r3, _v_r4);
        __m256 _v_t024c = _mm256_add_ps(_v_r5, _v_r6);
        __m256 _v_t135c = _mm256_sub_ps(_v_r5, _v_r6);

        __m256 _v_t0 = _mm256_add_ps(_mm256_add_ps(_v_r0, _v_t024a), _mm256_fmadd_ps(_v32, _v_t024c, _v_t024b));
        __m256 _v_t1 = _mm256_fmadd_ps(_v16, _v_t135c, _mm256_fmadd_ps(_v2, _v_t135b, _v_t135a));
        __m256 _v_t2 = _mm256_fmadd_ps(_v8, _v_t024c, _mm256_fmadd_ps(_v4, _v_t024b, _v_t024a));
        __m256 _v_t3 = _mm256_fmadd_ps(_v4, _v_t135c, _mm256_fmadd_ps(_v8, _v_t135b, _v_t135a));
        __m256 _v_t4 = _mm256_fmadd_ps(_v2, _v_t024c, _mm256_fmadd_ps(_v16, _v_t024b, _v_t024a));
        __m256 _v_t5 = _mm256_add_ps(_mm256_add_ps(_v_r7, _v_t135a), _mm256_fmadd_ps(_v32, _v_t135b, _v_t135c));

        dest[i * stride + 0] += reduce_add_ps(_v_t0);
        dest[i * stride + 1] += reduce_add_ps(_v_t1);
        dest[i * stride + 2] += reduce_add_ps(_v_t2);
        dest[i * stride + 3] += reduce_add_ps(_v_t3);
        dest[i * stride + 4] += reduce_add_ps(_v_t4);
        dest[i * stride + 5] += reduce_add_ps(_v_t5);
    }
}


inline void winogradUVTransUnit8K3S1Pack1(float* dest, int ldd, const float* source_v, const float* source_u, int lds) {
    float ATUV[48 * 1];
    uvTransLeftUnit8K3S1Pack1(ATUV, source_v, source_u, lds);
    uvTransRightUnit8K3S1Pack1(dest, ldd, ATUV);
}

inline void winogradUVTransUnit8K3S1Pack4(float* dest, int ldd, const float* source_v, const float* source_u, int lds) {
    float ATUV[48 * 4];
    uvTransLeftUnit8K3S1Pack4(ATUV, source_v, source_u, lds);
    uvTransRightUnit8K3S1Pack4(dest, ldd, ATUV);
}

inline void winogradUVTransUnit8K3S1Pack8(float* dest, int ldd, const float* source_v, const float* source_u, int lds) {
    float ATUV[48 * 8];
    uvTransLeftUnit8K3S1Pack8(ATUV, source_v, source_u, lds);
    uvTransRightUnit8K3S1Pack8(dest, ldd, ATUV);
}


}  // namespace kernel
}  // namespace cpu
}  // namespace fastnum