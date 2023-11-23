#pragma once

#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <algorithm>

#include <immintrin.h>
#include <nmmintrin.h>
#include <smmintrin.h>

#include "arithmetic_x86.hpp"
#include "stranspose_x86.hpp"
#include "vectorize_x86.hpp"

namespace fastnum {
namespace cpu {
namespace kernel {

inline void conv2dK1S1Ch6(float* dst, const float* src, const float* weight, int src_h, int src_w) {
    int ch_stride = src_h * src_w;

    const float* src_ptr0 = src + 0 * ch_stride;
    const float* src_ptr1 = src + 1 * ch_stride;
    const float* src_ptr2 = src + 2 * ch_stride;
    const float* src_ptr3 = src + 3 * ch_stride;
    const float* src_ptr4 = src + 4 * ch_stride;
    const float* src_ptr5 = src + 5 * ch_stride;

    __m256 _v_w0 = _mm256_broadcast_ss(weight + 0);
    __m256 _v_w1 = _mm256_broadcast_ss(weight + 1);
    __m256 _v_w2 = _mm256_broadcast_ss(weight + 2);
    __m256 _v_w3 = _mm256_broadcast_ss(weight + 3);
    __m256 _v_w4 = _mm256_broadcast_ss(weight + 4);
    __m256 _v_w5 = _mm256_broadcast_ss(weight + 5);

    int x = 0;
    for (x = 0; x < ch_stride - 7; x += 8) {
        __m256 _v_r0 = _mm256_loadu_ps(src_ptr0);
        __m256 _v_r1 = _mm256_loadu_ps(src_ptr1);
        __m256 _v_r2 = _mm256_loadu_ps(src_ptr2);
        __m256 _v_r3 = _mm256_loadu_ps(src_ptr3);
        __m256 _v_r4 = _mm256_loadu_ps(src_ptr4);
        __m256 _v_r5 = _mm256_loadu_ps(src_ptr5);

        __m256 _v_y = _mm256_loadu_ps(dst + x);

        __m256 _v_t0 = _mm256_fmadd_ps(_v_w0, _v_r0, _mm256_mul_ps(_v_w1, _v_r1));
        __m256 _v_t1 = _mm256_fmadd_ps(_v_w2, _v_r2, _mm256_mul_ps(_v_w3, _v_r3));
        __m256 _v_t2 = _mm256_fmadd_ps(_v_w4, _v_r4, _mm256_mul_ps(_v_w5, _v_r5));

        _v_y = _mm256_add_ps(_v_y, _mm256_add_ps(_mm256_add_ps(_v_t0, _v_t1), _v_t2));

        _mm256_store_ps(dst + x, _v_y);

        src_ptr0 += 8;
        src_ptr1 += 8;
        src_ptr2 += 8;
        src_ptr3 += 8;
        src_ptr4 += 8;
        src_ptr5 += 8;
    }

    for (; x < ch_stride; ++x) {
        float y = (*src_ptr0++) * weight[0] + (*src_ptr1++) * weight[1] + (*src_ptr2++) * weight[2] + (*src_ptr3++) * weight[3] +
                  (*src_ptr4++) * weight[4] + (*src_ptr5++) * weight[5];
        dst[x] += y;
    }
}

inline void conv2dK1S1Ch4(float* dst, const float* src, const float* weight, int src_h, int src_w) {
    int ch_stride = src_h * src_w;

    const float* src_ptr0 = src + 0 * ch_stride;
    const float* src_ptr1 = src + 1 * ch_stride;
    const float* src_ptr2 = src + 2 * ch_stride;
    const float* src_ptr3 = src + 3 * ch_stride;

    __m256 _v_w0 = _mm256_broadcast_ss(weight + 0);
    __m256 _v_w1 = _mm256_broadcast_ss(weight + 1);
    __m256 _v_w2 = _mm256_broadcast_ss(weight + 2);
    __m256 _v_w3 = _mm256_broadcast_ss(weight + 3);

    int x = 0;
    for (x = 0; x < ch_stride - 7; x += 8) {
        __m256 _v_r0 = _mm256_loadu_ps(src_ptr0);
        __m256 _v_r1 = _mm256_loadu_ps(src_ptr1);
        __m256 _v_r2 = _mm256_loadu_ps(src_ptr2);
        __m256 _v_r3 = _mm256_loadu_ps(src_ptr3);

        __m256 _v_y = _mm256_loadu_ps(dst + x);

        __m256 _v_t0 = _mm256_fmadd_ps(_v_w0, _v_r0, _mm256_mul_ps(_v_w1, _v_r1));
        __m256 _v_t1 = _mm256_fmadd_ps(_v_w2, _v_r2, _mm256_mul_ps(_v_w3, _v_r3));

        _v_y = _mm256_add_ps(_v_y, _mm256_add_ps(_v_t0, _v_t1));

        _mm256_store_ps(dst + x, _v_y);

        src_ptr0 += 8;
        src_ptr1 += 8;
        src_ptr2 += 8;
        src_ptr3 += 8;
    }

    for (; x < ch_stride; ++x) {
        float y = (*src_ptr0++) * weight[0] + (*src_ptr1++) * weight[1] + (*src_ptr2++) * weight[2] + (*src_ptr3++) * weight[3];
        dst[x] += y;
    }
}

inline void conv2dK1S1Ch1(float* dst, const float* src, const float* weight, int src_h, int src_w) {
    //fma(dst, src, weight[0], src_h*src_w);
    int ch_stride = src_h*src_w;
    const float* src_ptr0 = src + 0 * ch_stride;
    int x = 0;
    __m256 _v_w0 = _mm256_broadcast_ss(weight + 0);
    for (x = 0; x < ch_stride - 7; x += 8) {
        __m256 _v_r0 = _mm256_loadu_ps(src_ptr0);
        __m256 _v_y = _mm256_loadu_ps(dst + x);
        _v_y = _mm256_fmadd_ps(_v_r0, _v_w0, _v_y);
        _mm256_store_ps(dst + x, _v_y);
        src_ptr0 += 8;
    }
    for (; x < ch_stride; ++x) {
        float y = (*src_ptr0++) * weight[0];
        dst[x] += y;
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace fastnum