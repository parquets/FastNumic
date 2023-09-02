#pragma once

#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <algorithm>

#include <immintrin.h>
#include <nmmintrin.h>
#include <smmintrin.h>

#include "vectorize_x86.hpp"

namespace fastnum {
namespace cpu {
namespace kernel {

float reduce_min(const float* src, int size) {
    int i = 0;
    float ret = 9999999;

    __m256 _v_min = _mm256_broadcast_ss(&ret);
    for (i = 0; i < size - 31; i += 32) {
        __m256 _v_s0 = _mm256_loadu_ps(src);
        __m256 _v_s1 = _mm256_loadu_ps(src + 8);
        __m256 _v_s2 = _mm256_loadu_ps(src + 16);
        __m256 _v_s3 = _mm256_loadu_ps(src + 24);

        __m256 _v_m0 = _mm256_min_ps(_v_s0, _v_s1);
        __m256 _v_m1 = _mm256_min_ps(_v_s2, _v_s3);

        __m256 _v_m2 = _mm256_min_ps(_v_m0, _v_m1);
        _v_min = _mm256_min_ps(_v_min, _v_m2);
        src += 32;
    }

    for (; i < size - 15; i += 16) {
        __m256 _v_s0 = _mm256_loadu_ps(src);
        __m256 _v_s1 = _mm256_loadu_ps(src + 8);

        __m256 _v_m0 = _mm256_min_ps(_v_s0, _v_s1);
        ;
        _v_min = _mm256_min_ps(_v_min, _v_m0);
        src += 16;
    }

    for (; i < size - 7; i += 8) {
        __m256 _v_s0 = _mm256_loadu_ps(src);
        _v_min = _mm256_min_ps(_v_min, _v_s0);
        src += 8;
    }

    ret = reduce_min_ps(_v_min);

    for (; i < size; ++i) {
        ret = std::min(ret, *src);
        ++src;
    }

    return ret;
}

float reduce_max(const float* src, int size) {
    int i = 0;
    float ret = -9999999;

    __m256 _v_max = _mm256_broadcast_ss(&ret);
    for (i = 0; i < size - 31; i += 32) {
        __m256 _v_s0 = _mm256_loadu_ps(src);
        __m256 _v_s1 = _mm256_loadu_ps(src + 8);
        __m256 _v_s2 = _mm256_loadu_ps(src + 16);
        __m256 _v_s3 = _mm256_loadu_ps(src + 24);

        __m256 _v_m0 = _mm256_max_ps(_v_s0, _v_s1);
        __m256 _v_m1 = _mm256_max_ps(_v_s2, _v_s3);

        __m256 _v_m2 = _mm256_max_ps(_v_m0, _v_m1);
        _v_max = _mm256_max_ps(_v_max, _v_m2);
        src += 32;
    }

    for (; i < size - 15; i += 16) {
        __m256 _v_s0 = _mm256_loadu_ps(src);
        __m256 _v_s1 = _mm256_loadu_ps(src + 8);

        __m256 _v_m0 = _mm256_max_ps(_v_s0, _v_s1);
        ;
        _v_max = _mm256_max_ps(_v_max, _v_m0);
        src += 16;
    }

    for (; i < size - 7; i += 8) {
        __m256 _v_s0 = _mm256_loadu_ps(src);
        _v_max = _mm256_max_ps(_v_max, _v_s0);
        src += 8;
    }

    ret = reduce_max_ps(_v_max);

    for (; i < size; ++i) {
        ret = std::max(ret, *src);
        ++src;
    }

    return ret;
}

float reduce_add(const float* src, int size) {
    int i = 0;
    float ret = 0;

    __m256 _v_add = _mm256_broadcast_ss(&ret);
    for (i = 0; i < size - 31; i += 32) {
        __m256 _v_s0 = _mm256_loadu_ps(src);
        __m256 _v_s1 = _mm256_loadu_ps(src + 8);
        __m256 _v_s2 = _mm256_loadu_ps(src + 16);
        __m256 _v_s3 = _mm256_loadu_ps(src + 24);

        __m256 _v_m0 = _mm256_add_ps(_v_s0, _v_s1);
        __m256 _v_m1 = _mm256_add_ps(_v_s2, _v_s3);

        __m256 _v_m2 = _mm256_add_ps(_v_m0, _v_m1);
        _v_add = _mm256_add_ps(_v_add, _v_m2);
        src += 32;
    }

    for (; i < size - 15; i += 16) {
        __m256 _v_s0 = _mm256_loadu_ps(src);
        __m256 _v_s1 = _mm256_loadu_ps(src + 8);

        __m256 _v_m0 = _mm256_add_ps(_v_s0, _v_s1);
        ;
        _v_add = _mm256_add_ps(_v_add, _v_m0);
        src += 16;
    }

    for (; i < size - 7; i += 8) {
        __m256 _v_s0 = _mm256_loadu_ps(src);
        _v_add = _mm256_add_ps(_v_add, _v_s0);
        src += 8;
    }

    ret = reduce_add_ps(_v_add);

    for (; i < size; ++i) {
        ret += *src;
        ++src;
    }

    return ret;
}


int argmax(const float* src, int size) {
    int max_index;
    int i=0;
    float fmin = -9999999;
    __m256 _v_max = _mm256_broadcast_ss(&fmin);
    __m256i _v_indices = _mm256_setzero_si256();

    for(i=0; i<size-31; i+=32) {
        __m256 _v_x0 = _mm256_loadu_ps(src+0*8);
        __m256 _v_x1 = _mm256_loadu_ps(src+1*8);
        __m256 _v_x2 = _mm256_loadu_ps(src+2*8);
        __m256 _v_x3 = _mm256_loadu_ps(src+3*8);

        __m256 _v_t0 = _mm256_max_ps(_v_x0, _v_x1);
        __m256 _v_t1 = _mm256_max_ps(_v_x2, _v_x3);
        
        __m256 _v_cmax = _mm256_max_ps(_v_t0, _v_t1);
        
        __mmask8 m0 = _mm256_cmp_ps_mask(_v_cmax, _v_max, _CMP_GT_OS);

        if(m0) {
            _v_max = _mm256_mask_blend_ps(m0, _v_max, _v_cmax);
            _v_indices = _mm256_mask_set1_epi32(_v_indices, m0, i);
        }

        src += 32;
    }
}


}  // namespace kernel
}  // namespace cpu
}  // namespace fastnum
