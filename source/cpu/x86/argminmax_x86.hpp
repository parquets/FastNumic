#pragma once

#include <memory.h>
#include <stdio.h>
#include <cmath>
#include <float.h>
#include <immintrin.h>
#include <nmmintrin.h>
#include <smmintrin.h>



namespace fastnum {
namespace cpu {
namespace kernel {

inline void argmax(float* max_val, int* max_index, float* source, int size) {
    float max = -FLT_MAX, index = 0;

    __m256 _v_p = _mm256_set1_ps(max);

    int i=0;
    for(i=0; i < size - 31; i += 32) {
        __m256 _v_s0 = _mm256_loadu_ps(source + i + 0);
        __m256 _v_s1 = _mm256_loadu_ps(source + i + 8);
        __m256 _v_s2 = _mm256_loadu_ps(source + i + 16);
        __m256 _v_s3 = _mm256_loadu_ps(source + i + 24);

        _v_s0 = _mm256_max_ps(_v_s0, _v_s1);
        _v_s2 = _mm256_max_ps(_v_s2, _v_s3);
        _v_s0 = _mm256_max_ps(_v_s0, _v_s2);

        __m256 _v_mask = _mm256_cmp_ps(_v_p, _v_s0, _CMP_GT_OS);

        if(! _mm256_testz_ps(_v_mask, _v_mask)) {
            index = i;
            for (int j = i; j < i + 32; j++) {
                max = (source[j] > max ? source[j] : max);
            }
            _v_p = _mm256_set1_ps(max);
        }
    }

    for (int ii = index; ii < index + 31; ++ii) {
        if (source[ii] == max) {
            if(max_val != nullptr) *max_val = max;
            *max_index = ii;
            break;
        }
    }

    for(; i < size; ++i) {
        if(source[i] > max) {
            max = source[i];
            if(max_val != nullptr) *max_val = max;
            *max_index = i;
        }
    }
}

inline void argmax(int* max_val, int* max_index, int* source, int size) {
    int max = INT_MIN, index = 0;

    __m256i _v_p = _mm256_set1_epi32(max);

    int i=0;
    for(i=0; i<size - 31; i += 32) {
        __m256i _v_s0 = _mm256_loadu_si256((__m256i*)(source + 0));
        __m256i _v_s1 = _mm256_loadu_si256((__m256i*)(source + 8));
        __m256i _v_s2 = _mm256_loadu_si256((__m256i*)(source + 16));
        __m256i _v_s3 = _mm256_loadu_si256((__m256i*)(source + 24));

        _v_s0 = _mm256_max_epi32(_v_s0, _v_s1);
        _v_s2 = _mm256_max_epi32(_v_s2, _v_s3);
        _v_s0 = _mm256_max_epi32(_v_s0, _v_s2);

        __m256i _v_mask = _mm256_cmpgt_epi32(_v_p, _v_s0);

        if (!_mm256_testz_si256(_v_mask, _v_mask)) {
            index = i;
            for (int j = i; j < i + 32; j++) {
                max = (source[j] > max ? source[j] : max);
            }
            _v_p = _mm256_set1_epi32(max);
        }
    }

    for (int ii = index; ii < index + 31; ++ii) {
        if (source[ii] == max) {
            if(max_val != nullptr) *max_val = max;
            *max_index = ii;
            break;
        }
    }

    for(; i < size; ++i) {
        if(source[i] > max) {
            max = source[i];
            if(max_val != nullptr) *max_val = max;
            *max_index = i;
        }
    }
}

}
}
}